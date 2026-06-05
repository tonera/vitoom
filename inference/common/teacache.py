import inspect
import torch
from types import MethodType
from typing import Any, Callable, Dict, Optional, Tuple, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)

_COEFFICIENTS_BY_MODEL: dict = {
    # Flux / Flux-derivatives
    "flux": [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01],
    "flux-kontext": [-1.04655119e03, 3.12563399e02, -1.69500694e01, 4.10995971e-01, 3.74537863e-02],
    # NOTE: Chroma is a Flux-derivative in nunchaku; start with Flux coefficients as a reasonable default.
    "chroma": [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01],
    # QwenImage: default to identity (raw rel-L1) to avoid over-amplifying small changes.
    "qwenimage": None,
    # Debug/tuning option: no polynomial rescale, use raw relative L1 directly.
    "identity": None,
}


def _teacache_scaled_rel_l1(rel: float, *, load_name: str) -> float:
    coeffs = _COEFFICIENTS_BY_MODEL.get(load_name, None)
    if coeffs is None:
        return float(rel)
    return float(np.abs(np.poly1d(coeffs)(float(rel))))


def _extract_kw_or_pos(name: str, args: tuple, kwargs: dict, pos_idx: int) -> Any:
    if name in kwargs:
        return kwargs.get(name)
    if len(args) > pos_idx:
        return args[pos_idx]
    return None


def _to_scalar_timestep(timestep: Any) -> Optional[float]:
    try:
        if timestep is None:
            return None
        if isinstance(timestep, (int, float)):
            return float(timestep)
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                return None
            return float(timestep.reshape(-1)[0].detach().cpu().item())
        return None
    except Exception:
        return None


def _flux_single_blocks_separate_encoder_states(transformer: Any) -> bool:
    """
    Detect the native diffusers Flux single-block convention.

    Native diffusers Flux models pass `encoder_hidden_states` to each single block and
    receive `(encoder_hidden_states, hidden_states)` back. Older / nunchaku-style
    implementations operate on concatenated hidden states only.
    """
    cached = getattr(transformer, "_teacache_flux_single_blocks_separate_encoder_states", None)
    if cached is not None:
        return bool(cached)

    use_separate = False
    try:
        blocks = getattr(transformer, "single_transformer_blocks", None)
        if blocks is not None and len(blocks) > 0:
            params = inspect.signature(blocks[0].forward).parameters
            use_separate = "encoder_hidden_states" in params
    except Exception:
        use_separate = False

    try:
        setattr(transformer, "_teacache_flux_single_blocks_separate_encoder_states", bool(use_separate))
    except Exception:
        pass
    return bool(use_separate)


def _run_flux_single_transformer_blocks(
    self: Any,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Any,
    joint_attention_kwargs: Optional[Dict[str, Any]],
    controlnet_single_block_samples: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_separate = _flux_single_blocks_separate_encoder_states(self)

    if not use_separate:
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            if use_separate:
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

        else:
            if use_separate:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            if use_separate:
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]
            else:
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

    if not use_separate:
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    return encoder_hidden_states, hidden_states


def make_generic_teacache_wrapper(
    *,
    original_forward: Callable[..., Any],
    num_steps: int,
    rel_l1_thresh: float,
    skip_steps: int,
    load_name: str,
) -> Callable[..., Any]:
    """
    Generic TeaCache wrapper for transformer-like models.

    It does NOT assume a specific model architecture: it wraps the original forward and
    skips the call when the input hidden_states change is small, reusing an additive residual:
        out ≈ hidden_states + prev_residual

    This works best when the model output has the same shape as hidden_states (common for diffusion noise_pred heads).
    If shapes mismatch, it will fall back to full compute.
    """

    def _pack_like(prev_out: Any, sample: torch.Tensor) -> Any:
        try:
            if isinstance(prev_out, Transformer2DModelOutput):
                return Transformer2DModelOutput(sample=sample)
            if isinstance(prev_out, tuple):
                if len(prev_out) == 0:
                    return (sample,)
                return (sample,) + tuple(prev_out[1:])
            if isinstance(prev_out, torch.Tensor):
                return sample
            # best-effort: objects with .sample attr
            if hasattr(prev_out, "sample"):
                try:
                    return type(prev_out)(sample=sample)
                except Exception:
                    pass
            return sample
        except Exception:
            return sample

    def teacache_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        enable = bool(getattr(self, "enable_teacache", True))
        if not enable:
            return original_forward(*args, **kwargs)

        # Extract key tensors (best-effort).
        hidden_states = _extract_kw_or_pos("hidden_states", args, kwargs, 0)
        timestep = kwargs.get("timestep", None)
        if timestep is None:
            # common naming variants
            timestep = kwargs.get("timesteps", None)
        t0 = _to_scalar_timestep(timestep)

        if not isinstance(hidden_states, torch.Tensor):
            # Can't do caching without a tensor baseline.
            return original_forward(*args, **kwargs)

        # Per-branch cache context key (avoid CFG mixing).
        enc = kwargs.get("encoder_hidden_states", None)
        attn_mask = kwargs.get("attention_mask", None)
        enc_mask = kwargs.get("encoder_hidden_states_mask", None)
        guidance = kwargs.get("guidance", None)
        ctx_key = (
            int(enc.data_ptr()) if isinstance(enc, torch.Tensor) else id(enc),
            int(attn_mask.data_ptr()) if isinstance(attn_mask, torch.Tensor) else id(attn_mask),
            int(enc_mask.data_ptr()) if isinstance(enc_mask, torch.Tensor) else id(enc_mask),
            int(guidance.data_ptr()) if isinstance(guidance, torch.Tensor) else id(guidance),
        )

        teacache_ctx = getattr(self, "_generic_teacache_ctx", None)
        if teacache_ctx is None:
            teacache_ctx = {}
            setattr(self, "_generic_teacache_ctx", teacache_ctx)
        st = teacache_ctx.get(ctx_key)
        if st is None:
            st = {
                "step_idx": -1,
                "last_t": None,
                "run_last_t": None,
                "acc": 0.0,
                "prev_probe": None,
                "prev_res": None,
                "prev_out": None,
            }
            teacache_ctx[ctx_key] = st

        # Run boundary detection: timesteps often decrease. Jump up => new run.
        if t0 is not None:
            if st["run_last_t"] is not None and t0 > float(st["run_last_t"]):
                st["step_idx"] = -1
                st["acc"] = 0.0
                st["prev_probe"] = None
                st["prev_res"] = None
                st["prev_out"] = None
                st["last_t"] = None
            st["run_last_t"] = t0
            if st["last_t"] is None or t0 != float(st["last_t"]):
                st["step_idx"] += 1
                st["last_t"] = t0
        else:
            # Fallback: count calls (may double-count under CFG, but still safe).
            st["step_idx"] += 1

        step_idx = int(st["step_idx"])
        if num_steps <= 0:
            # No sequence boundary info; keep behavior correct (no skip).
            return original_forward(*args, **kwargs)

        probe = hidden_states
        should_calc: bool
        if step_idx <= 0 or step_idx >= (num_steps - 1) or st["prev_probe"] is None:
            should_calc = True
            st["acc"] = 0.0
        else:
            denom = float(st["prev_probe"].abs().mean().detach().cpu().item())
            denom = max(denom, 1e-8)
            rel = float(((probe - st["prev_probe"]).abs().mean() / denom).detach().cpu().item())
            scaled = _teacache_scaled_rel_l1(rel, load_name=load_name)
            st["acc"] += float(scaled)
            if st["acc"] < float(rel_l1_thresh):
                should_calc = False
            else:
                should_calc = True
                st["acc"] = 0.0

        st["prev_probe"] = probe.detach()

        if (
            step_idx > int(skip_steps)
            and (not should_calc)
            and isinstance(st.get("prev_res"), torch.Tensor)
            and st.get("prev_out") is not None
        ):
            prev_res = st["prev_res"]
            if isinstance(prev_res, torch.Tensor) and prev_res.shape == hidden_states.shape:
                sample = hidden_states + prev_res.to(device=hidden_states.device, dtype=hidden_states.dtype)
                return _pack_like(st["prev_out"], sample)

        out = original_forward(*args, **kwargs)

        # Extract sample tensor for residual cache.
        sample: Optional[torch.Tensor] = None
        if isinstance(out, Transformer2DModelOutput):
            sample = out.sample
        elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            sample = out[0]
        elif isinstance(out, torch.Tensor):
            sample = out

        if isinstance(sample, torch.Tensor) and sample.shape == hidden_states.shape:
            st["prev_res"] = (sample - hidden_states).detach()
            st["prev_out"] = out
        else:
            st["prev_res"] = None
            st["prev_out"] = out

        return out

    return teacache_forward


def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        if self.enable_teacache:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                load_name = str(getattr(self, "teacache_load_name", "flux") or "flux")
                denom = float(self.previous_modulated_input.abs().mean().detach().cpu().item())
                denom = max(denom, 1e-8)
                rel = float(((modulated_inp-self.previous_modulated_input).abs().mean() / denom).detach().cpu().item())
                self.accumulated_rel_l1_distance += _teacache_scaled_rel_l1(rel, load_name=load_name)
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp 
            self.cnt += 1 
            if self.cnt == self.num_steps:
                self.cnt = 0           
        
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for index_block, block in enumerate(self.transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                encoder_hidden_states, hidden_states = _run_flux_single_transformer_blocks(
                    self,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                )
                self.previous_residual = hidden_states - ori_hidden_states
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            encoder_hidden_states, hidden_states = _run_flux_single_transformer_blocks(
                self,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_single_block_samples=controlnet_single_block_samples,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

def chroma_teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    img_ids: Optional[torch.Tensor] = None,
    txt_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples: Optional[torch.Tensor] = None,
    controlnet_single_block_samples: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    TeaCache forward for `NunchakuChromaTransformer2dModel`.

    Notes:
    - Chroma forward signature differs from Flux (no pooled_projections/guidance).
    - Chroma may be called twice per diffusion step under classic CFG; this implementation uses per-branch cache state
      keyed by (encoder_hidden_states, attention_mask) pointer identity to avoid mixing unconditional/conditional branches.
    """
    # Chroma-specific utilities are only needed when this forward is actually used.
    from nunchaku.models.embeddings import pack_rotemb  # type: ignore
    from nunchaku.utils import pad_tensor  # type: ignore

    if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
        raise NotImplementedError("TeaCache(Chroma): ControlNet is not supported for NunchakuChromaTransformer2dModel")
    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        raise NotImplementedError("TeaCache(Chroma): IP-Adapter is not supported for NunchakuChromaTransformer2dModel")

    if txt_ids is None or img_ids is None or timestep is None or encoder_hidden_states is None:
        raise ValueError("TeaCache(Chroma) requires encoder_hidden_states/timestep/txt_ids/img_ids to be set.")

    teacache_ctx = getattr(self, "_nunchaku_teacache_ctx", None)
    if teacache_ctx is None:
        teacache_ctx = {}
        setattr(self, "_nunchaku_teacache_ctx", teacache_ctx)

    # Key by pointer identity (cheap) to separate CFG branches.
    e_ptr = int(encoder_hidden_states.data_ptr())
    m_ptr = int(attention_mask.data_ptr()) if attention_mask is not None else 0
    ctx_key = (e_ptr, m_ptr)
    st = teacache_ctx.get(ctx_key)
    if st is None:
        st = {"step_idx": -1, "last_t": None, "run_last_t": None, "acc": 0.0, "prev_mod": None, "prev_res": None}
        teacache_ctx[ctx_key] = st

    # Detect run boundary: timesteps usually decrease; a jump upwards indicates a new generation run.
    try:
        t0 = float(timestep.reshape(-1)[0].item())
    except Exception:
        t0 = None
    if t0 is not None:
        if st["run_last_t"] is not None and t0 > float(st["run_last_t"]):
            st["step_idx"] = -1
            st["acc"] = 0.0
            st["prev_mod"] = None
            st["prev_res"] = None
            st["last_t"] = None
        st["run_last_t"] = t0
        if st["last_t"] is None or t0 != float(st["last_t"]):
            st["step_idx"] += 1
            st["last_t"] = t0
    step_idx = int(st["step_idx"])

    enable = bool(getattr(self, "enable_teacache", True))
    num_steps = int(getattr(self, "num_steps", 0) or 0)
    rel_l1_thresh = float(getattr(self, "rel_l1_thresh", 0.6))
    skip_steps = int(getattr(self, "skip_steps", 0) or 0)
    load_name = str(getattr(self, "teacache_load_name", "chroma") or "chroma")

    hidden_states = self.x_embedder(hidden_states)
    baseline_hidden_states = hidden_states.clone()

    timestep_ = timestep.to(hidden_states.dtype) * 1000
    batch_size = int(hidden_states.shape[0])

    input_vec = self.time_text_embed(timestep_)
    pooled_temb = self.distilled_guidance_layer(input_vec)

    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    txt_tokens = int(encoder_hidden_states.shape[1])
    img_tokens = int(hidden_states.shape[1])
    attn_mask_1d = attention_mask
    valid_txt = None

    # Prepare packed RoPE for nunchaku fused kernels (copied from model forward).
    assert image_rotary_emb.ndim == 6
    assert image_rotary_emb.shape[0] == 1
    assert image_rotary_emb.shape[1] == 1
    assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
    image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
    rotary_emb_txt = pack_rotemb(pad_tensor(image_rotary_emb[:, :txt_tokens, ...], 256, 1))
    rotary_emb_img = pack_rotemb(pad_tensor(image_rotary_emb[:, txt_tokens:, ...], 256, 1))
    rotary_emb_single = pack_rotemb(pad_tensor(image_rotary_emb, 256, 1))

    if batch_size != int(rotary_emb_txt.shape[0]):
        rotary_emb_txt = rotary_emb_txt.expand(batch_size, -1, -1).contiguous()
    if batch_size != int(rotary_emb_img.shape[0]):
        rotary_emb_img = rotary_emb_img.expand(batch_size, -1, -1).contiguous()
    if batch_size != int(rotary_emb_single.shape[0]):
        rotary_emb_single = rotary_emb_single.expand(batch_size, -1, -1).contiguous()

    # TEA decision uses the image stream modulation from the first dual block.
    num_layers = len(self.transformer_blocks)
    num_single = len(self.single_transformer_blocks)
    img_offset = 3 * num_single
    txt_offset = img_offset + 6 * num_layers

    if enable and num_steps > 0:
        temb0 = torch.cat(
            (pooled_temb[:, img_offset : img_offset + 6], pooled_temb[:, txt_offset : txt_offset + 6]),
            dim=1,
        )
        temb0_img = temb0[:, :6].clone()
        inp0 = baseline_hidden_states.clone()
        modulated_inp, *_ = self.transformer_blocks[0].norm1(inp0, emb=temb0_img)

        if step_idx <= 0 or step_idx >= (num_steps - 1) or st["prev_mod"] is None:
            should_calc = True
            st["acc"] = 0.0
        else:
            denom = float(st["prev_mod"].abs().mean().cpu().item())
            denom = max(denom, 1e-8)
            rel = float(((modulated_inp - st["prev_mod"]).abs().mean() / denom).cpu().item())
            scaled = _teacache_scaled_rel_l1(rel, load_name=load_name)
            st["acc"] += float(scaled)
            if st["acc"] < rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                st["acc"] = 0.0
        st["prev_mod"] = modulated_inp
    else:
        should_calc = True

    # Skip heavy blocks if allowed and residual available.
    if enable and step_idx > skip_steps and (not should_calc) and st["prev_res"] is not None:
        hidden_states = baseline_hidden_states + st["prev_res"]
    else:
        # Full forward (copied from model forward), then update residual cache.
        import math
        import os

        use_cpp_ws = (
            os.getenv("NUNCHAKU_CHROMA_USE_CPP_ADDITIVE_ATTN", "1") == "1"
            and hidden_states.is_cuda
            and attention_mask is not None
            and int(self.config.attention_head_dim) == 128
        )
        ws_dual = None
        ws_single = None
        mask_dual = None
        mask_single = None
        if use_cpp_ws:
            heads = int(self.config.num_attention_heads)
            head_dim = int(self.config.attention_head_dim)
            pad_size = 256
            txt_pad = int(math.ceil(txt_tokens / pad_size) * pad_size)
            img_pad = int(math.ceil(img_tokens / pad_size) * pad_size)
            num_tokens_pad = int(txt_pad + img_pad)

            key_dual = (batch_size, num_tokens_pad, heads, head_dim, str(hidden_states.device), hidden_states.dtype)
            ws_dual = getattr(self, "_nunchaku_cpp_ws_dual_shared", None)
            if ws_dual is None or ws_dual.get("key") != key_dual:
                ws_dual = {
                    "key": key_dual,
                    "q": torch.empty((batch_size, heads, num_tokens_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "k": torch.empty((batch_size, heads, num_tokens_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "v": torch.empty((batch_size, heads, num_tokens_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "m": torch.empty((batch_size, num_tokens_pad), device=hidden_states.device, dtype=torch.float16),
                    "out": torch.empty((batch_size, num_tokens_pad, heads * head_dim), device=hidden_states.device, dtype=hidden_states.dtype),
                }
                self._nunchaku_cpp_ws_dual_shared = ws_dual

            s_total = int(txt_tokens + img_tokens)
            s_pad = int(math.ceil(s_total / pad_size) * pad_size)
            key_single = (batch_size, s_pad, heads, head_dim, str(hidden_states.device), hidden_states.dtype)
            ws_single = getattr(self, "_nunchaku_cpp_ws_single_shared", None)
            if ws_single is None or ws_single.get("key") != key_single:
                ws_single = {
                    "key": key_single,
                    "q": torch.empty((batch_size, heads, s_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "k": torch.empty((batch_size, heads, s_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "v": torch.empty((batch_size, heads, s_pad, head_dim), device=hidden_states.device, dtype=torch.float16),
                    "m": torch.empty((batch_size, s_pad), device=hidden_states.device, dtype=torch.float16),
                    "out": torch.empty((batch_size, s_pad, heads * head_dim), device=hidden_states.device, dtype=hidden_states.dtype),
                }
                self._nunchaku_cpp_ws_single_shared = ws_single

            if attention_mask is not None:
                attn_mask_fp16 = attention_mask.to(dtype=torch.float16)
                mask_single = ws_single["m"]
                mask_single.zero_()
                s_total = int(txt_tokens + img_tokens)
                mask_single[:, :s_total] = attn_mask_fp16

                mask_dual = ws_dual["m"]
                mask_dual.zero_()
                mask_dual[:, :txt_tokens] = attn_mask_fp16[:, :txt_tokens]
                mask_dual[:, txt_pad : txt_pad + img_tokens] = attn_mask_fp16[:, txt_tokens : txt_tokens + img_tokens]

        for i, block in enumerate(self.transformer_blocks):
            img_mod = img_offset + 6 * i
            txt_mod = txt_offset + 6 * i
            temb_i = torch.cat(
                (pooled_temb[:, img_mod : img_mod + 6], pooled_temb[:, txt_mod : txt_mod + 6]),
                dim=1,
            )
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb_i,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                attention_mask=None,
                valid_txt=valid_txt,
                joint_attention_kwargs=joint_attention_kwargs,
                attention_mask_1d=attn_mask_1d,
                cpp_workspace=ws_dual,
                cpp_mask=mask_dual,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for i, block in enumerate(self.single_transformer_blocks):
            start = 3 * i
            temb_i = pooled_temb[:, start : start + 3]
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb_i,
                image_rotary_emb=rotary_emb_single,
                attention_mask=None,
                attention_mask_1d=attn_mask_1d,
                txt_tokens=txt_tokens,
                valid_txt=valid_txt,
                joint_attention_kwargs=joint_attention_kwargs,
                cpp_workspace=ws_single,
                cpp_mask=mask_single,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        if enable:
            st["prev_res"] = hidden_states - baseline_hidden_states

    temb_out = pooled_temb[:, -2:]
    hidden_states = self.norm_out(hidden_states, temb_out)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def qwenimage_teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes=None,
    txt_seq_lens=None,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    TeaCache forward for `NunchakuQwenImageTransformer2DModel` (dual-stream img/txt).
    """
    teacache_ctx = getattr(self, "_nunchaku_teacache_ctx_qwenimage", None)
    if teacache_ctx is None:
        teacache_ctx = {}
        setattr(self, "_nunchaku_teacache_ctx_qwenimage", teacache_ctx)

    e_ptr = int(encoder_hidden_states.data_ptr()) if encoder_hidden_states is not None else 0
    m_ptr = int(encoder_hidden_states_mask.data_ptr()) if encoder_hidden_states_mask is not None else 0
    g_ptr = int(guidance.data_ptr()) if guidance is not None else 0
    ctx_key = (e_ptr, m_ptr, g_ptr)
    st = teacache_ctx.get(ctx_key)
    if st is None:
        st = {
            "step_idx": -1,
            "last_t": None,
            "run_last_t": None,
            "acc": 0.0,
            "prev_mod": None,
            "prev_res_img": None,
            "prev_res_txt": None,
        }
        teacache_ctx[ctx_key] = st

    # Run boundary detection.
    try:
        t0 = float(timestep.reshape(-1)[0].item())
    except Exception:
        t0 = None
    if t0 is not None:
        if st["run_last_t"] is not None and t0 > float(st["run_last_t"]):
            st["step_idx"] = -1
            st["acc"] = 0.0
            st["prev_mod"] = None
            st["prev_res_img"] = None
            st["prev_res_txt"] = None
            st["last_t"] = None
        st["run_last_t"] = t0
        if st["last_t"] is None or t0 != float(st["last_t"]):
            st["step_idx"] += 1
            st["last_t"] = t0
    step_idx = int(st["step_idx"])

    enable = bool(getattr(self, "enable_teacache", True))
    num_steps = int(getattr(self, "num_steps", 0) or 0)
    rel_l1_thresh = float(getattr(self, "rel_l1_thresh", 0.6))
    skip_steps = int(getattr(self, "skip_steps", 0) or 0)
    load_name = str(getattr(self, "teacache_load_name", "qwenimage") or "qwenimage")

    # Conservative: skip path won't apply per-block ControlNet residuals.
    force_calc = controlnet_block_samples is not None

    device = hidden_states.device
    if getattr(self, "offload", False):
        self.offload_manager.set_device(device)

    hidden_states = self.img_in(hidden_states)
    baseline_hidden_states = hidden_states.clone()

    timestep_ = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)
    baseline_encoder_hidden_states = encoder_hidden_states.clone()

    if guidance is not None:
        guidance_ = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance_ = None

    temb = (
        self.time_text_embed(timestep_, hidden_states)
        if guidance_ is None
        else self.time_text_embed(timestep_, guidance_, hidden_states)
    )

    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

    if enable and num_steps > 0 and (not force_calc):
        # TEA decision using first block's image modulation (closest to Flux modulated_inp).
        block0 = self.transformer_blocks[0]
        img_mod_params = block0.img_mod(temb)
        img_mod_params = img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(
            img_mod_params.shape[0], -1
        )
        img_mod1, _img_mod2 = img_mod_params.chunk(2, dim=-1)
        img_normed = block0.img_norm1(baseline_hidden_states)
        modulated_inp, _gate = block0._modulate(img_normed, img_mod1)

        if step_idx <= 0 or step_idx >= (num_steps - 1) or st["prev_mod"] is None:
            should_calc = True
            st["acc"] = 0.0
        else:
            denom = float(st["prev_mod"].abs().mean().cpu().item())
            denom = max(denom, 1e-8)
            rel = float(((modulated_inp - st["prev_mod"]).abs().mean() / denom).cpu().item())
            scaled = _teacache_scaled_rel_l1(rel, load_name=load_name)
            st["acc"] += float(scaled)
            if st["acc"] < rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                st["acc"] = 0.0
        st["prev_mod"] = modulated_inp
    else:
        should_calc = True

    if (
        enable
        and step_idx > skip_steps
        and (not should_calc)
        and st["prev_res_img"] is not None
        and st["prev_res_txt"] is not None
    ):
        hidden_states = baseline_hidden_states + st["prev_res_img"]
        encoder_hidden_states = baseline_encoder_hidden_states + st["prev_res_txt"]
    else:
        compute_stream = torch.cuda.current_stream() if hidden_states.is_cuda else None
        if getattr(self, "offload", False) and compute_stream is not None:
            self.offload_manager.initialize(compute_stream)

        for block_idx, block in enumerate(self.transformer_blocks):
            if compute_stream is not None:
                with torch.cuda.stream(compute_stream):
                    if getattr(self, "offload", False):
                        block = self.offload_manager.get_block(block_idx)

                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                            block,
                            hidden_states,
                            encoder_hidden_states,
                            encoder_hidden_states_mask,
                            temb,
                            image_rotary_emb,
                        )
                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=attention_kwargs,
                        )

                    if controlnet_block_samples is not None:
                        interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states = hidden_states + controlnet_block_samples[block_idx // interval_control]

                if getattr(self, "offload", False):
                    self.offload_manager.step(compute_stream)
            else:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        image_rotary_emb,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )

                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_block_samples[block_idx // interval_control]

        if enable:
            st["prev_res_img"] = hidden_states - baseline_hidden_states
            st["prev_res_txt"] = encoder_hidden_states - baseline_encoder_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def set_tea_cache(
    transformer: Any,
    num_inference_steps: int,
    rel_l1_thresh: float = 0.6,
    skip_steps: int = 0,
    *,
    load_name: Optional[str] = None,
) -> Any:
    """
    Enable TeaCache on a transformer model in-place.

    Backward compatible with old usage:
        - caller may patch class forward manually (Flux path),
        - and then call `set_tea_cache(transformer, steps)`.

    New behavior:
        - auto-select and bind the appropriate TeaCache forward for Flux / Chroma / QwenImage.
    """
    if load_name is None:
        load_name = "flux"
    load_name = str(load_name or "flux")

    # Store configuration on BOTH instance and class for compatibility with existing callers.
    # (Instance attrs take precedence during attribute lookup.)
    for obj in (transformer, transformer.__class__):
        try:
            setattr(obj, "enable_teacache", True)
            setattr(obj, "num_steps", int(num_inference_steps))
            setattr(obj, "rel_l1_thresh", float(rel_l1_thresh))
            setattr(obj, "skip_steps", int(skip_steps))
            setattr(obj, "teacache_load_name", str(load_name))
            # Flux-style state (some impls will override with their own dict states)
            setattr(obj, "cnt", 0)
            setattr(obj, "accumulated_rel_l1_distance", 0)
            setattr(obj, "previous_modulated_input", None)
            setattr(obj, "previous_residual", None)
        except Exception:
            pass

    # Bind per-instance forward (avoid global class pollution).
    try:
        # Prevent double-wrapping: remember the base forward once.
        if not hasattr(transformer, "_teacache_base_forward"):
            setattr(transformer, "_teacache_base_forward", transformer.forward)

        mn = str(load_name or "").strip().lower()
        if mn in {"chroma", "qwenimage"}:
            if not getattr(transformer, "_teacache_generic_wrapped", False):
                base = getattr(transformer, "_teacache_base_forward")
                transformer.forward = MethodType(
                    make_generic_teacache_wrapper(
                        original_forward=base,
                        num_steps=int(num_inference_steps),
                        rel_l1_thresh=float(rel_l1_thresh),
                        skip_steps=int(skip_steps),
                        load_name=mn,
                    ),
                    transformer,
                )
                setattr(transformer, "_teacache_generic_wrapped", True)
        else:
            if not getattr(transformer, "_teacache_flux_wrapped", False):
                transformer.forward = MethodType(teacache_forward, transformer)
                setattr(transformer, "_teacache_flux_wrapped", True)
    except Exception:
        # Best-effort: if binding fails, keep correctness (no caching) rather than crashing.
        try:
            setattr(transformer, "enable_teacache", False)
        except Exception:
            pass

    return transformer