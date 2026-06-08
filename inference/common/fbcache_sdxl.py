"""
Caching utilities for SDXL UNet2DConditionModel.
Implements forward pass with caching support for SDXL UNet.
**Main Functions**
- :func:`cached_forward_sdxl` : Forward pass for SDXL UNet with caching support.
"""
import torch
import functools
from diffusers import DiffusionPipeline
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers
from nunchaku.caching.fbcache import get_buffer, get_can_use_cache, set_buffer,cache_context, create_cache_context, get_current_cache_context
import math

def apply_cache_on_unet(unet, *, residual_diff_threshold=0.12, verbose=False):
    """
    Enable caching for a SDXL UNet2DConditionModel.
    This function wraps the UNet's forward method to use caching for faster inference.
    Uses single first-block caching with configurable similarity thresholds.
    Parameters
    ----------
    unet : UNet2DConditionModel
        The UNet to modify.
    residual_diff_threshold : float, optional
        Similarity threshold for caching (default: 0.12).
    verbose : bool, optional
        Print caching status messages (default: False).
    Returns
    -------
    UNet2DConditionModel
        The UNet with caching enabled.
    Notes
    -----
    If already cached, returns the UNet unchanged. Caching is only active within a cache context.
    """
    if getattr(unet, "_is_cached", False):
        return unet

    # Store original forward method
    unet._original_forward = unet.forward
    unet.residual_diff_threshold = residual_diff_threshold
    unet.verbose = verbose
    # Simple per-call stats (reset by apply_cache_on_pipe wrapper each pipeline call).
    if not hasattr(unet, "_fbcache_stats"):
        unet._fbcache_stats = {"hits": 0, "misses": 0, "diff_min": math.inf, "diff_max": 0.0}

    # Replace forward with cached version
    @functools.wraps(unet.forward)
    def new_forward(*args, **kwargs):
        cache_ctx = get_current_cache_context()
        if cache_ctx is not None:
            # Use cached forward
            return cached_forward_sdxl(unet, *args, **kwargs)
        else:
            # Use original forward
            return unet._original_forward(*args, **kwargs)

    unet.forward = new_forward
    unet._is_cached = True

    return unet


def apply_cache_on_pipe(pipe: DiffusionPipeline, *, residual_diff_threshold=0.12, verbose=False):
    """
    Enable caching for a complete SDXL diffusion pipeline.
    This function wraps the pipeline's ``__call__`` method to manage cache contexts,
    and applies UNet-level caching.
    Parameters
    ----------
    pipe : DiffusionPipeline
        The SDXL pipeline to modify.
    residual_diff_threshold : float, optional
        Similarity threshold for caching (default: 0.12).
    verbose : bool, optional
        Print caching status messages (default: False).
    Returns
    -------
    DiffusionPipeline
        The pipeline with caching enabled.
    Notes
    -----
    The pipeline is patched in an instance-isolated way (avoid global class pollution).
    """
    # Wrap pipeline __call__ with cache context (instance-isolated).
    # NOTE: special methods like __call__ are resolved on the *type*, not the instance dict,
    # so to avoid patching the shared class, we create a per-instance subclass and patch that.
    if not getattr(pipe, "_fbcache_call_isolated", False):
        base_cls = pipe.__class__
        original_call = base_cls.__call__

        patched_cls = type(f"{base_cls.__name__}FBCachePatched_{id(pipe)}", (base_cls,), {})
        # 记录原始 pipeline 信息，供外部（如推理参数构建器）在类名被 patch 时做兼容判断。
        # 注意：多个 runtime patch 可能叠加到同一个 pipe 上，这里要保留“根原始类”，
        # 同时为 FBCache 单独记录它自己的上一层类，避免回滚时串到别的 patch。
        try:
            root_base_cls = getattr(pipe, "_pipeline_base_class", None) or base_cls
            root_base_name = getattr(pipe, "_pipeline_base_class_name", None) or getattr(root_base_cls, "__name__", None)
            pipe._pipeline_base_class_name = root_base_name
            pipe._pipeline_base_class = root_base_cls
            pipe._fbcache_pipeline_base_class_name = getattr(base_cls, "__name__", None)
            pipe._fbcache_pipeline_base_class = base_cls
        except Exception:
            pass

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            # Reset stats per pipeline call to make it obvious whether caching is hitting.
            if hasattr(self, "unet") and hasattr(self.unet, "_fbcache_stats"):
                self.unet._fbcache_stats = {"hits": 0, "misses": 0, "diff_min": math.inf, "diff_max": 0.0}
            with cache_context(create_cache_context()):
                out = original_call(self, *args, **kwargs)
            if verbose and hasattr(self, "unet") and hasattr(self.unet, "_fbcache_stats"):
                s = self.unet._fbcache_stats
                total = max(1, int(s.get("hits", 0)) + int(s.get("misses", 0)))
                hit_rate = float(s.get("hits", 0)) / float(total)
                diff_min = s.get("diff_min", None)
                diff_max = s.get("diff_max", None)
                if isinstance(diff_min, (int, float)) and math.isinf(diff_min):
                    diff_min = None
                print(
                    f"[FBCache][SDXL] hits={s.get('hits', 0)} misses={s.get('misses', 0)} "
                    f"hit_rate={hit_rate:.2%} diff_min={diff_min} diff_max={diff_max}"
                )
            return out

        patched_cls.__call__ = new_call
        pipe.__class__ = patched_cls
        pipe._fbcache_call_isolated = True

    # Mark the instance (UNet has its own _is_cached flag).
    pipe._is_cached = True

    # Apply caching to UNet
    apply_cache_on_unet(pipe.unet, residual_diff_threshold=residual_diff_threshold, verbose=verbose)

    return pipe


def cached_forward_sdxl(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:
    r"""
    The [`UNet2DConditionModel`] forward method with caching support.
    Args:
        sample (`torch.Tensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.Tensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        class_labels (`torch.Tensor`, *optional*, defaults to `None`):
            Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
            Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
            through the `self.time_embedding` layer to obtain the timestep embeddings.
        attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
            A tuple of tensors that if specified are added to the residuals of down unet blocks.
        mid_block_additional_residual: (`torch.Tensor`, *optional*):
            A tensor that if specified is added to the residual of the middle unet block.
        down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
            tuple.
    Returns:
        [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
            otherwise a `tuple` is returned where the first element is the sample tensor.
    """
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
    # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
    if cross_attention_kwargs is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        lora_scale = cross_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)

    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
    is_adapter = down_intrablock_additional_residuals is not None
    # diffusers may pass tuples here; we mutate via pop(0), so ensure it's a list.
    if is_adapter and not isinstance(down_intrablock_additional_residuals, list):
        down_intrablock_additional_residuals = list(down_intrablock_additional_residuals)
    # maintain backward compatibility for legacy usage, where
    #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
    #       but can only use one or the other
    if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
        deprecate(
            "T2I should not use down_block_additional_residuals",
            "1.3.0",
            "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                   and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                   for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
            standard_warn=False,
        )
        down_intrablock_additional_residuals = down_block_additional_residuals
        is_adapter = True

    # 3.2 FBCache: Process first down_block
    # Keep the base residual (post conv_in) but delay building the full residual list until we know it's a miss.
    # Using list avoids repeated tuple concatenations on the miss path.
    down_block_res_samples_base = [sample]
    first_block = self.down_blocks[0]
    if hasattr(first_block, "has_cross_attention") and first_block.has_cross_attention:
        # For t2i-adapter CrossAttnDownBlock2D
        additional_residuals = {}
        if is_adapter and len(down_intrablock_additional_residuals) > 0:
            add_res = down_intrablock_additional_residuals.pop(0)
            if isinstance(add_res, torch.Tensor) and add_res.dtype != sample.dtype:
                add_res = add_res.to(dtype=sample.dtype)
            additional_residuals["additional_residuals"] = add_res

        sample, res_samples = first_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            encoder_attention_mask=encoder_attention_mask,
            **additional_residuals,
        )
    else:
        sample, res_samples = first_block(hidden_states=sample, temb=emb)
        if is_adapter and len(down_intrablock_additional_residuals) > 0:
            add_res = down_intrablock_additional_residuals.pop(0)
            if isinstance(add_res, torch.Tensor) and add_res.dtype != sample.dtype:
                add_res = add_res.to(dtype=sample.dtype)
            sample += add_res
    first_block_res_samples = res_samples

    # 3.3 FBCache: Check cache using first block output sample
    # 3.3 FBCache: Check cache using first block output sample
    # If the reference buffer isn't ready yet (first step in a fresh cache context),
    # some implementations return a sentinel `diff == threshold`, which would make
    # our diff_min "stick" to the threshold. Detect warmup explicitly and skip stats.
    prev_ref = get_buffer("first_single_hidden_states_residual")
    diff = None
    if prev_ref is None:
        can_use_cache = False
        if self.verbose:
            print("[FBCache][SDXL] warmup (no reference buffer yet)")
    else:
        can_use_cache, diff = get_can_use_cache(
            sample,
            threshold=self.residual_diff_threshold,
            parallelized=False,
            mode="single",
        )
        # NOTE: Some cache implementations use a strict `< threshold` comparison internally.
        # With float precision (fp16/fp32), a value that *prints* as exactly `threshold`
        # may still fail the strict check (e.g. 0.20000000298 vs 0.2). Add a tiny epsilon
        # so boundary cases can hit.
        try:
            d = float(diff)
            if (not can_use_cache) and d <= float(self.residual_diff_threshold) + 1e-6:
                can_use_cache = True
        except Exception:
            pass

    # 3.4 FBCache: Apply caching logic
    torch._dynamo.graph_break()
    if can_use_cache:
        target_dtype = sample.dtype
        target_device = sample.device
        if hasattr(self, "_fbcache_stats"):
            self._fbcache_stats["hits"] = int(self._fbcache_stats.get("hits", 0)) + 1
            try:
                if diff is not None:
                    d = float(diff)
                    self._fbcache_stats["diff_min"] = min(float(self._fbcache_stats.get("diff_min", math.inf)), d)
                    self._fbcache_stats["diff_max"] = max(float(self._fbcache_stats.get("diff_max", 0.0)), d)
            except Exception:
                pass
        if self.verbose:
            try:
                if diff is None:
                    print("[FBCache][SDXL] hit")
                else:
                    print(f"[FBCache][SDXL] hit diff={float(diff):.6f}")
            except Exception:
                print("[FBCache][SDXL] hit")
        # Skip all remaining computation and get final output (after post-process)
        sample = get_buffer("final_output")
        # Safety: if cache buffer isn't ready, fall back to full compute.
        if sample is None:
            can_use_cache = False
            if self.verbose:
                print("[FBCache][SDXL] cache buffer missing; fallback to miss")
        # 防止缓存实现/历史残留导致 dtype 不一致：Half/Float 混用会在 GroupNorm/Linear 里直接炸
        if can_use_cache and isinstance(sample, torch.Tensor):
            if sample.device != target_device:
                sample = sample.to(device=target_device)
            if sample.dtype != target_dtype:
                if self.verbose:
                    print(f"[FBCache][SDXL] cache output dtype cast: {sample.dtype} -> {target_dtype}")
                sample = sample.to(dtype=target_dtype)
    if not can_use_cache:
        if hasattr(self, "_fbcache_stats"):
            self._fbcache_stats["misses"] = int(self._fbcache_stats.get("misses", 0)) + 1
            try:
                if diff is not None:
                    d = float(diff)
                    self._fbcache_stats["diff_min"] = min(float(self._fbcache_stats.get("diff_min", math.inf)), d)
                    self._fbcache_stats["diff_max"] = max(float(self._fbcache_stats.get("diff_max", 0.0)), d)
            except Exception:
                pass
        if self.verbose:
            try:
                if diff is None:
                    print("[FBCache][SDXL] miss")
                else:
                    print(f"[FBCache][SDXL] miss diff={float(diff):.6f}")
            except Exception:
                print("[FBCache][SDXL] miss")
        # Store first block output for next comparison
        set_buffer("first_single_hidden_states_residual", sample)

        # Make a copy to preserve state
        down_intrablock_additional_residuals_copy = list(down_intrablock_additional_residuals) if is_adapter else []

        # Build residual samples only when we actually need them (miss path).
        down_block_res_samples = list(down_block_res_samples_base)
        down_block_res_samples.extend(first_block_res_samples)

        # Process remaining down blocks
        for downsample_block in self.down_blocks[1:]:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals_copy) > 0:
                    add_res = down_intrablock_additional_residuals_copy.pop(0)
                    if isinstance(add_res, torch.Tensor) and add_res.dtype != sample.dtype:
                        add_res = add_res.to(dtype=sample.dtype)
                    additional_residuals["additional_residuals"] = add_res

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals_copy) > 0:
                    add_res = down_intrablock_additional_residuals_copy.pop(0)
                    if isinstance(add_res, torch.Tensor) and add_res.dtype != sample.dtype:
                        add_res = add_res.to(dtype=sample.dtype)
                    sample += add_res

            down_block_res_samples.extend(res_samples)

        if is_controlnet:
            down_block_res_samples = [
                r + a for r, a in zip(down_block_res_samples, down_block_additional_residuals)
            ]

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals_copy) > 0
                and sample.shape == down_intrablock_additional_residuals_copy[0].shape
            ):
                add_res = down_intrablock_additional_residuals_copy.pop(0)
                if isinstance(add_res, torch.Tensor) and add_res.dtype != sample.dtype:
                    add_res = add_res.to(dtype=sample.dtype)
                sample += add_res

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            res_samples_tuple = tuple(res_samples)

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples_tuple,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples_tuple,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # Store final output (after post-process) for next inference
        set_buffer("final_output", sample)
    torch._dynamo.graph_break()

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)


def orig_forward_sdxl(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:
    r"""
    The [`UNet2DConditionModel`] forward method with caching support.
    Args:
        sample (`torch.Tensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.Tensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        class_labels (`torch.Tensor`, *optional*, defaults to `None`):
            Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
            Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
            through the `self.time_embedding` layer to obtain the timestep embeddings.
        attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
            A tuple of tensors that if specified are added to the residuals of down unet blocks.
        mid_block_additional_residual: (`torch.Tensor`, *optional*):
            A tensor that if specified is added to the residual of the middle unet block.
        down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
            tuple.
    Returns:
        [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
            otherwise a `tuple` is returned where the first element is the sample tensor.
    """
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
    # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
    if cross_attention_kwargs is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        lora_scale = cross_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)

    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
    is_adapter = down_intrablock_additional_residuals is not None
    # diffusers may pass tuples here; we mutate via pop(0), so ensure it's a list.
    if is_adapter and not isinstance(down_intrablock_additional_residuals, list):
        down_intrablock_additional_residuals = list(down_intrablock_additional_residuals)
    # maintain backward compatibility for legacy usage, where
    #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
    #       but can only use one or the other
    if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
        deprecate(
            "T2I should not use down_block_additional_residuals",
            "1.3.0",
            "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                   and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                   for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
            standard_warn=False,
        )
        down_intrablock_additional_residuals = down_block_additional_residuals
        is_adapter = True

    # Use list to avoid repeated tuple concatenations (Python overhead).
    down_block_res_samples = [sample]
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                add_res = down_intrablock_additional_residuals.pop(0)
                if isinstance(add_res, torch.Tensor) and add_res.dtype != sample.dtype:
                    add_res = add_res.to(dtype=sample.dtype)
                additional_residuals["additional_residuals"] = add_res

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                add_res = down_intrablock_additional_residuals.pop(0)
                if isinstance(add_res, torch.Tensor) and add_res.dtype != sample.dtype:
                    add_res = add_res.to(dtype=sample.dtype)
                sample += add_res

        down_block_res_samples.extend(res_samples)

    if is_controlnet:
        down_block_res_samples = [
            r + a for r, a in zip(down_block_res_samples, down_block_additional_residuals)
        ]

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)

        # To support T2I-Adapter-XL
        if (
            is_adapter
            and len(down_intrablock_additional_residuals) > 0
            and sample.shape == down_intrablock_additional_residuals[0].shape
        ):
            add_res = down_intrablock_additional_residuals.pop(0)
            if isinstance(add_res, torch.Tensor) and add_res.dtype != sample.dtype:
                add_res = add_res.to(dtype=sample.dtype)
            sample += add_res

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
        res_samples_tuple = tuple(res_samples)

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples_tuple,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples_tuple,
                upsample_size=upsample_size,
            )

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)