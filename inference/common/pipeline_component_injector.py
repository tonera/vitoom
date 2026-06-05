from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import re

from common.family_utils import to_model_family
from common.hf_weight_detector import infer_variant_and_use_safetensors
from common.Constant import FAMILY_ALLOWED_KEYS, FAMILY_SINGLEFILE_POLICY
from common.model_metadata import (
    read_int_from_json,
    safetensors_find_first_shape,
    safetensors_has_any_prefixes,
)


ComponentOverrides = dict[str, Any]
LazyComponentLoader = Callable[[], Any]


@dataclass(frozen=True)
class ComponentContext:
    family: str
    inference_config: Any
    device_plan: Any
    logger: Any
    model_info: Any
    num_inference_steps: int
    fast_mode: bool
    low_vram: bool

    @property
    def models_dir(self) -> str:
        return str(getattr(self.inference_config, "models_dir", "") or "")

    @property
    def weights_dir(self) -> str:
        return str(getattr(self.inference_config, "weights_dir", "") or "")

    @property
    def load_name(self) -> str:
        return str(getattr(self.model_info, "load_name", "") or "")


def _resolve_component_path(value: object, *, models_dir: str, weights_dir: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Empty component path: value={value!r}")
    p = Path(raw)
    if p.is_absolute():
        return p
    cand = Path(models_dir) / raw
    if cand.exists():
        return cand
    return Path(weights_dir) / raw


def _register_lazy_component_loader(
    overrides: ComponentOverrides,
    *,
    key: str,
    loader: LazyComponentLoader,
) -> None:
    lazy_loaders = overrides.get("__lazy_component_loaders")
    if not isinstance(lazy_loaders, dict):
        lazy_loaders = {}
        overrides["__lazy_component_loaders"] = lazy_loaders
    lazy_loaders[key] = loader


def _platform_nunchaku_precision() -> Optional[str]:
    """
    Return platform-supported nunchaku precision ("fp4"/"int4") if available.
    """
    try:
        from nunchaku.utils import get_precision  # type: ignore

        p = str(get_precision() or "").strip().lower()
    except Exception:
        return None
    if "fp4" in p:
        return "fp4"
    if "int4" in p:
        return "int4"
    return None


def _maybe_subfolder_root(root: Path, subfolder: str) -> tuple[Path, Optional[str]]:
    # 约定：优先使用 repo 根 + subfolder；若传入的就是子目录，也允许直接用
    if (root / subfolder).is_dir():
        return root / subfolder, subfolder
    return root, None


_SVT_GENERIC_RE = re.compile(r"^svdq-(?P<prec>fp4|int4)_r(?P<r>\d+)-.+\.safetensors$", re.IGNORECASE)
_NUNCHAKU_TEXT_RE = re.compile(r"^(?P<kind>svdq|awq)-(?P<prec>fp4|int4)-.+\.safetensors$", re.IGNORECASE)


def _candidate_from_existing_paths(paths: list[Path], *, preferred_precision: Optional[str]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    if preferred_precision in {"fp4", "int4"}:
        preferred = [p for p in existing if preferred_precision in p.name.lower()]
        if preferred:
            existing = preferred
    return sorted(existing, key=lambda p: str(p))[0]


def _model_dir_for_auto_detect(model_info: Any) -> Path:
    repo_path = Path(str(getattr(model_info, "repo_id", "") or ""))
    if repo_path.is_dir():
        return repo_path
    return repo_path.parent


def _load_name_candidates_for_auto_detect(ctx: ComponentContext) -> set[str]:
    names: set[str] = set()
    raw_values = [
        ctx.load_name,
        str(getattr(ctx.model_info, "repo_id", "") or ""),
    ]
    for raw in raw_values:
        s = str(raw or "").strip()
        if not s:
            continue
        names.add(s.lower())
        try:
            p = Path(s)
            if p.name:
                names.add(p.name.lower())
            if p.stem:
                names.add(p.stem.lower())
        except Exception:
            pass
    return names


def _project_component_config_json(*, family: str, component: str) -> Optional[Path]:
    inference_dir = Path(__file__).resolve().parents[1]
    p = inference_dir / "config" / str(family) / str(component) / "config.json"
    return p if p.is_file() else None


def _probe_flux2_klein_expected_context_in(ctx: ComponentContext) -> tuple[Optional[int], str]:
    repo_path = Path(str(getattr(ctx.model_info, "repo_id", "") or ""))
    method = str(getattr(ctx.model_info, "method", "") or "")

    if method != "from_single_file":
        detect_dir, _ = _maybe_subfolder_root(repo_path, "transformer")
        cfg_path = detect_dir / "config.json"
        v = read_int_from_json(cfg_path, "joint_attention_dim")
        if v is not None:
            return v, f"transformer_config:{str(cfg_path)}"

    if repo_path.is_file() and repo_path.suffix.lower() == ".safetensors":
        shape, key = safetensors_find_first_shape(
            repo_path,
            (
                "context_embedder.weight",
                "model.diffusion_model.context_embedder.weight",
                "transformer.context_embedder.weight",
            ),
        )
        if shape is not None and len(shape) == 2 and shape[1] > 0 and key:
            return int(shape[1]), f"single_file:{key}"

    return None, "unknown"


def _probe_flux2_klein_candidate_hidden_size(ctx: ComponentContext, ckpt: Path) -> tuple[Optional[int], str]:
    model_root = _model_dir_for_auto_detect(ctx.model_info)
    candidate_paths = []

    if ckpt.is_dir():
        candidate_paths.extend(
            [
                ckpt / "config.json",
                ckpt / "text_encoder" / "config.json",
            ]
        )
    else:
        candidate_paths.extend(
            [
                ckpt.parent / "config.json",
                ckpt.parent / "text_encoder" / "config.json",
            ]
        )

    project_cfg = _project_component_config_json(family=ctx.family, component="text_encoder")
    if project_cfg is not None:
        candidate_paths.append(project_cfg)

    # 只有当候选本身就在当前模型目录内时，才允许退回读取目标模型自己的 text_encoder config。
    # 对外部 fallback 候选（如全局 weights/Qwen3-text-Nunchaku），绝不能误用目标模型自带的 config。
    try:
        if model_root.is_dir() and str(ckpt).startswith(str(model_root) + "/"):
            candidate_paths.extend(
                [
                    model_root / "text_encoder" / "config.json",
                    model_root / "config.json",
                ]
            )
    except Exception:
        pass

    seen: set[str] = set()
    for path in candidate_paths:
        sp = str(path)
        if sp in seen:
            continue
        seen.add(sp)
        v = read_int_from_json(path, "hidden_size")
        if v is not None:
            return v, f"config:{sp}"
    return None, "unknown"


def _is_flux2_klein_nunchaku_text_encoder_compatible(
    ctx: ComponentContext,
    *,
    text_ckpt: Path,
) -> tuple[bool, str]:
    expected_context_in, expected_src = _probe_flux2_klein_expected_context_in(ctx)
    if expected_context_in is None:
        return False, f"requirement unavailable source={expected_src}"

    hidden_size, hidden_src = _probe_flux2_klein_candidate_hidden_size(ctx, text_ckpt)
    if hidden_size is None:
        return False, f"candidate hidden_size unavailable source={hidden_src}"

    if expected_context_in % hidden_size != 0:
        return (
            False,
            f"incompatible expected_context_in={expected_context_in} hidden_size={hidden_size} "
            f"requirement_source={expected_src} hidden_source={hidden_src}",
        )

    return (
        True,
        f"compatible expected_context_in={expected_context_in} hidden_size={hidden_size} "
        f"requirement_source={expected_src} hidden_source={hidden_src}",
    )


def _scan_nunchaku_main_component_ckpt_in_dir(model_dir: Path, *, preferred_precision: Optional[str]) -> Optional[Path]:
    """
    在模型目录中寻找主量化权重：
    svdq-{precision}_r{number}-*.safetensors
    """
    if not model_dir.is_dir():
        return None
    cands: list[tuple[str, int, Path]] = []
    try:
        for p in model_dir.iterdir():
            if not p.is_file() or p.suffix.lower() != ".safetensors":
                continue
            m = _SVT_GENERIC_RE.match(p.name)
            if not m:
                continue
            prec = str(m.group("prec")).lower()
            r = int(m.group("r"))
            cands.append((prec, r, p))
    except Exception:
        return None
    if not cands:
        return None

    def _pick_one(items: list[tuple[str, int, Path]]) -> Path:
        return sorted(items, key=lambda x: str(x[2]))[0][2]

    if preferred_precision in {"fp4", "int4"}:
        pref_items = [x for x in cands if x[0] == preferred_precision]
        if pref_items:
            return _pick_one(pref_items)
    return _pick_one(cands)


def _default_nunchaku_main_component_candidates(ctx: ComponentContext) -> list[Path]:
    load_names = _load_name_candidates_for_auto_detect(ctx)
    specs: list[tuple[str, str]] = []

    if ctx.family == "flux_kontext":
        specs.append(("nunchaku-flux.1-kontext-dev", "flux.1-kontext-dev"))
    elif ctx.family == "flux":
        by_load_name = {
            "flux.1-canny-dev": ("nunchaku-flux.1-canny-dev", "flux.1-canny-dev"),
            "flux.1-depth-dev": ("nunchaku-flux.1-depth-dev", "flux.1-depth-dev"),
            "flux.1-dev": ("nunchaku-flux.1-dev", "flux.1-dev"),
            "flux.1-krea-dev": ("nunchaku-flux.1-krea-dev", "krea-dev"),
        }
        for load_name, spec in by_load_name.items():
            if load_name in load_names:
                specs.append(spec)
                break
    elif ctx.family == "zimage" and "z-image-turbo" in load_names:
        specs.append(("nunchaku-z-image-turbo", "z-image-turbo"))

    if not specs:
        return []

    candidates: list[Path] = []
    seen: set[str] = set()
    roots = [
        Path(p)
        for p in (ctx.models_dir, ctx.weights_dir)
        if str(p or "").strip()
    ]
    for root in roots:
        for subdir, stem in specs:
            cand_dir = root / subdir
            if not cand_dir.is_dir():
                continue
            try:
                for cand in sorted(cand_dir.iterdir(), key=lambda p: str(p)):
                    if not cand.is_file() or cand.suffix.lower() != ".safetensors":
                        continue
                    m = _SVT_GENERIC_RE.match(cand.name)
                    if not m:
                        continue
                    if not cand.name.lower().endswith(f"-{stem.lower()}.safetensors"):
                        continue
                    sc = str(cand)
                    if sc in seen:
                        continue
                    seen.add(sc)
                    candidates.append(cand)
            except Exception:
                continue
    return candidates


def _pick_nunchaku_main_component_from_existing_paths(
    paths: list[Path], *, preferred_precision: Optional[str]
) -> Optional[Path]:
    cands: list[tuple[str, int, Path]] = []
    for p in paths:
        if not p.is_file() or p.suffix.lower() != ".safetensors":
            continue
        m = _SVT_GENERIC_RE.match(p.name)
        if not m:
            continue
        prec = str(m.group("prec")).lower()
        r = int(m.group("r"))
        cands.append((prec, r, p))
    if not cands:
        return None

    def _pick_one(items: list[tuple[str, int, Path]]) -> Path:
        return sorted(items, key=lambda x: str(x[2]))[0][2]

    if preferred_precision in {"fp4", "int4"}:
        pref_items = [x for x in cands if x[0] == preferred_precision]
        if pref_items:
            return _pick_one(pref_items)
    return _pick_one(cands)


def _detect_nunchaku_main_component_ckpt(
    ctx: ComponentContext, model_dir: Path, *, preferred_precision: Optional[str]
) -> Optional[Path]:
    ckpt = _scan_nunchaku_main_component_ckpt_in_dir(model_dir, preferred_precision=preferred_precision)
    if ckpt is not None:
        return ckpt
    return _pick_nunchaku_main_component_from_existing_paths(
        _default_nunchaku_main_component_candidates(ctx),
        preferred_precision=preferred_precision,
    )


def _default_nunchaku_text_encoder_candidates(ctx: ComponentContext) -> list[Path]:
    models_dir = Path(ctx.models_dir)
    weights_dir = Path(ctx.weights_dir)
    fam = ctx.family
    if fam in {"flux2", "flux2_klein"}:
        return [
            models_dir / "Qwen3-text-Nunchaku" / "svdq-int4-Qwen3-text-Nunchaku.safetensors",
            weights_dir / "Qwen3-text-Nunchaku" / "svdq-int4-Qwen3-text-Nunchaku.safetensors",
        ]
    if fam == "qwen":
        return [
            models_dir / "Qwen2.5vl-Nunchaku" / "svdq-int4-Qwen2.5vl-text-Nunchaku.safetensors",
            weights_dir / "Qwen2.5vl-Nunchaku" / "svdq-int4-Qwen2.5vl-text-Nunchaku.safetensors",
        ]
    if fam == "qwen.edit":
        return [
            models_dir / "Qwen2.5vl-Nunchaku" / "svdq-int4-Qwen2.5vl-Nunchaku.safetensors",
            weights_dir / "Qwen2.5vl-Nunchaku" / "svdq-int4-Qwen2.5vl-Nunchaku.safetensors",
        ]
    if fam in {"flux", "flux_kontext", "chroma"}:
        return [
            models_dir / "nunchaku-t5" / "awq-int4-flux.1-t5xxl.safetensors",
            weights_dir / "nunchaku-t5" / "awq-int4-flux.1-t5xxl.safetensors",
        ]
    return []


def _detect_nunchaku_text_encoder_ckpt(ctx: ComponentContext, model_dir: Path, *, preferred_precision: Optional[str]) -> tuple[Optional[Path], str]:
    """
    优先在模型目录中寻找：
    - svdq-{precision}-*.safetensors
    - awq-{precision}-*.safetensors
    否则根据 family 使用固定 fallback。
    """
    if model_dir.is_dir():
        cands: list[tuple[str, str, Path]] = []
        try:
            for p in model_dir.iterdir():
                if not p.is_file() or p.suffix.lower() != ".safetensors":
                    continue
                m = _NUNCHAKU_TEXT_RE.match(p.name)
                if not m:
                    continue
                cands.append((str(m.group("prec")).lower(), str(m.group("kind")).lower(), p))
        except Exception:
            cands = []
        if cands:
            if preferred_precision in {"fp4", "int4"}:
                preferred = [x for x in cands if x[0] == preferred_precision]
                if preferred:
                    cands = preferred
            cands = sorted(cands, key=lambda x: (0 if x[1] == "svdq" else 1, str(x[2])))
            return cands[0][2], "model_dir"
    fallback = _candidate_from_existing_paths(
        _default_nunchaku_text_encoder_candidates(ctx),
        preferred_precision=preferred_precision,
    )
    if fallback is not None:
        return fallback, "fallback"
    return None, "missing"


def _load_unet(ctx: ComponentContext, root: Path) -> Any:
    from diffusers import UNet2DConditionModel  # type: ignore

    detect_dir, subfolder = _maybe_subfolder_root(root, "unet")
    variant, use_safetensors = infer_variant_and_use_safetensors(detect_dir)
    ctx.logger.info(
        f"[inject] {ctx.family} diffusers unet: root={root} subfolder={subfolder} "
        f"variant={variant} use_safetensors={use_safetensors}"
    )
    kwargs: dict = {
        "torch_dtype": ctx.device_plan.torch_dtype,
        "use_safetensors": use_safetensors,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    if variant:
        kwargs["variant"] = variant
    return UNet2DConditionModel.from_pretrained(str(root), **kwargs)


def _load_nunchaku_unet_sdxl(ctx: ComponentContext, ckpt: Path) -> Any:
    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel  # type: ignore

    ctx.logger.info(f"[inject] sdxl nunchaku_unet: {ckpt}")
    return NunchakuSDXLUNet2DConditionModel.from_pretrained(str(ckpt))


def _load_vae(ctx: ComponentContext, root: Path) -> Any:
    from diffusers import AutoencoderKL  # type: ignore

    detect_dir, subfolder = _maybe_subfolder_root(root, "vae")
    variant, use_safetensors = infer_variant_and_use_safetensors(detect_dir)
    ctx.logger.info(
        f"[inject] {ctx.family} diffusers vae: root={root} subfolder={subfolder} "
        f"variant={variant} use_safetensors={use_safetensors}"
    )
    kwargs: dict = {
        "torch_dtype": ctx.device_plan.torch_dtype,
        "use_safetensors": use_safetensors,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    if variant:
        kwargs["variant"] = variant
    return AutoencoderKL.from_pretrained(str(root), **kwargs)


def _load_text_encoder_clip(ctx: ComponentContext, root: Path, *, with_projection: bool) -> Any:
    if with_projection:
        from transformers import CLIPTextModelWithProjection as M  # type: ignore
    else:
        from transformers import CLIPTextModel as M  # type: ignore

    sub = "text_encoder_2" if with_projection else "text_encoder"
    detect_dir, subfolder = _maybe_subfolder_root(root, sub)
    # transformers 通常不需要 variant/use_safetensors 检测；这里仅根据 repo 结构决定 subfolder
    ctx.logger.info(f"[inject] {ctx.family} text_encoder: root={root} subfolder={subfolder} class={M.__name__}")
    kwargs: dict = {
        "torch_dtype": ctx.device_plan.torch_dtype,
        "local_files_only": True,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    return M.from_pretrained(str(root), **kwargs)


def _load_text_encoder_auto(ctx: ComponentContext, root: Path, *, sub: str) -> Any:
    from transformers import AutoModel  # type: ignore

    detect_dir, subfolder = _maybe_subfolder_root(root, sub)
    ctx.logger.info(f"[inject] {ctx.family} {sub}: root={root} subfolder={subfolder} via=AutoModel(trust_remote_code)")
    kwargs: dict = {
        "torch_dtype": ctx.device_plan.torch_dtype,
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    return AutoModel.from_pretrained(str(root), **kwargs)


def _load_text_encoder_t5(ctx: ComponentContext, root: Path, *, sub: str) -> Any:
    from transformers import T5EncoderModel  # type: ignore

    detect_dir, subfolder = _maybe_subfolder_root(root, sub)
    ctx.logger.info(f"[inject] {ctx.family} {sub}: root={root} subfolder={subfolder} class=T5EncoderModel")
    kwargs: dict = {
        "torch_dtype": ctx.device_plan.torch_dtype,
        "local_files_only": True,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    return T5EncoderModel.from_pretrained(str(root), **kwargs)


def _load_text_encoder_qwen25vl(ctx: ComponentContext, root: Path) -> Any:
    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

    detect_dir, subfolder = _maybe_subfolder_root(root, "text_encoder")
    ctx.logger.info(
        f"[inject] {ctx.family} text_encoder: root={root} subfolder={subfolder} class=Qwen2_5_VLForConditionalGeneration"
    )
    kwargs: dict = {
        "torch_dtype": ctx.device_plan.torch_dtype,
        "local_files_only": True,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(str(root), **kwargs)


def _flux_transformer_config_json() -> str:
    # <repo>/inference/common/pipeline_component_injector.py -> parents[1] is <repo>/inference/
    inference_dir = Path(__file__).resolve().parents[1]
    return str(inference_dir / "config" / "flux" / "transformer" / "config.json")


def _load_flux_transformer(ctx: ComponentContext, p: Path) -> Any:
    from diffusers import FluxTransformer2DModel  # type: ignore

    if p.is_file():
        ctx.logger.info(f"[inject] flux transformer(single_file): {p}")
        return FluxTransformer2DModel.from_single_file(
            str(p),
            torch_dtype=ctx.device_plan.torch_dtype,
            offload=True,
            config=_flux_transformer_config_json(),
        )

    detect_dir, subfolder = _maybe_subfolder_root(p, "transformer")
    variant, use_safetensors = infer_variant_and_use_safetensors(detect_dir)
    ctx.logger.info(
        f"[inject] flux transformer(pretrained): root={p} subfolder={subfolder} "
        f"variant={variant} use_safetensors={use_safetensors}"
    )
    kwargs: dict = {
        "torch_dtype": ctx.device_plan.torch_dtype,
        "use_safetensors": use_safetensors,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    if variant:
        kwargs["variant"] = variant
    return FluxTransformer2DModel.from_pretrained(str(p), **kwargs)


def _load_zimage_transformer(ctx: ComponentContext, root: Path) -> Any:
    from diffusers import ZImageTransformer2DModel  # type: ignore

    detect_dir, subfolder = _maybe_subfolder_root(root, "transformer")
    variant, use_safetensors = infer_variant_and_use_safetensors(detect_dir)
    ctx.logger.info(
        f"[inject] zimage transformer: root={root} subfolder={subfolder} "
        f"variant={variant} use_safetensors={use_safetensors}"
    )
    kwargs: dict = {
        "torch_dtype": ctx.device_plan.torch_dtype,
        "use_safetensors": use_safetensors,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    if variant:
        kwargs["variant"] = variant
    return ZImageTransformer2DModel.from_pretrained(str(root), **kwargs)


def _load_qwen_transformer(ctx: ComponentContext, root: Path) -> Any:
    from diffusers import QwenImageTransformer2DModel  # type: ignore

    detect_dir, subfolder = _maybe_subfolder_root(root, "transformer")
    variant, use_safetensors = infer_variant_and_use_safetensors(detect_dir)
    ctx.logger.info(
        f"[inject] qwen transformer: root={root} subfolder={subfolder} "
        f"variant={variant} use_safetensors={use_safetensors}"
    )
    kwargs: dict = {
        "torch_dtype": ctx.device_plan.torch_dtype,
        "use_safetensors": use_safetensors,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    if variant:
        kwargs["variant"] = variant
    return QwenImageTransformer2DModel.from_pretrained(str(root), **kwargs)


def _load_chroma_transformer(ctx: ComponentContext, p: Path) -> Any:
    from diffusers import ChromaTransformer2DModel  # type: ignore

    if p.is_file():
        ctx.logger.info(f"[inject] chroma transformer(single_file): {p}")
        return ChromaTransformer2DModel.from_single_file(str(p), torch_dtype=ctx.device_plan.torch_dtype)
    ctx.logger.info(f"[inject] chroma transformer(pretrained): {p}")
    return ChromaTransformer2DModel.from_pretrained(str(p), torch_dtype=ctx.device_plan.torch_dtype, local_files_only=True)


def _load_nunchaku_transformer_flux(ctx: ComponentContext, ckpt: Path) -> Any:
    from nunchaku import NunchakuFluxTransformer2dModel  # type: ignore
    # from torchao.quantization import Int8WeightOnlyConfig, quantize_  # type: ignore

    ctx.logger.info(f"[inject] flux nunchaku_transformer: {ckpt}")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        str(ckpt), torch_dtype=ctx.device_plan.torch_dtype, offload=True
    )
    if hasattr(transformer, "set_attention_impl"):
        transformer.set_attention_impl("nunchaku-fp16")
    # quantize_(transformer, Int8WeightOnlyConfig())
    return transformer


def _load_nunchaku_transformer_qwen(ctx: ComponentContext, ckpt: Path) -> Any:
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel  # type: ignore

    ctx.logger.info(f"[inject] qwen nunchaku_transformer: {ckpt}")
    return NunchakuQwenImageTransformer2DModel.from_pretrained(
        str(ckpt),
        torch_dtype=ctx.device_plan.torch_dtype,
        # LoRA 会在运行期 fuse 到量化低秩权重里；若此时已预先构建 offload manager，
        # 后续 CPU slab 形状会基于“加 LoRA 前”的 rank，触发 size mismatch。
        # 因此 Qwen 系列默认先关闭内部 block offload，等 low_vram 分支再显式 set_offload(True)。
        pin_memory=False,
        offload=False,
    )


def _load_nunchaku_transformer_zimage(ctx: ComponentContext, ckpt: Path) -> Any:
    from nunchaku import NunchakuZImageTransformer2DModel  # type: ignore

    ctx.logger.info(f"[inject] zimage nunchaku_transformer: {ckpt}")
    return NunchakuZImageTransformer2DModel.from_pretrained(str(ckpt))


def _load_nunchaku_transformer_chroma(ctx: ComponentContext, ckpt: Path) -> Any:
    from nunchaku import NunchakuChromaTransformer2dModel  # type: ignore

    ctx.logger.info(f"[inject] chroma nunchaku_transformer: {ckpt}")
    return NunchakuChromaTransformer2dModel.from_pretrained(str(ckpt), torch_dtype=ctx.device_plan.torch_dtype)


def _load_nunchaku_transformer_flux2(ctx: ComponentContext, ckpt: Path) -> Any:
    from nunchaku import NunchakuFlux2Transformer2DModel  # type: ignore

    ctx.logger.info(f"[inject] {ctx.family} nunchaku_transformer: {ckpt}")
    return NunchakuFlux2Transformer2DModel.from_pretrained(str(ckpt), torch_dtype=ctx.device_plan.torch_dtype)


def _load_nunchaku_t5_encoder(ctx: ComponentContext, ckpt: Path, *, key: str) -> Any:
    from nunchaku import NunchakuT5EncoderModel  # type: ignore

    ctx.logger.info(f"[inject] {ctx.family} {key}(nunchaku-t5): {ckpt}")
    return NunchakuT5EncoderModel.from_pretrained(str(ckpt))


def _load_nunchaku_qwen_encoder(ctx: ComponentContext, ckpt: Path, *, key: str) -> Any:
    from nunchaku import NunchakuQwenEncoderModel  # type: ignore

    ctx.logger.info(f"[inject] {ctx.family} {key}(nunchaku-qwen): {ckpt}")
    return NunchakuQwenEncoderModel.from_pretrained(str(ckpt))


NUNCHAKU_TRANSFORMER_LOADER: dict[str, Callable[[ComponentContext, Path], Any]] = {
    "flux": _load_nunchaku_transformer_flux,
    "flux_kontext": _load_nunchaku_transformer_flux,
    "flux2": _load_nunchaku_transformer_flux2,
    "flux2_klein": _load_nunchaku_transformer_flux2,
    "qwen": _load_nunchaku_transformer_qwen,
    "qwen.edit": _load_nunchaku_transformer_qwen,
    "zimage": _load_nunchaku_transformer_zimage,
    "chroma": _load_nunchaku_transformer_chroma,
}


NUNCHAKU_TEXT_ENCODER_SLOT: dict[str, str] = {
    "flux": "text_encoder_2",
    "flux_kontext": "text_encoder_2",
    "flux2": "text_encoder",
    "flux2_klein": "text_encoder",
    "qwen": "text_encoder",
    "qwen.edit": "text_encoder",
    "chroma": "text_encoder",
}


def _apply_auto_nunchaku_overrides(
    *,
    ctx: ComponentContext,
    allowed: set[str],
    overrides: ComponentOverrides,
    sources: dict[str, str],
    component_sig: dict[str, str],
) -> None:
    # 开关职责约定：
    # - fast_mode：只控制是否自动发现/注入 nunchaku 组件
    # - low_vram：不参与组件注入决策，只影响运行期的设备/offload 策略
    if not ctx.fast_mode:
        return

    model_dir = _model_dir_for_auto_detect(ctx.model_info)
    preferred_precision = _platform_nunchaku_precision()
    strategy = "fast_mode"

    main_ckpt = _detect_nunchaku_main_component_ckpt(ctx, model_dir, preferred_precision=preferred_precision)
    if main_ckpt is None:
        try:
            ctx.logger.info(f"[inject] auto-detect main nunchaku miss family={ctx.family} dir={str(model_dir)!r}")
        except Exception:
            pass
    else:
        if ctx.family == "sdxl" and "nunchaku_unet" in allowed and "unet" not in overrides:
            _register_lazy_component_loader(
                overrides,
                key="unet",
                loader=lambda ctx=ctx, main_ckpt=main_ckpt: _load_nunchaku_unet_sdxl(ctx, main_ckpt),
            )
            sources["unet"] = f"{strategy}(auto-detect:nunchaku_unet)"
            component_sig["unet"] = f"{strategy}:nunchaku_unet:{str(main_ckpt)}"
            try:
                ctx.logger.info(
                    f"[inject-plan] auto-detect nunchaku unet family=sdxl path={str(main_ckpt)!r} "
                    f"(lazy; load on pipeline cache miss)"
                )
            except Exception:
                pass
        elif ctx.family == "sdxl":
            try:
                ctx.logger.info("[inject] auto-detect nunchaku unet unsupported/occupied family=sdxl")
            except Exception:
                pass

        if ctx.family in NUNCHAKU_TRANSFORMER_LOADER and "nunchaku_transformer" in allowed and "transformer" not in overrides:
            loader = NUNCHAKU_TRANSFORMER_LOADER[ctx.family]
            _register_lazy_component_loader(
                overrides,
                key="transformer",
                loader=lambda ctx=ctx, main_ckpt=main_ckpt, loader=loader: loader(ctx, main_ckpt),
            )
            sources["transformer"] = f"{strategy}(auto-detect:nunchaku_transformer)"
            component_sig["transformer"] = f"{strategy}:nunchaku_transformer:{str(main_ckpt)}"
            try:
                ctx.logger.info(
                    f"[inject-plan] auto-detect nunchaku transformer family={ctx.family} path={str(main_ckpt)!r} "
                    f"(lazy; load on pipeline cache miss)"
                )
            except Exception:
                pass
        elif ctx.family not in {"sdxl"}:
            try:
                ctx.logger.info(f"[inject] auto-detect nunchaku transformer unsupported family={ctx.family}")
            except Exception:
                pass

    text_slot = NUNCHAKU_TEXT_ENCODER_SLOT.get(ctx.family)
    if not text_slot:
        try:
            ctx.logger.info(f"[inject] auto-detect nunchaku text encoder unsupported family={ctx.family}")
        except Exception:
            pass
        return
    if text_slot in overrides:
        try:
            ctx.logger.info(f"[inject] auto-detect nunchaku text encoder skipped family={ctx.family} slot={text_slot}")
        except Exception:
            pass
        return

    if text_slot == "text_encoder_2" and "nunchaku_text_encoder_2" not in allowed:
        return
    if text_slot == "text_encoder" and "nunchaku_text_encoder" not in allowed:
        return

    text_ckpt, origin = _detect_nunchaku_text_encoder_ckpt(ctx, model_dir, preferred_precision=preferred_precision)
    if text_ckpt is None:
        try:
            ctx.logger.info(f"[inject] auto-detect nunchaku text encoder miss family={ctx.family} dir={str(model_dir)!r}")
        except Exception:
            pass
        return

    if ctx.family == "flux2_klein":
        ok, detail = _is_flux2_klein_nunchaku_text_encoder_compatible(ctx, text_ckpt=text_ckpt)
        try:
            level = ctx.logger.info if ok else ctx.logger.warning
            level(
                f"[inject-check] flux2_klein nunchaku text encoder path={str(text_ckpt)!r} origin={origin} {detail}"
            )
        except Exception:
            pass
        if not ok:
            try:
                ctx.logger.warning(
                    f"[inject-plan] skip flux2_klein nunchaku text encoder slot={text_slot} path={str(text_ckpt)!r}"
                )
            except Exception:
                pass
            return

    if ctx.family in {"flux", "flux_kontext", "chroma"}:
        loader = lambda ctx=ctx, text_ckpt=text_ckpt, text_slot=text_slot: _load_nunchaku_t5_encoder(  # noqa: E731
            ctx, text_ckpt, key=text_slot
        )
    else:
        loader = lambda ctx=ctx, text_ckpt=text_ckpt, text_slot=text_slot: _load_nunchaku_qwen_encoder(  # noqa: E731
            ctx, text_ckpt, key=text_slot
        )
    _register_lazy_component_loader(
        overrides,
        key=text_slot,
        loader=loader,
    )
    src = f"{strategy}(auto-detect:nunchaku_text_encoder:{origin})"
    sources[text_slot] = src
    component_sig[text_slot] = f"{strategy}:nunchaku_text_encoder:{str(text_ckpt)}"
    try:
        ctx.logger.info(
            f"[inject-plan] auto-detect nunchaku text encoder family={ctx.family} slot={text_slot} "
            f"origin={origin} path={str(text_ckpt)!r} (lazy; load on pipeline cache miss)"
        )
    except Exception:
        pass


def _single_file_config_dir_for_family(family: str) -> Optional[str]:
    """
    单文件 pipeline 统一约定：
    config 固定在项目内 inference/config/<模型家族名> （绝对路径）。
    """
    inference_dir = Path(__file__).resolve().parents[1]
    cfg_dir = inference_dir / "config" / str(family)
    if not cfg_dir.exists():
        return None
    return str(cfg_dir)


def build_component_overrides(
    *,
    family: str,
    model_config: Optional[dict],
    inference_config: Any,
    device_plan: Any,
    model_info: Any,
    logger: Any,
    num_inference_steps: int,
    fast_mode: bool,
    low_vram: bool,
) -> ComponentOverrides:
    fam = to_model_family(family)
    overrides: ComponentOverrides = {}
    sources: dict[str, str] = {}
    component_sig: dict[str, str] = {}

    # ===== Anima (non-diffusers runtime) =====
    # 设计目标：不污染 diffusers 体系的注入逻辑；仅在 family=anima 时解析 model_config 并透传给 AnimaPipeline。
    if fam == "anima":
        try:
            repo_id = str(getattr(model_info, "repo_id", "") or "")
            root = Path(repo_id).expanduser()
            root_dir = root if root.is_dir() else root.parent
        except Exception:
            root_dir = Path(".")

        # backend 偏好：model_config.anima.backend = auto/runtime/diffusers
        backend_pref = ""
        try:
            if isinstance(model_config, dict) and isinstance(model_config.get("anima"), dict):
                backend_pref = str(model_config["anima"].get("backend") or "").strip().lower()
        except Exception:
            backend_pref = ""
        if backend_pref not in {"", "auto", "runtime", "diffusers"}:
            backend_pref = ""

        has_manifest = False
        try:
            has_manifest = bool(
                (root_dir / "anima_paths.json").is_file()
                or (root_dir / "anime_paths.json").is_file()  # 兼容常见拼写误差
                or (root_dir / "anima.json").is_file()
            )
        except Exception:
            has_manifest = False

        def _resolve_rel_to_model_root(v: object) -> str:
            raw = str(v or "").strip()
            if not raw:
                raise ValueError("empty path")
            p = Path(raw)
            if p.is_absolute():
                return str(p)
            cand = root_dir / p
            if cand.exists():
                return str(cand)
            # 兜底：沿用通用规则（models_dir / weights_dir）
            return str(_resolve_component_path(raw, models_dir=str(getattr(inference_config, "models_dir", "") or ""), weights_dir=str(getattr(inference_config, "weights_dir", "") or "")))

        # paths：允许两种写法
        # 1) model_config.anima = {dit_path, vae_path, qwen3:{...}, t5_tokenizer_dir, ...}
        # 2) model_config.anima_paths = {dit_path, vae_path, qwen3:{...}, t5_tokenizer_dir}
        anima_cfg: dict = {}
        anima_paths_cfg: dict = {}
        if isinstance(model_config, dict):
            if isinstance(model_config.get("anima"), dict):
                anima_cfg = dict(model_config.get("anima") or {})
            if isinstance(model_config.get("anima_paths"), dict):
                anima_paths_cfg = dict(model_config.get("anima_paths") or {})

        def _looks_like_paths(d: dict) -> bool:
            if not isinstance(d, dict) or not d:
                return False
            # 最少包含一个关键路径字段，才认为它是“paths 配置”
            if any(k in d for k in ("dit_path", "vae_path", "t5_tokenizer_dir")):
                return True
            q3 = d.get("qwen3")
            return isinstance(q3, dict) and any(k in q3 for k in ("model_or_weights_path", "config_dir", "tokenizer_dir"))

        # 优先级：anima_paths（专用） > anima（当 anima 同时承载 paths 时）
        raw_paths = anima_paths_cfg if _looks_like_paths(anima_paths_cfg) else (anima_cfg if _looks_like_paths(anima_cfg) else {})
        # 只有 runtime backend 才进行注入；diffusers backend 应保持“完全无特判”（避免 kwargs 被过滤/掉参日志污染）
        runtime_backend = False
        if backend_pref == "runtime":
            runtime_backend = True
        elif backend_pref == "diffusers":
            runtime_backend = False
        else:
            # auto：有 manifest 或显式提供 anima_paths 时，认为是 runtime bundle
            runtime_backend = bool(has_manifest or raw_paths)

        if not runtime_backend:
            return {}

        # 设备参数（runtime）：优先使用 device_plan，确保 AnimaInferencer 在目标设备构建/加载
        try:
            overrides["device"] = str(getattr(device_plan, "device", "") or "cuda")
            sources["device"] = "device_plan"
        except Exception:
            pass

        # paths（runtime）：可由 model_config 注入；若未提供，AnimaPipeline 也可自行从 manifest 读取
        if raw_paths:
            try:
                from third_party.anima_runtime import AnimaPaths  # type: ignore
                from third_party.anima_runtime.tokenizers import Qwen3LocalPaths  # type: ignore

                q3 = raw_paths.get("qwen3")
                if not isinstance(q3, dict):
                    raise ValueError("model_config.anima.qwen3 must be a dict")
                paths = AnimaPaths(
                    dit_path=_resolve_rel_to_model_root(raw_paths.get("dit_path")),
                    vae_path=_resolve_rel_to_model_root(raw_paths.get("vae_path")),
                    qwen3=Qwen3LocalPaths(
                        model_or_weights_path=_resolve_rel_to_model_root(q3.get("model_or_weights_path")),
                        config_dir=_resolve_rel_to_model_root(q3.get("config_dir")),
                        tokenizer_dir=_resolve_rel_to_model_root(q3.get("tokenizer_dir")),
                    ),
                    t5_tokenizer_dir=_resolve_rel_to_model_root(raw_paths.get("t5_tokenizer_dir")),
                )
                overrides["anima_paths"] = paths
                sources["anima_paths"] = "model_config"
                component_sig["anima_paths"] = (
                    f"dit={paths.dit_path}|vae={paths.vae_path}|qwen3={paths.qwen3.model_or_weights_path}|"
                    f"qwen3_cfg={paths.qwen3.config_dir}|qwen3_tok={paths.qwen3.tokenizer_dir}|t5_tok={paths.t5_tokenizer_dir}"
                )
            except Exception as e:
                raise ValueError(f"[anima] invalid model_config paths: {e}") from e

        # 运行时可选参数（仅 family=anima 时透传；不参与 diffusers kwargs 过滤会自动 drop 无关项）
        for k in (
            "text_device",
            "text_dtype",
            "attn_mode",
            "split_attn",
            "vae_spatial_chunk_size",
            "vae_disable_cache",
            "enable_block_swap",
            "dit_loading_device",
            "qwen3_loading_device",
            "materialize_cpu_tensors_before_to_cuda",
        ):
            v = anima_cfg.get(k) if isinstance(anima_cfg, dict) else None
            if v is None:
                continue
            overrides[k] = v
            sources[k] = "model_config(anima)"
            component_sig[k] = str(v)

        if sources:
            overrides["__component_sources"] = dict(sources)
        if component_sig:
            overrides["__component_sig"] = dict(component_sig)
        return overrides

    # from_single_file: 统一要求项目内 inference/config/<family> 存在
    if getattr(model_info, "method", None) == "from_single_file":
        cfg = _single_file_config_dir_for_family(fam)
        if cfg is None:
            try:
                logger.warning(
                    f"[{fam}] from_single_file: config dir not found, skip injecting `config`: "
                    f"{str(Path(__file__).resolve().parents[1] / 'config' / str(fam))!r}"
                )
            except Exception:
                pass
        else:
            overrides["config"] = cfg

        # from_single_file: family special kwargs（收敛 SDXL/SD15 的 original_config/cache_dir）
        # 注意：尽量保持现有相对路径语义，避免影响部署目录结构。
        if fam == "sdxl":
            overrides.setdefault("original_config", "inference/config/sdxl/sd_xl_base.yaml")
            overrides.setdefault("cache_dir", "resources/cache")
        elif fam == "sd15":
            overrides.setdefault("original_config", "inference/config/sd15/v1-inference.yaml")
            overrides.setdefault("cache_dir", "resources/cache")

        # from_single_file: safetensors hint
        try:
            repo_id = str(getattr(model_info, "repo_id", "") or "")
            if repo_id.lower().endswith(".safetensors"):
                overrides.setdefault("use_safetensors", True)
        except Exception:
            pass

    allowed = FAMILY_ALLOWED_KEYS.get(fam, set())
    ctx = ComponentContext(
        family=fam,
        inference_config=inference_config,
        device_plan=device_plan,
        logger=logger,
        model_info=model_info,
        num_inference_steps=int(num_inference_steps or 0),
        fast_mode=bool(fast_mode),
        low_vram=bool(low_vram),
    )
    _apply_auto_nunchaku_overrides(
        ctx=ctx,
        allowed=allowed,
        overrides=overrides,
        sources=sources,
        component_sig=component_sig,
    )

    # ===== zimage single-file converter (only fill missing) =====
    if fam == "zimage" and getattr(model_info, "method", None) == "from_single_file":
        need_text_encoder = ("text_encoder" in allowed) and ("text_encoder" not in overrides)
        need_vae = ("vae" in allowed) and ("vae" not in overrides)
        need_transformer = ("transformer" in allowed) and ("transformer" not in overrides)

        # 只要有一个组件需要转换才触发
        if need_text_encoder or need_vae or need_transformer:
            cfg_dir = _single_file_config_dir_for_family(fam)
            if cfg_dir is None:
                try:
                    logger.warning(
                        f"[zimage] from_single_file: config dir not found, skip single-file component conversion: "
                        f"{str(Path(__file__).resolve().parents[1] / 'config' / 'zimage')!r}"
                    )
                except Exception:
                    pass
            else:
                try:
                    from common.single_file_component_converter import build_zimage_overrides_for_single_file

                    ckpt_path = str(getattr(model_info, "repo_id", "") or "")
                    # from_single_file 场景：repo_id 就是单文件路径
                    conv = build_zimage_overrides_for_single_file(
                        ckpt_path=ckpt_path,
                        config_dir=cfg_dir,
                        torch_dtype=ctx.device_plan.torch_dtype,
                        local_files_only=True,
                        logger=logger,
                        need_text_encoder=need_text_encoder,
                        need_vae=need_vae,
                        need_transformer=need_transformer,
                    )
                    # 只补缺，不覆盖已有
                    for k, v in conv.items():
                        if k not in overrides:
                            overrides[k] = v
                            sources.setdefault(k, "single_file(converter)")
                            component_sig.setdefault(k, "single_file:converter")
                except Exception as e:
                    # 不中断：记录错误并继续（让 diffusers 自己尝试）
                    try:
                        logger.warning(f"[zimage] single-file component conversion failed (ignored): {e}")
                    except Exception:
                        pass

    # ===== from_single_file: assemble missing runtime components (auto overrides -> single_file -> base_model) =====
    if getattr(model_info, "method", None) == "from_single_file":
        policy = FAMILY_SINGLEFILE_POLICY.get(fam)
        if isinstance(policy, dict):
            required = set(policy.get("required_components") or set())
            main = str(policy.get("main_component") or "").strip()
            base_path = str(policy.get("base_path") or "").strip()
            presence = policy.get("presence_prefixes") if isinstance(policy.get("presence_prefixes"), dict) else {}

            ckpt_has: set[str] = set()
            ckpt = Path(str(getattr(model_info, "repo_id", "") or ""))
            if main and (main not in overrides):
                ckpt_has.add(main)
            for comp in required:
                if comp in overrides or comp == main:
                    continue
                prefixes = tuple(presence.get(comp) or ()) if isinstance(presence, dict) else ()
                if safetensors_has_any_prefixes(ckpt, prefixes):
                    ckpt_has.add(comp)

            # base_model fallback for components that既不在自动 overrides 中，也不在单文件中
            missing_for_base = required - set(overrides.keys()) - ckpt_has
            if missing_for_base:
                if not base_path:
                    raise ValueError(f"[{fam}] from_single_file missing components {sorted(missing_for_base)} but base_path is empty")
                base_root = Path(ctx.models_dir) / base_path
                if not base_root.exists():
                    raise ValueError(
                        f"[{fam}] from_single_file missing components {sorted(missing_for_base)}; base model not found: {str(base_root)!r}"
                    )

                def _load_from_base(comp: str) -> Any:
                    if comp == "vae":
                        return _load_vae(ctx, base_root)
                    if comp == "unet":
                        return _load_unet(ctx, base_root)
                    if comp == "text_encoder":
                        if fam in {"sd15", "sdxl", "flux", "flux_kontext"}:
                            return _load_text_encoder_clip(ctx, base_root, with_projection=False)
                        if fam == "chroma":
                            return _load_text_encoder_t5(ctx, base_root, sub="text_encoder")
                        if fam in {"qwen", "qwen.edit"}:
                            return _load_text_encoder_qwen25vl(ctx, base_root)
                        return _load_text_encoder_auto(ctx, base_root, sub="text_encoder")
                    if comp == "text_encoder_2":
                        if fam == "sdxl":
                            return _load_text_encoder_clip(ctx, base_root, with_projection=True)
                        if fam in {"flux", "flux_kontext"}:
                            return _load_text_encoder_t5(ctx, base_root, sub="text_encoder_2")
                        return _load_text_encoder_auto(ctx, base_root, sub="text_encoder_2")
                    raise ValueError(f"[{fam}] base fallback for component {comp!r} is not supported")

                for comp in sorted(missing_for_base):
                    _register_lazy_component_loader(
                        overrides,
                        key=comp,
                        loader=lambda comp=comp: _load_from_base(comp),
                    )
                    sources.setdefault(comp, "base_model")
                    component_sig.setdefault(comp, f"base_model:{str(base_root)}")

            # 最终校验：缺任何必选组件就显式报错
            final_missing = required - set(overrides.keys()) - ckpt_has
            if final_missing:
                raise ValueError(f"[{fam}] from_single_file missing required components: {sorted(final_missing)}")

            # 输出来源日志（逐组件，便于排障）
            for comp in sorted(required):
                if comp in sources:
                    src = sources[comp]
                elif comp in overrides:
                    src = "override"
                elif comp in ckpt_has:
                    src = "single_file"
                else:
                    src = "missing"
                try:
                    logger.info(f"[single_file_assemble] {fam} {comp} <- {src}")
                except Exception:
                    pass

    # 内部观测/缓存用：不应透传给 diffusers pipeline 工厂（由 PipelineService 做 kwargs 过滤）
    if sources:
        overrides["__component_sources"] = dict(sources)
    if component_sig:
        overrides["__component_sig"] = dict(component_sig)

    return overrides

