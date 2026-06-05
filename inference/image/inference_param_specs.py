"""
按模型家族（family）组织的推理参数 Spec。

里程碑A目标：
- 先把新增/变动频繁的家族（Flux2/Flux2Klein）迁移进 spec
- `inference_params_builder.py` 逐步变成“薄调度层”，避免到处 if/elif
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from schemas import InferenceRequestParams
from common.Constant import JT_ED, JT_POSE
from common.lpw_stable_diffusion_xl import get_weighted_text_embeddings_sdxl
from common.image_utils import load_image, load_images_from_list
import torch


class InferenceParamSpec:
    families: set[str] = set()
    pipeline_class_names: set[str] = set()

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class Flux2InferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"flux2", "flux2_klein"})
        object.__setattr__(
            self,
            "pipeline_class_names",
            {"Flux2Pipeline", "Flux2KleinPipeline", "Flux2KleinKVPipeline"},
        )

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        # Flux2 系列：不使用 negative_prompt
        base.pop("negative_prompt", None)

        if request_params.job_type in {JT_ED, JT_POSE}:
            images = load_images_from_list(request_params.tpl_list or [])
            if not images:
                raise ValueError(f"{request_params.job_type} requires non-empty tpl_list with loadable images")
            base["image"] = images
        elif request_params.url:
            image = load_image(request_params.url)
            if image:
                base["image"] = image

        base.update(
            {
                "height": int(height),
                "width": int(width),
                "guidance_scale": 4,
            }
        )

        return base


@dataclass(frozen=True)
class FluxInferenceParamSpec(InferenceParamSpec):
    """
    Flux(1) 家族推理参数：
    迁移自 inference_params_builder.py 的 Flux 分支，行为尽量保持一致。
    """
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        # Flux 与 Flux Kontext：pipeline 调参规则保持一致（Kontext 仍是一条独立 family）
        object.__setattr__(self, "families", {"flux", "flux_kontext"})
        object.__setattr__(
            self,
            "pipeline_class_names",
            {
                "FluxPipeline",
                "FluxImg2ImgPipeline",
                "FluxInpaintPipeline",
                "FluxControlNetInpaintPipeline",
                "FluxControlNetImg2ImgPipeline",
                "FluxControlPipeline",
                "FluxControlImg2ImgPipeline",
                "FluxPriorReduxPipeline",
                "FluxFillPipeline",
                "FluxKontextPipeline",
                "FluxKontextInpaintPipeline",
            },
        )

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        # 与原逻辑一致：仅部分 Flux pipeline 不传 negative_prompt
        if pipeline_class_name in {"FluxControlPipeline", "FluxFillPipeline", "FluxControlImg2ImgPipeline"}:
            base.pop("negative_prompt", None)

        # FluxPriorReduxPipeline 需要 image 参数，不需要 height/width
        if pipeline_class_name == "FluxPriorReduxPipeline":
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
        else:
            base.update({"height": int(height), "width": int(width)})

        # 编辑任务：flux_kontext 只支持处理第一张图片
        if request_params.job_type in {JT_ED, JT_POSE}:
            tpl_list = request_params.tpl_list or []
            if not tpl_list:
                raise ValueError(f"{request_params.job_type} requires non-empty tpl_list")
            image = load_image(tpl_list[0])
            if not image:
                raise ValueError(f"{request_params.job_type} requires loadable image in tpl_list")
            base["image"] = image

        if pipeline_class_name in {"FluxImg2ImgPipeline", "FluxKontextPipeline"}:
            base["guidance_scale"] = 30

        # 图生图相关
        if pipeline_class_name in {"FluxImg2ImgPipeline", "FluxControlNetImg2ImgPipeline", "FluxControlImg2ImgPipeline"}:
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
                base["strength"] = request_params.strength

        # Inpaint相关
        if "Inpaint" in pipeline_class_name or pipeline_class_name == "FluxFillPipeline":
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
            if request_params.image_file2:
                mask_image = load_image(request_params.image_file2)
                if mask_image:
                    base["mask_image"] = mask_image
            if pipeline_class_name == "FluxFillPipeline":
                base["guidance_scale"] = 30

        # ControlNet相关
        if ("Control" in pipeline_class_name) or ("ControlNet" in pipeline_class_name):
            if request_params.image_file2:
                control_image = load_image(request_params.image_file2)
                if control_image:
                    base["control_image"] = control_image
            base["controlnet_conditioning_scale"] = getattr(request_params, "controlnet_conditioning_scale", 1.0)

        return base


@dataclass(frozen=True)
class SdxlAutoPipelineInferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"sdxl"})
        object.__setattr__(
            self,
            "pipeline_class_names",
            {"AutoPipelineForText2Image", "AutoPipelineForImage2Image"},
        )

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        # SDXL AutoPipeline：需要 height/width；img2img 需要 image/strength
        base.update({"height": int(height), "width": int(width)})

        if pipeline_class_name == "AutoPipelineForImage2Image":
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
                base["strength"] = request_params.strength

        # SDXL: 若 pipeline 实例可用，生成 prompt_embeds 并移除 prompt 文本
        if pipeline_instance is not None:
            try:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                    get_weighted_text_embeddings_sdxl(
                        pipe=pipeline_instance,
                        prompt=request_params.prompt,
                        neg_prompt=request_params.negative_prompt or "",
                    )
                )
                base["prompt_embeds"] = prompt_embeds
                base["negative_prompt_embeds"] = negative_prompt_embeds
                base["pooled_prompt_embeds"] = pooled_prompt_embeds
                base["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
                base.pop("prompt", None)
                base.pop("negative_prompt", None)
            except Exception as e:
                # 失败则回退为文本 prompt
                pass
        return base


@dataclass(frozen=True)
class Sd15AutoPipelineInferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"sd15"})
        object.__setattr__(
            self,
            "pipeline_class_names",
            {"AutoPipelineForText2Image", "AutoPipelineForImage2Image"},
        )

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        # SD15 AutoPipeline：Text2Img 需要 height/width；Image2Image 不需要 height/width
        if pipeline_class_name == "AutoPipelineForText2Image":
            base.update({"height": int(height), "width": int(width)})
            return base

        if pipeline_class_name == "AutoPipelineForImage2Image":
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
                base["strength"] = request_params.strength
            return base

        return base


@dataclass(frozen=True)
class SdxlMainInferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"sdxl"})
        object.__setattr__(
            self,
            "pipeline_class_names",
            {
                "StableDiffusionXLPipeline",
                "StableDiffusionXLImg2ImgPipeline",
                "StableDiffusionXLInpaintPipeline",
                "StableDiffusionXLControlNetPipeline",
                "StableDiffusionXLControlNetImg2ImgPipeline",
                "StableDiffusionXLControlNetInpaintPipeline",
            },
        )

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        base.update({"height": int(height), "width": int(width)})

        # 图生图相关
        if "Img2Img" in pipeline_class_name:
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
                base["strength"] = request_params.strength

        # Inpaint相关
        if "Inpaint" in pipeline_class_name:
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
            if request_params.image_file2:
                mask_image = load_image(request_params.image_file2)
                if mask_image:
                    base["mask_image"] = mask_image

        # ControlNet相关
        if "ControlNet" in pipeline_class_name:
            if request_params.image_file2:
                control_image = load_image(request_params.image_file2)
                if control_image:
                    base["control_image"] = control_image
            base["controlnet_conditioning_scale"] = getattr(request_params, "controlnet_conditioning_scale", 1.0)

        # SDXL prompt embeds（仅在 pipeline 已初始化时）
        if pipeline_instance is not None:
            try:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                    get_weighted_text_embeddings_sdxl(
                        pipe=pipeline_instance,
                        prompt=request_params.prompt,
                        neg_prompt=request_params.negative_prompt or "",
                    )
                )
                base["prompt_embeds"] = prompt_embeds
                base["negative_prompt_embeds"] = negative_prompt_embeds
                base["pooled_prompt_embeds"] = pooled_prompt_embeds
                base["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
                base.pop("prompt", None)
                base.pop("negative_prompt", None)
            except Exception:
                pass

        return base


@dataclass(frozen=True)
class Sd15MainInferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"sd15"})
        object.__setattr__(
            self,
            "pipeline_class_names",
            {
                "StableDiffusionPipeline",
                "StableDiffusionImg2ImgPipeline",
            },
        )

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        base.update({"height": int(height), "width": int(width)})
        if "Img2Img" in pipeline_class_name:
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
                base["strength"] = request_params.strength
        return base


@dataclass(frozen=True)
class QwenInferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"qwen", "qwen.edit"})
        object.__setattr__(
            self,
            "pipeline_class_names",
            {
                "QwenImagePipeline",
                "QwenImageImg2ImgPipeline",
                "QwenImageInpaintPipeline",
                "QwenImageEditPipeline",
                "QwenImageEditInpaintPipeline",
                "QwenImageControlNetPipeline",
                "QwenImageEditPlusPipeline",
            },
        )

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        base.update({"height": int(height), "width": int(width)})

        # EditPlus 只移除 pipeline 明确不接受的通用参数，不再自动改写用户请求。
        if pipeline_class_name == "QwenImageEditPlusPipeline":
            base.pop("guidance_scale", None)
            base.pop("generator", None)

            # 加速相关（保持原行为）
            if bool(getattr(request_params, "fast_mode", False)):
                base["true_cfg_scale"] = 1.0

            # 对齐官方 lightning 示例：true_cfg_scale=1.0 时不传 negative_prompt。
            # 我们的通用构建器会默认塞入 negative_prompt=""，但在 CFG 未启用时这会触发
            # diffusers 内部不必要的分支/告警，某些版本组合下甚至会抛异常。
            try:
                tcs = float(base.get("true_cfg_scale", 1.0) or 1.0)
            except Exception:
                tcs = 1.0
            if tcs <= 1.0:
                np = base.get("negative_prompt", None)
                if np is None or (isinstance(np, str) and not np.strip()):
                    base.pop("negative_prompt", None)

        # 图生图相关
        if "Img2Img" in pipeline_class_name:
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
                base["strength"] = request_params.strength

        # Inpaint相关
        if "Inpaint" in pipeline_class_name:
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
            if request_params.image_file2:
                mask_image = load_image(request_params.image_file2)
                if mask_image:
                    base["mask_image"] = mask_image

        # Edit相关（只处理 image 参数）
        if "Edit" in pipeline_class_name:
            if request_params.job_type in {JT_ED, JT_POSE}:
                images = load_images_from_list(request_params.tpl_list or [])
                if not images:
                    raise ValueError(f"{request_params.job_type} requires non-empty tpl_list with loadable images")
                base["image"] = images
            else:
                if request_params.url:
                    image = load_image(request_params.url)
                    if not image:
                        raise ValueError("Edit pipeline requires a loadable image from url")
                    base["image"] = image
                else:
                    raise ValueError("Edit pipeline requires image url or tpl_list")

        # ControlNet相关
        if "ControlNet" in pipeline_class_name:
            if request_params.image_file2:
                control_image = load_image(request_params.image_file2)
                if control_image:
                    base["control_image"] = control_image
            base["controlnet_conditioning_scale"] = getattr(request_params, "controlnet_conditioning_scale", 1.0)

        return base


@dataclass(frozen=True)
class ZImageInferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"zimage"})
        object.__setattr__(self, "pipeline_class_names", {"ZImagePipeline", "ZImageImg2ImgPipeline"})

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        # 保持原实现：ZImage 使用 request_params.height/width（而非 keep_size 计算后的 width/height）
        base.update({"height": request_params.height, "width": request_params.width})

        # Edit（JT_ED）：flux 只支持处理第一张图片
        # if request_params.job_type == JT_ED:
        #     tpl_list = request_params.tpl_list or []
        #     if not tpl_list:
        #         raise ValueError("JT_ED requires non-empty tpl_list")
        #     image = load_image(tpl_list[0])
        #     if not image:
        #         raise ValueError("JT_ED requires loadable image in tpl_list")
        #     base["image"] = image
        #     w_ed, h_ed = resolve_edit_size_1024_square_multiple16([image])
        #     if w_ed and h_ed:
        #         if "width" in base:
        #             base["width"] = int(w_ed)
        #         if "height" in base:
        #             base["height"] = int(h_ed)
        #     base["strength"] = request_params.strength

        if "Img2Img" in pipeline_class_name:
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image
                base["strength"] = request_params.strength
        return base


@dataclass(frozen=True)
class AuraInferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"aura"})
        object.__setattr__(self, "pipeline_class_names", {"AuraFlowPipeline"})

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        base.update({"height": request_params.height, "width": request_params.width})
        return base


@dataclass(frozen=True)
class WanInferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"wan"})
        object.__setattr__(
            self,
            "pipeline_class_names",
            {
                "WanPipeline",
                "WanImageToVideoPipeline",
                "WanVACEPipeline",
                "WanVideoToVideoPipeline",
                "WanAnimatePipeline",
            },
        )

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        base.update({"height": request_params.height, "width": request_params.width})

        if ("ImageToVideo" in pipeline_class_name) or ("VACE" in pipeline_class_name):
            if request_params.url:
                image = load_image(request_params.url)
                if image:
                    base["image"] = image

        # VideoToVideo/VACE 的 video 参数目前未在 builder 构建，这里保持原样（无额外参数）

        if "Animate" in pipeline_class_name:
            base["num_frames"] = int(request_params.duration) * 8

        return base

@dataclass(frozen=True)
class ChromaInferenceParamSpec(InferenceParamSpec):
    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"chroma"})
        object.__setattr__(self, "pipeline_class_names", {"ChromaPipeline"})

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        w_ed, h_ed = 0, 0
        if request_params.job_type == JT_ED:
            raise ValueError("This model does not support EDIT mode")
        elif request_params.url:
            image = load_image(request_params.url)
            if image:
                base["image"] = image

        base.update(
            {
                "height": int(height),
                "width": int(width),
                "guidance_scale": request_params.guidance_scale,
                # "generator": torch.Generator("cpu").manual_seed(request_params.seed),
            }
        )
        # generator=torch.Generator("cpu").manual_seed(seed),
        return base


@dataclass(frozen=True)
class AnimaInferenceParamSpec(InferenceParamSpec):
    """
    Anima（非 diffusers 格式）推理参数：
    - 仅支持 text2img（MK）；不支持 url/img2img、ED/SED 等编辑分支。
    - 通过额外字段把 anima 的运行时参数透传给 AnimaPipeline.__call__。
    """

    families: set[str] = None  # type: ignore[assignment]
    pipeline_class_names: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "families", {"anima"})
        # 未来官方 diffusers pipeline 类名可能变化；用 "*" 保证同一 family 不必频繁改 spec
        object.__setattr__(self, "pipeline_class_names", {"*"})

    def build(
        self,
        *,
        pipeline_class_name: str,
        request_params: InferenceRequestParams,
        base: Dict[str, Any],
        width: int,
        height: int,
        pipeline_instance: Any,
    ) -> Dict[str, Any]:
        if request_params.job_type == JT_ED or str(getattr(request_params, "job_type", "")).strip().upper() == "SED":
            raise ValueError("Anima model does not support EDIT/SED mode")
        if getattr(request_params, "url", None):
            raise ValueError("Anima model does not support img2img (params.url)")

        # anima 不支持 diffusers 的 num_images_per_prompt 语义（DiffusionHandler 已统一按迭代产出 1 张）
        base.pop("num_images_per_prompt", None)
        base.update({"height": int(height), "width": int(width)})

        # anima 专用可调参数：从 model_cfg.anima 读取（避免污染其它家族字段）
        cfg = getattr(request_params, "model_cfg", None)
        a = cfg.get("anima") if isinstance(cfg, dict) and isinstance(cfg.get("anima"), dict) else {}
        if isinstance(a, dict):
            if "flow_shift" in a and a.get("flow_shift") is not None:
                base["flow_shift"] = float(a.get("flow_shift"))
            if "qwen3_max_len" in a and a.get("qwen3_max_len") is not None:
                base["qwen3_max_len"] = int(a.get("qwen3_max_len"))
            if "t5_max_len" in a and a.get("t5_max_len") is not None:
                base["t5_max_len"] = int(a.get("t5_max_len"))

        return base


def _discover_specs() -> list[InferenceParamSpec]:
    """
    自动发现并实例化本模块内的 InferenceParamSpec 子类，避免新增家族时忘记往 SPECS 列表里手工加。
    约束：Spec 子类必须支持无参构造。
    """
    out: list[InferenceParamSpec] = []
    for cls in InferenceParamSpec.__subclasses__():
        try:
            out.append(cls())  # type: ignore[call-arg]
        except TypeError:
            # 非无参构造的子类不自动实例化
            continue
    # 稳定排序，避免不同 Python 版本/导入顺序导致的非确定性
    out.sort(key=lambda x: type(x).__name__)
    return out


SPECS: list[InferenceParamSpec] = _discover_specs()


def pick_spec(family: str, pipeline_class_name: str) -> InferenceParamSpec | None:
    fam = str(family or "").strip().lower()
    if not fam:
        return None
    pcl = str(pipeline_class_name or "").strip()
    if not pcl:
        return None
    for s in SPECS:
        fam_ok = fam in getattr(s, "families", set())
        pcls = getattr(s, "pipeline_class_names", set())
        pcl_ok = ("*" in pcls) or (pcl in pcls)
        if fam_ok and pcl_ok:
            return s
    return None

