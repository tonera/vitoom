"""
推理参数构建器
根据pipeline类型和InferenceRequestParams构建推理参数
"""
from typing import Any, Dict

from schemas import InferenceRequestParams
from common.logger import get_logger
from common.image_utils import get_image_size, constrain_size
from common.model_registry import MODEL_REGISTRY
from image.inference_param_specs import pick_spec

logger = get_logger(__name__)

def _normalize_pipeline_class_name(name: str) -> str:
    """
    将运行时被动态 patch 的 pipeline 类名归一化为原始 diffusers pipeline 名称。

    背景：
    - fast_mode(fbcache) 可能会把 pipe.__class__ 换成 per-instance 的动态子类，
      类名形如：StableDiffusionXLPipelineFBCachePatched_139992...
    - 本模块历史实现用“类名字符串匹配”决定是否注入 width/height 等参数。
      若不归一化，会导致缓存复用时漏传 width/height，从而回落到默认 1024x1024。
    """
    if not name:
        return name

    # 本项目 fbcache 的命名规则：{BaseName}FBCachePatched_{id}
    if "FBCachePatched_" in name:
        base = name.split("FBCachePatched_")[0]
        if base:
            return base

    # 兜底：若出现其它 “Patched_...” 形式，尽量取 Patched_ 之前的部分
    if "Patched_" in name:
        base = name.split("Patched_")[0]
        if base:
            return base

    return name


def build_inference_params(
    pipeline: Any,
    request_params: InferenceRequestParams,
) -> Dict[str, Any]:
    """
    根据pipeline类型和InferenceRequestParams构建推理参数
    
    Args:
        pipeline: diffusers pipeline类或实例
        request_params: 推理请求参数
        pipeline_params: 管道参数
        
    Returns:
        推理参数字典
    """
    # 获取pipeline类名（处理类或实例）
    if isinstance(pipeline, type):
        pipeline_class_name = _normalize_pipeline_class_name(pipeline.__name__)
        pipeline_instance = None
    else:
        pipeline_instance = pipeline
        # 优先使用 runtime patch 记录的“原始 pipeline 类名”（例如 fbcache 会把 __class__ 换成动态子类）
        base_name = getattr(pipeline_instance, "_pipeline_base_class_name", None)
        if isinstance(base_name, str) and base_name:
            pipeline_class_name = base_name
        else:
            pipeline_class_name = _normalize_pipeline_class_name(pipeline_instance.__class__.__name__)
    
    # 基础参数（所有pipeline通用）
    inference_params: Dict[str, Any] = {
        "prompt": request_params.prompt,
        "negative_prompt": request_params.negative_prompt or "",
        "num_inference_steps": request_params.num_inference_steps,
        "guidance_scale": request_params.guidance_scale,
        # 统一由 ImageInferrer 的迭代驱动生成多张图；这里始终单次产出 1 张，避免重复生成浪费
        "num_images_per_prompt": 1,
    }
    # 打印prompt
    logger.debug(f"prompt: {request_params.prompt}")
    
    # 根据keep_size规则设置width和height
    width = request_params.width
    height = request_params.height
    
    if request_params.keep_size == 'init_images' and request_params.url:
        # 宽高为req.url的宽高
        size = get_image_size(request_params.url)
        if size:
            width, height = size
            logger.debug(f"Using image size from url: {width}x{height}")
    elif request_params.keep_size == 'image_file2' and request_params.image_file2:
        # 宽高为req.image_file2的宽高
        size = get_image_size(request_params.image_file2)
        if size:
            width, height = size
            logger.debug(f"Using image size from image_file2: {width}x{height}")
    elif request_params.keep_size == 'ref_image' and request_params.tpl_list:
        # 编辑任务下，宽高为 tpl_list[0] 的原始宽高
        size = get_image_size(request_params.tpl_list[0])
        if size:
            width, height = size
            logger.debug(f"Using image size from tpl_list[0]: {width}x{height}")

    old_w, old_h = int(width), int(height)
    width, height = constrain_size(old_w, old_h)
    if (width, height) != (old_w, old_h):
        logger.info(f"constrain_size: ({old_w}x{old_h}) -> ({width}x{height})")

    # ===== Milestone A: 按 family/spec 构建（先迁移 Flux2/Flux2Klein） =====
    family = MODEL_REGISTRY.to_family(getattr(request_params, "family", None))
    spec = pick_spec(family, pipeline_class_name)
    if spec is not None:
        return spec.build(
            pipeline_class_name=pipeline_class_name,
            request_params=request_params,
            base=inference_params,
            width=int(width),
            height=int(height),
            pipeline_instance=pipeline_instance,
        )

    return inference_params

