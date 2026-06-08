"""
根据模型文件或目录自动判定应使用的 diffusers Pipeline 类。

改为类供外部调用：
    detector = PipelineDetector()
    pipeline_cls = detector.from_file(params) 或 detector.from_model_index(params)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING
from schemas import InferenceRequestParams
from .logger import get_logger
# 注意：这里不要在运行时 import inference.image.*，否则会触发 image/__init__.py 的副作用导入，
# 造成 common.pipeline_detector <-> image.inferrer 的循环导入，pytest 收集阶段就会失败。
if TYPE_CHECKING:  # pragma: no cover
    from image.runtime.device_planner import DevicePlan  # noqa: F401
    from image.runtime.model_locator import ModelInfo  # noqa: F401
from .pipeline_component_injector import build_component_overrides

logger = get_logger(__name__)

from common.model_catalog import get_catalog
from common.model_catalog.types import PipelineRef
from common.model_metadata import read_family_name, safetensors_get_shape, safetensors_list_keys
from common.family_utils import to_model_family


class PipelineDetector:
    """提供基于文件或model_index的Pipeline类选择"""
    family: str = ""
    
    def __init__(self):
        """初始化Pipeline检测器，加载推理配置"""
        from .config_loader import load_inference_config
        self.inference_config = load_inference_config()

    def _resolve_model_path(self, load_name: Optional[str]) -> Path:
        """
        将 params.load_name 解析为实际模型路径。
        - 兼容传入绝对路径（便于单测/脚本）
        - 兼容传入相对名称（基于 inference_config.models_dir）
        """
        raw = str(load_name or "").strip()
        if not raw:
            return Path(self.inference_config.models_dir)
        p = Path(raw)
        if p.is_absolute():
            return p
        # 若外部已经传入可直接访问的相对路径，也允许直接用
        if p.exists():
            return p
        return Path(self.inference_config.models_dir) / raw
    
    @staticmethod
    def _detect_model_type_from_file_keys(file_path: Path) -> Optional[str]:
        """
        精简版权重键检测：区分 SDXL / Flux / SD15 / ZImage / Qwen。
        """
        if not file_path.exists() or not file_path.is_file():
            return None
        if file_path.suffix not in (".safetensors", ".ckpt"):
            return None
        if file_path.suffix == ".ckpt":
            return None  # 未实现 ckpt 解析

        try:
            keys = safetensors_list_keys(file_path)
            if not keys:
                return None

            # ZImage 特征（必须优先于 Flux，避免被 text_encoders 规则误判）
            # ZImage 的 text_encoder 通常是 Qwen3Model，单文件权重里常见前缀：
            # - text_encoders.qwen3_4b.transformer.*
            # - text_encoders.qwen3_4b.*
            if any(k.lower().startswith("text_encoders.qwen") for k in keys):
                return "zimage"

            # SDXL 特征
            if any("text_encoder_2" in k.lower() or "text_encoder2" in k.lower() for k in keys):
                return "sdxl"
            if any("conditioner" in k.lower() for k in keys):
                return "sdxl"

            # Qwen 特征（Qwen-Image 单文件常见只有 model.diffusion_model.* 的 DiT/Transformer 权重）
            # 规则设计：
            # - 优先：命中 >=2 个 DiT 结构信号（time_text_embed/pos_embed/txt_in/img_in/transformer_blocks）
            # - 兜底：几乎全是 model.diffusion_model.* 且包含 time_text_embed，并且不具备 SDXL/SD15/Flux 的关键特征
            has_input_blocks = any("input_blocks" in k for k in keys)
            has_double_blocks = any("double_blocks" in k for k in keys)
            has_single_blocks = any("single_transformer_blocks" in k for k in keys)
            has_context = any("context_embedder" in k.lower() for k in keys)
            has_text_encoders = any("text_encoders" in k.lower() for k in keys)
            has_cond_stage_model = any("cond_stage_model" in k.lower() for k in keys)

            has_qwen_time_text_embed = any("model.diffusion_model.time_text_embed." in k for k in keys)
            has_qwen_pos_embed = any("model.diffusion_model.pos_embed" in k for k in keys)
            has_qwen_txt_in = any("model.diffusion_model.txt_in" in k for k in keys)
            has_qwen_img_in = any("model.diffusion_model.img_in" in k for k in keys)
            has_qwen_transformer_blocks = any("model.diffusion_model.transformer_blocks" in k for k in keys)
            qwen_hits = sum(
                [
                    1 if has_qwen_time_text_embed else 0,
                    1 if has_qwen_pos_embed else 0,
                    1 if has_qwen_txt_in else 0,
                    1 if has_qwen_img_in else 0,
                    1 if has_qwen_transformer_blocks else 0,
                ]
            )
            diffusion_prefix_cnt = sum(1 for k in keys if k.startswith("model.diffusion_model."))
            diffusion_prefix_ratio = diffusion_prefix_cnt / max(len(keys), 1)
            qwen_fallback_ok = (
                has_qwen_time_text_embed
                and diffusion_prefix_ratio >= 0.8
                and (not has_input_blocks)
                and (not has_double_blocks)
                and (not has_single_blocks)
                and (not has_context)
                and (not has_text_encoders)
                and (not has_cond_stage_model)
            )
            if (qwen_hits >= 2 and (not has_input_blocks) and (not has_double_blocks)) or qwen_fallback_ok:
                return "qwen"

            # Flux 特征
            if any("double_blocks" in k for k in keys):
                return "flux"
            if has_single_blocks and has_context:
                return "flux"
            if any("text_encoders" in k.lower() for k in keys) and not any(
                "cond_stage_model" in k.lower() for k in keys
            ):
                return "flux"

            # SD15 特征
            if any("input_blocks" in k for k in keys) and any("cond_stage_model" in k.lower() for k in keys):
                return "sd15"

            # 兜底：UNet 输入通道
            unet_keys = [
                k
                for k in keys
                if "input_blocks.0.0.weight" in k or "model.diffusion_model.input_blocks.0.0.weight" in k
            ]
            for k in unet_keys:
                shape = safetensors_get_shape(file_path, k)
                if shape is None or len(shape) < 2:
                    continue
                ch = shape[0]
                if ch == 9:
                    return "sdxl"
                if ch == 4:
                    return "sd15"
        except Exception:
            return None
        return None

    @staticmethod
    def _pick_single_file_pipeline_ref(family: str, model_path: Path, *, is_img2img: bool) -> Optional[PipelineRef]:
        fam = to_model_family(family)
        if fam == "flux2_klein":
            stem = model_path.stem.lower()
            if "-kv" in stem or "_kv" in stem:
                return PipelineRef("diffusers", "Flux2KleinKVPipeline")

        cat = get_catalog()
        return cat.default_pipeline_ref(fam, is_img2img=is_img2img)

    def from_file(self, params: InferenceRequestParams):
        """
        根据模型文件权重键选择对应的 Pipeline 类。
        返回 diffusers 的 Pipeline 类对象，未识别则抛 ValueError。
        """
        model_path = self._resolve_model_path(params.load_name)
        self.family = self._detect_model_type_from_file_keys(model_path)
        is_img2img = bool(params.url)

        fam = to_model_family(self.family)
        self.family = fam
        pref = self._pick_single_file_pipeline_ref(fam, model_path, is_img2img=is_img2img)
        if pref is None:
            raise ValueError(f"Unsupported or undetected model family for file: {model_path} (family={fam})")
        return pref.resolve()

    def from_model_index(self, params: InferenceRequestParams):
        """
        读取 model_index.json 的 _class_name 映射到具体 Pipeline 类。
        """
        model_dir = self._resolve_model_path(params.load_name)
        model_index_path = model_dir / "model_index.json"
        if not model_index_path.exists():
            raise ValueError(f"model_index.json not found in {model_dir}")

        try:
            class_name = read_family_name(model_dir)
            logger.debug(
                f"[from_model_index] 读取model_index.json: class_name={class_name}, "
                f"load_name={params.load_name}"
            )
        except Exception as e:
            raise ValueError(f"Failed to read model_index.json: {e}") from e

        is_img2img = bool(params.url)

        cat = get_catalog()
        picked = cat.choose_pipeline_ref(str(class_name or ""), is_img2img=is_img2img)
        if picked:
            family, pipeline_ref = picked
            # detector.family 始终保存 canonical family（全链路分支统一使用）
            self.family = str(family)
            try:
                logger.debug(
                    f"[from_model_index] catalog 命中: class_name={class_name}, family={family}, pipeline={pipeline_ref.class_name}"
                )
            except Exception:
                pass
            return pipeline_ref.resolve()

        raise ValueError(f"Unsupported _class_name in model_index.json: {class_name}")

    def get_pipeline(self, params: InferenceRequestParams):
        """
        根据模型路径自动选择（文件/目录）并返回对应的 Pipeline 类。
        """
        model_path = self._resolve_model_path(params.load_name)

        # ===== Anima: dual-backend (runtime/diffusers/auto) =====
        # 长远目标：未来官方推出 diffusers pipeline 时，尽量只需更新 model_families/anima.py 的 model_index_rules，
        # 而不需要再改 detector / injector / param spec。
        try:
            fam = to_model_family(getattr(params, "family", None))
        except Exception:
            fam = ""

        # 读取 backend 偏好（可选）：params.model_cfg.anima.backend = "auto" | "runtime" | "diffusers"
        backend_pref = ""
        try:
            cfg = getattr(params, "model_cfg", None)
            if isinstance(cfg, dict) and isinstance(cfg.get("anima"), dict):
                backend_pref = str(cfg["anima"].get("backend") or "").strip().lower()
        except Exception:
            backend_pref = ""
        if backend_pref not in {"", "auto", "runtime", "diffusers"}:
            backend_pref = ""

        has_anima_manifest = False
        try:
            if model_path.is_dir():
                has_anima_manifest = bool(
                    (model_path / "anima_paths.json").is_file()
                    or (model_path / "anime_paths.json").is_file()  # 兼容常见拼写误差
                    or (model_path / "anima.json").is_file()
                )
        except Exception:
            has_anima_manifest = False

        # 若显式标记为 anima family，或配置/manifest 暗示是 anima，则进入 anima 分支
        if fam == "anima" or backend_pref or has_anima_manifest:
            self.family = "anima"

            def _runtime_ref():
                cat = get_catalog()
                pref = cat.default_pipeline_ref("anima", is_img2img=False)
                if pref is None:
                    raise ValueError("AnimaPipeline default ref missing in catalog")
                return pref.resolve()

            if backend_pref == "runtime":
                return _runtime_ref()

            if backend_pref == "diffusers":
                # 强制 diffusers：要求 model_index.json 存在且能被 catalog 识别，否则直接报错
                return self.from_model_index(params)

            # auto：优先 diffusers（若 model_index.json 存在且 catalog 支持），否则回退 runtime
            try:
                if model_path.is_dir() and (model_path / "model_index.json").is_file():
                    return self.from_model_index(params)
            except Exception:
                pass
            # 回退：manifest 或显式 family=anima 时，走 runtime（paths 可来自 manifest 或 model_config 注入）
            return _runtime_ref()

        logger.debug(
            f"[get_pipeline] 检测模型路径: {model_path}, "
            f"is_file={model_path.is_file()}, suffix={model_path.suffix if model_path.is_file() else 'N/A'}"
        )
        try:
            if model_path.is_file() and model_path.suffix in [".ckpt", ".safetensors"]:
                pipeline_cls = self.from_file(params)
                logger.debug(f"[get_pipeline] 从文件检测，family={self.family}")
                return pipeline_cls
            pipeline_cls = self.from_model_index(params)
            logger.debug(f"[get_pipeline] 从model_index检测，family={self.family}")
            return pipeline_cls
        except Exception as e:
            # 兜底：任何情况下未检测到可用 pipeline，就退回通用 DiffusionPipeline
            try:
                logger.warning(
                    f"[get_pipeline] 未检测到可用pipeline，fallback=DiffusionPipeline. "
                    f"load_name={params.load_name}, resolved={model_path}, err={e}"
                )
            except Exception:
                pass
            self.family = "diffusion"
            cat = get_catalog()
            pref = cat.default_pipeline_ref("diffusion", is_img2img=False)
            if pref is None:
                raise ImportError("DiffusionPipeline default ref missing in catalog")
            return pref.resolve()

    def build_pipeline_params(
        self,
        params: InferenceRequestParams,
        *,
        device_plan: Any,
        model_info: Any,
    ) -> dict:
        """
        根据模型与运行环境构建 pipeline 参数字典。
        返回的键可能包含：transformer、text_encoder_2、original_config、cache_dir、torch_dtype、repo_id、method
        """
        args: dict = {
            "torch_dtype": device_plan.torch_dtype,
        }

        # 注意：此处必须基于“最终决策的 canonical family”，而不是 detector 的隐式状态，
        # 否则会出现 entry(family) 与 detector(family) 不一致导致的注入错配。
        mv = to_model_family(getattr(params, "family", None) or self.family)

        # 统一注入：按 family + 运行模式规划组件覆写（仅保留 model_cfg 的非组件用途）
        overrides = build_component_overrides(
            family=mv,
            model_config=getattr(params, "model_cfg", None),
            inference_config=self.inference_config,
            device_plan=device_plan,
            model_info=model_info,
            logger=logger,
            num_inference_steps=getattr(params, "num_inference_steps", 0),
            fast_mode=bool(getattr(params, "fast_mode", False)),
            low_vram=bool(getattr(params, "low_vram", False)),
        )
        args.update(overrides)

        # 部分家族：降低 CPU 峰值内存（避免加载阶段被 OOM killer 杀）
        if mv in {"qwen", "qwen.edit", "zimage"}:
            args["low_cpu_mem_usage"] = True

        # 3) original_config / cache_dir / use_safetensors 等单文件 special kwargs 已收敛到 build_component_overrides()
        # 4)如果是本地文件模型，需 local_files_only
        args["local_files_only"] = True
        return args
