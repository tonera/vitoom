from dataclasses import dataclass
from pathlib import Path

from common.model_registry import MODEL_REGISTRY


@dataclass
class ModelInfo:
    repo_id: str
    method: str  # from_pretrained 或 from_single_file
    load_name: str


class ModelLocator:
    """
    负责模型路径与加载方式的解析
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def locate(self, params) -> ModelInfo:
        raw = str(getattr(params, "load_name", "") or "").strip()
        if not raw:
            raise ValueError("load_name is required")

        p = Path(raw)
        # 支持绝对路径 / 直接可访问的相对路径（便于脚本与单测）
        if p.is_absolute() or p.exists():
            model_path = p
        else:
            model_path = Path(self.models_dir) / raw

        repo_id = str(model_path)
        if not model_path.exists():
            raise ValueError(f"Model not found at path: {repo_id}")

        method = "from_pretrained" if model_path.is_dir() else "from_single_file"
        # 触发一次 family 归一化（兼容上游传入别名/非 canonical 值）；此处不做特殊改写。
        _ = MODEL_REGISTRY.to_family(getattr(params, "family", None))

        return ModelInfo(repo_id=repo_id, method=method, load_name=raw)

