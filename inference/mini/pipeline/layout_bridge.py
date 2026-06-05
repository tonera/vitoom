"""
版面分析薄封装（layout bridge）

职责：
- 对一张 PIL 图像做版面检测，返回统一 schema 的 BlockDetection 列表
- 首发实现走 `doclayout-yolo`（纯 PyTorch、无 paddle 依赖），未来可并列接入
  GLM-OCR 官方 SDK 里的 PP-DocLayoutV3 或其他 layout 模型

返回的 kind 统一归并到 7 类：
    text / title / table / figure / formula / caption / list

注意：
- 模型权重位置固定，约定与 audio/image 推理器一致：依次在
  {models_dir}/<dir_name>/<weight_file>、{weights_dir}/<dir_name>/<weight_file> 查找。
  不支持运行时配置切换。
- 不同 doclayout-yolo 预训练权重的类别标签可能有差异；映射表在本文件内部收敛。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# pyright: reportMissingImports=false

from common.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 固定常量：文档版面检测模型（与 audio/image 推理器的权重约定保持一致）
#   {models_dir or weights_dir} / DEFAULT_LAYOUT_DIR_NAME / DEFAULT_LAYOUT_WEIGHT_FILE
# 目前本服务只启用一种版面后端（doclayout-yolo，DocStructBench 权重），故写死不暴露配置。
# 若未来需要换后端，在 build_layout_detector 里加分支 + 再加一组常量即可。
# ---------------------------------------------------------------------------

DEFAULT_LAYOUT_DIR_NAME = "DocLayout_YOLO_DocStructBench_imgsz1280_2501"
DEFAULT_LAYOUT_WEIGHT_FILE = "doclayout_yolo_docstructbench_imgsz1280_2501.pt"
# DEFAULT_LAYOUT_WEIGHT_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"


# 推理参数固定值（凭经验选择；conf/iou 按 doclayout-yolo 推荐，imgsz 与权重命名一致）
DEFAULT_LAYOUT_CONF_THR = 0.25
DEFAULT_LAYOUT_IOU_THR = 0.45
DEFAULT_LAYOUT_IMGSZ = 1280


def resolve_layout_weights_path(
    *,
    models_dir: Optional[str] = None,
    weights_dir: Optional[str] = None,
) -> str:
    """在 models_dir / weights_dir 下按约定路径查找版面模型权重。

    约定与 audio 推理器一致：依次尝试
        {root} / DocLayout_YOLO_DocStructBench_imgsz1280_2501 / doclayout_yolo_docstructbench_imgsz1280_2501.pt
    找不到时抛 FileNotFoundError，并把查过的路径一起带出便于排查。
    """
    tried: List[str] = []
    for root in (models_dir, weights_dir):
        root_text = str(root or "").strip()
        if not root_text:
            continue
        candidate = (
            Path(root_text).expanduser().resolve()
            / DEFAULT_LAYOUT_DIR_NAME
            / DEFAULT_LAYOUT_WEIGHT_FILE
        )
        tried.append(str(candidate))
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "DocLayout-YOLO weights not found. Please place "
        f"{DEFAULT_LAYOUT_WEIGHT_FILE} under "
        f"<models_dir>/{DEFAULT_LAYOUT_DIR_NAME}/ (or weights_dir). "
        f"Searched: {tried or ['<no models_dir/weights_dir configured>']}"
    )


# ---------------------------------------------------------------------------
# 统一 schema
# ---------------------------------------------------------------------------

# 7 种标准 kind
BLOCK_KIND_TEXT = "text"
BLOCK_KIND_TITLE = "title"
BLOCK_KIND_TABLE = "table"
BLOCK_KIND_FIGURE = "figure"
BLOCK_KIND_FORMULA = "formula"
BLOCK_KIND_CAPTION = "caption"
BLOCK_KIND_LIST = "list"

_ALL_BLOCK_KINDS = {
    BLOCK_KIND_TEXT,
    BLOCK_KIND_TITLE,
    BLOCK_KIND_TABLE,
    BLOCK_KIND_FIGURE,
    BLOCK_KIND_FORMULA,
    BLOCK_KIND_CAPTION,
    BLOCK_KIND_LIST,
}


@dataclass
class BlockDetection:
    kind: str                             # 见 _ALL_BLOCK_KINDS
    bbox: Tuple[int, int, int, int]       # (x1, y1, x2, y2) 像素整数
    score: float                          # 置信度
    page_index: int                       # 所属页（0-based）
    reading_order: int = 0                # 页内阅读顺序（外部按 (y,x) 排序后填充）

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def x_center(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------


class DocLayoutDetector:
    """版面检测器统一接口。"""

    def detect(self, image: Any, page_index: int = 0) -> List[BlockDetection]:
        raise NotImplementedError

    def close(self) -> None:
        """可选：释放 GPU/内存。默认 no-op。"""
        return None


# ---------------------------------------------------------------------------
# doclayout-yolo 实现
# ---------------------------------------------------------------------------


# doclayout-yolo 默认发布的 DocStructBench 权重含以下类别（v0.0.3 附近版本）：
#   ["title", "plain text", "abandon", "figure", "figure_caption",
#    "table", "table_caption", "table_footnote", "isolate_formula",
#    "formula_caption"]
# 这里把原始 label 映射到统一 kind；未识别的忽略（返回空列表时上游会忽略这块）。
_DOCLAYOUT_YOLO_LABEL_MAP: Dict[str, str] = {
    "title": BLOCK_KIND_TITLE,
    "plain text": BLOCK_KIND_TEXT,
    "plain_text": BLOCK_KIND_TEXT,
    "text": BLOCK_KIND_TEXT,
    "list": BLOCK_KIND_LIST,
    "figure": BLOCK_KIND_FIGURE,
    "figure_caption": BLOCK_KIND_CAPTION,
    "figure caption": BLOCK_KIND_CAPTION,
    "table": BLOCK_KIND_TABLE,
    "table_caption": BLOCK_KIND_CAPTION,
    "table caption": BLOCK_KIND_CAPTION,
    "table_footnote": BLOCK_KIND_CAPTION,
    "isolate_formula": BLOCK_KIND_FORMULA,
    "isolate formula": BLOCK_KIND_FORMULA,
    "formula": BLOCK_KIND_FORMULA,
    "formula_caption": BLOCK_KIND_CAPTION,
    "formula caption": BLOCK_KIND_CAPTION,
    # "abandon" / "header" / "footer" 等视为噪声，不映射
}


class DocLayoutYoloDetector(DocLayoutDetector):
    """doclayout-yolo 实现。

    懒加载：__init__ 不触发 import；首次 detect 时才真正 load。
    权重文件必须是本地已下载好的 .pt 路径（由上层 resolve_layout_weights_path 解析）。
    """

    def __init__(
        self,
        weights_path: str,
        *,
        conf_thr: float = DEFAULT_LAYOUT_CONF_THR,
        iou_thr: float = DEFAULT_LAYOUT_IOU_THR,
        imgsz: int = DEFAULT_LAYOUT_IMGSZ,
    ):
        self.weights_path = weights_path
        self.conf_thr = float(conf_thr)
        self.iou_thr = float(iou_thr)
        self.imgsz = int(imgsz)

        self._model: Any = None
        self._names: Dict[int, str] = {}
        self._torch_device: str = "cpu"

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> None:
        try:
            from doclayout_yolo import YOLOv10  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "doclayout-yolo is not installed. "
                "Please `pip install doclayout-yolo` (see inference/mini/requirements.txt)."
            ) from e

        p = Path(self.weights_path)
        if not p.is_file():
            raise FileNotFoundError(f"DocLayout-YOLO weights file not found: {p}")

        self._torch_device = self._resolve_device()
        logger.info(
            "Loading DocLayoutYolo detector weights=%s device=%s conf=%.2f iou=%.2f imgsz=%d",
            str(p), self._torch_device, self.conf_thr, self.iou_thr, self.imgsz,
        )
        self._model = YOLOv10(str(p))
        # ultralytics 风格：model.names 是 {int: str}
        try:
            names = getattr(self._model, "names", None) or {}
            self._names = {int(k): str(v) for k, v in dict(names).items()}
        except Exception:
            self._names = {}

    # ------------------------------------------------------------------

    def detect(self, image: Any, page_index: int = 0) -> List[BlockDetection]:
        if self._model is None:
            self._load()

        # YOLOv10.predict 接受 PIL.Image / ndarray / 路径；我们统一传 PIL
        results = None
        try:
            try:
                results = self._model.predict(
                    image,
                    imgsz=self.imgsz,
                    conf=self.conf_thr,
                    iou=self.iou_thr,
                    device=self._torch_device,
                    verbose=False,
                )
            except Exception as e:
                raise RuntimeError(f"doclayout-yolo predict failed: {type(e).__name__}: {e}") from e

            blocks: List[BlockDetection] = []
            if not results:
                return blocks

            res = results[0]
            # 优先走 .boxes；不同 ultralytics 版本字段一致
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                return blocks

            try:
                xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
                cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
                confs = boxes.conf.cpu().numpy()
            except Exception as e:
                logger.warning("doclayout-yolo boxes extraction failed: %s", e)
                return blocks

            names = getattr(res, "names", None) or self._names or {}
            for (x1, y1, x2, y2), cls_id, score in zip(xyxy, cls_ids, confs):
                label = str(names.get(int(cls_id), "")).strip().lower()
                kind = _DOCLAYOUT_YOLO_LABEL_MAP.get(label)
                if not kind:
                    # 未知/噪声类别（header/footer/abandon/…）直接丢弃
                    continue
                blocks.append(BlockDetection(
                    kind=kind,
                    bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    score=float(score),
                    page_index=int(page_index),
                    reading_order=0,
                ))

            return blocks
        finally:
            # 关键：ultralytics 的 predictor 会把上一次 predict 的 Results（含 CUDA tensor、orig_img）
            # 挂在 self.predictor.results / self.predictor.batch 里，持续叠加会吃显存和 RAM。
            # 每页处理完就主动切断引用，避免"每跑一份文档显存都增长"。
            try:
                # 切断局部引用
                results = None  # noqa: F841
            except Exception:
                pass
            self._release_predict_cache()

    def _release_predict_cache(self) -> None:
        """切断 ultralytics predictor 内部对上一批推理结果的强引用。
        保留 predictor 本身（包含权重/前处理器），只是不留 per-image 中间产物。
        """
        model = self._model
        if model is None:
            return
        predictor = getattr(model, "predictor", None)
        if predictor is None:
            return
        for attr in ("results", "batch", "plotted_img"):
            try:
                if hasattr(predictor, attr):
                    setattr(predictor, attr, None)
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._release_predict_cache()
        except Exception:
            pass
        self._model = None
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_layout_detector(
    *,
    models_dir: Optional[str] = None,
    weights_dir: Optional[str] = None,
) -> DocLayoutDetector:
    """按固定约定创建默认 detector（DocLayout-YOLO + DocStructBench 权重）。"""
    weights_path = resolve_layout_weights_path(
        models_dir=models_dir,
        weights_dir=weights_dir,
    )
    return DocLayoutYoloDetector(
        weights_path=weights_path,
        conf_thr=DEFAULT_LAYOUT_CONF_THR,
        iou_thr=DEFAULT_LAYOUT_IOU_THR,
        imgsz=DEFAULT_LAYOUT_IMGSZ,
    )


__all__ = [
    "BLOCK_KIND_TEXT",
    "BLOCK_KIND_TITLE",
    "BLOCK_KIND_TABLE",
    "BLOCK_KIND_FIGURE",
    "BLOCK_KIND_FORMULA",
    "BLOCK_KIND_CAPTION",
    "BLOCK_KIND_LIST",
    "DEFAULT_LAYOUT_DIR_NAME",
    "DEFAULT_LAYOUT_WEIGHT_FILE",
    "DEFAULT_LAYOUT_CONF_THR",
    "DEFAULT_LAYOUT_IOU_THR",
    "DEFAULT_LAYOUT_IMGSZ",
    "BlockDetection",
    "DocLayoutDetector",
    "DocLayoutYoloDetector",
    "resolve_layout_weights_path",
    "build_default_layout_detector",
]
