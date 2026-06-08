import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _pil_to_bgr_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to OpenCV-style BGR numpy array."""
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


class GenericFaceProcessor:
    """
    通用人脸处理类（参考 SwapFaceProcessor 的 insightface 初始化方式）：

    1) detect_face_bboxes(image):
       - 输入一张图片，返回图中所有人脸的 bbox 坐标列表 (x1, y1, x2, y2)
       - 无人脸返回 None

    2) mask_from_bboxes(image, bboxes):
       - 输入一张图片和一组 bbox 坐标，返回与原图尺寸一致的 mask 图片（PIL, mode='L'）
       - bbox 区域为 255，其余为 0
    """

    def __init__(self, models_dir: str, providers: Optional[List[str]] = None) -> None:
        """
        Args:
            models_dir: 模型目录（支持两种常见结构）：
              - A) "{base_dir}/weights/models" 且存在 "{models_dir}/buffalo_l"
              - B) "models/roop" 且存在 "{models_dir}/models/buffalo_l"
            providers: onnxruntime providers，默认会按是否可用 CUDA 推断
        """
        self.models_dir = models_dir
        # 两种可能的 buffalo_l 位置
        self._buffalo_dir_flat = os.path.join(self.models_dir, "buffalo_l")  # A)
        self._buffalo_dir_nested = os.path.join(self.models_dir, "models", "buffalo_l")  # B)

        # 推理后端优先级（尽量复用 SwapFaceProcessor 的策略）
        if providers is None:
            try:
                import torch

                cuda_available = torch.cuda.is_available()
            except Exception:
                cuda_available = False
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_available else ["CPUExecutionProvider"]
        self.providers = providers

        # 让 insightface 在指定目录下查找缓存/模型
        # - 若 models_dir 已经包含 models/buffalo_l，则直接把 root 指向 models_dir
        # - 否则沿用 SwapFaceProcessor 的约定：root 指向 models_dir 的父目录，模型落在 {root}/models/buffalo_l
        if os.path.isdir(self._buffalo_dir_nested):
            self._insightface_root = self.models_dir
        else:
            self._insightface_root = os.path.dirname(self.models_dir)

        os.environ["INSIGHTFACE_HOME"] = self._insightface_root

        self._face_analyser = None

    def _prepare_insightface_models(self) -> None:
        """
        insightface 默认会在 {INSIGHTFACE_HOME}/models 下查找套件。
        若本地已存在 buffalo_l，但 INSIGHTFACE_HOME 期望位置不存在，则软链/复制就位。
        """
        models_cache_dir = os.path.join(self._insightface_root, "models")
        os.makedirs(models_cache_dir, exist_ok=True)
        dst = os.path.join(models_cache_dir, "buffalo_l")
        if os.path.isdir(dst):
            return
        # 选择可用的 src
        src = None
        if os.path.isdir(self._buffalo_dir_nested):
            # models_dir/models/buffalo_l 已经就位：这种情况下 dst==src，直接返回即可
            return
        if os.path.isdir(self._buffalo_dir_flat):
            src = self._buffalo_dir_flat

        if src is None:
            # 不自动下载，直接给出清晰错误，避免悄悄拉网下载到意外目录
            raise FileNotFoundError(
                "未找到本地 buffalo_l 模型目录。请确认以下之一存在：\n"
                f"- {self._buffalo_dir_flat}\n"
                f"- {self._buffalo_dir_nested}\n"
                "否则 insightface 会尝试联网下载。"
            )

        try:
            os.symlink(src, dst)
        except Exception:
            try:
                import shutil

                shutil.copytree(src, dst, dirs_exist_ok=True)
            except Exception:
                pass

    def _ensure_face_analyser(self) -> None:
        if self._face_analyser is not None:
            return
        from insightface.app import FaceAnalysis

        self._prepare_insightface_models()

        self._face_analyser = FaceAnalysis(name="buffalo_l", root=self._insightface_root, providers=self.providers)
        ctx_id = 0 if ("CUDAExecutionProvider" in self.providers) else -1
        self._face_analyser.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect_face_bboxes(self, image: Image.Image) -> Optional[List[BBox]]:
        """
        检测图片中所有人脸 bbox。

        Returns:
            - None: 未检测到人脸
            - List[BBox]: 检测到的人脸 bbox 列表（按面积从大到小排序）
        """
        self._ensure_face_analyser()
        bgr = _pil_to_bgr_numpy(image)
        faces = self._face_analyser.get(bgr)
        if not faces:
            return None

        bboxes: List[BBox] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(float).tolist()
            bboxes.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))

        # 按面积从大到小（便于取最大脸时直接 bboxes[0]）
        bboxes.sort(key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]), reverse=True)
        return bboxes

    @staticmethod
    def mask_from_bboxes(
        image: Image.Image,
        bboxes: Optional[Sequence[Union[BBox, Sequence[Union[int, float]]]]] = None,
        fill: int = 255,
    ) -> Image.Image:
        """
        根据 bbox 生成 mask（与 image 同尺寸）。

        Args:
            image: 原图（用于确定输出尺寸）
            bboxes: bbox 列表，每个 bbox 是 (x1, y1, x2, y2)。传 None/空列表会返回全黑 mask。
            fill: bbox 区域的填充值（0-255）

        Returns:
            PIL.Image (mode='L')，size 与 image 一致。
        """
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        if not bboxes:
            return Image.fromarray(mask, mode="L")

        def _norm_box(box: Union[BBox, Sequence[Union[int, float]]]) -> BBox:
            if len(box) != 4:
                raise ValueError(f"bbox 必须是 4 个数字 (x1,y1,x2,y2)，实际为: {box}")
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            # 允许输入乱序，自动纠正
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            # clip 到图像范围
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            return (x1, y1, x2, y2)

        v = int(max(0, min(255, fill)))
        for box in bboxes:
            x1, y1, x2, y2 = _norm_box(box)
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = v

        return Image.fromarray(mask, mode="L")


