"""
换脸处理器：参考 aiservice/diffusers/SwapFaceProcessor.py

约定（与参考实现一致）：
- generate(images): images[0] 为源脸，其余为目标图；返回等长列表，首图原样返回，其余为换脸结果。

依赖：
- insightface, onnxruntime
- numpy, pillow
- 可选：gfpgan（用于增强）

权重目录约定（weights_root 下；推荐指向 `{models_dir}/roop`）：
- buffalo_l/（人脸检测/识别套件目录）
- inswapper_128.onnx
- GFPGANv1.4.pth（可选）
- detection_Resnet50_Final.pth / parsing_parsenet.pth（可选，用于 GFPGAN 辅助）
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PIL import Image

from .logger import get_logger

logger = get_logger(__name__)


def _pil_to_bgr_numpy(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _bgr_numpy_to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = image_bgr[:, :, ::-1]
    return Image.fromarray(rgb)


class SwapFaceProcessor:
    def __init__(
        self,
        weights_root: str,
        use_enhancer: bool = True,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.weights_root = weights_root
        self.use_enhancer = use_enhancer

        if providers is None:
            # 默认策略：优先 CUDA（若系统真的可用），否则回退 CPU。
            # 这里不能只看 torch.cuda.is_available()：常见情况是 onnxruntime-gpu 已安装、
            # 但缺少 CUDA 运行库（例如 libcublasLt.so.12），会导致 ORT 反复报错后再回退 CPU。
            want_cuda = False
            try:
                import torch

                want_cuda = bool(torch.cuda.is_available())
            except Exception:
                want_cuda = False

            providers = ["CPUExecutionProvider"]
            if want_cuda:
                try:
                    import onnxruntime as ort  # type: ignore

                    # 如果探测时抛错/不可用，直接退回 CPU，避免大量 provider_bridge 报错日志。
                    available = set(getattr(ort, "get_available_providers", lambda: [])() or [])
                    if "CUDAExecutionProvider" in available:
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    else:
                        logger.info(
                            "onnxruntime CUDAExecutionProvider is not available; falling back to CPUExecutionProvider. "
                            "If you expect GPU, install CUDA 12 runtime (libcublasLt.so.12) and a matching onnxruntime-gpu."
                        )
                except Exception:
                    logger.info(
                        "onnxruntime CUDA provider probe failed; falling back to CPUExecutionProvider. "
                        "If you expect GPU, install CUDA 12 runtime (libcublasLt.so.12) and a matching onnxruntime-gpu."
                    )
        self.providers = providers

        self.buffalo_dir = os.path.join(self.weights_root, "buffalo_l")
        self.inswapper_path = os.path.join(self.weights_root, "inswapper_128.onnx")
        self.gfpgan_path = os.path.join(self.weights_root, "GFPGANv1.4.pth")

        # 让 insightface 在 weights_root 下落盘 models 缓存，避免跑到用户家目录
        os.environ["INSIGHTFACE_HOME"] = self.weights_root

        self._face_analyser = None
        self._swapper = None
        self._gfpgan = None

    def _prepare_insightface_models(self) -> None:
        models_dir = os.path.join(self.weights_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        src = os.path.join(self.weights_root, "buffalo_l")
        dst = os.path.join(models_dir, "buffalo_l")
        if os.path.isdir(dst):
            return
        if os.path.isdir(src):
            try:
                os.symlink(src, dst)
            except Exception:
                try:
                    import shutil

                    shutil.copytree(src, dst, dirs_exist_ok=True)
                except Exception:
                    pass

    def _prepare_gfpgan_aux_weights(self) -> None:
        default_dir = os.path.abspath(os.path.join(os.getcwd(), "gfpgan", "weights"))
        os.makedirs(default_dir, exist_ok=True)
        candidates = ["detection_Resnet50_Final.pth", "parsing_parsenet.pth"]
        for fname in candidates:
            src = os.path.join(self.weights_root, fname)
            dst = os.path.join(default_dir, fname)
            if os.path.isfile(dst):
                continue
            if os.path.isfile(src):
                try:
                    os.symlink(src, dst)
                except Exception:
                    try:
                        import shutil

                        shutil.copy2(src, dst)
                    except Exception:
                        pass

    def _ensure_face_analyser(self):
        if self._face_analyser is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:  # pragma: no cover
            raise RuntimeError("FS requires package 'insightface'") from e

        self._prepare_insightface_models()

        self._face_analyser = FaceAnalysis(
            name="buffalo_l", root=self.weights_root, providers=self.providers
        )
        ctx_id = 0 if ("CUDAExecutionProvider" in self.providers) else -1
        self._face_analyser.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def _ensure_swapper(self):
        if self._swapper is not None:
            return
        try:
            from insightface import model_zoo
        except Exception as e:  # pragma: no cover
            raise RuntimeError("FS requires package 'insightface'") from e

        if not os.path.isfile(self.inswapper_path):
            raise FileNotFoundError(f"inswapper_128.onnx not found: {self.inswapper_path}")
        self._swapper = model_zoo.get_model(self.inswapper_path, providers=self.providers)

    def _ensure_gfpgan(self):
        if self._gfpgan is not None or not self.use_enhancer:
            return
        try:
            from gfpgan.utils import GFPGANer
        except Exception:
            self.use_enhancer = False
            return

        self._prepare_gfpgan_aux_weights()
        if not os.path.isfile(self.gfpgan_path):
            self.use_enhancer = False
            return
        device = "cuda" if ("CUDAExecutionProvider" in self.providers) else "cpu"
        self._gfpgan = GFPGANer(model_path=self.gfpgan_path, upscale=1, device=device)

    @staticmethod
    def _pick_largest_face(faces):
        if not faces:
            return None

        def area(f):
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            return max(0, x2 - x1) * max(0, y2 - y1)

        return max(faces, key=area)

    def generate(self, images: List[Image.Image]) -> List[Image.Image]:
        if images is None or len(images) < 2:
            raise ValueError("FS requires at least 2 images: [source, target1, ...]")

        self._ensure_face_analyser()
        self._ensure_swapper()
        self._ensure_gfpgan()

        source_bgr = _pil_to_bgr_numpy(images[0])
        source_faces = self._face_analyser.get(source_bgr)
        source_face = self._pick_largest_face(source_faces)
        if source_face is None:
            raise RuntimeError("No face detected in source image")

        results: List[Image.Image] = [images[0]]
        for idx in range(1, len(images)):
            target_bgr = _pil_to_bgr_numpy(images[idx])
            target_faces = self._face_analyser.get(target_bgr)
            if not target_faces:
                results.append(images[idx])
                continue
            swapped = target_bgr
            for tf in target_faces:
                swapped = self._swapper.get(swapped, tf, source_face, paste_back=True)

            if self.use_enhancer and self._gfpgan is not None:
                try:
                    _, _, restored = self._gfpgan.enhance(swapped, paste_back=True)
                    swapped = restored if restored is not None else swapped
                except Exception:
                    pass

            results.append(_bgr_numpy_to_pil(swapped))

        return results


