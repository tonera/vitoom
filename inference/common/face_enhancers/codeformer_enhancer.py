from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from common.logger import get_logger

logger = get_logger(__name__)


_ARC_FACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _pick_largest_face(faces: List[Any]) -> Optional[Any]:
    if not faces:
        return None

    def area(f) -> float:
        x1, y1, x2, y2 = f.bbox.astype(float).tolist()
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    return max(faces, key=area)


class CodeFormerEnhancer:
    def __init__(
        self,
        *,
        ckpt_path: str,
        weights_dir: str,
        upscale: int,
        bg_upsampler: Optional[Any],
        w: float = 0.5,
        face_size: int = 512,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.weights_dir = weights_dir
        self.upscale = int(upscale or 1)
        self.bg_upsampler = bg_upsampler
        self.w = float(w)
        self.face_size = int(face_size)

        self._device = None
        self._net = None
        self._face_analyser = None

    # -------- insightface helpers --------
    def _resolve_roop_dir(self) -> str:
        return os.path.join(self.weights_dir, "roop")

    def _prepare_insightface_models(self, root: str) -> None:
        models_dir = os.path.join(root, "models")
        os.makedirs(models_dir, exist_ok=True)
        src = os.path.join(root, "buffalo_l")
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

    def _ensure_face_analyser(self):
        if self._face_analyser is not None:
            return

        roop_dir = self._resolve_roop_dir()
        os.makedirs(roop_dir, exist_ok=True)
        os.environ["INSIGHTFACE_HOME"] = roop_dir

        providers = ["CPUExecutionProvider"]
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    import onnxruntime as ort  # type: ignore

                    available = set(getattr(ort, "get_available_providers", lambda: [])() or [])
                    if "CUDAExecutionProvider" in available:
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                except Exception:
                    providers = ["CPUExecutionProvider"]
        except Exception:
            providers = ["CPUExecutionProvider"]

        from insightface.app import FaceAnalysis  # type: ignore

        self._prepare_insightface_models(roop_dir)
        self._face_analyser = FaceAnalysis(name="buffalo_l", root=roop_dir, providers=providers)
        ctx_id = 0 if ("CUDAExecutionProvider" in providers) else -1
        self._face_analyser.prepare(ctx_id=ctx_id, det_size=(640, 640))

    # -------- codeformer model helpers --------
    def _ensure_model(self):
        if self._net is not None:
            return

        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        from third_party.codeformer.model import CodeFormer  # local vendored

        net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        )

        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"CodeFormer checkpoint not found: {self.ckpt_path}")

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("params_ema") or ckpt.get("params") or ckpt.get("state_dict") or ckpt
        else:
            state = ckpt
        net.load_state_dict(state, strict=True)
        net.eval()
        net.to(self._device)
        self._net = net

    def _restore_face_bgr(self, face_bgr_512: np.ndarray) -> np.ndarray:
        import torch

        self._ensure_model()
        assert self._net is not None
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")

        # BGR uint8 -> RGB float32 [0,1]
        rgb = cv2.cvtColor(face_bgr_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # to tensor BCHW
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        # normalize to [-1, 1] with mean=0.5 std=0.5
        x = (x - 0.5) / 0.5
        x = x.to(device)

        with torch.no_grad():
            out = self._net(x, w=float(self.w), adain=True)[0]  # type: ignore[misc]
        out = out.detach().float().clamp(-1, 1)
        out = (out + 1.0) / 2.0
        out = (out * 255.0).round().clamp(0, 255).byte()
        out_rgb = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr

    # -------- geometry helpers --------
    def _estimate_affine_5p(self, kps5: np.ndarray) -> Optional[np.ndarray]:
        if kps5 is None or np.asarray(kps5).shape != (5, 2):
            return None
        src = np.asarray(kps5, dtype=np.float32)
        scale = float(self.face_size) / 112.0
        dst = _ARC_FACE_TEMPLATE_112 * scale
        m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        return m

    @staticmethod
    def _soft_mask(mask: np.ndarray) -> np.ndarray:
        # mask: HxW uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # blur edges
        k = 31
        if k % 2 == 0:
            k += 1
        m = cv2.GaussianBlur(mask, (k, k), 0)
        return m

    def enhance(
        self,
        bgr: np.ndarray,
        *,
        has_aligned: bool = False,
        only_center_face: bool = False,
        paste_back: bool = True,
    ) -> np.ndarray:
        if bgr is None:
            return bgr
        if not isinstance(bgr, np.ndarray):
            return bgr
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            return bgr

        try:
            if bool(has_aligned):
                face = cv2.resize(bgr, (self.face_size, self.face_size), interpolation=cv2.INTER_LINEAR)
                restored = self._restore_face_bgr(face)
                return restored

            self._ensure_face_analyser()
            assert self._face_analyser is not None

            faces = self._face_analyser.get(bgr)
            if not faces:
                return bgr

            if bool(only_center_face):
                f = _pick_largest_face(faces)
                faces = [f] if f is not None else []

            h, w = bgr.shape[:2]
            out = bgr.copy()

            for f in faces:
                kps = getattr(f, "kps", None)
                m = self._estimate_affine_5p(kps)
                if m is None:
                    continue

                face_crop = cv2.warpAffine(
                    bgr,
                    m,
                    (self.face_size, self.face_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                restored_face = self._restore_face_bgr(face_crop)

                if not paste_back:
                    out = restored_face
                    continue

                minv = cv2.invertAffineTransform(m)
                restored_back = cv2.warpAffine(
                    restored_face,
                    minv,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                mask = np.full((self.face_size, self.face_size), 255, dtype=np.uint8)
                mask_back = cv2.warpAffine(mask, minv, (w, h), flags=cv2.INTER_LINEAR)
                mask_back = self._soft_mask(mask_back)

                alpha = (mask_back.astype(np.float32) / 255.0)[:, :, None]
                out = (alpha * restored_back.astype(np.float32) + (1.0 - alpha) * out.astype(np.float32)).round().clip(
                    0, 255
                ).astype(np.uint8)

            # 若需要 upscale，则与 GFPGAN 行为对齐：增强后输出即为目标倍率
            if int(self.upscale or 1) in (2, 4) and self.bg_upsampler is not None:
                try:
                    up, _ = self.bg_upsampler.enhance(out, outscale=int(self.upscale))
                    out = up if up is not None else out
                except TypeError:
                    # 某些 RealESRGANer 返回 (out, _)；另一些可能只返回 out
                    try:
                        up = self.bg_upsampler.enhance(out, outscale=int(self.upscale))
                        out = up[0] if isinstance(up, (list, tuple)) else up
                    except Exception:
                        pass

            return out
        except Exception as e:
            logger.warning(f"CodeFormer enhance failed: {e}")
            return bgr

    def close(self) -> None:
        self._face_analyser = None
        self._net = None
        self._device = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

