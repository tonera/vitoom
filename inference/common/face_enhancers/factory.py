from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from common.logger import get_logger

from .base import FaceEnhancer, FaceEnhancerBuildConfig

logger = get_logger(__name__)

_CODEFORMER_DEFAULT_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"


def _truthy(v: str) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _resolve_weight(weights_dir: str, fname: str) -> str:
    wd = Path(weights_dir)
    candidates = [
        wd / "Real-ESRGAN" / fname,
        wd / "Real-ESRGAN" / "weights" / fname,
        wd / "roop" / fname,
        wd / fname,
        wd / "weights" / fname,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


def _read_codeformer_w(default: float) -> float:
    raw = os.getenv("VITOOM_CODEFORMER_W")
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        w = float(raw)
    except Exception:
        return float(default)
    # 与原项目一致：通常在 [0,1] 之间，但不强行 clamp，便于实验
    return w


def _download_url_to_path(url: str, dst: Path, *, timeout_seconds: float = 60.0) -> bool:
    """
    best-effort 下载到指定路径（原子落盘：先写 .part，再 rename）。

    注意：该函数不做任何“可信校验”（hash/signature）。如需更严格的供应链控制，请改为
    预置权重到 `{models_dir}/roop/` 并关闭外网访问。
    """
    try:
        import urllib.request
    except Exception:
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=float(timeout_seconds)) as r:  # nosec - trusted model URL by ops
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        os.replace(str(tmp), str(dst))
        logger.info(f"Download succeeded: url={url} dst={dst}")
        return True
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        logger.warning(f"Download failed: url={url} dst={dst} err={e}")
        return False


def build_face_enhancer(cfg: FaceEnhancerBuildConfig) -> Optional[FaceEnhancer]:
    """
    根据环境变量构建人脸增强后端。

    环境变量：
    - VITOOM_FACE_ENHANCER=codeformer|gfpgan（默认 gfpgan）
    - VITOOM_FACE_ENHANCER_STRICT=1（严格模式：构建失败直接抛错；否则 best-effort 回退）
    - VITOOM_CODEFORMER_W=0.5（可选）
    - VITOOM_CODEFORMER_CKPT=codeformer.pth（可选，默认在 {models_dir}/roop 下寻找）
    """

    backend = (os.getenv("VITOOM_FACE_ENHANCER") or "gfpgan").strip().lower()
    strict = _truthy(os.getenv("VITOOM_FACE_ENHANCER_STRICT", "0"))

    w = _read_codeformer_w(cfg.codeformer_w)

    def _build_codeformer() -> tuple[FaceEnhancer, str]:
        from .codeformer_enhancer import CodeFormerEnhancer

        ckpt_name = (os.getenv("VITOOM_CODEFORMER_CKPT") or "codeformer.pth").strip()
        ckpt_path = _resolve_weight(cfg.weights_dir, ckpt_name)
        if not os.path.isfile(ckpt_path):
            # 约定：默认从 {models_dir}/roop/codeformer.pth 读取；若缺失则尝试下载到 roop 目录
            roop_dst = Path(cfg.weights_dir) / "roop" / ckpt_name
            # 仅对默认文件名做自动下载（用户自定义文件名时，避免猜测 URL 造成误下载）
            if ckpt_name == "codeformer.pth":
                logger.info(f"CodeFormer checkpoint missing. Try downloading to: {roop_dst}")
                _download_url_to_path(_CODEFORMER_DEFAULT_URL, roop_dst)
            if roop_dst.is_file():
                ckpt_path = str(roop_dst)
            else:
                raise FileNotFoundError(f"CodeFormer checkpoint not found: {roop_dst}")
        enh = CodeFormerEnhancer(
            ckpt_path=ckpt_path,
            weights_dir=cfg.weights_dir,
            upscale=cfg.upscale,
            bg_upsampler=cfg.bg_upsampler,
            w=w,
        )
        info = f"ckpt={ckpt_path} w={w}"
        return enh, info

    def _build_gfpgan() -> tuple[FaceEnhancer, str]:
        from .gfpgan_enhancer import GFPGANEnhancer

        load_name = "RestoreFormer.pth" if (cfg.arch or "").strip() == "RestoreFormer" else "GFPGANv1.4.pth"
        model_path = _resolve_weight(cfg.weights_dir, load_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"GFPGAN checkpoint not found: {model_path}")
        upscale = cfg.upscale if int(cfg.upscale or 0) >= 1 else 1
        enh = GFPGANEnhancer(
            model_path=model_path,
            weights_dir=cfg.weights_dir,
            arch=(cfg.arch or "clean"),
            upscale=upscale,
            bg_upsampler=cfg.bg_upsampler,
        )
        info = f"ckpt={model_path} arch={cfg.arch or 'clean'} upscale={upscale}"
        return enh, info

    def _log_selected(*, requested: str, selected: str, detail: str) -> None:
        try:
            logger.info(f"FaceEnhancer selected backend={selected} requested={requested} {detail}")
        except Exception:
            pass

    # 1) preferred backend
    try:
        if backend == "gfpgan":
            enh, detail = _build_gfpgan()
            _log_selected(requested=backend, selected="gfpgan", detail=detail)
            return enh
        enh, detail = _build_codeformer()
        _log_selected(requested=backend, selected="codeformer", detail=detail)
        return enh
    except Exception as e:
        if strict:
            raise
        logger.warning(f"FaceEnhancer backend '{backend}' init failed: {e}. Falling back if possible.")

    # 2) fallback backend
    try:
        if backend != "gfpgan":
            enh, detail = _build_gfpgan()
            _log_selected(requested=backend, selected="gfpgan", detail=detail)
            return enh
        enh, detail = _build_codeformer()
        _log_selected(requested=backend, selected="codeformer", detail=detail)
        return enh
    except Exception as e:
        if strict:
            raise
        logger.warning(f"FaceEnhancer fallback init failed: {e}. Face enhancement will be disabled.")
        return None

