from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 允许在 pytest 下直接导入 inference 下的模块（common/*）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_default_backend_is_gfpgan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # default backend is gfpgan
    monkeypatch.delenv("VITOOM_FACE_ENHANCER", raising=False)
    monkeypatch.delenv("VITOOM_FACE_ENHANCER_STRICT", raising=False)
    monkeypatch.delenv("VITOOM_CODEFORMER_CKPT", raising=False)

    _touch(tmp_path / "roop" / "GFPGANv1.4.pth")

    from common.face_enhancers import FaceEnhancerBuildConfig, build_face_enhancer  # type: ignore

    cfg = FaceEnhancerBuildConfig(weights_dir=str(tmp_path), arch="clean", upscale=1, bg_upsampler=None)
    enh = build_face_enhancer(cfg)
    assert enh is not None
    assert enh.__class__.__name__ == "GFPGANEnhancer"


def test_backend_switch_to_gfpgan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VITOOM_FACE_ENHANCER", "gfpgan")
    monkeypatch.delenv("VITOOM_FACE_ENHANCER_STRICT", raising=False)

    _touch(tmp_path / "roop" / "GFPGANv1.4.pth")

    from common.face_enhancers import FaceEnhancerBuildConfig, build_face_enhancer  # type: ignore

    cfg = FaceEnhancerBuildConfig(weights_dir=str(tmp_path), arch="clean", upscale=1, bg_upsampler=None)
    enh = build_face_enhancer(cfg)
    assert enh is not None
    assert enh.__class__.__name__ == "GFPGANEnhancer"


def test_fallback_to_codeformer_when_gfpgan_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # default is gfpgan, but its ckpt missing -> fallback to codeformer when available
    monkeypatch.delenv("VITOOM_FACE_ENHANCER", raising=False)
    monkeypatch.delenv("VITOOM_FACE_ENHANCER_STRICT", raising=False)
    monkeypatch.delenv("VITOOM_CODEFORMER_CKPT", raising=False)

    _touch(tmp_path / "roop" / "codeformer.pth")

    from common.face_enhancers import FaceEnhancerBuildConfig, build_face_enhancer  # type: ignore

    # avoid real network call in tests
    import common.face_enhancers.factory as fac  # type: ignore

    monkeypatch.setattr(fac, "_download_url_to_path", lambda *a, **k: False)

    cfg = FaceEnhancerBuildConfig(weights_dir=str(tmp_path), arch="clean", upscale=1, bg_upsampler=None)
    enh = build_face_enhancer(cfg)
    assert enh is not None
    assert enh.__class__.__name__ == "CodeFormerEnhancer"


def test_strict_mode_raises_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VITOOM_FACE_ENHANCER", "codeformer")
    monkeypatch.setenv("VITOOM_FACE_ENHANCER_STRICT", "1")
    monkeypatch.setenv("VITOOM_CODEFORMER_CKPT", "codeformer.pth")

    from common.face_enhancers import FaceEnhancerBuildConfig, build_face_enhancer  # type: ignore

    # avoid real network call in tests
    import common.face_enhancers.factory as fac  # type: ignore

    monkeypatch.setattr(fac, "_download_url_to_path", lambda *a, **k: False)

    cfg = FaceEnhancerBuildConfig(weights_dir=str(tmp_path), arch="clean", upscale=1, bg_upsampler=None)
    with pytest.raises(FileNotFoundError):
        build_face_enhancer(cfg)

