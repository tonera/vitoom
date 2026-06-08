from pathlib import Path
import sys
import types

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vitoom_setup.model_hub import _download_ms_sdk  # noqa: E402


def test_modelscope_sdk_download_uses_local_dir(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def snapshot_download(**kwargs):
        captured.update(kwargs)

    modelscope_module = types.ModuleType("modelscope")
    hub_module = types.ModuleType("modelscope.hub")
    snapshot_module = types.ModuleType("modelscope.hub.snapshot_download")
    snapshot_module.snapshot_download = snapshot_download

    monkeypatch.setitem(sys.modules, "modelscope", modelscope_module)
    monkeypatch.setitem(sys.modules, "modelscope.hub", hub_module)
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", snapshot_module)

    local_dir = tmp_path / "resources" / "models" / "FLUX.2-klein-9B-Nunchaku"
    _download_ms_sdk("en-US", "tonera/FLUX.2-klein-9B-Nunchaku", local_dir, ())

    assert captured["model_id"] == "tonera/FLUX.2-klein-9B-Nunchaku"
    assert captured["local_dir"] == str(local_dir)
    assert "cache_dir" not in captured
