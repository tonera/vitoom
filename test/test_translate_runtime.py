from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

from translate.runtime.runtime_resolver import merge_translate_runtime_cfg, resolve_translate_backend


def test_merge_translate_runtime_cfg():
    merged = merge_translate_runtime_cfg(
        {
            "backend": "transformers",
            "trust_remote_code": True,
            "transformers": {"device_map": "cuda:0", "dtype": "bf16"},
        },
        backend="transformers",
    )
    assert merged["device_map"] == "cuda:0"
    assert merged["dtype"] == "bf16"
    assert merged["trust_remote_code"] is True
    assert "backend" not in merged


def test_resolve_translate_backend():
    assert resolve_translate_backend({"runtime": {"backend": "transformers"}}) == "transformers"
    assert resolve_translate_backend({"runtime": {"backend": "hf"}}) == "transformers"
