from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.catalog_meta import (
    get_catalog_meta_payload,
    is_valid_modality,
    list_modality_ids,
    normalize_display_family,
    reload_catalog_meta_cache,
)


def test_catalog_meta_loads_modalities_and_families():
    reload_catalog_meta_cache()
    payload = get_catalog_meta_payload()
    ids = list_modality_ids()
    assert "translate" in ids
    assert "image" in ids
    assert len(payload["families"]) >= 40
    assert any(x["value"] == "TranslateGemma" for x in payload["families"])


def test_is_valid_modality():
    reload_catalog_meta_cache()
    assert is_valid_modality("translate")
    assert is_valid_modality("IMAGE")
    assert not is_valid_modality("voice")
    assert not is_valid_modality("")


def test_normalize_display_family():
    reload_catalog_meta_cache()
    assert normalize_display_family("TranslateGemma") == "TranslateGemma"
    assert normalize_display_family("pony") == "Pony"
    assert normalize_display_family("custom-family") == "custom-family"
