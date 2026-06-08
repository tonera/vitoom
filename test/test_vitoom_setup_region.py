"""Vitoom setup locale/region helpers."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vitoom_setup.constants import CN_MIRROR  # noqa: E402
from vitoom_setup.region import (  # noqa: E402
    locale_from_env_value,
    region_env_updates,
    region_from_env_value,
    region_from_locale,
)


def test_locale_from_env_value():
    assert locale_from_env_value("") is None
    assert locale_from_env_value("zh-CN") == "zh-CN"
    assert locale_from_env_value("ja-JP") == "ja-JP"
    assert locale_from_env_value("en-US") == "en-US"
    assert locale_from_env_value("invalid") is None


def test_region_from_env_value():
    assert region_from_env_value("") is None
    assert region_from_env_value("cn") == "cn"
    assert region_from_env_value("china") == "cn"
    assert region_from_env_value("intl") == "intl"
    assert region_from_env_value("global") == "intl"


def test_region_from_locale_legacy_fallback():
    assert region_from_locale("zh-CN") == "cn"
    assert region_from_locale("ja-JP") == "intl"
    assert region_from_locale("en-US") == "intl"


def test_region_env_updates_cn_sets_aliyun_mirrors():
    updates = region_env_updates("cn")
    assert updates["VITOOM_REGION"] == "cn"
    assert updates["APT_MIRROR"] == CN_MIRROR["APT_MIRROR"]
    assert updates["PIP_INDEX_URL"] == CN_MIRROR["PIP_INDEX_URL"]


def test_region_env_updates_intl_clears_mirrors():
    updates = region_env_updates("intl")
    assert updates["VITOOM_REGION"] == "intl"
    assert updates["APT_MIRROR"] == ""
    assert updates["PIP_INDEX_URL"] == ""
