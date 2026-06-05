import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parent.parent / "backend" / "models" / "video_profiles.py"
SPEC = importlib.util.spec_from_file_location("video_profiles", MODULE_PATH)
assert SPEC and SPEC.loader
VIDEO_PROFILES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIDEO_PROFILES)

list_video_task_modes = VIDEO_PROFILES.list_video_task_modes
resolve_video_profile = VIDEO_PROFILES.resolve_video_profile


def test_list_video_task_modes_contains_expected_order():
    modes = list_video_task_modes()
    keys = [item["key"] for item in modes]
    assert keys[:4] == ["t2v", "i2v", "ti2v", "vicv"]
    assert "s2v" in keys
    assert "ccv" in keys


def test_resolve_video_profile_for_multi_capability_model():
    profile = resolve_video_profile(
        {
            "type": "video",
            "name": "Wan2.2-TI2V-5B-FP8",
            "full_name": "Wan2.2-TI2V-5B-FP8",
            "local_path": "/models/Wan2.2-TI2V-5B-FP8",
        }
    )

    assert profile is not None
    assert profile["default_resolution"] == 720
    assert profile["supported_resolutions"] == [720]
    assert [mode["key"] for mode in profile["task_modes"]] == ["t2v", "i2v", "ti2v"]


def test_resolve_video_profile_returns_none_for_unknown_model():
    profile = resolve_video_profile(
        {
            "type": "video",
            "name": "Custom-Video-Model",
            "full_name": "Custom-Video-Model",
        }
    )

    assert profile is None


def test_resolve_video_profile_can_match_by_family():
    profile = resolve_video_profile(
        {
            "type": "video",
            "name": "Some Friendly Name",
            "full_name": "Some Friendly Name",
            "family": "TI2V-5B",
        }
    )

    assert profile is not None
    assert [mode["key"] for mode in profile["task_modes"]] == ["t2v", "i2v", "ti2v"]
