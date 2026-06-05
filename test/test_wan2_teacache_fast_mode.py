from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest


def _load_wan2_call_utils_module():
    """
    通过文件路径加载模块，避免 import `video` 包触发 inference/video/__init__.py 的副作用。
    同时注入 inference/ 到 sys.path，保证 `common.*` / `schemas` 这类“运行期 sys.path 约定”可用。
    """
    repo_root = Path(__file__).resolve().parents[1]
    inference_root = repo_root / "inference"
    if str(inference_root) not in sys.path:
        sys.path.insert(0, str(inference_root))

    # 只按文件路径注入 wan2_call_utils 依赖的轻量 common 子模块，
    # 避免触发 common/__init__.py 里的重型导入（如 redis.asyncio）。
    if "common" not in sys.modules:
        pkg = types.ModuleType("common")
        pkg.__path__ = [str(inference_root / "common")]
        sys.modules["common"] = pkg

    for name in ("logger", "task_cancel"):
        full_name = f"common.{name}"
        if full_name in sys.modules:
            continue
        sub_path = inference_root / "common" / f"{name}.py"
        sub_spec = importlib.util.spec_from_file_location(full_name, str(sub_path))
        assert sub_spec is not None and sub_spec.loader is not None
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[full_name] = sub_mod
        sub_spec.loader.exec_module(sub_mod)  # type: ignore[attr-defined]

    p = repo_root / "inference" / "video" / "runtime" / "wan2_call_utils.py"
    spec = importlib.util.spec_from_file_location("_wan2_call_utils_for_test", str(p))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_fast_mode_enables_teacache_for_wan21_i2v_720p():
    mod = _load_wan2_call_utils_module()
    req = SimpleNamespace(
        fast_mode=True,
        model_name="Wan2.1-I2V-14B",
        height=720,
        width=1280,
    )
    kw = mod.build_wan2_teacache_kwargs(req, pipe_name="i2v")
    assert kw["tea_cache_model_id"] == "Wan2.1-I2V-14B-720P"
    assert float(kw["tea_cache_l1_thresh"]) > 0


def test_fast_mode_skips_teacache_for_wan22_by_default():
    mod = _load_wan2_call_utils_module()
    req = SimpleNamespace(
        fast_mode=True,
        model_name="Wan2.2-T2V-A14B",
        height=480,
        width=832,
    )
    kw = mod.build_wan2_teacache_kwargs(req, pipe_name="t2v")
    assert kw == {}


def test_call_pipe_raises_cancelled_before_pipe_call():
    mod = _load_wan2_call_utils_module()

    class _Pipe:
        def __call__(self, prompt: str):
            raise AssertionError("pipe should not run when already cancelled")

    with pytest.raises(mod.TaskCancelledError, match="before pipe call"):
        mod.call_pipe(
            _Pipe(),
            task_id="task-1",
            stage="mkv.t2v denoise",
            is_task_cancelled=lambda: True,
            prompt="hello",
        )


def test_call_pipe_raises_cancelled_during_progress_steps():
    mod = _load_wan2_call_utils_module()
    consumed: list[int] = []
    check_count = {"value": 0}

    class _Pipe:
        def __call__(self, prompt: str, progress_bar_cmd=None):
            assert callable(progress_bar_cmd)
            for step in progress_bar_cmd(range(5)):
                consumed.append(step)
            return prompt

    def _is_task_cancelled() -> bool:
        check_count["value"] += 1
        return check_count["value"] >= 3

    with pytest.raises(mod.TaskCancelledError, match=r"step=2"):
        mod.call_pipe(
            _Pipe(),
            task_id="task-2",
            stage="mkv.t2v denoise",
            is_task_cancelled=_is_task_cancelled,
            prompt="hello",
        )

    assert consumed == [0]

