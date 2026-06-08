from __future__ import annotations

from pathlib import Path


def test_backend_task_create_request_allows_guidance_scale_zero():
    # Import inside test to avoid side effects during collection.
    from backend.api.tasks.routes import TaskCreateRequest

    req = TaskCreateRequest(task_type="image", guidance_scale=0)
    assert req.guidance_scale == 0


def test_inference_request_params_allows_guidance_scale_zero():
    from inference.schemas import InferenceRequestParams

    params = InferenceRequestParams(
        type="image",
        job_type="MK",
        id="msg-1",
        user_id="u-1",
        task_id="t-1",
        prompt="hello",
        guidance_scale=0,
    )
    assert params.guidance_scale == 0


def test_no_truthy_fallback_overwrites_guidance_scale_zero_in_inference_handlers():
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "inference" / "image" / "handlers" / "pose_handler.py",
        repo_root / "inference" / "image" / "handlers" / "id_handler.py",
    ]

    bad_snippets = [
        'getattr(params, "guidance_scale", 7.5) or 7.5',
        'getattr(params, "guidance_scale", 3.5) or 3.5',
    ]

    for fp in targets:
        text = fp.read_text(encoding="utf-8")
        for s in bad_snippets:
            assert s not in text, f"Found legacy truthy fallback in {fp}: {s}"


def test_no_truthy_fallback_overwrites_guidance_scale_zero_in_frontend_image_views():
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "frontend" / "src" / "views" / "image" / "ImageEdit.vue",
        repo_root / "frontend" / "src" / "views" / "image" / "ImageControler.vue",
        repo_root / "frontend" / "src" / "views" / "image" / "ImageGenerate.vue",
    ]

    for fp in targets:
        text = fp.read_text(encoding="utf-8")
        # 旧逻辑会把 0 当成 falsy：Number(x) || form.guidanceScale
        assert "|| form.guidanceScale" not in text, f"Found legacy truthy fallback in {fp}"
        assert "form.guidanceScale = Number(" not in text, f"Found legacy assignment in {fp}"

