import hashlib
import json
from types import SimpleNamespace


def _make_min_params(InferenceRequestParams, **overrides):
    base = dict(
        type="image",
        job_type="MK",
        id="msg-1",
        user_id="u1",
        task_id="t1",
        prompt="a cat",
        family="sdxl",
        model_name="foo",
    )
    base.update(overrides)
    return InferenceRequestParams(**base)


def test_is_v_prediction_model_uses_path_for_cache_dir(monkeypatch, tmp_path):
    """
    回归测试：models_dir 为 str 时，_is_v_prediction_model 内部不能出现 `str / str`。
    并且应能正确定位 cache 文件并命中返回，不触发 heavy 的 quick-latent-test。
    """
    from inference.schemas import InferenceRequestParams
    from inference.image.runtime import scheduler_loader

    models_dir_str = str(tmp_path)
    model_name = "foo"
    cache_key = f"{models_dir_str}|{model_name}"
    sha = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()

    cache_dir = tmp_path / "cache" / "prediction_type"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sha}.json"
    cache_path.write_text(json.dumps({"key": cache_key, "is_v_prediction": True}), encoding="utf-8")

    monkeypatch.setattr(scheduler_loader, "load_inference_config", lambda: SimpleNamespace(models_dir=models_dir_str))
    params = _make_min_params(InferenceRequestParams, model_name=model_name)
    assert scheduler_loader._is_v_prediction_model(params) is True


def test_is_v_prediction_model_returns_false_when_models_dir_missing(monkeypatch):
    from inference.schemas import InferenceRequestParams
    from inference.image.runtime import scheduler_loader

    monkeypatch.setattr(scheduler_loader, "load_inference_config", lambda: SimpleNamespace(models_dir=""))
    params = _make_min_params(InferenceRequestParams, model_name="foo")
    assert scheduler_loader._is_v_prediction_model(params) is False

