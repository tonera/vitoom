from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

import pytest

from text.runtime.runtime_resolver import (
    resolve_text_model_ref,
    resolve_text_runtime,
    resolve_text_runtime_policy,
    resolve_speculative_config,
    try_resolve_local_model_ref,
)


def test_resolve_text_runtime_reads_configured_backend():
    assert (
        resolve_text_runtime(
            SimpleNamespace(
                family="Qwen-text",
                service_runtime={"backend": "vllm"},
            )
        )
        == "vllm"
    )
    assert (
        resolve_text_runtime(
            SimpleNamespace(
                family="Qwen-text",
                service_runtime={"backend": "transformers"},
            )
        )
        == "transformers"
    )


def test_resolve_text_runtime_requires_backend():
    with pytest.raises(ValueError):
        resolve_text_runtime(SimpleNamespace(family="Qwen-text"))


def test_resolve_text_runtime_ignores_model_cfg_runtime():
    """请求侧 model_cfg.runtime 不得影响解析（与 TextInferrer 行为一致）。"""
    assert (
        resolve_text_runtime(
            SimpleNamespace(
                family="Qwen-text",
                service_runtime={"backend": "vllm"},
                model_cfg={"runtime": {"backend": "transformers"}},
            )
        )
        == "vllm"
    )


def test_resolve_text_runtime_policy_reads_runtime_config():
    policy = resolve_text_runtime_policy(
        SimpleNamespace(
            family="Gemma",
            service_runtime={
                "backend": "vllm",
                "max_model_len": 32768,
                "enable_thinking": True,
                "vllm": {
                    "tensor_parallel_size": 2,
                    "gpu_memory_utilization": 0.85,
                    "engine_kwargs": {
                        "limit_mm_per_prompt": {"image": 0, "audio": 0},
                    },
                },
            },
        )
    )

    assert policy.runtime == "vllm"
    assert policy.tensor_parallel_size == 2
    assert policy.gpu_memory_utilization == 0.85
    assert policy.max_model_len == 32768
    assert policy.enable_thinking is True
    assert policy.engine_kwargs["limit_mm_per_prompt"]["image"] == 0


def test_resolve_text_runtime_policy_reads_transformers_config():
    policy = resolve_text_runtime_policy(
        SimpleNamespace(
            family="Qwen-text",
            service_runtime={
                "backend": "transformers",
                "trust_remote_code": True,
                "enable_thinking": True,
                "transformers": {
                    "dtype": "bfloat16",
                    "device_map": "auto",
                    "model_kwargs": {"attn_implementation": "sdpa"},
                },
            },
        )
    )

    assert policy.runtime == "transformers"
    assert policy.dtype == "bfloat16"
    assert policy.device_map == "auto"
    assert policy.enable_thinking is True
    assert policy.allow_cpu_offload is False
    assert policy.model_kwargs == {"attn_implementation": "sdpa"}


def test_resolve_text_runtime_policy_reads_allow_cpu_offload():
    policy = resolve_text_runtime_policy(
        SimpleNamespace(
            family="Qwen-text",
            service_runtime={
                "backend": "transformers",
                "transformers": {"allow_cpu_offload": True},
            },
        )
    )

    assert policy.runtime == "transformers"
    assert policy.allow_cpu_offload is True


def test_resolve_text_runtime_policy_service_max_tokens_absent_means_none():
    policy = resolve_text_runtime_policy(
        SimpleNamespace(
            family="Qwen-text",
            service_runtime={"backend": "vllm"},
        )
    )
    assert policy.service_max_tokens is None


def test_resolve_text_runtime_policy_service_max_tokens_from_yaml():
    policy = resolve_text_runtime_policy(
        SimpleNamespace(
            family="Qwen-text",
            service_runtime={"backend": "vllm", "max_tokens": 8192},
        )
    )
    assert policy.service_max_tokens == 8192


def test_resolve_text_runtime_policy_service_max_tokens_empty_defaults_2048():
    policy = resolve_text_runtime_policy(
        SimpleNamespace(
            family="Qwen-text",
            service_runtime={"backend": "vllm", "max_tokens": None},
        )
    )
    assert policy.service_max_tokens == 2048


def test_resolve_text_model_ref_uses_models_dir(tmp_path: Path):
    model_dir = tmp_path / "nvidia" / "Gemma-4-31B-IT-NVFP4"
    model_dir.mkdir(parents=True)

    resolved = resolve_text_model_ref(
        SimpleNamespace(model_name="nvidia/Gemma-4-31B-IT-NVFP4"),
        models_dir=str(tmp_path),
    )

    assert resolved == str(model_dir.resolve())


def test_resolve_text_model_ref_requires_existing_path(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_text_model_ref(
            SimpleNamespace(model_name="Qwen/Qwen3-8B"),
            models_dir=str(tmp_path),
        )


def test_try_resolve_local_model_ref_checks_weights_dir(tmp_path: Path):
    weights_root = tmp_path / "weights"
    model_dir = weights_root / "gemma-4-31B-it-assistant"
    model_dir.mkdir(parents=True)

    resolved = try_resolve_local_model_ref(
        "gemma-4-31B-it-assistant",
        models_dir=str(tmp_path / "empty_models"),
        weights_dir=str(weights_root),
    )

    assert resolved == str(model_dir.resolve())


def test_resolve_speculative_config_resolves_relative_model(tmp_path: Path):
    assistant_dir = tmp_path / "gemma-4-31B-it-assistant"
    assistant_dir.mkdir()

    resolved = resolve_speculative_config(
        {
            "method": "mtp",
            "model": "gemma-4-31B-it-assistant",
            "num_speculative_tokens": 4,
        },
        models_dir=str(tmp_path),
    )

    assert resolved is not None
    assert resolved["method"] == "mtp"
    assert resolved["model"] == str(assistant_dir.resolve())
    assert resolved["num_speculative_tokens"] == 4


def test_resolve_speculative_config_missing_assistant_returns_none(tmp_path: Path):
    assert (
        resolve_speculative_config(
            {"method": "mtp", "model": "gemma-4-31B-it-assistant"},
            models_dir=str(tmp_path),
        )
        is None
    )


def test_resolve_text_runtime_policy_reads_runtime_speculative_config():
    policy = resolve_text_runtime_policy(
        SimpleNamespace(
            family="Gemma",
            service_runtime={
                "backend": "vllm",
                "speculative_config": {
                    "method": "mtp",
                    "model": "gemma-4-31B-it-assistant",
                    "num_speculative_tokens": 4,
                },
            },
        )
    )

    assert policy.speculative_config == {
        "method": "mtp",
        "model": "gemma-4-31B-it-assistant",
        "num_speculative_tokens": 4,
    }


def test_resolve_text_runtime_policy_ignores_vllm_nested_speculative_config():
    policy = resolve_text_runtime_policy(
        SimpleNamespace(
            family="Gemma",
            service_runtime={
                "backend": "vllm",
                "vllm": {
                    "speculative_config": {
                        "method": "mtp",
                        "model": "gemma-4-31B-it-assistant",
                    },
                },
            },
        )
    )

    assert policy.speculative_config is None


def test_resolve_text_model_ref_returns_ollama_tag_when_configured():
    resolved = resolve_text_model_ref(
        SimpleNamespace(
            model_name="qwen3.6:35b",
            service_runtime={"backend": "ollama", "ollama": {"model_source": "tag"}},
        ),
        models_dir="/tmp/unused",
    )

    assert resolved == "qwen3.6:35b"
