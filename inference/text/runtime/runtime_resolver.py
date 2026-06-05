from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TextRuntimePolicy:
    runtime: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: Optional[int]
    trust_remote_code: bool
    enable_thinking: bool
    allow_cpu_offload: bool = False
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: str = "auto"
    device_map: str = "auto"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    # ollama 专属参数原样透传（host / tag_prefix / keep_alive / num_gpu /
    # num_thread / repeat_penalty / modelfile_extra 等）。不为每个 key 开单独字段，
    # 避免和 vllm/transformers 的字段命名空间混在一起。
    ollama_cfg: Dict[str, Any] = field(default_factory=dict)
    # Gemma 4 MTP 等投机解码（vLLM / transformers 共用）；assistant ``model`` 为相对名。
    speculative_config: Optional[Dict[str, Any]] = None
    # 当服务 YAML 的 ``config.runtime`` 中出现 ``max_tokens`` 键时解析到此：单轮补全上限，
    # 且推理侧不再使用请求里的 ``max_tokens``。键存在但值为空/非法时用 2048。
    # 故意不参与 ``cache_key``，避免仅改补全上限就触发 vLLM 引擎重建。
    service_max_tokens: Optional[int] = None

    @property
    def cache_key(self) -> str:
        return (
            f"runtime={self.runtime}|tp={self.tensor_parallel_size}|"
            f"gpu_mem={self.gpu_memory_utilization:.2f}|"
            f"max_model_len={self.max_model_len or 0}|"
            f"trust_remote_code={int(self.trust_remote_code)}|"
            f"enable_thinking={int(self.enable_thinking)}|"
            f"allow_cpu_offload={int(self.allow_cpu_offload)}|"
            f"dtype={self.dtype}|device_map={self.device_map}|"
            f"engine_kwargs={sorted(self.engine_kwargs.items())}|"
            f"speculative_config={sorted((self.speculative_config or {}).items())}|"
            f"model_kwargs={sorted(self.model_kwargs.items())}|"
            f"ollama_cfg={sorted(self.ollama_cfg.items())}"
        )


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except Exception:
        return default
    return coerced if coerced > 0 else default


def _coerce_float(value: Any, default: float) -> float:
    try:
        coerced = float(value)
    except Exception:
        return default
    if coerced <= 0:
        return default
    return coerced


def _runtime_cfg(params: Any) -> Dict[str, Any]:
    """读取**仅来自服务 YAML** 的 ``config.runtime``，与 audio/mini 一致。

    必须由 ``TextInferrer._build_request_spec`` 注入 ``params.service_runtime``（服务端
    ``config.runtime`` 的快照）。**不使用** ``params.model_cfg[\"runtime\"]``，请求侧
    不得覆盖引擎/backend 级配置。
    """
    sr = getattr(params, "service_runtime", None)
    if not isinstance(sr, dict):
        return {}
    return dict(sr)


def _backend_cfg(params: Any, backend_key: str) -> Dict[str, Any]:
    runtime_cfg = _runtime_cfg(params)
    backend_cfg = runtime_cfg.get(backend_key)
    return dict(backend_cfg) if isinstance(backend_cfg, dict) else {}


def _ollama_cfg(params: Any) -> Dict[str, Any]:
    return _backend_cfg(params, "ollama")


def _ollama_model_source(params: Any) -> str:
    cfg = _ollama_cfg(params)
    raw = str(cfg.get("model_source") or "local_gguf").strip().lower()
    aliases = {
        "": "local_gguf",
        "local": "local_gguf",
        "gguf": "local_gguf",
        "path": "local_gguf",
        "directory": "local_gguf",
        "ollama_tag": "tag",
        "registry": "tag",
        "name": "tag",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"local_gguf", "tag"}:
        raise ValueError(
            f"Unsupported ollama.model_source={raw!r}; expected one of local_gguf/tag"
        )
    return normalized


def resolve_text_runtime(params: Any) -> str:
    runtime_cfg = _runtime_cfg(params)
    backend_raw = str(runtime_cfg.get("backend") or "").strip().lower()
    if not backend_raw:
        raise ValueError(
            "text runtime.backend is not configured; "
            "please set `config.runtime.backend` (vllm|transformers) in the service YAML"
        )
    if backend_raw == "vllm":
        return "vllm"
    if backend_raw in {"hf", "huggingface", "transformers"}:
        return "transformers"
    if backend_raw == "ollama":
        return "ollama"
    raise ValueError(
        f"Unsupported text runtime.backend={backend_raw!r}; expected one of transformers/vllm/ollama"
    )


def try_resolve_local_model_ref(
    load_name: str,
    *,
    models_dir: str | None = None,
    weights_dir: str | None = None,
) -> str | None:
    """在 models_dir / weights_dir 下解析相对模型名；已存在则返回绝对路径，否则 None。"""
    name = str(load_name or "").strip()
    if not name:
        return None

    candidate = Path(name).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())

    searched: list[str] = []
    for root in (models_dir, weights_dir):
        root_text = str(root or "").strip()
        if not root_text:
            continue
        rooted = (Path(root_text).expanduser().resolve() / name).resolve()
        searched.append(str(rooted.parent))
        if rooted.exists():
            return str(rooted)

    return None


def resolve_text_model_ref(
    params: Any,
    *,
    models_dir: str | None = None,
    weights_dir: str | None = None,
) -> str:
    load_name = str(
        getattr(params, "load_name", None) or getattr(params, "model_name", None) or ""
    ).strip()
    if not load_name:
        raise ValueError("text task is missing load_name")

    runtime_cfg = _runtime_cfg(params)
    backend_raw = str(runtime_cfg.get("backend") or "").strip().lower()
    if backend_raw == "ollama" and _ollama_model_source(params) == "tag":
        return load_name

    resolved = try_resolve_local_model_ref(
        load_name,
        models_dir=models_dir,
        weights_dir=weights_dir,
    )
    if resolved:
        return resolved

    if not models_dir and not weights_dir:
        raise ValueError("models_dir is required to resolve local text model path")

    roots = [
        str(Path(r).expanduser().resolve())
        for r in (models_dir, weights_dir)
        if str(r or "").strip()
    ]
    raise ValueError(
        f"Text model path not found for load_name='{load_name}'. "
        f"Expected under one of: {', '.join(roots) or '<models_dir>'}"
    )


def resolve_speculative_config(
    raw: Any,
    *,
    models_dir: str | None = None,
    weights_dir: str | None = None,
) -> Dict[str, Any] | None:
    """解析 ``runtime.speculative_config``，将 assistant ``model`` 相对名落到本地目录。

    若配置了非空 ``model`` 但在 models_dir/weights_dir 下找不到，返回 None（不启用 MTP）。
    未配置 ``model`` 时原样返回（适用于目标模型自带 MTP 层的场景）。
    """
    if not isinstance(raw, dict) or not raw:
        return None

    config = dict(raw)
    model_name = str(config.get("model") or "").strip()
    if not model_name:
        return config

    resolved = try_resolve_local_model_ref(
        model_name,
        models_dir=models_dir,
        weights_dir=weights_dir,
    )
    if not resolved:
        return None

    config["model"] = resolved
    method = str(config.get("method") or "").strip().lower()
    if not method:
        config["method"] = "mtp"
    return config


def resolve_text_runtime_policy(params: Any) -> TextRuntimePolicy:
    runtime = resolve_text_runtime(params)
    runtime_cfg = _runtime_cfg(params)
    vllm_cfg = _backend_cfg(params, "vllm")
    transformers_cfg = _backend_cfg(params, "transformers")

    tensor_parallel_size = _coerce_int(vllm_cfg.get("tensor_parallel_size"), 1)
    gpu_memory_utilization = _coerce_float(vllm_cfg.get("gpu_memory_utilization"), 0.90)
    max_model_len_raw = runtime_cfg.get("max_model_len")
    try:
        max_model_len = int(max_model_len_raw) if max_model_len_raw not in (None, "") else None
    except Exception:
        max_model_len = None

    service_max_tokens: Optional[int] = None
    if "max_tokens" in runtime_cfg:
        raw_mt = runtime_cfg.get("max_tokens")
        try:
            if raw_mt in (None, ""):
                service_max_tokens = 2048
            else:
                parsed_mt = int(raw_mt)
                service_max_tokens = parsed_mt if parsed_mt > 0 else 2048
        except Exception:
            service_max_tokens = 2048

    trust_remote_code = _coerce_bool(runtime_cfg.get("trust_remote_code"), True)
    enable_thinking = _coerce_bool(runtime_cfg.get("enable_thinking"), False)
    allow_cpu_offload = _coerce_bool(transformers_cfg.get("allow_cpu_offload"), False)
    engine_kwargs = (
        dict(vllm_cfg.get("engine_kwargs"))
        if isinstance(vllm_cfg.get("engine_kwargs"), dict)
        else {}
    )
    speculative_config: Optional[Dict[str, Any]] = None
    raw_spec = runtime_cfg.get("speculative_config")
    if isinstance(raw_spec, dict) and raw_spec:
        speculative_config = dict(raw_spec)
    dtype = str(transformers_cfg.get("dtype") or "auto").strip() or "auto"
    device_map = str(transformers_cfg.get("device_map") or "auto").strip() or "auto"
    model_kwargs = (
        dict(transformers_cfg.get("model_kwargs"))
        if isinstance(transformers_cfg.get("model_kwargs"), dict)
        else {}
    )
    # GB10/DGX Spark 等平台的加载调优项（见 docs/GB10.md），由 transformers_bridge 消费。
    for loader_key in ("disable_mmap", "pin_memory", "patch_accelerate_pin_memory"):
        if loader_key in transformers_cfg and loader_key not in model_kwargs:
            model_kwargs[loader_key] = transformers_cfg[loader_key]
    ollama_cfg = _ollama_cfg(params)

    return TextRuntimePolicy(
        runtime=runtime,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        enable_thinking=enable_thinking,
        allow_cpu_offload=allow_cpu_offload,
        engine_kwargs=dict(engine_kwargs),
        speculative_config=speculative_config,
        dtype=dtype,
        device_map=device_map,
        model_kwargs=dict(model_kwargs),
        ollama_cfg=dict(ollama_cfg),
        service_max_tokens=service_max_tokens,
    )
