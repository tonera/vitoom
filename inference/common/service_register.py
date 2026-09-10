"""WS ``service_register`` 消息构建（推理服务统一注册协议）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def _normalize_string_list(
    values: Optional[List[Any]],
    *,
    lowercase: bool = False,
) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        normalized.append(text.lower() if lowercase else text)
    return list(dict.fromkeys(normalized))


def build_service_register_message(
    *,
    service_type: str,
    service_config: Optional[Dict[str, Any]] = None,
    supports_task: Optional[bool] = None,
    supported_models: Optional[List[str]] = None,
    capabilities: Optional[List[str]] = None,
    fixed_model: Optional[str] = None,
    fixed_family: Optional[str] = None,
    queue_length: Optional[int] = None,
) -> Dict[str, Any]:
    """构建 ``service_register`` WS 帧。

    与 ``backend/services/inference/service.py::sync_service_registration`` 字段对齐。
    显式传入的 ``supported_models`` / ``capabilities`` / pin 字段优先于 ``service_config``。
    """
    cfg = dict(service_config or {})

    if supports_task is None:
        raw_supports_task = cfg.get("supports_task", True)
        supports_task = bool(raw_supports_task) if isinstance(raw_supports_task, bool) else True

    models = _normalize_string_list(supported_models)
    if not models:
        models = _normalize_string_list(cfg.get("supported_models"))

    caps = _normalize_string_list(capabilities, lowercase=True)
    if not caps:
        caps = _normalize_string_list(cfg.get("capabilities"), lowercase=True)

    resolved_fixed_model = str(fixed_model if fixed_model is not None else cfg.get("fixed_model") or "").strip()
    resolved_fixed_family = str(
        fixed_family if fixed_family is not None else cfg.get("fixed_family") or ""
    ).strip()

    resolved_type = str(service_type or "").strip()
    if not resolved_type:
        raise ValueError("service_type is required")

    message: Dict[str, Any] = {
        "type": "service_register",
        "service_type": resolved_type,
        "supports_task": bool(supports_task),
        "timestamp": datetime.utcnow().isoformat(),
    }
    if models:
        message["supported_models"] = models
    if caps:
        message["capabilities"] = caps
    if resolved_fixed_model:
        message["fixed_model"] = resolved_fixed_model
    if resolved_fixed_family:
        message["fixed_family"] = resolved_fixed_family
    if queue_length is not None:
        message["queue_length"] = max(0, int(queue_length))
    return message
