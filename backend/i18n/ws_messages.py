"""WebSocket message enrichment with stable message_code keys."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

TASK_STATUS_MESSAGE_CODES = {
    "completed": "task.completed",
    "failed": "task.failed",
    "cancelled": "task.cancelled",
    "running": "task.running",
    "processing": "task.running",
    "pending": "task.pending",
}

_INFERENCE_ERROR_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"cuda.*out.*of.*memory|out of memory|oom", re.I), "inference.cudaOutOfMemory"),
    (re.compile(r"model.*not.*found|model_not_found", re.I), "inference.modelNotFound"),
    (re.compile(r"cancelled|canceled", re.I), "task.cancelled"),
    (re.compile(r"timeout", re.I), "task.timeout"),
)


def infer_message_code_from_error(error_text: str) -> Optional[str]:
    code, _params = infer_message_code_and_params_from_error(error_text)
    return code


def infer_message_code_and_params_from_error(error_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    text = str(error_text or "").strip()
    if not text:
        return None, {}

    load_name_match = re.search(r"requested load_name=([^\s,)]+)", text, re.I)
    if load_name_match:
        return "inference.modelNotAvailable", {"model": load_name_match.group(1)}

    capability_match = re.search(r"capability=(\w+)", text, re.I)
    if re.search(r"No running inference service available", text, re.I):
        if capability_match and not load_name_match:
            service_type_match = re.search(r"task_type=(\w+)", text, re.I)
            return "inference.capabilityNotAvailable", {
                "capability": capability_match.group(1),
                "serviceType": (service_type_match.group(1) if service_type_match else "audio"),
            }
        return "inference.serviceNotRunning", {}

    for pattern, message_code in _INFERENCE_ERROR_PATTERNS:
        if pattern.search(text):
            return message_code, {}
    return None, {}


def build_task_message_params(message: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    progress = message.get("progress")
    if progress is not None:
        try:
            params["progress"] = int(progress)
        except (TypeError, ValueError):
            params["progress"] = progress
    for key in ("seconds", "model", "model_key", "task_id"):
        if key in message and message[key] is not None:
            params[key] = message[key]
    if "model_key" in params and "model" not in params:
        params["model"] = params["model_key"]
    return params


def enrich_task_ws_message(message: Dict[str, Any]) -> Dict[str, Any]:
    if message.get("message_code"):
        return message

    enriched = dict(message)
    status = str(enriched.get("status") or "").lower().strip()
    message_code = TASK_STATUS_MESSAGE_CODES.get(status)
    params = build_task_message_params(enriched)

    if status == "failed":
        error_text = str(enriched.get("error") or enriched.get("message") or "").strip()
        inferred_code, inferred_params = infer_message_code_and_params_from_error(error_text)
        if inferred_code:
            message_code = inferred_code
            params.update(inferred_params)
            if inferred_code == "task.timeout" and "seconds" not in params:
                match = re.search(r"(\d+)", error_text)
                if match:
                    params["seconds"] = match.group(1)
        else:
            inferred = infer_message_code_from_error(error_text)
            if inferred:
                message_code = inferred
                if inferred == "task.timeout" and "seconds" not in params:
                    match = re.search(r"(\d+)", error_text)
                    if match:
                        params["seconds"] = match.group(1)

    if message_code:
        enriched["message_code"] = message_code
        if params:
            enriched["message_params"] = params
    return enriched
