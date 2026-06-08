"""User-facing feedback when inference dispatch cannot pick a service."""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.core.exceptions import InferenceDispatchUnavailableException
from backend.services.chat.router import DispatchSelectionError, DispatchSpec, get_dispatch_router


def build_dispatch_unavailable_feedback(
    *,
    service_type: str,
    load_name: str = "",
    capability: str = "",
) -> Dict[str, Any]:
    model = str(load_name or "").strip()
    cap = str(capability or "").strip()
    normalized_type = str(service_type or "").strip().lower()

    if model:
        return {
            "message_code": "inference.modelNotAvailable",
            "message_params": {"model": model},
        }
    if cap:
        return {
            "message_code": "inference.capabilityNotAvailable",
            "message_params": {"capability": cap, "serviceType": normalized_type or "audio"},
        }
    return {
        "message_code": "inference.serviceNotRunning",
        "message_params": {},
    }


async def assert_inference_dispatch_available(
    *,
    service_type: str,
    load_name: str = "",
    capability: str = "",
    require_supports_task: bool = True,
) -> None:
    from backend.websocket.manager import get_websocket_manager

    connected_service_ids = await get_websocket_manager().get_connected_inference_service_ids()
    spec = DispatchSpec(
        service_type=str(service_type or "").strip().lower(),
        require_supports_task=require_supports_task,
        reason=f"task_type={service_type}",
        load_name=str(load_name or "").strip(),
        capability=str(capability or "").strip(),
    )
    try:
        get_dispatch_router().pick_service(
            spec,
            connected_service_ids=connected_service_ids,
        )
    except DispatchSelectionError as exc:
        feedback = build_dispatch_unavailable_feedback(
            service_type=service_type,
            load_name=load_name,
            capability=capability,
        )
        raise InferenceDispatchUnavailableException(
            str(exc),
            message_code=str(feedback["message_code"]),
            message_params=dict(feedback.get("message_params") or {}),
        ) from exc
