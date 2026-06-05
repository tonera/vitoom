"""管理员用户管理接口。"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from backend.api.auth.service import get_user_by_id
from backend.auth import get_current_admin_user_id
from backend.core.exceptions import (
    InferenceServiceNotFoundException,
    PermissionDeniedException,
    UserAlreadyExistsException,
    UserNotFoundException,
)
from backend.core.config import get_config
from backend.core.logger import get_app_logger
from backend.core.response import ok
from backend.database import InferenceService, User
from backend.database.db import get_db_context
from backend.services.inference.service import get_inference_service_manager
from backend.utils import generate_uuid, hash_password
from backend.websocket.manager import get_websocket_manager

logger = get_app_logger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin"])

USER_STATUSES = frozenset({"active", "disabled"})
HEARTBEAT_STALE_SECONDS = 90
SENSITIVE_CONFIG_KEYS = ("token", "secret", "password", "key")


def _normalize_user_status(status: Optional[str]) -> str:
    normalized = (status or "active").strip().lower()
    if normalized == "inactive":
        return "disabled"
    return normalized


class AdminUserCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="Password (minimum 6 characters)")
    nickname: Optional[str] = Field(None, max_length=100, description="User nickname")
    status: str = Field(default="active", description="User status")
    is_admin: bool = Field(default=False, description="Whether the user is an admin")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = _normalize_user_status(value)
        if normalized not in USER_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(USER_STATUSES))}")
        return normalized


class AdminUserUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: Optional[EmailStr] = Field(None, description="User email")
    password: Optional[str] = Field(None, min_length=6, description="New password (optional)")
    nickname: Optional[str] = Field(None, max_length=100, description="User nickname")
    status: Optional[str] = Field(None, description="User status")
    is_admin: Optional[bool] = Field(None, description="Whether the user is an admin")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = _normalize_user_status(value)
        if normalized not in USER_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(USER_STATUSES))}")
        return normalized


class AdminInferenceServiceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    service_id: str = Field(..., min_length=1, max_length=100, description="Global service id")
    service_type: str = Field(..., min_length=1, max_length=50, description="Task content type")
    module: str = Field(..., min_length=1, max_length=255, description="Inference module path")
    program_name: Optional[str] = Field(None, max_length=100, description="Supervisor program name")
    supervisor_url: str = Field(..., min_length=1, max_length=500, description="Supervisor Agent base URL")
    template: Optional[str] = Field(None, max_length=255, description="Source config template")
    config: Optional[dict[str, Any]] = Field(None, description="Service config")
    enabled: bool = Field(default=False, description="Whether service should autostart")


class AdminInferenceConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config: dict[str, Any] = Field(default_factory=dict, description="Service config replacement")


def _public_user(user_dict: dict) -> dict:
    return {
        "id": user_dict["id"],
        "email": user_dict["email"],
        "nickname": user_dict.get("nickname"),
        "status": _normalize_user_status(user_dict.get("status")),
        "is_admin": bool(user_dict.get("is_admin", False)),
        "created_at": user_dict.get("created_at"),
        "updated_at": user_dict.get("updated_at"),
    }


def _service_config(service: dict[str, Any]) -> dict[str, Any]:
    config = service.get("config")
    return config if isinstance(config, dict) else {}


def _redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if any(marker in normalized_key for marker in SENSITIVE_CONFIG_KEYS):
                redacted[key] = "***"
            else:
                redacted[key] = _redact_config(item)
        return redacted
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


def _public_inference_service(service: dict[str, Any]) -> dict[str, Any]:
    public = dict(service)
    public["config"] = _redact_config(_service_config(service))
    return public


def _resolve_supervisor_url(service: dict[str, Any]) -> str:
    config = _service_config(service)
    value = config.get("supervisor_url") or os.getenv("VITOOM_SUPERVISOR_URL", "")
    return str(value or "").strip().rstrip("/")


def _resolve_supervisor_token(service: dict[str, Any]) -> str:
    value = (
        os.getenv("VITOOM_INFERENCE_UPLOAD_AUTH_SECRET", "")
        or get_config("inference.upload_auth_secret", "")
    )
    return str(value or "").strip()


def _control_unavailable_reason(service: dict[str, Any]) -> str:
    if not _resolve_supervisor_url(service):
        return "supervisor_url is not configured"
    if not _resolve_supervisor_token(service):
        return "inference upload auth secret is not configured"
    return ""


def _resolve_program_name(service: dict[str, Any]) -> str:
    config = _service_config(service)
    return str(config.get("program_name") or service.get("id") or "").strip()


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _heartbeat_fresh(service: dict[str, Any]) -> bool:
    dt = _parse_datetime(service.get("last_heartbeat_at"))
    if not dt:
        return False
    return (datetime.now(timezone.utc) - dt).total_seconds() <= HEARTBEAT_STALE_SECONDS


def _program_by_service_id(agent_snapshot: dict[str, Any], service_id: str) -> Optional[dict[str, Any]]:
    programs = agent_snapshot.get("programs")
    if not isinstance(programs, list) or not service_id:
        return None
    for program in programs:
        if not isinstance(program, dict):
            continue
        if program.get("service_id") == service_id:
            return program
        if program.get("name") == service_id:
            return program
    return None


def _runtime_state(
    service: dict[str, Any],
    *,
    agent_snapshot: Optional[dict[str, Any]],
    ws_online: bool,
) -> str:
    if not agent_snapshot:
        return "agent_unreachable" if _resolve_supervisor_url(service) else "unknown"
    if agent_snapshot.get("unreachable"):
        return "agent_unreachable"

    service_id = str(service.get("id") or "").strip()
    program = _program_by_service_id(agent_snapshot, service_id)
    if not program:
        return "unknown"

    state = str(program.get("state") or "").upper()
    if state == "RUNNING":
        return "online" if ws_online and _heartbeat_fresh(service) else "degraded"
    if state in {"STOPPED", "EXITED", "FATAL", "BACKOFF"}:
        return "offline"
    return "unknown"


def _agent_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _fetch_agent_programs(
    client: httpx.AsyncClient,
    *,
    supervisor_url: str,
    token: str,
) -> dict[str, Any]:
    if not supervisor_url:
        return {"unreachable": True, "reachable": False, "detail": "missing supervisor_url"}

    health_reachable = False
    try:
        health_response = await client.get(urljoin(f"{supervisor_url}/", "health"))
        health_response.raise_for_status()
        health_reachable = True
    except Exception as exc:
        return {"unreachable": True, "reachable": False, "detail": str(exc)}

    if not token:
        return {
            "unreachable": False,
            "reachable": True,
            "programs_error": "missing inference upload auth secret",
            "programs_error_status": 400,
        }

    try:
        response = await client.get(
            urljoin(f"{supervisor_url}/", "v1/programs"),
            headers=_agent_headers(token),
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            data["reachable"] = True
            return data
        return {"programs": [], "reachable": True}
    except httpx.HTTPStatusError as exc:
        detail: Any
        try:
            detail = exc.response.json()
        except Exception:
            detail = exc.response.text
        return {
            "unreachable": False,
            "reachable": health_reachable,
            "programs_error": detail,
            "programs_error_status": exc.response.status_code,
        }
    except Exception as exc:
        return {
            "unreachable": False,
            "reachable": health_reachable,
            "programs_error": str(exc),
        }


async def _call_agent(
    service: dict[str, Any],
    path: str,
    *,
    method: Literal["GET", "POST", "PUT"] = "POST",
    params: Optional[dict[str, Any]] = None,
    json_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    supervisor_url = _resolve_supervisor_url(service)
    token = _resolve_supervisor_token(service)
    if not supervisor_url:
        raise HTTPException(status_code=400, detail="supervisor_url is not configured")
    if not token:
        raise HTTPException(status_code=400, detail="inference upload auth secret is not configured")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.request(
                method,
                urljoin(f"{supervisor_url}/", path.lstrip("/")),
                headers=_agent_headers(token),
                params=params,
                json=json_body,
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {"data": data}
    except httpx.HTTPStatusError as exc:
        detail: Any
        try:
            detail = exc.response.json()
        except Exception:
            detail = exc.response.text
        upstream_status = exc.response.status_code
        if upstream_status in {400, 401, 403, 404, 409, 422}:
            raise HTTPException(
                status_code=upstream_status,
                detail={
                    "supervisor_url": supervisor_url,
                    "agent_path": path,
                    "agent_response": detail,
                },
            ) from exc
        raise HTTPException(
            status_code=502,
            detail={
                "supervisor_url": supervisor_url,
                "agent_path": path,
                "agent_response": detail,
            },
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "supervisor_url": supervisor_url,
                "agent_path": path,
                "message": str(exc),
            },
        ) from exc


def _public_config_document(agent_response: dict[str, Any]) -> dict[str, Any]:
    config = agent_response.get("config")
    if not isinstance(config, dict):
        config = agent_response if isinstance(agent_response, dict) else {}
    return {
        "path": agent_response.get("path"),
        "config": _redact_config(config),
    }


async def _notify_inference_changed(service_id: str, reason: str) -> None:
    try:
        await get_websocket_manager().notify_inference_services_changed(
            service_id=service_id,
            reason=reason,
        )
    except Exception as exc:
        logger.warning("Failed to notify inference service change: %s", exc)


def _count_active_admins() -> int:
    with get_db_context() as db:
        return int(
            db.query(User)
            .filter(User.is_admin.is_(True), User.status == "active")
            .count()
        )


def _ensure_not_last_admin(user_id: str, *, next_is_admin: Optional[bool] = None) -> None:
    """避免移除系统中最后一位 active 管理员。"""
    user_dict = User.get_by_id(user_id)
    if not user_dict:
        return

    currently_admin = bool(user_dict.get("is_admin"))
    is_currently_active = user_dict.get("status") == "active"
    will_remain_admin = currently_admin if next_is_admin is None else next_is_admin
    if currently_admin and is_currently_active and not will_remain_admin:
        if _count_active_admins() <= 1:
            raise PermissionDeniedException("Cannot remove the last active admin user")


def _ensure_not_last_active_admin(user_id: str) -> None:
    """避免禁用系统中最后一位 active 管理员。"""
    user_dict = User.get_by_id(user_id)
    if not user_dict or not user_dict.get("is_admin") or user_dict.get("status") != "active":
        return
    if _count_active_admins() <= 1:
        raise PermissionDeniedException("Cannot disable the last active admin user")


@router.get("/users")
async def list_users(
    keyword: Optional[str] = Query(None, description="Search by email or nickname"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _admin_id: str = Depends(get_current_admin_user_id),
):
    total = User.count_all(keyword=keyword)
    rows = User.list_all(limit=limit, offset=offset, keyword=keyword)
    items = [_public_user(row) for row in rows]
    return ok(data={"items": items, "total": total}, msg="ok")


@router.post("/users", status_code=201)
async def create_user(
    request: AdminUserCreateRequest,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    email = request.email.lower()
    with get_db_context() as db:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            raise UserAlreadyExistsException(email)

    user_dict = User.create(
        id=generate_uuid(),
        email=email,
        password_hash=hash_password(request.password),
        nickname=request.nickname,
        status=request.status,
        is_admin=request.is_admin,
    )
    if not user_dict:
        raise HTTPException(status_code=500, detail="Failed to create user")

    logger.info("Admin created user: %s", email)
    return ok(data=_public_user(user_dict), msg="created")


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    request: AdminUserUpdateRequest,
    admin_id: str = Depends(get_current_admin_user_id),
):
    user_dict = get_user_by_id(user_id)
    if not user_dict:
        raise UserNotFoundException(user_id)

    updates = request.model_dump(exclude_unset=True)
    if not updates:
        return ok(data=_public_user(user_dict), msg="ok")

    if "email" in updates and updates["email"]:
        updates["email"] = updates["email"].lower()
        with get_db_context() as db:
            existing = (
                db.query(User)
                .filter(User.email == updates["email"], User.id != user_id)
                .first()
            )
            if existing:
                raise UserAlreadyExistsException(updates["email"])

    if "password" in updates:
        password = updates.pop("password")
        if password:
            updates["password_hash"] = hash_password(password)

    if "is_admin" in updates:
        next_is_admin = bool(updates["is_admin"])
        if user_id == admin_id and not next_is_admin:
            raise PermissionDeniedException("Cannot revoke your own admin privileges")
        _ensure_not_last_admin(user_id, next_is_admin=next_is_admin)

    if updates.get("status") == "disabled":
        if user_id == admin_id:
            raise PermissionDeniedException("Cannot disable your own account")
        _ensure_not_last_active_admin(user_id)

    updated = User.update(user_id, **updates)
    if not updated:
        raise UserNotFoundException(user_id)

    logger.info("Admin updated user: %s", user_id)
    return ok(data=_public_user(updated), msg="ok")


@router.delete("/users/{user_id}")
async def disable_user(
    user_id: str,
    admin_id: str = Depends(get_current_admin_user_id),
):
    """软删除：将用户状态设为 disabled，保留关联数据。"""
    if user_id == admin_id:
        raise PermissionDeniedException("Cannot disable your own account")

    user_dict = get_user_by_id(user_id)
    if not user_dict:
        raise UserNotFoundException(user_id)

    if _normalize_user_status(user_dict.get("status")) == "disabled":
        return ok(data={"id": user_id, "status": "disabled"}, msg="already_disabled")

    _ensure_not_last_active_admin(user_id)

    updated = User.update(user_id, status="disabled")
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to disable user")

    logger.info("Admin disabled user: %s", user_id)
    return ok(data={"id": user_id, "status": "disabled"}, msg="disabled")


async def _decorate_inference_service(
    service: dict[str, Any],
    *,
    agent_snapshot: Optional[dict[str, Any]],
    connected_service_ids: set[str],
) -> dict[str, Any]:
    service_id = str(service.get("id") or "")
    program = _program_by_service_id(agent_snapshot or {}, service_id)
    program_name = str(program.get("name") or "") if program else _resolve_program_name(service)
    ws_online = service_id in connected_service_ids
    control_unavailable_reason = _control_unavailable_reason(service)
    public = _public_inference_service(service)
    public.update(
        {
            "program_name": program_name,
            "supervisor_url": _resolve_supervisor_url(service),
            "control_available": not control_unavailable_reason,
            "control_unavailable_reason": control_unavailable_reason or None,
            "agent_reachable": bool(
                agent_snapshot and (agent_snapshot.get("reachable") or not agent_snapshot.get("unreachable"))
            ),
            "agent_detail": (agent_snapshot or {}).get("detail"),
            "agent_programs_error": (agent_snapshot or {}).get("programs_error"),
            "agent_programs_error_status": (agent_snapshot or {}).get("programs_error_status"),
            "supervisor_program": program,
            "ws_online": ws_online,
            "heartbeat_fresh": _heartbeat_fresh(service),
            "runtime_state": _runtime_state(
                service,
                agent_snapshot=agent_snapshot,
                ws_online=ws_online,
            ),
        }
    )
    return public


@router.get("/inference/services")
async def list_admin_inference_services(
    type: Optional[str] = Query(None, description="Engine type, e.g. diffusers/vllm"),
    service_type: Optional[str] = Query(None, description="Task content type, e.g. image/video"),
    status_filter: Optional[str] = Query(None, alias="status", description="DB status filter"),
    _admin_id: str = Depends(get_current_admin_user_id),
):
    manager = get_inference_service_manager()
    services = manager.list_services(
        service_type=type,
        content_service_type=service_type,
        status=status_filter,
    )
    connected_service_ids = await get_websocket_manager().get_connected_inference_service_ids()

    agent_keys: dict[tuple[str, str], None] = {}
    for service in services:
        supervisor_url = _resolve_supervisor_url(service)
        token = _resolve_supervisor_token(service)
        if supervisor_url and token:
            agent_keys[(supervisor_url, token)] = None

    snapshots: dict[tuple[str, str], dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        results = await asyncio.gather(
            *[
                _fetch_agent_programs(client, supervisor_url=url, token=token)
                for url, token in agent_keys.keys()
            ],
            return_exceptions=False,
        )
    for key, snapshot in zip(agent_keys.keys(), results):
        snapshots[key] = snapshot

    items = [
        await _decorate_inference_service(
            service,
            agent_snapshot=snapshots.get((_resolve_supervisor_url(service), _resolve_supervisor_token(service))),
            connected_service_ids=connected_service_ids,
        )
        for service in services
    ]
    return ok(data={"items": items, "total": len(items)}, msg="ok")


@router.post("/inference/services", status_code=status.HTTP_201_CREATED)
async def create_admin_inference_service(
    request: AdminInferenceServiceCreateRequest,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    existing = InferenceService.get_by_id(request.service_id)
    if existing:
        raise HTTPException(status_code=409, detail="Inference service already exists")

    service_config = dict(request.config or {})
    service_config.update(
        {
            "module": request.module,
            "program_name": request.program_name or request.service_id,
            "supervisor_url": request.supervisor_url,
        }
    )
    if request.template:
        service_config["template"] = request.template

    created = InferenceService.create(
        id=request.service_id,
        name=request.service_id,
        service_type=str(service_config.get("type") or "unknown"),
        config=service_config,
        status="stopped",
        auto_start=request.enabled,
        content_service_type=request.service_type,
    )
    if not created:
        raise HTTPException(status_code=500, detail="Failed to create inference service")

    await _notify_inference_changed(request.service_id, "created")
    logger.info("Admin created inference service: %s", request.service_id)
    return ok(data=_public_inference_service(created), msg="created")


@router.get("/inference/services/{service_id}")
async def get_admin_inference_service(
    service_id: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    service = get_inference_service_manager().get_service(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    supervisor_url = _resolve_supervisor_url(service)
    token = _resolve_supervisor_token(service)
    async with httpx.AsyncClient(timeout=5.0) as client:
        agent_snapshot = await _fetch_agent_programs(client, supervisor_url=supervisor_url, token=token)
    connected_service_ids = await get_websocket_manager().get_connected_inference_service_ids()
    decorated = await _decorate_inference_service(
        service,
        agent_snapshot=agent_snapshot,
        connected_service_ids=connected_service_ids,
    )
    return ok(data=decorated, msg="ok")


@router.delete("/inference/services/{service_id}")
async def delete_admin_inference_service(
    service_id: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    connected_service_ids = await get_websocket_manager().get_connected_inference_service_ids()
    if service_id in connected_service_ids:
        raise HTTPException(status_code=409, detail="Cannot delete a connected inference service")

    manager = get_inference_service_manager()
    try:
        success = manager.delete_service(service_id)
    except InferenceServiceNotFoundException as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete service")

    await _notify_inference_changed(service_id, "deleted")
    logger.info("Admin deleted inference service: %s", service_id)
    return ok(data={"service_id": service_id}, msg="deleted")


@router.get("/inference/services/{service_id}/config")
async def get_admin_inference_service_config(
    service_id: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    service = get_inference_service_manager().get_service(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    agent_response = await _call_agent(
        service,
        f"/v1/config/services/{service_id}",
        method="GET",
    )
    return ok(data=_public_config_document(agent_response), msg="ok")


@router.put("/inference/services/{service_id}/config")
async def update_admin_inference_service_config(
    service_id: str,
    request: AdminInferenceConfigUpdateRequest,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    service = get_inference_service_manager().get_service(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    agent_response = await _call_agent(
        service,
        f"/v1/config/services/{service_id}",
        method="PUT",
        json_body={"config": request.config},
    )
    await _notify_inference_changed(service_id, "config_updated")
    logger.info("Admin updated inference service config via agent: %s", service_id)
    return ok(data=_public_config_document(agent_response), msg="updated")


@router.get("/inference/services/{service_id}/global-config")
async def get_admin_inference_global_config(
    service_id: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    service = get_inference_service_manager().get_service(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    agent_response = await _call_agent(service, "/v1/config/global", method="GET")
    return ok(data=_public_config_document(agent_response), msg="ok")


@router.put("/inference/services/{service_id}/global-config")
async def update_admin_inference_global_config(
    service_id: str,
    request: AdminInferenceConfigUpdateRequest,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    service = get_inference_service_manager().get_service(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    agent_response = await _call_agent(
        service,
        "/v1/config/global",
        method="PUT",
        json_body={"config": request.config},
    )
    await _notify_inference_changed(service_id, "global_config_updated")
    logger.info("Admin updated inference global config via agent: %s (service=%s)", service_id, service_id)
    return ok(data=_public_config_document(agent_response), msg="updated")


@router.post("/inference/services/{service_id}/start")
async def start_admin_inference_service(
    service_id: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    service = get_inference_service_manager().get_service(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    agent_response = await _call_agent(service, f"/v1/services/{service_id}/start")
    updated = InferenceService.update(service_id, status="starting") or service
    await _notify_inference_changed(service_id, "start_requested")
    logger.info("Admin requested inference service start: %s", service_id)
    return ok(
        data={
            "service": _public_inference_service(updated),
            "agent": agent_response,
        },
        msg="start_requested",
    )


@router.post("/inference/services/{service_id}/stop")
async def stop_admin_inference_service(
    service_id: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    service = get_inference_service_manager().get_service(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    agent_response = await _call_agent(service, f"/v1/services/{service_id}/stop")
    updated = InferenceService.update(service_id, status="stopped") or service
    await _notify_inference_changed(service_id, "stop_requested")
    logger.info("Admin requested inference service stop: %s", service_id)
    return ok(
        data={
            "service": _public_inference_service(updated),
            "agent": agent_response,
        },
        msg="stop_requested",
    )


@router.post("/inference/services/{service_id}/restart")
async def restart_admin_inference_service(
    service_id: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    service = get_inference_service_manager().get_service(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    agent_response = await _call_agent(service, f"/v1/services/{service_id}/restart")
    updated = InferenceService.update(service_id, status="starting") or service
    await _notify_inference_changed(service_id, "restart_requested")
    logger.info("Admin requested inference service restart: %s", service_id)
    return ok(
        data={
            "service": _public_inference_service(updated),
            "agent": agent_response,
        },
        msg="restart_requested",
    )


@router.get("/inference/services/{service_id}/logs")
async def get_admin_inference_service_logs(
    service_id: str,
    stream: Literal["stdout", "stderr"] = Query("stdout"),
    tail: int = Query(200, ge=1, le=1000),
    _admin_id: str = Depends(get_current_admin_user_id),
):
    service = get_inference_service_manager().get_service(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")

    agent_response = await _call_agent(
        service,
        f"/v1/services/{service_id}/logs",
        method="GET",
        params={"stream": stream, "tail": tail},
    )
    return ok(data=agent_response, msg="ok")
