from __future__ import annotations

import hmac
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
import yaml

from .config_io import (
    read_global_config,
    read_service_config,
    write_global_config,
    write_service_config,
)
from .schemas import (
    ConfigDocumentResponse,
    ConfigUpdateRequest,
    HealthResponse,
    ProgramActionResponse,
    ProgramLogsResponse,
    ProgramsResponse,
)
from .supervisorctl import (
    SupervisorCtlError,
    control_program,
    list_programs,
    resolve_supervisor_program_name,
    tail_program_logs,
    validate_program_name,
)


SELF_PROGRAM_NAME = "supervisor-agent"
MAX_LOG_LINES = 1000

app = FastAPI(title="Vitoom Supervisor Agent", version="1.0.0")


def _configured_token() -> str:
    config_path = Path(__file__).resolve().parents[1] / "config" / "inference.yaml"
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""

    storage = data.get("storage") if isinstance(data, dict) else {}
    server = storage.get("server") if isinstance(storage, dict) else {}
    auth = server.get("auth") if isinstance(server, dict) else {}
    if isinstance(auth, dict):
        return str(auth.get("secret") or "").strip()
    return ""


def require_bearer_token(authorization: str | None = Header(default=None)) -> None:
    token = _configured_token()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference upload auth secret is not configured.",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    supplied_token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied_token, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        )


def _safe_program_name(name: str) -> str:
    try:
        return validate_program_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _ensure_not_self(name: str) -> None:
    if name == SELF_PROGRAM_NAME:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The supervisor agent program cannot be controlled through this API.",
        )


def _supervisor_error(exc: SupervisorCtlError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=str(exc),
    )


def _safe_service_id(service_id: str) -> str:
    try:
        return validate_program_name(service_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _program_state(name: str) -> str | None:
    for program in list_programs():
        if program.name == name:
            return program.state
    return None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get(
    "/v1/programs",
    response_model=ProgramsResponse,
    dependencies=[Depends(require_bearer_token)],
)
def get_programs() -> ProgramsResponse:
    try:
        return ProgramsResponse(programs=list_programs())
    except SupervisorCtlError as exc:
        raise _supervisor_error(exc) from exc


@app.post(
    "/v1/programs/{name}/start",
    response_model=ProgramActionResponse,
    dependencies=[Depends(require_bearer_token)],
)
def start_program(name: str) -> ProgramActionResponse:
    program_name = _safe_program_name(name)
    _ensure_not_self(program_name)
    try:
        result = control_program(program_name, "start")
    except SupervisorCtlError as exc:
        raise _supervisor_error(exc) from exc
    return ProgramActionResponse(name=program_name, action="start", output=result.output)


@app.post(
    "/v1/programs/{name}/stop",
    response_model=ProgramActionResponse,
    dependencies=[Depends(require_bearer_token)],
)
def stop_program(name: str) -> ProgramActionResponse:
    program_name = _safe_program_name(name)
    _ensure_not_self(program_name)
    try:
        result = control_program(program_name, "stop")
    except SupervisorCtlError as exc:
        raise _supervisor_error(exc) from exc
    return ProgramActionResponse(name=program_name, action="stop", output=result.output)


@app.post(
    "/v1/programs/{name}/restart",
    response_model=ProgramActionResponse,
    dependencies=[Depends(require_bearer_token)],
)
def restart_program(name: str) -> ProgramActionResponse:
    program_name = _safe_program_name(name)
    _ensure_not_self(program_name)
    try:
        stopped_states = {"STOPPED", "EXITED", "FATAL", "BACKOFF"}
        action = "start" if _program_state(program_name) in stopped_states else "restart"
        result = control_program(program_name, action)
    except SupervisorCtlError as exc:
        raise _supervisor_error(exc) from exc
    return ProgramActionResponse(name=program_name, action="restart", output=result.output)


@app.get(
    "/v1/programs/{name}/logs",
    response_model=ProgramLogsResponse,
    dependencies=[Depends(require_bearer_token)],
)
def get_program_logs(
    name: str,
    stream: Literal["stdout", "stderr"] = "stdout",
    tail: int = Query(default=200, ge=1, le=MAX_LOG_LINES),
) -> ProgramLogsResponse:
    program_name = _safe_program_name(name)
    lines = tail_program_logs(program_name, stream, tail)
    return ProgramLogsResponse(name=program_name, stream=stream, lines=lines)


@app.post(
    "/v1/services/{service_id}/start",
    response_model=ProgramActionResponse,
    dependencies=[Depends(require_bearer_token)],
)
def start_service(service_id: str) -> ProgramActionResponse:
    normalized = _safe_service_id(service_id)
    try:
        program_name = resolve_supervisor_program_name(normalized)
        result = control_program(program_name, "start")
    except SupervisorCtlError as exc:
        raise _supervisor_error(exc) from exc
    return ProgramActionResponse(name=program_name, action="start", output=result.output)


@app.post(
    "/v1/services/{service_id}/stop",
    response_model=ProgramActionResponse,
    dependencies=[Depends(require_bearer_token)],
)
def stop_service(service_id: str) -> ProgramActionResponse:
    normalized = _safe_service_id(service_id)
    try:
        program_name = resolve_supervisor_program_name(normalized)
        result = control_program(program_name, "stop")
    except SupervisorCtlError as exc:
        raise _supervisor_error(exc) from exc
    return ProgramActionResponse(name=program_name, action="stop", output=result.output)


@app.post(
    "/v1/services/{service_id}/restart",
    response_model=ProgramActionResponse,
    dependencies=[Depends(require_bearer_token)],
)
def restart_service(service_id: str) -> ProgramActionResponse:
    normalized = _safe_service_id(service_id)
    _ensure_not_self(normalized)
    try:
        program_name = resolve_supervisor_program_name(normalized)
        stopped_states = {"STOPPED", "EXITED", "FATAL", "BACKOFF"}
        action = "start" if _program_state(program_name) in stopped_states else "restart"
        result = control_program(program_name, action)
    except SupervisorCtlError as exc:
        raise _supervisor_error(exc) from exc
    return ProgramActionResponse(name=program_name, action="restart", output=result.output)


@app.get(
    "/v1/services/{service_id}/logs",
    response_model=ProgramLogsResponse,
    dependencies=[Depends(require_bearer_token)],
)
def get_service_logs(
    service_id: str,
    stream: Literal["stdout", "stderr"] = "stdout",
    tail: int = Query(default=200, ge=1, le=MAX_LOG_LINES),
) -> ProgramLogsResponse:
    normalized = _safe_service_id(service_id)
    try:
        program_name = resolve_supervisor_program_name(normalized)
        lines = tail_program_logs(program_name, stream, tail)
    except SupervisorCtlError as exc:
        raise _supervisor_error(exc) from exc
    return ProgramLogsResponse(name=program_name, stream=stream, lines=lines)


def _config_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.get(
    "/v1/config/global",
    response_model=ConfigDocumentResponse,
    dependencies=[Depends(require_bearer_token)],
)
def get_global_config() -> ConfigDocumentResponse:
    try:
        config = read_global_config()
    except Exception as exc:
        raise _config_http_error(exc) from exc
    return ConfigDocumentResponse(path="inference.yaml", config=config)


@app.put(
    "/v1/config/global",
    response_model=ConfigDocumentResponse,
    dependencies=[Depends(require_bearer_token)],
)
def put_global_config(request: ConfigUpdateRequest) -> ConfigDocumentResponse:
    try:
        config = write_global_config(request.config)
    except Exception as exc:
        raise _config_http_error(exc) from exc
    return ConfigDocumentResponse(path="inference.yaml", config=config)


@app.get(
    "/v1/config/services/{service_id}",
    response_model=ConfigDocumentResponse,
    dependencies=[Depends(require_bearer_token)],
)
def get_service_config(service_id: str) -> ConfigDocumentResponse:
    try:
        config, path = read_service_config(service_id)
    except Exception as exc:
        raise _config_http_error(exc) from exc
    return ConfigDocumentResponse(path=path, config=config)


@app.put(
    "/v1/config/services/{service_id}",
    response_model=ConfigDocumentResponse,
    dependencies=[Depends(require_bearer_token)],
)
def put_service_config(service_id: str, request: ConfigUpdateRequest) -> ConfigDocumentResponse:
    try:
        config, path = write_service_config(service_id, request.config)
    except Exception as exc:
        raise _config_http_error(exc) from exc
    return ConfigDocumentResponse(path=path, config=config)

