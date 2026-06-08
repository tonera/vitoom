from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ProgramStatus(BaseModel):
    name: str
    state: str
    description: str = ""
    pid: Optional[int] = None
    uptime: Optional[str] = None
    source: str = "supervisor"
    command: Optional[str] = None
    # 与 supervisord program 名解耦；由 cmdline / 配置解析，供 Backend 按 service_id 管理。
    service_id: Optional[str] = None


class ProgramsResponse(BaseModel):
    programs: list[ProgramStatus]


class ProgramActionResponse(BaseModel):
    name: str
    action: str
    output: str = ""


class ProgramLogsResponse(BaseModel):
    name: str
    stream: str
    lines: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"


class ConfigDocumentResponse(BaseModel):
    path: str
    config: dict = Field(default_factory=dict)


class ConfigUpdateRequest(BaseModel):
    config: dict = Field(default_factory=dict)

