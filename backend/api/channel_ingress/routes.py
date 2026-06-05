from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.auth import get_optional_user_id
from backend.core.response import ok
from backend.services.agent import AgentCommand, AgentRuntimeError, AgentValidationError, create_agent_run

router = APIRouter(prefix="/v1", tags=["Channel Ingress"])


class ChannelIngressRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: Optional[str] = Field(default=None, description="Internal user ID; required for unauthenticated calls")
    agent_id: str = Field(..., description="Target agent template ID")
    message: str = Field(..., description="Normalized message body")
    source_type: str = Field(default="hook", description="Message source")
    source_ref: Optional[str] = Field(default=None, description="Channel message reference ID")
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    runtime_config: Dict[str, Any] = Field(default_factory=dict)


@router.post("/channel-ingress", status_code=202)
async def channel_ingress(
    request: ChannelIngressRequest,
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    effective_user_id = str(user_id or request.user_id or "").strip()
    if not effective_user_id:
        raise HTTPException(status_code=400, detail="user_id is required when no authenticated user is present")

    try:
        agent_run = await create_agent_run(
            AgentCommand(
                user_id=effective_user_id,
                agent_id=request.agent_id,
                message=request.message,
                source_type=request.source_type,
                source_ref=request.source_ref,
                attachments=request.attachments,
                context=request.context,
                runtime_config=request.runtime_config,
            )
        )
    except AgentValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except AgentRuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return ok(
        data={
            "run_id": agent_run["id"],
            "task_id": agent_run["task_id"],
            "status": agent_run["status"],
        },
        msg="accepted",
    )
