"""用户 API Key 管理接口。

管理接口只接受登录态 JWT；API Key 本身不能用来管理 API Key。
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.auth import get_current_user_id
from backend.core.response import ok
from backend.services.api_keys import create_api_key, delete_api_key, list_api_keys

router = APIRouter(prefix="/api/api-keys", tags=["API Keys"])


class ApiKeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = Field(default=None, max_length=100)
    expires_in: str = Field(default="never", description="1d / 1m / 1y / never")


@router.get("")
async def list_user_api_keys(user_id: str = Depends(get_current_user_id)):
    return ok(data={"items": list_api_keys(user_id)}, msg="ok")


@router.post("", status_code=201)
async def create_user_api_key(
    request: ApiKeyCreateRequest,
    user_id: str = Depends(get_current_user_id),
):
    created = create_api_key(user_id, name=request.name, expires_in=request.expires_in)
    return ok(data=created, msg="created")


@router.delete("/{key_id}")
async def delete_user_api_key(
    key_id: str,
    user_id: str = Depends(get_current_user_id),
):
    if not delete_api_key(user_id, key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    return ok(data={"id": key_id}, msg="deleted")
