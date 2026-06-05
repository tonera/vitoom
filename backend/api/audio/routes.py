from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.auth import get_current_user_id
from backend.core.response import ok
from backend.services.tts_speakers import load_tts_speakers

router = APIRouter(prefix="/v1/audio", tags=["Audio"])


@router.get("/tts-speakers")
async def list_tts_speakers(_user_id: str = Depends(get_current_user_id)):
    return ok(data=load_tts_speakers())
