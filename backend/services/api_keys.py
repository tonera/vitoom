"""用户 API Key 生成、存储与校验。"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.core.config import get_config
from backend.core.exceptions import InvalidParameterException, InvalidTokenException
from backend.database.db import get_db_context
from backend.database.models import ApiKey

API_KEY_PREFIX = "vt_sk_"
API_KEY_PREFIX_LENGTH = 14
EXPIRATION_OPTIONS = {"1d", "1m", "1y", "never"}


def _get_hmac_secret() -> bytes:
    configured = str(
        get_config("security.api_keys.hmac_secret", "")
        or get_config("security.api_key_hmac_secret", "")
        or ""
    ).strip()
    jwt_secret = str(get_config("security.jwt.secret_key", "") or "").strip()
    secret = configured or jwt_secret or "vitoom-default-secret-key-change-in-production"
    return secret.encode("utf-8")


def hash_api_key(raw_key: str) -> str:
    """对明文 API Key 做 HMAC-SHA256，数据库只保存结果。"""
    normalized = str(raw_key or "").strip()
    return hmac.new(_get_hmac_secret(), normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def looks_like_api_key(raw_key: str) -> bool:
    return str(raw_key or "").strip().startswith(API_KEY_PREFIX)


def _normalize_name(name: Optional[str]) -> str:
    value = str(name or "").strip()
    if not value:
        return "未命名 API Key"
    return value[:100]


def _resolve_expires_at(expires_in: str) -> Optional[datetime]:
    value = str(expires_in or "").strip()
    if value not in EXPIRATION_OPTIONS:
        raise InvalidParameterException("expires_in", "expires_in must be one of: 1d, 1m, 1y, never")
    if value == "never":
        return None
    now = datetime.utcnow()
    if value == "1d":
        return now + timedelta(days=1)
    if value == "1m":
        return now + timedelta(days=30)
    return now + timedelta(days=365)


def generate_raw_api_key() -> str:
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def create_api_key(user_id: str, *, name: Optional[str], expires_in: str) -> Dict[str, Any]:
    raw_key = generate_raw_api_key()
    row = ApiKey(
        id=str(uuid4()),
        user_id=user_id,
        name=_normalize_name(name),
        key_prefix=raw_key[:API_KEY_PREFIX_LENGTH],
        key_hash=hash_api_key(raw_key),
        expires_at=_resolve_expires_at(expires_in),
        created_at=datetime.utcnow(),
    )

    with get_db_context() as db:
        db.add(row)
        db.commit()
        db.refresh(row)
        data = row.to_dict()

    data["key"] = raw_key
    return data


def list_api_keys(user_id: str) -> List[Dict[str, Any]]:
    with get_db_context() as db:
        rows = (
            db.query(ApiKey)
            .filter(ApiKey.user_id == user_id)
            .order_by(ApiKey.created_at.desc())
            .all()
        )
        return [row.to_dict() for row in rows]


def delete_api_key(user_id: str, key_id: str) -> bool:
    with get_db_context() as db:
        row = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == user_id).first()
        if not row:
            return False
        db.delete(row)
        db.commit()
        return True


def authenticate_api_key(raw_key: str) -> str:
    normalized = str(raw_key or "").strip()
    if not looks_like_api_key(normalized):
        raise InvalidTokenException("Invalid API key")

    digest = hash_api_key(normalized)
    now = datetime.utcnow()
    with get_db_context() as db:
        row = db.query(ApiKey).filter(ApiKey.key_hash == digest).first()
        if not row:
            raise InvalidTokenException("Invalid API key")
        if row.expires_at is not None and row.expires_at <= now:
            raise InvalidTokenException("API key expired")

        row.last_used_at = now
        db.commit()
        return str(row.user_id)


__all__ = [
    "API_KEY_PREFIX",
    "EXPIRATION_OPTIONS",
    "authenticate_api_key",
    "create_api_key",
    "delete_api_key",
    "hash_api_key",
    "list_api_keys",
    "looks_like_api_key",
]
