"""
认证基础模块
提供JWT工具、密码加密、认证中间件等基础功能
"""
from .jwt_utils import (
    create_access_token,
    create_refresh_token,
    verify_token,
    decode_token,
    get_user_id_from_token,
    refresh_access_token,
    get_jwt_secret,
    get_jwt_algorithm,
    get_access_token_expire,
    get_refresh_token_expire,
)
from .middleware import (
    get_current_user_id,
    get_current_admin_user_id,
    get_current_user_id_or_api_key,
    get_optional_user_id,
    create_auth_dependency,
    security,
)

# 密码加密工具（从utils模块导入）
from backend.utils import hash_password, verify_password

__all__ = [
    # JWT工具
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "decode_token",
    "get_user_id_from_token",
    "refresh_access_token",
    "get_jwt_secret",
    "get_jwt_algorithm",
    "get_access_token_expire",
    "get_refresh_token_expire",
    # 认证中间件
    "get_current_user_id",
    "get_current_admin_user_id",
    "get_current_user_id_or_api_key",
    "get_optional_user_id",
    "create_auth_dependency",
    "security",
    # 密码加密工具
    "hash_password",
    "verify_password",
]

