"""
认证中间件基础
提供FastAPI认证依赖和中间件
"""
from typing import Optional
from fastapi import Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.auth.jwt_utils import get_user_id_from_token
from backend.core.exceptions import AuthRequiredException, InvalidTokenException, TokenExpiredException
from backend.core.logger import get_app_logger
from backend.services.api_keys import authenticate_api_key, looks_like_api_key

logger = get_app_logger(__name__)

# HTTP Bearer认证方案
security = HTTPBearer(auto_error=False)


async def get_current_user_id(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    从请求中获取当前用户ID（FastAPI依赖）
    
    Args:
        request: FastAPI请求对象
        credentials: HTTP Bearer认证凭据（可选）
    
    Returns:
        用户ID字符串
    
    Raises:
        AuthRequiredException: 未提供认证信息
        InvalidTokenException: Token无效
        TokenExpiredException: Token已过期
    
    Example:
        >>> @app.get("/users/me")
        >>> async def get_current_user(user_id: str = Depends(get_current_user_id)):
        >>>     return {"user_id": user_id}
    """
    # 优先从HTTP Bearer获取Token
    if credentials:
        token = credentials.credentials
    else:
        # 从Authorization头获取Token
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise AuthRequiredException("Authorization header missing")
        
        # 检查Bearer格式
        if not authorization.startswith("Bearer "):
            raise AuthRequiredException("Invalid authorization header format")
        
        token = authorization.replace("Bearer ", "").strip()
    
    if not token:
        raise AuthRequiredException("Token not provided")
    
    # 验证Token并获取用户ID
    try:
        user_id = get_user_id_from_token(token)
        if not user_id:
            raise InvalidTokenException("Token missing user ID")

        _ensure_user_is_active(user_id)
        return user_id
    
    except (InvalidTokenException, TokenExpiredException) as e:
        logger.warning(f"Token validation failed: {e.message}")
        raise


def _ensure_user_is_active(user_id: str) -> None:
    """校验用户存在且处于 active 状态。"""
    from backend.core.exceptions import InvalidCredentialsException
    from backend.database import User

    user_dict = User.get_by_id(user_id)
    if not user_dict:
        raise InvalidTokenException(f"User not found: {user_id}")
    if user_dict.get("status") != "active":
        raise InvalidCredentialsException("User account is not active")


async def get_current_user_id_or_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """获取当前用户 ID，支持 JWT Bearer、API Key Bearer 和 X-Api-Key。

    当 Authorization: Bearer 与 X-Api-Key 同时存在时，按业务约定以 Bearer 为准。
    """
    if credentials:
        token = str(credentials.credentials or "").strip()
        if not token:
            raise AuthRequiredException("Token not provided")
        if looks_like_api_key(token):
            return authenticate_api_key(token)
        return await get_current_user_id(request, credentials)

    x_api_key = str(request.headers.get("X-Api-Key", "") or "").strip()
    if x_api_key:
        return authenticate_api_key(x_api_key)

    return await get_current_user_id(request, credentials)


async def get_current_admin_user_id(
    user_id: str = Depends(get_current_user_id),
) -> str:
    """要求当前用户为 active 管理员。"""
    from backend.database import User
    from backend.core.exceptions import PermissionDeniedException

    user_dict = User.get_by_id(user_id)
    if not user_dict or not user_dict.get("is_admin"):
        raise PermissionDeniedException("Admin access required")
    if user_dict.get("status") != "active":
        raise PermissionDeniedException("Admin access required")
    return user_id


async def get_optional_user_id(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    从请求中获取当前用户ID（可选，如果未认证返回None）
    
    Args:
        request: FastAPI请求对象
        credentials: HTTP Bearer认证凭据（可选）
    
    Returns:
        用户ID字符串，如果未认证则返回None
    
    Example:
        >>> @app.get("/public")
        >>> async def public_endpoint(user_id: Optional[str] = Depends(get_optional_user_id)):
        >>>     if user_id:
        >>>         return {"user_id": user_id, "authenticated": True}
        >>>     return {"authenticated": False}
    """
    try:
        return await get_current_user_id(request, credentials)
    except (AuthRequiredException, InvalidTokenException, TokenExpiredException):
        return None


def create_auth_dependency(required: bool = True):
    """
    创建认证依赖函数
    
    Args:
        required: 是否必需认证，如果为False则未认证时返回None
    
    Returns:
        认证依赖函数
    """
    if required:
        return get_current_user_id
    else:
        return get_optional_user_id

