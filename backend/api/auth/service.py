"""
用户认证服务层
处理用户注册、登录、登出等业务逻辑
"""
from typing import Optional, Dict, Any
from backend.database import User
from backend.database.db import get_db_context
from backend.auth.jwt_utils import (
    create_access_token,
    create_refresh_token,
    refresh_access_token,
)
from backend.utils import generate_uuid, hash_password, verify_password
from backend.core.exceptions import (
    UserAlreadyExistsException,
    UserNotFoundException,
    InvalidCredentialsException,
)
from backend.core.logger import get_app_logger
from backend.api.auth.constants import is_default_admin_email

logger = get_app_logger(__name__)


def register_user(
    email: str,
    password: str,
    nickname: Optional[str] = None
) -> Dict[str, Any]:
    """
    注册新用户
    
    Args:
        email: 用户邮箱
        password: 用户密码
        nickname: 用户昵称（可选）
    
    Returns:
        用户信息字典
    
    Raises:
        UserAlreadyExistsException: 用户已存在
    """
    # 检查用户是否已存在
    with get_db_context() as db:
        existing_user = db.query(User).filter(User.email == email.lower()).first()
        if existing_user:
            raise UserAlreadyExistsException(email)
    
    # 哈希密码
    password_hash = hash_password(password)
    
    # 创建用户
    user_id = generate_uuid()
    is_admin = is_default_admin_email(email.lower())
    
    user_dict = User.create(
        id=user_id,
        email=email.lower(),
        password_hash=password_hash,
        nickname=nickname,
        status="active",
        is_admin=is_admin
    )
    
    logger.info(f"User registered: {email}")
    
    return user_dict


def login_user(email: str, password: str) -> Dict[str, Any]:
    """
    用户登录
    
    Args:
        email: 用户邮箱
        password: 用户密码
    
    Returns:
        包含Token和用户信息的字典
    
    Raises:
        UserNotFoundException: 用户不存在
        InvalidCredentialsException: 密码错误
    """
    # 查找用户
    with get_db_context() as db:
        user = db.query(User).filter(User.email == email.lower()).first()
        if not user:
            raise UserNotFoundException(email)
        
        # 检查用户状态
        if user.status != "active":
            raise InvalidCredentialsException("User account is not active")
        
        # 验证密码
        if not verify_password(password, user.password_hash):
            raise InvalidCredentialsException("Invalid password")
        
        user_dict = user.to_dict()
    
    # 生成Token
    access_token = create_access_token({
        "sub": user_dict["id"],
        "email": user_dict["email"]
    })
    
    refresh_token = create_refresh_token({
        "sub": user_dict["id"]
    })
    
    logger.info(f"User logged in: {email}")
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": user_dict["id"],
            "email": user_dict["email"],
            "nickname": user_dict.get("nickname"),
            "status": user_dict["status"],
            "is_admin": user_dict.get("is_admin", False),
        }
    }


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """
    根据用户ID获取用户信息
    
    Args:
        user_id: 用户ID
    
    Returns:
        用户信息字典，如果不存在则返回None
    """
    user_dict = User.get_by_id(user_id)
    if user_dict:
        # 移除敏感信息
        user_dict.pop("password_hash", None)
    return user_dict


def update_user_profile(user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """更新当前用户昵称或密码。"""
    user_dict = get_user_by_id(user_id)
    if not user_dict:
        raise UserNotFoundException(user_id)

    mapped: Dict[str, Any] = {}
    if "nickname" in updates:
        normalized = (updates.get("nickname") or "").strip()
        mapped["nickname"] = normalized or None
    if updates.get("new_password"):
        mapped["password_hash"] = hash_password(updates["new_password"])

    if not mapped:
        return user_dict

    updated = User.update(user_id, **mapped)
    if not updated:
        raise UserNotFoundException(user_id)

    updated.pop("password_hash", None)
    logger.info("User profile updated: %s", user_id)
    return updated


def refresh_user_token(refresh_token: str) -> Dict[str, Any]:
    """
    刷新用户Token
    
    Args:
        refresh_token: 刷新Token
    
    Returns:
        新的访问Token
    
    Raises:
        InvalidTokenException: Token无效
        TokenExpiredException: Token已过期
    """
    from backend.auth.jwt_utils import verify_token
    from backend.core.exceptions import InvalidTokenException, TokenExpiredException
    
    try:
        # 验证刷新Token
        payload = verify_token(refresh_token, token_type="refresh")
        user_id = payload.get("sub")
        
        if not user_id:
            raise InvalidTokenException("Refresh token missing user ID")
        
        # 检查用户是否存在
        user = get_user_by_id(user_id)
        if not user:
            raise UserNotFoundException(user_id)
        
        if user["status"] != "active":
            raise InvalidCredentialsException("User account is not active")
        
        # 生成新的访问Token
        new_access_token = create_access_token({
            "sub": user_id,
            "email": user["email"]
        })
        
        logger.info(f"Token refreshed for user: {user_id}")
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
    
    except (InvalidTokenException, TokenExpiredException) as e:
        logger.warning(f"Token refresh failed: {e.message}")
        raise

