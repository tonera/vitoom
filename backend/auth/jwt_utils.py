"""
JWT工具类
支持JWT Token的生成、验证、刷新等功能
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt, JWTError
from backend.core.config import get_security_config
from backend.core.logger import get_app_logger
from backend.core.exceptions import InvalidTokenException, TokenExpiredException

logger = get_app_logger(__name__)


def get_jwt_secret() -> str:
    """获取JWT密钥"""
    security_config = get_security_config()
    secret_key = security_config.get("jwt", {}).get("secret_key", "")
    
    if not secret_key:
        # 如果没有配置密钥，使用默认密钥（生产环境应该配置）
        logger.warning("JWT secret key not configured, using default key")
        return "vitoom-default-secret-key-change-in-production"
    
    return secret_key


def get_jwt_algorithm() -> str:
    """获取JWT算法"""
    security_config = get_security_config()
    return security_config.get("jwt", {}).get("algorithm", "HS256")


def get_access_token_expire() -> int:
    """获取访问Token过期时间（秒）"""
    security_config = get_security_config()
    return security_config.get("jwt", {}).get("access_token_expire", 86400)  # 默认24小时


def get_refresh_token_expire() -> int:
    """获取刷新Token过期时间（秒）"""
    security_config = get_security_config()
    return security_config.get("jwt", {}).get("refresh_token_expire", 604800)  # 默认7天


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    创建访问Token
    
    Args:
        data: Token中要包含的数据（通常是用户ID、邮箱等）
        expires_delta: 过期时间增量，如果为None则使用配置的默认值
    
    Returns:
        JWT Token字符串
    
    Example:
        >>> token = create_access_token({"sub": "user_id", "email": "user@example.com"})
        >>> len(token) > 0
        True
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(seconds=get_access_token_expire())
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        get_jwt_secret(),
        algorithm=get_jwt_algorithm()
    )
    
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    创建刷新Token
    
    Args:
        data: Token中要包含的数据（通常是用户ID）
        expires_delta: 过期时间增量，如果为None则使用配置的默认值
    
    Returns:
        JWT Token字符串
    
    Example:
        >>> token = create_refresh_token({"sub": "user_id"})
        >>> len(token) > 0
        True
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(seconds=get_refresh_token_expire())
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        get_jwt_secret(),
        algorithm=get_jwt_algorithm()
    )
    
    return encoded_jwt


def verify_token(token: str, token_type: Optional[str] = None) -> Dict[str, Any]:
    """
    验证Token并返回payload
    
    Args:
        token: JWT Token字符串
        token_type: Token类型（"access"或"refresh"），如果为None则不检查类型
    
    Returns:
        Token的payload字典
    
    Raises:
        InvalidTokenException: Token无效
        TokenExpiredException: Token已过期
    
    Example:
        >>> token = create_access_token({"sub": "user_id"})
        >>> payload = verify_token(token)
        >>> payload["sub"] == "user_id"
        True
    """
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[get_jwt_algorithm()]
        )
        
        # 检查Token类型
        if token_type and payload.get("type") != token_type:
            raise InvalidTokenException("Invalid token type")
        
        return payload
    
    except jwt.ExpiredSignatureError:
        raise TokenExpiredException("Token has expired")
    
    except JWTError as e:
        raise InvalidTokenException(f"Invalid token: {str(e)}")


def decode_token(token: str) -> Dict[str, Any]:
    """
    解码Token（不验证签名和过期时间）
    
    Args:
        token: JWT Token字符串
    
    Returns:
        Token的payload字典
    
    Note:
        此方法不验证Token的有效性，仅用于调试或特殊场景
    """
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[get_jwt_algorithm()],
            options={"verify_signature": False, "verify_exp": False}
        )
        return payload
    except JWTError as e:
        raise InvalidTokenException(f"Failed to decode token: {str(e)}")


def get_user_id_from_token(token: str) -> Optional[str]:
    """
    从Token中提取用户ID
    
    Args:
        token: JWT Token字符串
    
    Returns:
        用户ID，如果Token无效则返回None
    """
    try:
        payload = verify_token(token)
        return payload.get("sub")  # "sub"是JWT标准中的subject字段，通常存储用户ID
    except (InvalidTokenException, TokenExpiredException):
        return None


def refresh_access_token(refresh_token: str) -> str:
    """
    使用刷新Token生成新的访问Token
    
    Args:
        refresh_token: 刷新Token字符串
    
    Returns:
        新的访问Token字符串
    
    Raises:
        InvalidTokenException: 刷新Token无效
        TokenExpiredException: 刷新Token已过期
    """
    # 验证刷新Token
    payload = verify_token(refresh_token, token_type="refresh")
    
    # 提取用户信息
    user_id = payload.get("sub")
    if not user_id:
        raise InvalidTokenException("Refresh token missing user ID")
    
    # 创建新的访问Token
    new_token_data = {"sub": user_id}
    
    # 保留其他有用的字段（如email等）
    if "email" in payload:
        new_token_data["email"] = payload["email"]
    
    return create_access_token(new_token_data)

