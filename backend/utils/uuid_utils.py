"""
UUID生成工具
"""
import uuid
from typing import Optional


def generate_uuid() -> str:
    """
    生成UUID字符串（默认使用UUID4）
    
    Returns:
        UUID字符串（不含连字符）
    
    Example:
        >>> uuid_str = generate_uuid()
        >>> len(uuid_str) == 32
        True
    """
    return generate_uuid4()


def generate_uuid4() -> str:
    """
    生成UUID4字符串（随机UUID）
    
    Returns:
        UUID字符串（不含连字符）
    
    Example:
        >>> uuid_str = generate_uuid4()
        >>> len(uuid_str) == 32
        True
    """
    return uuid.uuid4().hex


def is_valid_uuid(uuid_string: str, version: Optional[int] = None) -> bool:
    """
    验证UUID字符串是否有效
    
    Args:
        uuid_string: 待验证的UUID字符串
        version: UUID版本（1-5），如果为None则验证所有版本
    
    Returns:
        如果UUID有效返回True，否则返回False
    
    Example:
        >>> is_valid_uuid("550e8400-e29b-41d4-a716-446655440000")
        True
        >>> is_valid_uuid("invalid-uuid")
        False
    """
    try:
        uuid_obj = uuid.UUID(uuid_string)
        if version is not None:
            return uuid_obj.version == version
        return True
    except (ValueError, AttributeError):
        return False

