"""
加密工具（密码哈希等）
"""
import secrets
import hashlib
from typing import Optional

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    bcrypt = None

# 延迟初始化密码上下文（使用passlib作为备选）
_pwd_context = None


def _get_pwd_context():
    """获取密码上下文（延迟初始化，优先使用bcrypt）"""
    global _pwd_context
    if _pwd_context is None:
        if BCRYPT_AVAILABLE:
            # 直接使用bcrypt，避免passlib的初始化问题
            return None  # 返回None表示使用直接bcrypt
        else:
            from passlib.context import CryptContext
            _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return _pwd_context


def hash_password(password: str) -> str:
    """
    哈希密码（使用bcrypt）
    
    Args:
        password: 原始密码（bcrypt限制最大72字节，超长密码会先进行SHA256哈希）
    
    Returns:
        哈希后的密码字符串
    
    Example:
        >>> hashed = hash_password("mypassword")
        >>> len(hashed) > 0
        True
    """
    if not BCRYPT_AVAILABLE:
        raise ImportError("bcrypt is required for password hashing")
    
    # bcrypt限制密码长度不超过72字节
    # 如果密码超过72字节，先进行SHA256哈希
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = hashlib.sha256(password_bytes).digest()
    
    # 直接使用bcrypt
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码
    
    Args:
        plain_password: 原始密码（bcrypt限制最大72字节，超长密码会先进行SHA256哈希）
        hashed_password: 哈希后的密码
    
    Returns:
        如果密码匹配返回True，否则返回False
    
    Example:
        >>> hashed = hash_password("mypassword")
        >>> verify_password("mypassword", hashed)
        True
        >>> verify_password("wrongpassword", hashed)
        False
    """
    if not BCRYPT_AVAILABLE:
        raise ImportError("bcrypt is required for password verification")
    
    # bcrypt限制密码长度不超过72字节
    # 如果密码超过72字节，先进行SHA256哈希（与hash_password保持一致）
    password_bytes = plain_password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = hashlib.sha256(password_bytes).digest()
    
    # 直接使用bcrypt验证
    try:
        return bcrypt.checkpw(password_bytes, hashed_password.encode('utf-8'))
    except Exception:
        return False


def generate_random_string(length: int = 32) -> str:
    """
    生成随机字符串（用于token、salt等）
    
    Args:
        length: 字符串长度（十六进制字符数），默认为32
    
    Returns:
        随机字符串（十六进制）
    
    Example:
        >>> token = generate_random_string(32)
        >>> len(token) == 32  # 32 hex chars
        True
    """
    # token_hex(n) 生成 n*2 个十六进制字符
    # 所以要生成length个字符，需要length//2个字节（向上取整）
    byte_count = (length + 1) // 2
    result = secrets.token_hex(byte_count)
    # 如果结果长度超过要求，截取到指定长度
    return result[:length]


def generate_token(length: int = 32) -> str:
    """
    生成token（URL安全的随机字符串）
    
    Args:
        length: token长度（字节数），默认为32
    
    Returns:
        token字符串（URL安全）
    
    Example:
        >>> token = generate_token()
        >>> len(token) > 0
        True
    """
    return secrets.token_urlsafe(length)


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """
    计算字符串的哈希值
    
    Args:
        text: 要哈希的字符串
        algorithm: 哈希算法（md5, sha1, sha256等），默认为sha256
    
    Returns:
        哈希值（十六进制字符串）
    
    Example:
        >>> hash_value = hash_string("hello")
        >>> len(hash_value) == 64  # SHA256 produces 64 hex chars
        True
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode("utf-8"))
    return hash_obj.hexdigest()

