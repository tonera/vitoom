"""
JWT密钥生成工具
用于首次启动时生成安全的JWT密钥
"""
import secrets
from pathlib import Path
from typing import Optional


def generate_jwt_secret_key(length: int = 32) -> str:
    """
    生成JWT密钥
    
    Args:
        length: 密钥长度（字节数），默认为32
    
    Returns:
        安全的随机密钥字符串（URL安全）
    
    Example:
        >>> key = generate_jwt_secret_key()
        >>> len(key) > 0
        True
    """
    return secrets.token_urlsafe(length)


def ensure_jwt_secret_key(config_path: Optional[Path] = None) -> str:
    """
    确保JWT密钥存在，如果不存在则生成并保存到配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
    
    Returns:
        JWT密钥字符串
    
    Note:
        此函数会修改配置文件，添加或更新JWT密钥
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "app.yaml"
    
    # 读取现有配置
    from backend.core.config import get_config_manager
    config_manager = get_config_manager()
    
    # 检查是否已有密钥
    existing_key = config_manager.get("security.jwt.secret_key", "")
    
    if existing_key:
        return existing_key
    
    # 生成新密钥
    new_key = generate_jwt_secret_key()
    
    # 读取配置文件内容
    import yaml
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}
    
    # 更新配置
    if "security" not in config_data:
        config_data["security"] = {}
    if "jwt" not in config_data["security"]:
        config_data["security"]["jwt"] = {}
    
    config_data["security"]["jwt"]["secret_key"] = new_key
    
    # 写入配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    return new_key

