"""认证相关常量。"""

DEFAULT_ADMIN_EMAIL = "admin@vitoom.ai"


def is_default_admin_email(email: str) -> bool:
    """注册时判定是否为内置默认管理员邮箱（大小写不敏感）。"""
    return bool(email) and email.lower() == DEFAULT_ADMIN_EMAIL.lower()
