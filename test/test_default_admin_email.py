"""
默认管理员邮箱常量测试
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.auth.constants import DEFAULT_ADMIN_EMAIL, is_default_admin_email


def test_default_admin_email():
    assert DEFAULT_ADMIN_EMAIL == "admin@vitoom.ai"
    assert is_default_admin_email("admin@vitoom.ai")
    assert is_default_admin_email("ADMIN@vitoom.AI")
    assert not is_default_admin_email("user@example.com")
    assert not is_default_admin_email("")


if __name__ == "__main__":
    test_default_admin_email()
    print("test_default_admin_email passed")
