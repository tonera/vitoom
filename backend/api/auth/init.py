"""
首次启动初始化
创建默认管理员账户
"""
import os
from pathlib import Path

from backend.api.auth.constants import DEFAULT_ADMIN_EMAIL
from backend.database import User
from backend.database.db import get_db_context
from backend.core.logger import get_app_logger
from backend.utils import generate_uuid
from backend.auth import hash_password

logger = get_app_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _REPO_ROOT / ".env"


def _read_dotenv_value(key: str) -> str | None:
    if not _ENV_PATH.is_file():
        return None
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        env_key, value = stripped.split("=", 1)
        if env_key.strip() == key:
            return value.strip().strip("'\"")
    return None


def get_default_admin_password() -> str | None:
    """从进程环境或项目根目录 .env 读取默认管理员密码。"""
    password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "").strip()
    if password:
        return password
    from_file = _read_dotenv_value("DEFAULT_ADMIN_PASSWORD")
    if from_file:
        return from_file.strip()
    return None


def init_default_admin():
    """
    初始化默认管理员账户

    若数据库中尚无用户，则创建内置默认管理员（admin@vitoom.ai）。
    """
    with get_db_context() as db:
        user_count = db.query(User).count()

        if user_count > 0:
            logger.info("Users already exist, skipping default admin creation")
            return

        admin_email = DEFAULT_ADMIN_EMAIL.lower()

        try:
            existing_user = db.query(User).filter(User.email == admin_email).first()
            if existing_user:
                logger.info(f"Admin user already exists: {admin_email}")
                return

            admin_password = get_default_admin_password()
            if not admin_password:
                logger.error(
                    "DEFAULT_ADMIN_PASSWORD is not set. "
                    "Run scripts/setup_vitoom.py to generate .env or set it in the environment."
                )
                return

            user_id = generate_uuid()
            password_hash = hash_password(admin_password)

            admin_user = User(
                id=user_id,
                email=admin_email,
                password_hash=password_hash,
                nickname="Administrator",
                status="active",
                is_admin=True,
            )

            db.add(admin_user)
            db.commit()

            logger.info(
                f"Default admin user created: {admin_email} (password: {admin_password})"
            )
            logger.warning(
                f"IMPORTANT: Please change the default password for {admin_email} after first login!"
            )

        except Exception as e:
            logger.error(f"Failed to create admin user {admin_email}: {e}")
            db.rollback()


def init_test_user():
    """
    初始化测试账号

    如果测试账号不存在，则创建测试账号
    """
    test_email = "tonera@gmail.com"
    test_password = "test123456"

    with get_db_context() as db:
        existing_user = db.query(User).filter(User.email == test_email.lower()).first()
        if existing_user:
            logger.info(f"Test user already exists: {test_email}")
            return

        try:
            user_id = generate_uuid()
            password_hash = hash_password(test_password)

            test_user = User(
                id=user_id,
                email=test_email.lower(),
                password_hash=password_hash,
                nickname="Test User",
                status="active",
                is_admin=False,
            )

            db.add(test_user)
            db.commit()

            logger.info(f"Test user created: {test_email} (password: {test_password})")

        except Exception as e:
            logger.error(f"Failed to create test user {test_email}: {e}")
            db.rollback()


def check_and_init():
    """检查并初始化（应用启动时调用）。"""
    try:
        init_default_admin()
        # init_test_user()
    except Exception as e:
        logger.error(f"Failed to initialize default admin: {e}")
