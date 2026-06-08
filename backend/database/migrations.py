"""
数据库迁移脚本 - SQLAlchemy版本
用于数据库版本管理和升级
python backend/database/migrations.py
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, inspect, text

# 如果作为脚本直接运行，添加项目根目录到路径
if __name__ == "__main__":
    backend_dir = Path(__file__).parent.parent
    project_root = backend_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from backend.database.db import Base, get_engine
    from backend.database import models as _models
else:
    from .db import Base, get_engine
    from . import models as _models

logger = logging.getLogger(__name__)

VERSION_TABLE = "db_version"
TARGET_VERSION = 6
TARGET_DESCRIPTION = "Rename output storage columns to storage"


def _ensure_version_table() -> None:
    """确保版本表存在。"""
    engine = get_engine()
    inspector = inspect(engine)
    if VERSION_TABLE in inspector.get_table_names():
        return

    metadata = MetaData()
    Table(
        VERSION_TABLE,
        metadata,
        Column("version", Integer, primary_key=True),
        Column("applied_at", DateTime, nullable=False),
        Column("description", Text),
    )
    metadata.create_all(engine)


def get_current_version() -> int:
    """获取当前数据库版本。"""
    try:
        _ensure_version_table()
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT MAX(version) AS version FROM {VERSION_TABLE}")
            )
            row = result.fetchone()
            return row[0] if row and row[0] else 0
    except Exception as exc:
        logger.error("Failed to get database version: %s", exc)
        return 0


def set_version(version: int, description: str = "") -> None:
    """追加一个数据库版本记录。"""
    _ensure_version_table()
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text(f"SELECT version FROM {VERSION_TABLE} WHERE version = :version"),
            {"version": version},
        )
        if result.fetchone():
            logger.info("Version %s already exists, skipping", version)
            return

        conn.execute(
            text(
                f"""
                INSERT INTO {VERSION_TABLE} (version, applied_at, description)
                VALUES (:version, :applied_at, :description)
                """
            ),
            {
                "version": version,
                "applied_at": datetime.utcnow(),
                "description": description,
            },
        )


def _squash_version_history(version: int, description: str) -> None:
    """把旧的多版本历史压成单个基线版本。"""
    _ensure_version_table()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {VERSION_TABLE}"))
        conn.execute(
            text(
                f"""
                INSERT INTO {VERSION_TABLE} (version, applied_at, description)
                VALUES (:version, :applied_at, :description)
                """
            ),
            {
                "version": version,
                "applied_at": datetime.utcnow(),
                "description": description,
            },
        )


def _migrate_to_current_schema() -> None:
    """按当前 SQLAlchemy models 创建缺失表。"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_task_model_key_column()
    _ensure_storage_columns()
    _ensure_inference_service_client_ip_column()


def _ensure_inference_service_client_ip_column() -> None:
    """确保 inference_services 表记录 /start 请求来源 IP。"""
    columns = _table_columns("inference_services")
    if not columns or "client_ip" in columns:
        return

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE inference_services ADD COLUMN client_ip VARCHAR(45) NULL"))


def _table_columns(table_name: str) -> set[str]:
    inspector = inspect(get_engine())
    if table_name not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def _table_column_info(table_name: str) -> dict[str, dict]:
    inspector = inspect(get_engine())
    if table_name not in inspector.get_table_names():
        return {}
    return {col["name"]: col for col in inspector.get_columns(table_name)}


def _table_indexes(table_name: str) -> set[str]:
    inspector = inspect(get_engine())
    if table_name not in inspector.get_table_names():
        return set()
    return {idx["name"] for idx in inspector.get_indexes(table_name)}


def _ensure_task_model_key_column() -> None:
    """确保任务表使用 tasks.model_key 记录模型稳定键。"""
    columns = _table_columns("tasks")
    if not columns or "model_key" in columns:
        return

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tasks ADD COLUMN model_key VARCHAR(64) NULL"))
        if "idx_tasks_model_key" not in _table_indexes("tasks"):
            conn.execute(text("CREATE INDEX idx_tasks_model_key ON tasks (model_key)"))


def _ensure_storage_columns() -> None:
    """确保任务表和文件表使用 storage 字段。"""
    engine = get_engine()
    tasks_columns = _table_columns("tasks")
    if tasks_columns and "storage" not in tasks_columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN storage VARCHAR(20) NULL"))
            if "storage_mode" in tasks_columns:
                conn.execute(text("UPDATE tasks SET storage = storage_mode"))
            elif "storage_type" in tasks_columns:
                conn.execute(text("UPDATE tasks SET storage = storage_type"))
            conn.execute(text("UPDATE tasks SET storage = 'local' WHERE storage IS NULL OR storage = ''"))

    files_columns = _table_columns("files")
    if files_columns and "storage" not in files_columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE files ADD COLUMN storage VARCHAR(20) NULL"))
            if "storage_mode" in files_columns:
                conn.execute(text("UPDATE files SET storage = storage_mode"))
            elif "storage_type" in files_columns:
                conn.execute(text("UPDATE files SET storage = storage_type"))
            conn.execute(text("UPDATE files SET storage = 'local' WHERE storage IS NULL OR storage = ''"))

    _drop_legacy_storage_columns()


def _drop_legacy_storage_columns() -> None:
    """移除旧存储列，避免旧 NOT NULL 约束阻断新字段写入。"""
    engine = get_engine()
    dialect = engine.dialect.name

    if dialect == "sqlite":
        _rebuild_sqlite_storage_tables()
        return

    tasks_columns = _table_columns("tasks")
    with engine.begin() as conn:
        if "storage_mode" in tasks_columns:
            conn.execute(text("UPDATE tasks SET storage = storage_mode WHERE storage IS NULL OR storage = ''"))
            conn.execute(text("ALTER TABLE tasks DROP COLUMN storage_mode"))
        if "storage_type" in tasks_columns:
            conn.execute(text("UPDATE tasks SET storage = storage_type WHERE storage IS NULL OR storage = ''"))
            conn.execute(text("ALTER TABLE tasks DROP COLUMN storage_type"))

    files_columns = _table_columns("files")
    with engine.begin() as conn:
        if "storage_mode" in files_columns:
            conn.execute(text("UPDATE files SET storage = storage_mode WHERE storage IS NULL OR storage = ''"))
            conn.execute(text("ALTER TABLE files DROP COLUMN storage_mode"))
        if "storage_type" in files_columns:
            conn.execute(text("UPDATE files SET storage = storage_type WHERE storage IS NULL OR storage = ''"))
            conn.execute(text("ALTER TABLE files DROP COLUMN storage_type"))


def _rebuild_sqlite_storage_tables() -> None:
    """SQLite 需要重建表才能删除列和修正 NOT NULL 约束。"""
    engine = get_engine()
    tasks_columns = _table_column_info("tasks")
    files_columns = _table_column_info("files")

    rebuild_tasks = _needs_sqlite_tasks_rebuild(tasks_columns)
    rebuild_files = _needs_sqlite_files_rebuild(files_columns)
    if not rebuild_tasks and not rebuild_files:
        return

    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        conn.commit()
        trans = conn.begin()
        try:
            if rebuild_tasks:
                _rebuild_sqlite_tasks_table(conn, tasks_columns)
            if rebuild_files:
                _rebuild_sqlite_files_table(conn, files_columns)
            trans.commit()
        except Exception:
            trans.rollback()
            raise
        finally:
            conn.execute(text("PRAGMA foreign_keys=ON"))
            conn.commit()


def _needs_sqlite_tasks_rebuild(columns: dict[str, dict]) -> bool:
    if not columns:
        return False
    legacy_columns = {"storage_type", "storage_mode", "model_id", "is_local_model"}
    storage = columns.get("storage")
    return any(column in columns for column in legacy_columns) or bool(storage and storage.get("nullable"))


def _needs_sqlite_files_rebuild(columns: dict[str, dict]) -> bool:
    if not columns:
        return False
    legacy_columns = {"storage_type", "storage_mode"}
    storage = columns.get("storage")
    return any(column in columns for column in legacy_columns) or bool(storage and storage.get("nullable"))


def _rebuild_sqlite_tasks_table(conn, columns: dict[str, dict]) -> None:
    conn.execute(text("DROP TABLE IF EXISTS tasks_new"))
    conn.execute(
        text(
            """
            CREATE TABLE tasks_new (
                id VARCHAR(36) NOT NULL,
                user_id VARCHAR(36) NOT NULL,
                type VARCHAR(20) NOT NULL,
                status VARCHAR(20) NOT NULL,
                prompt TEXT NOT NULL,
                params TEXT,
                progress INTEGER,
                error TEXT,
                priority INTEGER,
                model_key VARCHAR(64),
                agent_run_id VARCHAR(36),
                storage VARCHAR(20) NOT NULL,
                created_at DATETIME NOT NULL,
                started_at DATETIME,
                completed_at DATETIME,
                PRIMARY KEY (id),
                FOREIGN KEY(user_id) REFERENCES users (id),
                FOREIGN KEY(model_key) REFERENCES model_catalog (model_key),
                FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id)
            )
            """
        )
    )

    model_key_expr = "model_key" if "model_key" in columns else "NULL"
    agent_run_id_expr = "agent_run_id" if "agent_run_id" in columns else "NULL"
    storage_expr = _sqlite_task_storage_expr(columns)
    conn.execute(
        text(
            f"""
            INSERT INTO tasks_new (
                id, user_id, type, status, prompt, params, progress, error, priority,
                model_key, agent_run_id, storage, created_at, started_at, completed_at
            )
            SELECT
                id, user_id, type, status, prompt, params, progress, error, priority,
                {model_key_expr}, {agent_run_id_expr}, {storage_expr}, created_at, started_at, completed_at
            FROM tasks
            """
        )
    )
    conn.execute(text("DROP TABLE tasks"))
    conn.execute(text("ALTER TABLE tasks_new RENAME TO tasks"))
    _create_tasks_indexes(conn)


def _rebuild_sqlite_files_table(conn, columns: dict[str, dict]) -> None:
    conn.execute(text("DROP TABLE IF EXISTS files_new"))
    conn.execute(
        text(
            """
            CREATE TABLE files_new (
                id VARCHAR(36) NOT NULL,
                task_id VARCHAR(36),
                user_id VARCHAR(36) NOT NULL,
                category VARCHAR(20) NOT NULL,
                storage VARCHAR(20) NOT NULL,
                storage_path TEXT NOT NULL,
                file_name VARCHAR(255),
                file_size BIGINT,
                mime_type VARCHAR(100),
                http_url TEXT,
                metadata TEXT,
                created_at DATETIME NOT NULL,
                PRIMARY KEY (id),
                FOREIGN KEY(task_id) REFERENCES tasks (id),
                FOREIGN KEY(user_id) REFERENCES users (id)
            )
            """
        )
    )

    storage_expr = _sqlite_output_storage_expr(columns)
    conn.execute(
        text(
            f"""
            INSERT INTO files_new (
                id, task_id, user_id, category, storage, storage_path, file_name,
                file_size, mime_type, http_url, metadata, created_at
            )
            SELECT
                id, task_id, user_id, category, {storage_expr}, storage_path, file_name,
                file_size, mime_type, http_url, metadata, created_at
            FROM files
            """
        )
    )
    conn.execute(text("DROP TABLE files"))
    conn.execute(text("ALTER TABLE files_new RENAME TO files"))
    _create_files_indexes(conn)


def _sqlite_output_storage_expr(columns: dict[str, dict]) -> str:
    candidates = []
    if "storage" in columns:
        candidates.append("NULLIF(storage, '')")
    if "storage_mode" in columns:
        candidates.append("NULLIF(storage_mode, '')")
    if "storage_type" in columns:
        candidates.append("NULLIF(storage_type, '')")
    candidates.append("'local'")
    return f"COALESCE({', '.join(candidates)})"


def _sqlite_task_storage_expr(columns: dict[str, dict]) -> str:
    candidates = []
    if "storage" in columns:
        candidates.append("NULLIF(storage, '')")
    if "storage_mode" in columns:
        candidates.append("NULLIF(storage_mode, '')")
    if "storage_type" in columns:
        candidates.append("NULLIF(storage_type, '')")
    candidates.append("'local'")
    return f"COALESCE({', '.join(candidates)})"


def _create_tasks_indexes(conn) -> None:
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tasks_user_id ON tasks (user_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (status)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tasks_model_key ON tasks (model_key)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tasks_agent_run_id ON tasks (agent_run_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tasks_created_at ON tasks (created_at)"))


def _create_files_indexes(conn) -> None:
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_files_task_id ON files (task_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_files_user_id ON files (user_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_files_category ON files (category)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_files_created_at ON files (created_at)"))


# 显式引用，避免静态分析把仅用于注册 metadata 的导入视为无用。
_ = _models


def migrate() -> None:
    """执行数据库迁移。"""
    try:
        current_version = get_current_version()

        if current_version > TARGET_VERSION:
            logger.info(
                "Squashing legacy migration history from version %s to version %s",
                current_version,
                TARGET_VERSION,
            )
            _migrate_to_current_schema()
            _squash_version_history(TARGET_VERSION, TARGET_DESCRIPTION)
            logger.info(
                "Database migration history squashed to version %s",
                TARGET_VERSION,
            )
            return

        if current_version == TARGET_VERSION:
            _migrate_to_current_schema()
            logger.info("Database is already at version %s", current_version)
            return

        logger.info(
            "Migrating database from version %s to %s",
            current_version,
            TARGET_VERSION,
        )
        _migrate_to_current_schema()
        set_version(TARGET_VERSION, TARGET_DESCRIPTION)
        logger.info("Migration to version %s completed", TARGET_VERSION)
    except Exception as exc:
        logger.error("Migration failed: %s", exc, exc_info=True)
        raise


def check_migration_needed() -> bool:
    """检查是否需要迁移。"""
    return get_current_version() != TARGET_VERSION


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("Starting database migration...")
    migrate()
    print("Migration completed!")
