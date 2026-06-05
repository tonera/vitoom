"""
数据库连接管理模块 - SQLAlchemy版本
支持SQLite、PostgreSQL、MySQL等多种数据库
"""
import os
from pathlib import Path
from typing import Generator
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# 数据库文件路径（SQLite）
DB_DIR = Path(__file__).parent.parent.parent / "resources" / "data"
DB_FILE = DB_DIR / "vitoom.db"

# SQLAlchemy Base类
Base = declarative_base()

# 数据库引擎和会话工厂
_engine = None
_SessionLocal = None


def _ensure_sqlite_parent_dir(db_url: str) -> None:
    """为 SQLite 文件 URL 创建父目录（含 DATABASE_URL 自定义路径）。"""
    if not db_url.startswith("sqlite"):
        return
    path_part = db_url.split("?", 1)[0]
    if not path_part.startswith("sqlite:///"):
        return
    raw = path_part[len("sqlite:///") :]
    if not raw or raw == ":memory:" or raw.startswith(":"):
        return
    db_path = Path(raw)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)


def get_database_url() -> str:
    """
    获取数据库连接URL
    支持环境变量配置，默认使用SQLite
    """
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        _ensure_sqlite_parent_dir(db_url)
        return db_url

    ensure_db_dir()
    return f"sqlite:///{DB_FILE}?check_same_thread=False"


def get_engine():
    """获取数据库引擎（单例模式）"""
    global _engine
    if _engine is None:
        db_url = get_database_url()
        
        # SQLite特殊配置
        if db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
            # 使用StaticPool避免连接问题
            engine = create_engine(
                db_url,
                connect_args=connect_args,
                poolclass=StaticPool,
                echo=False,  # 设置为True可以看到SQL语句
            )
            
            # SQLite外键支持
            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            # PostgreSQL、MySQL等其他数据库
            engine = create_engine(
                db_url,
                pool_pre_ping=True,  # 连接前ping，自动重连
                echo=False,
            )
        
        _engine = engine
        logger.info(f"Database engine created: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    
    return _engine


def get_session_local():
    """获取会话工厂（单例模式）"""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话（用于FastAPI依赖注入）
    
    Usage:
        from database import get_db
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    获取数据库会话的上下文管理器（兼容旧接口）
    
    Usage:
        with get_db_context() as db:
            user = db.query(User).get(user_id)
    """
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        db.close()


def init_db():
    """
    初始化数据库：创建所有表结构
    """
    ensure_db_dir()
    
    engine = get_engine()
    db_url = get_database_url()
    logger.info(f"Initializing database at {db_url}")
    
    # 导入所有模型，确保它们被注册到Base.metadata
    from .models import (
        User, ApiKey, Task, Model, File, UserUpload, Config, ErrorLog,
        InferenceService, InferenceServiceLog,
        Agent, AgentRun, Conversation, ConversationMessage
    )
    
    # 创建所有表
    Base.metadata.create_all(bind=engine)
    
    logger.info("Database initialized successfully")


def ensure_db_dir():
    """确保数据库目录存在"""
    DB_DIR.mkdir(parents=True, exist_ok=True)


def get_db_path() -> Path:
    """获取数据库文件路径（SQLite）"""
    return DB_FILE


def check_db_exists() -> bool:
    """检查数据库文件是否存在（SQLite）"""
    if get_database_url().startswith("sqlite"):
        return DB_FILE.exists()
    # 对于其他数据库，总是返回True（连接时检查）
    return True


def get_db_size() -> int:
    """获取数据库文件大小（字节，仅SQLite）"""
    if get_database_url().startswith("sqlite") and check_db_exists():
        return DB_FILE.stat().st_size
    return 0


def close_all_connections():
    """关闭所有数据库连接"""
    global _engine
    if _engine:
        _engine.dispose()
        _engine = None
        logger.info("All database connections closed")
