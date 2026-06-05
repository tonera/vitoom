"""
数据库初始化脚本
用于首次启动时初始化数据库
"""
import logging
import sys
from pathlib import Path

# 添加backend目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import init_db, check_db_exists
from database.migrations import migrate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数：初始化数据库"""
    logger.info("Starting database initialization...")
    
    try:
        # 初始化数据库（创建表结构）
        init_db()
        logger.info("Database tables created successfully")
        
        # 执行迁移（如果有）
        migrate()
        logger.info("Database migrations applied successfully")
        
        logger.info("Database initialization completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

