"""
测试彩色日志输出
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.logger import setup_logging, get_app_logger


def main():
    """测试彩色日志输出"""
    print("=" * 50)
    print("Testing Colorful Logging")
    print("=" * 50)
    print()
    
    # 初始化日志系统
    setup_logging()
    
    # 获取logger
    logger = get_app_logger("color_test")
    
    print("Testing different log levels with colors:")
    print("-" * 50)
    
    logger.debug("This is a DEBUG message (cyan)")
    logger.info("This is an INFO message (green)")
    logger.warning("This is a WARNING message (yellow)")
    logger.error("This is an ERROR message (red)")
    logger.critical("This is a CRITICAL message (red with white background)")
    
    print("-" * 50)
    print()
    print("If you see colors above, colorful logging is working!")
    print("Note: Colors only appear in terminal, not in log files.")


if __name__ == "__main__":
    main()

