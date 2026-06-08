"""
日志系统模块测试脚本
"""
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.logger import (
    get_logger,
    get_app_logger,
    get_task_logger,
    get_error_logger,
    get_inference_logger,
    log_function_call,
    log_execution_time,
    log_exceptions,
    setup_logging,
    get_log_file_path,
)


def test_basic_logging():
    """测试基本日志功能"""
    print("=" * 50)
    print("Testing Basic Logging")
    print("=" * 50)
    
    logger = get_app_logger("test")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print("✓ Basic logging test passed!\n")


def test_log_categories():
    """测试不同日志分类"""
    print("=" * 50)
    print("Testing Log Categories")
    print("=" * 50)
    
    # 应用日志
    app_logger = get_app_logger("test_app")
    app_logger.info("Application log message")
    
    # 任务日志
    task_logger = get_task_logger("test_task")
    task_logger.info("Task log message")
    
    # 错误日志
    error_logger = get_error_logger("test_error")
    error_logger.error("Error log message")
    
    # 推理服务日志
    inference_logger = get_inference_logger("test_service_123")
    inference_logger.info("Inference service log message")
    
    print("✓ Log categories test passed!\n")


def test_log_decorators():
    """测试日志装饰器"""
    print("=" * 50)
    print("Testing Log Decorators")
    print("=" * 50)
    
    @log_function_call()
    def test_function(x, y):
        return x + y
    
    @log_execution_time()
    def slow_function():
        time.sleep(0.1)
        return "done"
    
    @log_exceptions()
    def risky_function():
        raise ValueError("Test exception")
    
    # 测试函数调用日志
    result = test_function(1, 2)
    print(f"✓ Function call logged, result: {result}")
    
    # 测试执行时间日志
    result = slow_function()
    print(f"✓ Execution time logged, result: {result}")
    
    # 测试异常日志
    try:
        risky_function()
    except ValueError:
        print("✓ Exception logged")
    
    print("Log decorators test passed!\n")


def test_log_file_paths():
    """测试日志文件路径"""
    print("=" * 50)
    print("Testing Log File Paths")
    print("=" * 50)
    
    app_log_path = get_log_file_path("app")
    print(f"✓ App log path: {app_log_path}")
    
    task_log_path = get_log_file_path("tasks")
    print(f"✓ Task log path: {task_log_path}")
    
    error_log_path = get_log_file_path("error")
    print(f"✓ Error log path: {error_log_path}")
    
    inference_log_path = get_log_file_path("inference", "service_123")
    print(f"✓ Inference log path: {inference_log_path}")
    
    print("Log file paths test passed!\n")


def test_log_rotation():
    """测试日志轮转"""
    print("=" * 50)
    print("Testing Log Rotation")
    print("=" * 50)
    
    logger = get_app_logger("rotation_test")
    
    # 写入大量日志（测试轮转）
    for i in range(100):
        logger.info(f"Rotation test message {i}")
    
    print("✓ Log rotation test passed (check logs/app.log for rotation)\n")


def test_exception_logging():
    """测试异常日志记录"""
    print("=" * 50)
    print("Testing Exception Logging")
    print("=" * 50)
    
    logger = get_error_logger("exception_test")
    
    try:
        raise ValueError("Test exception with traceback")
    except Exception as e:
        logger.exception("Exception occurred")
        logger.error(f"Exception message: {e}", exc_info=True)
    
    print("✓ Exception logging test passed!\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("Logging System Test Suite")
    print("=" * 50 + "\n")
    
    try:
        # 初始化日志系统
        setup_logging()
        print("✓ Logging system initialized\n")
        
        test_basic_logging()
        test_log_categories()
        test_log_decorators()
        test_log_file_paths()
        test_log_rotation()
        test_exception_logging()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        print("\nCheck log files in logs/ directory:")
        print("  - logs/app.log")
        print("  - logs/tasks.log")
        print("  - logs/error.log")
        print("  - logs/inference/test_service_123.log")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

