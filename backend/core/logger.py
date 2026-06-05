"""
日志系统模块
支持多种日志分类、日志轮转、日志级别控制、彩色输出
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
import traceback

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

from .config import get_logging_config, get_config


# 日志目录
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 日志分类
LOG_CATEGORIES = {
    "app": "app.log",                    # 应用日志
    "tasks": "tasks.log",                # 任务日志
    "error": "error.log",                # 错误日志
    "inference": "inference",            # 推理服务日志目录
}

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# 已初始化的logger缓存
_loggers: Dict[str, logging.Logger] = {}


def _get_log_level(level_name: str) -> int:
    """获取日志级别"""
    return LOG_LEVELS.get(level_name.upper(), logging.INFO)


def _setup_file_handler(
    logger: logging.Logger,
    log_file: Path,
    level: int,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    rotation: str = "size"
) -> logging.Handler:
    """
    设置文件处理器
    
    Args:
        logger: Logger实例
        log_file: 日志文件路径
        level: 日志级别
        max_bytes: 最大文件大小（字节）
        backup_count: 备份文件数量
        rotation: 轮转方式，"size" 或 "time"
    
    Returns:
        配置好的Handler
    """
    # 确保日志目录存在
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    if rotation == "size":
        # 按大小轮转
        handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
    elif rotation == "time":
        # 按时间轮转（每天）
        handler = logging.handlers.TimedRotatingFileHandler(
            str(log_file),
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8"
        )
    else:
        # 默认使用按大小轮转
        handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
    
    handler.setLevel(level)
    
    # 设置格式
    formatter = logging.Formatter(
        get_config("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handler.setFormatter(formatter)
    
    return handler


def _setup_console_handler(logger: logging.Logger, level: int, use_color: bool = True) -> logging.Handler:
    """
    设置控制台处理器
    
    Args:
        logger: Logger实例
        level: 日志级别
        use_color: 是否使用彩色输出
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # 检查是否支持彩色输出（需要colorlog库且终端支持颜色）
    use_color_output = use_color and COLORLOG_AVAILABLE
    
    # 检查终端是否支持颜色
    if use_color_output:
        # 检查是否在终端中运行（不是重定向到文件）
        is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        use_color_output = use_color_output and is_tty
    
    if use_color_output:
        # 使用colorlog的彩色格式
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
    else:
        # 使用普通格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    
    return handler


def get_logger(name: str, category: str = "app") -> logging.Logger:
    """
    获取Logger实例
    
    Args:
        name: Logger名称（通常是模块名）
        category: 日志分类（app, tasks, error, inference）
    
    Returns:
        Logger实例
    
    Example:
        >>> logger = get_logger(__name__, "app")
        >>> logger.info("Application started")
    """
    # 生成logger的唯一键
    logger_key = f"{category}:{name}"
    
    # 如果已经初始化，直接返回
    if logger_key in _loggers:
        return _loggers[logger_key]
    
    # 创建logger
    logger = logging.getLogger(logger_key)
    logger.setLevel(logging.DEBUG)  # 设置最低级别，由handler控制实际输出
    
    # 避免重复添加handler和传播到root logger
    logger.propagate = False
    
    # 避免重复添加handler
    if logger.handlers:
        _loggers[logger_key] = logger
        return logger
    
    # 获取日志配置
    logging_config = get_logging_config()
    log_level = _get_log_level(logging_config.get("level", "INFO"))
    file_config = logging_config.get("file", {})
    use_color = logging_config.get("color", True)  # 默认启用彩色输出
    
    # 添加控制台处理器（总是启用）
    console_handler = _setup_console_handler(logger, log_level, use_color=use_color)
    logger.addHandler(console_handler)
    
    # 添加文件处理器（如果启用）
    if file_config.get("enabled", True):
        log_file_name = LOG_CATEGORIES.get(category, "app.log")
        
        if category == "inference":
            # 推理服务日志需要特殊处理（目录结构）
            log_file = LOG_DIR / "inference" / f"{name}.log"
        else:
            log_file = LOG_DIR / log_file_name
        
        max_bytes = file_config.get("max_bytes", 10 * 1024 * 1024)
        backup_count = file_config.get("backup_count", 5)
        rotation = file_config.get("rotation", "size")
        
        file_handler = _setup_file_handler(
            logger,
            log_file,
            log_level,
            max_bytes=max_bytes,
            backup_count=backup_count,
            rotation=rotation
        )
        logger.addHandler(file_handler)
    
    # 缓存logger
    _loggers[logger_key] = logger
    
    return logger


def get_app_logger(name: str = None) -> logging.Logger:
    """
    获取应用日志Logger（便捷函数）
    
    Args:
        name: Logger名称，默认为调用模块名
    
    Returns:
        Logger实例
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "app")
    
    return get_logger(name, "app")


def get_task_logger(name: str = None) -> logging.Logger:
    """
    获取任务日志Logger（便捷函数）
    
    Args:
        name: Logger名称，默认为调用模块名
    
    Returns:
        Logger实例
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "tasks")
    
    return get_logger(name, "tasks")


def get_error_logger(name: str = None) -> logging.Logger:
    """
    获取错误日志Logger（便捷函数）
    
    Args:
        name: Logger名称，默认为调用模块名
    
    Returns:
        Logger实例
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "error")
    
    return get_logger(name, "error")


def get_inference_logger(service_id: str) -> logging.Logger:
    """
    获取推理服务日志Logger
    
    Args:
        service_id: 推理服务ID
    
    Returns:
        Logger实例
    """
    return get_logger(service_id, "inference")


def log_function_call(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    日志装饰器：记录函数调用
    
    Args:
        logger: Logger实例，如果为None则自动获取
        level: 日志级别
    
    Example:
        >>> @log_function_call()
        >>> def my_function(arg1, arg2):
        >>>     return arg1 + arg2
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_app_logger(func.__module__)
            else:
                func_logger = logger
            
            # 记录函数调用
            func_logger.log(
                level,
                f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            )
            
            try:
                result = func(*args, **kwargs)
                func_logger.log(
                    level,
                    f"{func.__name__} completed successfully"
                )
                return result
            except Exception as e:
                func_logger.error(
                    f"{func.__name__} failed with error: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    日志装饰器：记录函数执行时间
    
    Args:
        logger: Logger实例，如果为None则自动获取
        level: 日志级别
    
    Example:
        >>> @log_execution_time()
        >>> def slow_function():
        >>>     time.sleep(1)
    """
    import time
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_app_logger(func.__module__)
            else:
                func_logger = logger
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                func_logger.log(
                    level,
                    f"{func.__name__} executed in {execution_time:.2f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                func_logger.error(
                    f"{func.__name__} failed after {execution_time:.2f}s: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_exceptions(logger: Optional[logging.Logger] = None, level: int = logging.ERROR):
    """
    日志装饰器：记录异常
    
    Args:
        logger: Logger实例，如果为None则自动获取
        level: 日志级别
    
    Example:
        >>> @log_exceptions()
        >>> def risky_function():
        >>>     raise ValueError("Something went wrong")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_app_logger(func.__module__)
            else:
                func_logger = logger
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_logger.log(
                    level,
                    f"Exception in {func.__name__}: {e}\n{traceback.format_exc()}"
                )
                raise
        
        return wrapper
    return decorator


def setup_logging():
    """
    初始化日志系统
    应该在应用启动时调用
    """
    logging_config = get_logging_config()
    log_level = _get_log_level(logging_config.get("level", "INFO"))
    use_color = logging_config.get("color", True)  # 默认启用彩色输出
    
    # 设置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除已有的handlers（避免重复）
    root_logger.handlers.clear()
    
    # 添加控制台handler
    console_handler = _setup_console_handler(root_logger, log_level, use_color=use_color)
    root_logger.addHandler(console_handler)
    
    # 添加文件handler（如果启用）
    file_config = logging_config.get("file", {})
    if file_config.get("enabled", True):
        log_file = LOG_DIR / "app.log"
        max_bytes = file_config.get("max_bytes", 10 * 1024 * 1024)
        backup_count = file_config.get("backup_count", 5)
        rotation = file_config.get("rotation", "size")
        
        file_handler = _setup_file_handler(
            root_logger,
            log_file,
            log_level,
            max_bytes=max_bytes,
            backup_count=backup_count,
            rotation=rotation
        )
        root_logger.addHandler(file_handler)
    
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


def get_log_file_path(category: str, name: Optional[str] = None) -> Path:
    """
    获取日志文件路径
    
    Args:
        category: 日志分类
        name: 日志名称（用于推理服务等）
    
    Returns:
        日志文件路径
    """
    if category == "inference" and name:
        return LOG_DIR / "inference" / f"{name}.log"
    else:
        log_file_name = LOG_CATEGORIES.get(category, "app.log")
        return LOG_DIR / log_file_name

