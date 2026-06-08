"""
日志模块
提供统一的日志配置
支持彩色日志输出（使用colorlog库）
"""
import logging
import sys
import os
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np

# 尝试导入colorlog用于彩色日志
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


def _parse_log_level(value: str) -> Optional[int]:
    """
    将环境变量/字符串解析为 logging level。
    支持：DEBUG/INFO/WARN/WARNING/ERROR/CRITICAL/FATAL，以及纯数字级别。
    解析失败返回 None。
    """
    v = (value or "").strip()
    if not v:
        return None
    vu = v.upper()
    if vu.isdigit():
        try:
            return int(vu)
        except Exception:
            return None
    if vu == "WARN":
        vu = "WARNING"
    if vu == "FATAL":
        vu = "CRITICAL"
    return getattr(logging, vu, None) if hasattr(logging, vu) else None


def get_logger(
    name: str,
    log_level: Optional[int] = None,
    log_file: Optional[str] = None,
    service_id: Optional[str] = None
) -> logging.Logger:
    """
    获取配置好的日志记录器
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别（可选）。若为 None，则读取 INFERENCE_LOG_LEVEL/LOG_LEVEL，未设置或解析失败则默认 INFO。
        log_file: 日志文件路径（可选）
        service_id: 服务ID，用于日志文件名（可选）
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    # 避免日志同时被本 logger 的 handler 和 root logger 重复输出
    # （例如外部代码调用了 logging.basicConfig 导致 root 也有 handler）
    logger.propagate = False
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 确定日志级别：优先使用传入参数；否则从环境变量读取，最后回退 INFO
    if log_level is None:
        env_level = os.getenv("INFERENCE_LOG_LEVEL") or os.getenv("LOG_LEVEL")
        parsed = _parse_log_level(env_level) if env_level else None
        log_level = parsed if parsed is not None else logging.INFO
    
    logger.setLevel(log_level)
    
    # 检查是否支持彩色输出（需要colorlog库且终端支持颜色）
    use_color = COLORLOG_AVAILABLE and sys.stdout.isatty()
    
    if use_color:
        # 使用colorlog的彩色格式
        formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s%(reset)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            reset=True,
            style='%'
        )
    else:
        # 普通格式（文件输出或不支持颜色时使用）
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # 控制台handler（使用彩色格式）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    elif service_id:
        # 如果没有指定日志文件但提供了service_id，使用默认路径
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"inferrer_{service_id}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 打印变量,调试利器
def print_info(data, prefix=""):
    logger = get_logger(__name__)
    if isinstance(data, dict):
        logger.debug(f"{prefix}dict长度: {len(data)}")
        for k, v in data.items():
            print_info(v, prefix=f"{prefix}{k}: ")
    elif isinstance(data, list):
        logger.debug(f"{prefix}list, 长度: {len(data)}")
        # 可选：递归打印每个元素
        # for idx, item in enumerate(data):
        #     print_info(item, prefix=f"{prefix}[{idx}]: ")
    elif isinstance(data, (int, float, str)):
        logger.debug(f"{prefix}{data}")
    elif isinstance(data, Image.Image):
        logger.debug(f"{prefix}图片, 大小: {data.size}, 宽: {data.width}, 高: {data.height}")
    elif isinstance(data, np.ndarray):
        logger.debug(f"{prefix}图片(numpy数组), 形状: {data.shape}")
    else:
        # 针对 torch.Generator：打印 device/seed，方便判断随机数发生在 CPU 还是 CUDA。
        # 这里用懒导入，避免 logger 模块在非推理环境下强依赖 torch。
        try:
            import torch  # type: ignore

            if isinstance(data, torch.Generator):
                dev = getattr(data, "device", None)
                seed = None
                try:
                    # torch.Generator.initial_seed() exists on both CPU/CUDA generators
                    seed = data.initial_seed()
                except Exception:
                    seed = None
                logger.debug(f"{prefix}torch.Generator, device: {dev}, seed: {seed}")
                return
        except Exception:
            pass

        if hasattr(data, "__class__") and not isinstance(data, type):
            logger.debug(f"{prefix}类: {data.__class__.__name__}")
        else:
            logger.debug(f"{prefix}类型: {type(data).__name__}")