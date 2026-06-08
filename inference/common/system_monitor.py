"""
系统监控模块
获取系统信息：GPU内存、系统负载、内存使用等
"""
import platform
import psutil
from typing import Dict, Any, Optional
from .logger import get_logger

logger = get_logger(__name__)

# 尝试导入GPU监控库
try:
    import pynvml
    PYNVML_AVAILABLE = True
    try:
        pynvml.nvmlInit()
    except Exception as e:
        logger.warning(f"Failed to initialize NVML: {e}")
        PYNVML_AVAILABLE = False
except ImportError:
    PYNVML_AVAILABLE = False
    logger.info("pynvml not available, GPU monitoring disabled")


class SystemMonitor:
    """系统监控类"""
    
    def __init__(self):
        """初始化系统监控"""
        self._gpu_available = PYNVML_AVAILABLE
    
    def get_hostname(self) -> str:
        """获取主机名"""
        return platform.node()
    
    def get_system_load(self) -> float:
        """
        获取系统负载（1分钟平均负载）
        
        Returns:
            系统负载值
        """
        try:
            # Unix系统返回1分钟平均负载
            if platform.system() != "Windows":
                load_avg = psutil.getloadavg()
                return float(load_avg[0])  # 1分钟负载
            else:
                # Windows系统使用CPU使用率作为近似值
                return psutil.cpu_percent(interval=1) / 100.0
        except Exception as e:
            logger.warning(f"Failed to get system load: {e}")
            return 0.0
    
    def get_memory_info(self) -> Dict[str, int]:
        """
        获取内存信息
        
        Returns:
            包含total和available的字典（单位：字节）
        """
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {
                "total": 0,
                "available": 0,
                "used": 0,
                "percent": 0
            }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        获取GPU信息
        
        Returns:
            包含gpu_total_memory和gpu_available_memory的字典（单位：字节）
        """
        if not self._gpu_available:
            return {
                "gpu_total_memory": 0,
                "gpu_available_memory": 0,
                "gpu_count": 0
            }
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return {
                    "gpu_total_memory": 0,
                    "gpu_available_memory": 0,
                    "gpu_count": 0
                }
            
            # 获取第一个GPU的信息（通常推理器只使用一个GPU）
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                "gpu_total_memory": mem_info.total,
                "gpu_available_memory": mem_info.free,
                "gpu_used_memory": mem_info.used,
                "gpu_count": device_count
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return {
                "gpu_total_memory": 0,
                "gpu_available_memory": 0,
                "gpu_count": 0
            }
    
    def get_all_info(self) -> Dict[str, Any]:
        """
        获取所有系统信息
        
        Returns:
            包含所有系统信息的字典
        """
        info = {
            "host": self.get_hostname(),
            "system_load": self.get_system_load(),
            "memory": self.get_memory_info(),
            "gpu": self.get_gpu_info()
        }
        return info

