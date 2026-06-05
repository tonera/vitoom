import torch
import time
from datetime import datetime

def print_gpu_memory_usage(stage=""):
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        utilization = (allocated / total_memory) * 100 if total_memory > 0 else 0
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{stage}] GPU内存: {allocated:.2f}GB/{total_memory:.2f}GB ({utilization:.1f}%), 已保留={reserved:.2f}GB, 峰值={max_allocated:.2f}GB")
    else:
        print(f"[{stage}] GPU不可用")

def print_detailed_gpu_info(stage=""):
    """打印详细的GPU信息"""
    if not torch.cuda.is_available():
        print(f"[{stage}] GPU不可用")
        return
    
    print(f"\n=== 详细GPU信息 [{stage}] ===")
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        
        # 内存信息
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        # 设备信息
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        
        print(f"GPU {i}: {device_name}")
        print(f"  计算能力: {device_capability[0]}.{device_capability[1]}")
        print(f"  总内存: {total_memory:.2f}GB")
        print(f"  已分配: {allocated:.2f}GB ({allocated/total_memory*100:.1f}%)")
        print(f"  已保留: {reserved:.2f}GB ({reserved/total_memory*100:.1f}%)")
        print(f"  峰值使用: {max_allocated:.2f}GB ({max_allocated/total_memory*100:.1f}%)")
        print(f"  空闲内存: {total_memory-reserved:.2f}GB")
    
    print("=" * 50)

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU缓存已清理")

def reset_gpu_memory_stats():
    """重置GPU内存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print("GPU内存统计已重置")

