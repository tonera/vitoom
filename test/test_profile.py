import torch
from torch.profiler import profile, record_function, ProfilerActivity
# 确保 NunchakuFluxTransformer2dModel 和 weight_dir 变量已定义

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision
model_dir = "/home/tonera/models"
weight_dir = "/home/tonera/weights"
precision = get_precision()
# 定义 profiler 的活动类型：同时监控 CPU 和 CUDA 活动
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

# 使用 with profile() 上下文管理器来运行加载操作
with profile(
    activities=activities,
    profile_memory=True,  # 监控内存使用
    record_shapes=True,   # 记录张量形状
    with_stack=True       # 记录调用堆栈以精确定位代码位置（需要更高版本的 PyTorch）
) as prof:
    with record_function("Model_Loading_and_Quantization"):
        # transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"weights/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors")
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"{weight_dir}/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
            device=torch.device("cuda"),
        )

# 打印结果的总结
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# 将详细结果保存到文件中，可以使用 TensorBoard 查看
# prof.export_chrome_trace("transformer_load_trace.json")
