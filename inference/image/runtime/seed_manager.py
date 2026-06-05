import random
from dataclasses import dataclass
import torch

MAX_SEED_VALUE = 4294967295


@dataclass
class SeedPlan:
    seed: int
    generator: torch.Generator


class SeedManager:
    """
    统一处理种子与随机数生成器
    """

    def __init__(self, max_seed: int = MAX_SEED_VALUE):
        self.max_seed = max_seed

    def _normalize_seed(self, seed: int | None) -> int:
        """
        约定（与 video 侧统一）：
        - seed is None 或 seed < 0 表示随机
        - seed >= 0 则按其值使用（包括 0）
        """
        return int(seed) if (seed is not None and int(seed) >= 0) else random.randint(0, self.max_seed)

    def initial_seed(self, user_seed: int | None) -> int:
        return self._normalize_seed(user_seed)

    def batch_seed(self, user_seed: int | None) -> int:
        return self._normalize_seed(user_seed)

    def iteration_seed(self) -> int:
        return self._normalize_seed(-1)

    def create_generator(self, device: str, seed: int) -> torch.Generator:
        # 统一使用 CPU generator：
        # - 一些模型/自定义 pipeline 在 CUDA generator 下表现不一致或直接报错
        # - diffusers 的 randn_tensor 通常支持 “CPU generator + CUDA device”（先在 CPU 采样再搬到目标 device）
        # 这里保留 device 参数仅为兼容现有调用方签名。
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

