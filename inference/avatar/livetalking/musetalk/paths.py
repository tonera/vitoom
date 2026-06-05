"""vitoom 侧 MuseTalk 资源路径常量。

设计与上游 ``LiveTalking`` 的差异：

* 上游约定模型放 ``./models/``、avatar 放 ``./data/avatars/``，依赖 cwd；
  本内核完全用绝对路径，不依赖 cwd / 环境变量，跑测试 / 子进程 / sidecar
  都稳定。
* 通过 ``__file__`` 反推 vitoom 项目根（``inference/avatar/livetalking/musetalk``
  的 4 级父目录）。
* 子目录命名按 **vitoom 侧实际目录** 维护，与上游可能不一致的拼写以本地为准
  （如 ``face-parse-bisenet`` vs 上游 ``face-parse-bisent``）。

"""

import os

# inference/avatar/livetalking/musetalk/paths.py
#   parents[0] = musetalk
#   parents[1] = livetalking
#   parents[2] = avatar
#   parents[3] = inference
#   parents[4] = <vitoom 项目根>
_VITOOM_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir, os.pardir, os.pardir, os.pardir,
    )
)

LIVETALKING_RESOURCES_ROOT = os.path.join(
    _VITOOM_ROOT, "resources", "models", "livetalking"
)

# 等价于上游 "./models/" 与 "./data/avatars/"
MODELS_DIR = LIVETALKING_RESOURCES_ROOT
AVATARS_DIR = os.path.join(LIVETALKING_RESOURCES_ROOT, "avatars")

# musetalk v15 主模型
MUSETALK_V15_DIR = os.path.join(MODELS_DIR, "musetalkV15")
MUSETALK_V15_UNET = os.path.join(MUSETALK_V15_DIR, "unet.pth")
MUSETALK_V15_CONFIG = os.path.join(MUSETALK_V15_DIR, "musetalk.json")

# 子模型（运行时强依赖：whisper + sd-vae；dwpose / face-parse 仅 genavatar 用）
WHISPER_DIR = os.path.join(MODELS_DIR, "whisper")
SD_VAE_DIR = os.path.join(MODELS_DIR, "sd-vae")
DWPOSE_DIR = os.path.join(MODELS_DIR, "dwpose")
FACE_PARSE_DIR = os.path.join(MODELS_DIR, "face-parse-bisenet")


def avatar_dir(avatar_id: str) -> str:
    """指定 avatar_id 的预计算资源目录（含 full_imgs/, latents.pt 等）。"""
    return os.path.join(AVATARS_DIR, avatar_id)


__all__ = [
    "AVATARS_DIR",
    "DWPOSE_DIR",
    "FACE_PARSE_DIR",
    "LIVETALKING_RESOURCES_ROOT",
    "MODELS_DIR",
    "MUSETALK_V15_CONFIG",
    "MUSETALK_V15_DIR",
    "MUSETALK_V15_UNET",
    "SD_VAE_DIR",
    "WHISPER_DIR",
    "avatar_dir",
]
