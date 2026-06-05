"""模型 + avatar 资源加载（vendored & 改写）。

对应上游：

* ``avatars/musetalk/utils/utils.load_all_model`` → 这里 ``load_all_model``
* ``avatars/musetalk_avatar.load_model`` → 这里 ``load_models``（同时返回
  ``audio_processor``，跟 ``MuseTalkRuntime`` 期待的 bundle 对齐）
* ``avatars/musetalk_avatar.load_avatar`` → 这里 ``load_avatar``

关键差异：

* 全部走 ``musetalk.paths`` 的绝对路径常量，不依赖 cwd
* GPU/CPU/MPS device 选择交给调用方传入（默认走 ``initialize_device()``）
* 缺资源时抛 ``FileNotFoundError`` 并附带 vitoom 侧路径，方便排错
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
from typing import List, Optional

import torch
from common.logger import get_logger  # type: ignore[import-not-found]

from .audio2feature import Audio2Feature
from .paths import (
    MUSETALK_V15_CONFIG,
    MUSETALK_V15_UNET,
    SD_VAE_DIR,
    WHISPER_DIR,
    avatar_dir,
)
from .unet import UNet, PositionalEncoding
from .vae import VAE

logger = get_logger(__name__)

# idle_cycle_frames 描述里允许的 token：整数、整数范围（含 hyphen）
_IDLE_FRAME_TOKEN_RE = re.compile(r"^\s*\d+(?:-\d+)?\s*$")


def parse_idle_cycle_frames_spec(spec: str, num_frames: int) -> List[int]:
    """把 ``idle_cycle_frames`` 配置的字符串解析为 **0-based** 索引列表。

    约定（与 ``full_imgs`` 文件名序一致）：配置里写的是 **人的第几帧**，从 **1**
    开始计数（``00000000.png`` → 帧 ``1``，与文件名数字相差 1）。

    支持的写法：

    * ``"27-81"`` ：闭区间连续帧；
    * ``"1,3,5,38-99"`` ：逗号分隔，可混写单帧与区间；
    * 区间若写成 ``81-27`` 会自动按从小到大展开。

    同一帧出现多次时只保留**首次**出现顺序。解析结果至少含 1 个索引，否则
    抛 ``ValueError``。

    Args:
        spec: 非空配置串
        num_frames: 当前 avatar 总帧数（``len(frame_list_cycle)``）

    Returns:
        升序不保证；顺序与配置中首次出现的顺序一致，供待机镜像轮播使用。
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    indices_1based: List[int] = []
    seen: set[int] = set()

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not _IDLE_FRAME_TOKEN_RE.match(part):
            raise ValueError(
                f"idle_cycle_frames invalid token {part!r}; "
                "use integers or ranges like 27-81 or 1,3,38-99"
            )
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a_f, b_f = int(a_str.strip()), int(b_str.strip())
            lo, hi = (a_f, b_f) if a_f <= b_f else (b_f, a_f)
            for n in range(lo, hi + 1):
                if n not in seen:
                    seen.add(n)
                    indices_1based.append(n)
        else:
            n = int(part)
            if n not in seen:
                seen.add(n)
                indices_1based.append(n)

    if not indices_1based:
        raise ValueError("idle_cycle_frames: empty after parsing")

    out_0based: List[int] = []
    for n in indices_1based:
        if n < 1 or n > num_frames:
            raise ValueError(
                f"idle_cycle_frames: frame {n} out of valid range 1..{num_frames}"
            )
        out_0based.append(n - 1)
    return out_0based


def _idle_indices_from_avator_json_value(
    raw: object, num_frames: int,
) -> Optional[List[int]]:
    """``avator_info.json`` 里 ``idle_cycle_frames`` 字段：str / int 列表均可。"""
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        return parse_idle_cycle_frames_spec(stripped, num_frames)
    if isinstance(raw, (list, tuple)):
        order: List[int] = []
        seen: set[int] = set()
        for x in raw:
            n = int(x)
            if n not in seen:
                seen.add(n)
                order.append(n)
        if not order:
            return None
        return parse_idle_cycle_frames_spec(
            ",".join(str(n) for n in order), num_frames,
        )
    raise TypeError(
        "idle_cycle_frames must be str, list of int, or null (got "
        f"{type(raw).__name__})"
    )


def read_idle_cycle_indices(avatar_path: str, num_frames: int) -> Optional[List[int]]:
    """读 ``<avatar_path>/avator_info.json`` 中的 ``idle_cycle_frames``；缺省返回 None。"""
    info_path = os.path.join(avatar_path, "avator_info.json")
    if not os.path.isfile(info_path):
        return None
    try:
        with open(info_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeError) as exc:
        logger.warning("avator_info.json read failed %s: %s", info_path, exc)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("avator_info.json JSON invalid %s: %s", info_path, exc)
        return None

    if not isinstance(data, dict):
        return None

    raw = data.get("idle_cycle_frames")
    if raw is None:
        return None

    try:
        idxs = _idle_indices_from_avator_json_value(raw, num_frames)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "avatar %s idle_cycle_frames invalid (ignored): %s",
            avatar_path,
            exc,
        )
        return None

    if not idxs:
        return None

    logger.info(
        "idle_cycle_frames: %d-frame subset for standby (total avatar frames=%d)",
        len(idxs),
        num_frames,
    )
    return idxs


def initialize_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _check_path(p: str, what: str) -> None:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"MuseTalk {what} not found at {p}. "
            "Please prepare resources/models/livetalking/* per docs/部署指南.md."
        )


def load_all_model(device: Optional[torch.device] = None):
    """Load VAE + UNet + PositionalEncoding (上游 load_all_model 等价)."""
    _check_path(MUSETALK_V15_UNET, "UNet weights")
    _check_path(MUSETALK_V15_CONFIG, "UNet config")
    _check_path(SD_VAE_DIR, "SD-VAE directory")

    vae = VAE(model_path=SD_VAE_DIR)
    logger.info("loaded MuseTalk SD-VAE from %s", SD_VAE_DIR)
    unet = UNet(unet_config=MUSETALK_V15_CONFIG, model_path=MUSETALK_V15_UNET, device=device)
    logger.info("loaded MuseTalk UNet from %s", MUSETALK_V15_UNET)
    pe = PositionalEncoding(d_model=384)
    return vae, unet, pe


def load_models(device: Optional[torch.device] = None):
    """组装 (vae, unet, pe, timesteps, audio_processor) bundle。

    跟上游 ``avatars/musetalk_avatar.load_model`` 等价；额外把模型权重转
    half/move 到 device 提前做完，调用方可直接交给 ``MuseTalkRuntime``。
    """
    if device is None:
        device = initialize_device()

    _check_path(WHISPER_DIR, "Whisper model directory")

    vae, unet, pe = load_all_model(device=device)
    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    audio_processor = Audio2Feature(model_path=WHISPER_DIR)
    logger.info("loaded MuseTalk Audio2Feature from %s on device=%s", WHISPER_DIR, device)

    return vae, unet, pe, timesteps, audio_processor


def load_avatar(avatar_id: str):
    """加载 avatar 预计算资源 bundle。

    资源约定（``resources/models/livetalking/avatars/<avatar_id>/``）：

    * ``avator_info.json`` —— 元数据（avatar_id、video_path、bbox_shift）；
      可选 ``idle_cycle_frames``：待机全图轮播只使用部分帧（见
      ``parse_idle_cycle_frames_spec``）
    * ``coords.pkl``       —— 每帧 face bbox
    * ``mask_coords.pkl``  —— 每帧 mask crop_box
    * ``latents.pt``       —— 每帧 VAE 编码后的 latent
    * ``full_imgs/``       —— 全身原图序列（按文件名数字升序）
    * ``mask/``            —— 每帧 mask 图像（同上）

    返回 ``(frame_list_cycle, mask_list_cycle, coord_list_cycle,
    mask_coords_list_cycle, input_latent_list_cycle, idle_cycle_indices)``。
    ``idle_cycle_indices`` 为 **0-based** 下标列表；若未配置则为 ``None``，
    表示待机时仍按完整 ``frame_list_cycle`` 做镜像轮播。
    """
    avatar_path = avatar_dir(avatar_id)
    _check_path(avatar_path, f"avatar directory '{avatar_id}'")

    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    coords_path = os.path.join(avatar_path, "coords.pkl")
    latents_out_path = os.path.join(avatar_path, "latents.pt")
    mask_out_path = os.path.join(avatar_path, "mask")
    mask_coords_path = os.path.join(avatar_path, "mask_coords.pkl")

    _check_path(full_imgs_path, "avatar full_imgs/")
    _check_path(coords_path, "avatar coords.pkl")
    _check_path(latents_out_path, "avatar latents.pt")
    _check_path(mask_out_path, "avatar mask/")
    _check_path(mask_coords_path, "avatar mask_coords.pkl")

    input_latent_list_cycle = torch.load(latents_out_path)
    with open(coords_path, "rb") as f:
        coord_list_cycle = pickle.load(f)
    with open(mask_coords_path, "rb") as f:
        mask_coords_list_cycle = pickle.load(f)

    frame_list_cycle = _read_image_sequence(full_imgs_path)
    mask_list_cycle = _read_image_sequence(mask_out_path)

    nframes = len(frame_list_cycle)
    idle_cycle_indices = read_idle_cycle_indices(avatar_path, nframes)

    logger.info(
        "loaded avatar '%s': %d frames / %d masks / %d latents",
        avatar_id, nframes, len(mask_list_cycle), len(input_latent_list_cycle),
    )
    return (
        frame_list_cycle,
        mask_list_cycle,
        coord_list_cycle,
        mask_coords_list_cycle,
        input_latent_list_cycle,
        idle_cycle_indices,
    )


def _read_image_sequence(dir_path: str):
    """按文件名数字升序读取目录下所有 jpg/jpeg/png/PNG 图像（BGR ndarray 列表）."""
    import cv2  # 局部 import：load 模块在没装 opencv 的 dev 环境也能 import

    paths = glob.glob(os.path.join(dir_path, "*.[jpJP][pnPN]*[gG]"))
    paths = sorted(paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if not paths:
        raise FileNotFoundError(f"No image frames in {dir_path}")
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"cv2.imread failed for {p}")
        frames.append(img)
    return frames


def mirror_index(size: int, index: int) -> int:
    """循环往复索引（避免 avatar 视频末尾跳回开头时出现卡顿/跳变）。"""
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    return size - res - 1


__all__ = [
    "Audio2Feature",
    "initialize_device",
    "load_all_model",
    "load_avatar",
    "load_models",
    "mirror_index",
    "parse_idle_cycle_frames_spec",
    "read_idle_cycle_indices",
]
