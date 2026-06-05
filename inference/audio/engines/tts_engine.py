"""TTS 引擎协议与公共数据结构。

两条推理链路共享：
- task 通道（`inference/audio/handlers/*_tts_handler.py`）
- session 通道（未来 `inference/audio/session_runtime.py` 的 TTS role）

引擎只负责纯合成 + 流式输出，**不建 task、不写 file、不依赖 ResultHandler**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
)

import numpy as np

from schemas import InferenceRequestParams


CancelCheck = Callable[[], Awaitable[bool]]


@dataclass
class VoiceConfig:
    """一次合成请求的纯语音参数。

    task 通道里的 `storage` / `response_format` / `agent_run_id` / `job_type` /
    `stream` 等 Task 语义字段**不进入** VoiceConfig。
    """

    tts_mode: str = "custom_voice"
    speaker_name: Optional[str] = None
    voice_preset: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only: bool = False
    language: Optional[str] = None
    sample_rate: Optional[int] = None
    file_type: str = "wav"
    load_name: Optional[str] = None
    design_seed_text: Optional[str] = None
    design_instruct: Optional[str] = None
    clone_base_load_name: Optional[str] = None
    # VoxCPM 专属：参考音频 / 文本（与 ref_audio/ref_text 语义不同，保持向后兼容）
    prompt_wav_path: Optional[str] = None
    prompt_text: Optional[str] = None
    # 当 True：``prompt_wav_path`` 仅用作 VoxCPM2 "Ultimate Cloning" 的韵律 prompt，
    # 不参与 reference_wav 解析；reference_wav 继续走 ``speaker_name`` preset
    # 或下方的 ``continuation_reference_wav_path``。
    # 用途：``session_runtime`` 在做 chained prompt 注入（方案 D）时设置此标志，
    # 让上一段合成结果只续韵律、不抢音色锚点，避免跨段音色累积漂移。
    # task 通道（drama 用户自带音频）保持默认 False，行为不变。
    continuation_prompt: bool = False
    # 仅在 ``continuation_prompt=True`` 时生效：显式指定 reference_wav 来源
    # （比 speaker_name preset 优先级更高），用于 drama 路径——drama 内每个 character
    # 的 reference 是动态合成出来的 seed wav，不在 speaker_presets 字典里，
    # 必须通过本字段把 character seed_path 直接传给 engine 作为音色锚点。
    # chat 实时路径不传此字段，reference 走 ``speaker_name`` preset。
    continuation_reference_wav_path: Optional[str] = None
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    drama: Optional[Dict[str, Any]] = None
    # 原 params.model_cfg 里暴露给合成层的生成参数（temperature / top_k /
    # stream_chunk_seconds 等），保持 dict 形态以最小化侵入。
    generation_cfg: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class AudioChunk:
    """引擎流式输出的单块音频。

    sequence 由引擎自增；上层（handler / session runtime）可按需重编号。
    终态用 `is_final=True` 表示：最后一条 chunk 的 pcm 可以为空（纯 end 标记）。
    """

    pcm: np.ndarray
    sample_rate: int
    sequence: int = 0
    is_final: bool = False


@runtime_checkable
class TtsEngine(Protocol):
    """纯合成底座。

    `synthesize_stream` 是 async generator，在产出 chunk 前应检查 `cancel_check`
    是否返回 True；命中则优雅退出（不再 yield 新数据，仍应 yield 一条
    `is_final=True` 的结尾 chunk 以便上层收尾）。
    """

    def synthesize_stream(
        self,
        *,
        text: str,
        voice_cfg: VoiceConfig,
        cancel_check: Optional[CancelCheck] = None,
    ) -> AsyncIterator[AudioChunk]: ...


def normalize_audio_array(audio: Any) -> np.ndarray:
    """把 numpy/torch/list 统一成 1D float32 mono 波形。

    关键语义：**高维输入只抽取第一声道 / 第一候选**，绝不做 ``reshape(-1)``
    串联。理由：
      - TTS 模型（如 Qwen3-TTS）在某些配置下会返回 ``(N_candidates, T)``，
        直接 flatten 会把 N 条不同 speaker / 不同语种的音频**顺序拼接**，下
        游前端按 mono 24kHz 播放会出现"多个人声依次说话"的错听（实测就是
        这个 bug）。
      - stereo ``(T, 2)`` 或 ``(2, T)`` 若 interleave 拍平为 2×T 序列，按
        mono 播放会出现左右声道叠加 + 2× 变调。

    启发式：squeeze 掉所有长度为 1 的维度；2D 情况下按"较短的那一维视为
    声道/候选轴"取第 0 项；更高维按首下标递归降到 1D。
    """
    if hasattr(audio, "detach"):
        arr = audio.detach().cpu().float().numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32)
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim > 1:
        arr = np.squeeze(arr).astype(np.float32, copy=False)

    if arr.ndim == 0:
        return np.zeros((0,), dtype=np.float32)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr)

    if arr.ndim == 2:
        h, w = arr.shape
        # 较短轴认定为声道 / 候选轴；等长时按 (channels, samples) 约定取 arr[0]
        if h <= w:
            picked = arr[0]
        else:
            picked = arr[:, 0]
        return np.ascontiguousarray(picked.astype(np.float32, copy=False))

    # ndim >= 3：反复取首下标直到降到 1D（不做交错 flatten）
    while arr.ndim > 1:
        arr = arr[0]
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _resolve_generation_cfg(model_cfg: Any) -> Dict[str, Any]:
    if not isinstance(model_cfg, dict):
        return {}
    generation_cfg = model_cfg.get("generation")
    if isinstance(generation_cfg, dict):
        return dict(generation_cfg)
    return dict(model_cfg)


def voice_config_from_params(params: InferenceRequestParams) -> VoiceConfig:
    """从 InferenceRequestParams 抽出纯语音配置。

    用于 task 通道 handler 把 InferenceRequestParams 喂给引擎；session 通道会直接
    由后端按 schema 构造 VoiceConfig。
    """
    generation_cfg = _resolve_generation_cfg(getattr(params, "model_cfg", None))
    return VoiceConfig(
        tts_mode=str(getattr(params, "tts_mode", "custom_voice") or "custom_voice").strip().lower(),
        speaker_name=getattr(params, "speaker_name", None),
        voice_preset=getattr(params, "voice_preset", None),
        instruct=getattr(params, "instruct", None),
        ref_audio=getattr(params, "ref_audio", None),
        ref_text=getattr(params, "ref_text", None),
        x_vector_only=bool(getattr(params, "x_vector_only", False)),
        language=getattr(params, "language", None),
        sample_rate=getattr(params, "sample_rate", None),
        file_type=str(getattr(params, "file_type", "wav") or "wav"),
        load_name=getattr(params, "load_name", None),
        design_seed_text=getattr(params, "design_seed_text", None),
        design_instruct=getattr(params, "design_instruct", None),
        clone_base_load_name=getattr(params, "clone_base_load_name", None),
        prompt_wav_path=getattr(params, "prompt_wav_path", None),
        prompt_text=getattr(params, "prompt_text", None),
        guidance_scale=getattr(params, "guidance_scale", None),
        num_inference_steps=getattr(params, "num_inference_steps", None),
        drama=getattr(params, "drama", None),
        generation_cfg=generation_cfg or None,
    )


__all__ = [
    "AudioChunk",
    "CancelCheck",
    "TtsEngine",
    "VoiceConfig",
    "normalize_audio_array",
    "voice_config_from_params",
]
