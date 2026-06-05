"""Qwen3-TTS task 通道薄壳 handler。

职责：
  - 承接 task 通道的 `InferenceRequestParams`
  - 委托 [`QwenTtsEngine`](../engines/qwen_tts_engine.py) 做纯合成
  - 维持 task 通道的外部行为：`audio_stream_*` 流事件 + `result_handler.process_single_result` 落盘
  - 当 ``params.drama`` 非空时，本地编排 design + voice_clone-prompt-cache 多角色合成
    （参考 ``voxcpm_tts_handler._run_drama`` 的 character/dialogue blocks 切分思路）

session 通道复用同一个 `QwenTtsEngine`，但不经过本文件。
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

from common.logger import get_logger
from common.result_handler import ResultHandler
from schemas import InferenceRequestParams

from audio.engines.qwen_tts_engine import QwenTtsEngine
from audio.engines.tts_engine import (
    VoiceConfig,
    normalize_audio_array,
    voice_config_from_params,
)
from audio.runtime.audio_wav_utils import audio_tensor_to_pcm16_bytes
from audio.runtime.qwen_tts_bridge import (
    normalize_qwen_language,
    resolve_qwen_custom_speaker,
)
from audio.handlers.progress_estimator import (
    estimate_tts_generation_seconds,
    tick_estimated_progress,
)

logger = get_logger(__name__)

# 单一来源在 engine：handler/drama 复用同一组默认权重；普通 TTS 的"按 voice_cfg 选权重"
# 规则也在 engine 里实现，本 handler 不再做二次默认。
from audio.engines.qwen_tts_engine import (
    DEFAULT_QWEN_BASE_MODEL as DEFAULT_DRAMA_CLONE_BASE_MODEL_NAME,
    DEFAULT_QWEN_CUSTOM_VOICE_MODEL,
    DEFAULT_QWEN_VOICE_DESIGN_MODEL,
)


# bundle_loader 的精确签名：``mode`` 是必填位置参数（"tts"/"asr" 等），
# ``load_name`` 是可选 kwarg，用来在同一推理任务里串行加载多份权重，
# 触发 ``LRU=1`` 的 bundle_cache 自动驱逐前一份、释放 VRAM。
BundleLoader = Callable[..., Awaitable[Dict[str, Any]]]


class QwenTtsHandler:
    """Qwen3-TTS task 通道 handler（薄壳）。

    纯合成逻辑已迁移至 `QwenTtsEngine`；本文件仅处理 task 通道的流事件编排与
    落盘回调。drama 路径直接持有 design / Base 两份 bundle，绕过 engine 的
    单段 ``synthesize_stream``，对每个 character 复用同一份 ``voice_clone_prompt``。
    """

    def __init__(
        self,
        *,
        audio_mode: str,
        result_handler: ResultHandler,
        service_id: str,
        logger=logger,
        bundle_loader: BundleLoader,
        stream_sender: Callable[[Dict[str, Any]], Awaitable[bool]],
        check_cancelled: Optional[Callable[[str], Awaitable[bool]]] = None,
        status_sender: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        if audio_mode != "tts":
            raise ValueError(
                f"Qwen-tts runtime currently supports only audio_mode=tts, got {audio_mode}"
            )
        self.audio_mode = audio_mode
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.stream_sender = stream_sender
        self.check_cancelled = check_cancelled
        self.status_sender = status_sender
        self._bundle_loader = bundle_loader
        self._engine = QwenTtsEngine(bundle_loader=bundle_loader, logger=logger)

    async def run(self, params: InferenceRequestParams, *, task_id: str) -> None:
        if params.type != "audio":
            raise ValueError(f"QwenTtsHandler expected req.type='audio', got '{params.type}'")

        # 单段 TTS 的"按 voice_cfg 选权重 + 对齐 mode"已下沉到 engine
        # (``QwenTtsEngine._resolve_weight_and_mode``)；drama 自己 per-character 决定。
        drama = getattr(params, "drama", None)
        is_drama = isinstance(drama, dict) and drama.get("characters") and drama.get("dialogues")

        voice_cfg = voice_config_from_params(params)
        stream_enabled = bool(params.stream)
        started_at = time.time()
        await self._send_progress(task_id, 5, "准备音频生成")

        if is_drama:
            await self._run_drama(
                params=params,
                task_id=task_id,
                voice_cfg=voice_cfg,
                started_at=started_at,
            )
            return

        collected: list[np.ndarray] = []
        final_audio: np.ndarray | None = None
        final_sr: int | None = None
        sequence = 0
        started = False

        self.logger.info(
            "[qwen-tts] task_id=%s tts_mode=%s stream=%s",
            task_id,
            voice_cfg.tts_mode,
            stream_enabled,
        )

        progress_ticker: Optional[asyncio.Task[None]] = None
        progress_stop_event: Optional[asyncio.Event] = None
        try:
            progress_cursor = 15
            await self._send_progress(task_id, progress_cursor, "开始合成音频")
            estimated_seconds: Optional[float] = None
            if not stream_enabled:
                estimated_seconds = await estimate_tts_generation_seconds(
                    engine=self._engine,
                    task_id=task_id,
                    text=params.prompt or "",
                    voice_cfg=voice_cfg,
                    send_progress=self._send_progress,
                    cancel_check=self._cancel_check,
                    logger=self.logger,
                    log_prefix="qwen-tts",
                )
                if estimated_seconds is not None:
                    progress_cursor = 20
                    await self._send_progress(task_id, progress_cursor, "测速完成，开始合成音频")
                    progress_stop_event = asyncio.Event()
                    progress_ticker = asyncio.create_task(
                        tick_estimated_progress(
                            task_id=task_id,
                            send_progress=self._send_progress,
                            start=progress_cursor,
                            end=88,
                            estimated_seconds=estimated_seconds,
                            message="正在合成音频",
                            stop_event=progress_stop_event,
                        )
                    )
            async for chunk in self._engine.synthesize_stream(
                text=params.prompt or "",
                voice_cfg=voice_cfg,
                cancel_check=self._cancel_check,
                stream_mode=stream_enabled,
            ):
                if stream_enabled and not started:
                    await self._emit_event(
                        task_id=task_id,
                        event_type="audio_stream_start",
                        sequence=sequence,
                        sample_rate=chunk.sample_rate,
                        is_final=False,
                    )
                    sequence += 1
                    started = True

                if chunk.is_final:
                    final_audio = chunk.pcm
                    final_sr = chunk.sample_rate
                    continue

                collected.append(chunk.pcm)
                if stream_enabled:
                    next_progress = min(90, progress_cursor + 3)
                    if next_progress > progress_cursor:
                        progress_cursor = next_progress
                        await self._send_progress(task_id, progress_cursor, "正在合成音频")
                if stream_enabled and chunk.pcm.size > 0:
                    await self._emit_event(
                        task_id=task_id,
                        event_type="audio_stream_chunk",
                        sequence=sequence,
                        sample_rate=chunk.sample_rate,
                        is_final=False,
                        data=base64.b64encode(audio_tensor_to_pcm16_bytes(chunk.pcm)).decode("ascii"),
                    )
                    sequence += 1
        finally:
            if progress_ticker:
                if progress_stop_event:
                    progress_stop_event.set()
                try:
                    await progress_ticker
                except asyncio.CancelledError:
                    pass
            if stream_enabled and started:
                await self._emit_event(
                    task_id=task_id,
                    event_type="audio_stream_end",
                    sequence=sequence,
                    sample_rate=final_sr or (collected[-1].size and 24000) or 24000,
                    is_final=True,
                )

        if self.check_cancelled and await self.check_cancelled("after qwen-tts generation"):
            return

        if final_audio is None or final_audio.size == 0:
            if collected:
                final_audio = np.concatenate(collected, axis=0)
                final_sr = final_sr or 24000
        if final_audio is None or final_audio.size == 0:
            raise RuntimeError("qwen-tts engine produced no audio")

        self.logger.info(
            "[qwen-tts] task_id=%s tts_mode=%s generation finished samples=%s seconds=%.3f",
            task_id,
            voice_cfg.tts_mode,
            final_audio.shape[0],
            (final_audio.shape[0] / final_sr) if final_sr else -1.0,
        )

        params.file_type = "wav"
        await self._send_progress(task_id, 95, "正在保存音频")
        await self.result_handler.process_single_result(
            file_data=_audio_bytes(final_audio, final_sr or 24000),
            request_params=params,
            generate_time=time.time() - started_at,
            service_id=self.service_id,
            index=0,
            total=1,
        )

    # =====================  drama 多角色合成  =====================

    async def _run_drama(
        self,
        *,
        params: InferenceRequestParams,
        task_id: str,
        voice_cfg: VoiceConfig,
        started_at: float,
    ) -> None:
        """Qwen3-TTS 的 drama 实现：per-character 决定 seed 权重 → swap to Base → 复用克隆。

        权重选型（per character）跟"单段 TTS 默认规则"对齐：
          - character 有 ``instruct`` 或 ``voice_mode='voice_design'`` → 走 VoiceDesign
            权重，``generate_voice_design`` 把声线设计成 seed；
          - character 是 ``custom_voice + speaker_name`` → 走 CustomVoice 权重，
            ``generate_custom_voice`` 用预置说话人产出 seed（不强行从 speaker meta 派生
            instruct 再走 design，保留预置音色的原貌）；
          - 两者都没有 → 直接报错让 LLM 补齐。

        显存策略：``bundle_cache`` 是 LRU=1，cache_key 一变就会立刻调 ``release_fn``
        把上一份权重的 VRAM 收回，所以同一时刻 GPU 上最多只持一份 Qwen-TTS 权重。
        组合 swap 序列（按需触发）：
          - 仅 voice_design 角色：VoiceDesign → Base，2 次 load；
          - 仅 custom_voice 角色：CustomVoice → Base，2 次 load；
          - 两类混合：VoiceDesign → CustomVoice → Base，3 次 load。
        每段 character 的 seed 立即搬到 CPU numpy，design/custom_voice bundle 在阶段
        结束时被丢掉强引用，等下一次 ``bundle_loader`` 触发 LRU evict。

        pin 模式：``_coerce_params_for_pin`` 已把所有 ``load_name`` 改写成 fixed_model，
        这里 swap 的 override 也会被 coerce 回 fixed_model。如果 fixed_model 不暴露
        某阶段需要的 capability（典型如 pin=Base 不能做 voice_design），会在该阶段的
        bundle 加载后立即抛出"primary weight does not expose ..."的明确错误，提示
        操作员去掉 pin 或换权重。
        """
        drama = getattr(params, "drama", None)
        if not isinstance(drama, dict):
            raise ValueError("audio drama requires params.drama")
        characters = self._normalize_drama_characters(drama.get("characters"))
        dialogues = self._normalize_drama_dialogues(drama.get("dialogues"))
        if not characters:
            raise ValueError("audio drama requires at least one character")
        if not dialogues:
            raise ValueError("audio drama requires at least one dialogue line")

        # 解析每段会用到的权重名（design / custom_voice / clone-base）。
        # voice_cfg.load_name 在 drama 场景下被解读为"design 阶段的偏好权重"，
        # 没传就回落到默认 1.7B-VoiceDesign；CustomVoice 阶段不接受用户的 load_name
        # 偏好（用户传的多半是 VoiceDesign 名字，对 custom_voice 角色无意义），
        # 直接用 1.7B-CustomVoice 默认。
        design_weight = (
            str(voice_cfg.load_name or "").strip() or DEFAULT_QWEN_VOICE_DESIGN_MODEL
        )
        custom_voice_weight = DEFAULT_QWEN_CUSTOM_VOICE_MODEL
        clone_base_weight = (
            (voice_cfg.clone_base_load_name or "").strip()
            or DEFAULT_DRAMA_CLONE_BASE_MODEL_NAME
        )

        # 把 character 按 seed 阶段需要的权重分组。一组只 load 一次权重，组内串行处理。
        voice_design_group: List[tuple[str, Dict[str, Any]]] = []
        custom_voice_group: List[tuple[str, Dict[str, Any]]] = []
        for character_id, character in characters.items():
            plan = self._classify_drama_character(character)
            if plan == "voice_design":
                voice_design_group.append((character_id, character))
            elif plan == "custom_voice":
                custom_voice_group.append((character_id, character))
            else:
                raise ValueError(
                    f"character {character_id} 缺少声音定义；qwen-tts drama 要求每个 character 至少给出 "
                    "instruct（走 voice_design）或 speaker_name（走 custom_voice 预置）"
                )

        if self.check_cancelled and await self.check_cancelled("before drama bundle load"):
            return

        self.logger.info(
            "[qwen-tts][drama] task_id=%s characters=%d voice_design=%d custom_voice=%d "
            "weights={design=%r custom_voice=%r clone_base=%r}",
            task_id,
            len(characters),
            len(voice_design_group),
            len(custom_voice_group),
            design_weight,
            custom_voice_weight,
            clone_base_weight,
        )

        character_seed_state: Dict[str, Dict[str, Any]] = {}
        sample_rate = 24000
        await self._send_progress(task_id, 8, "准备角色对白")

        # --- Phase 1a: VoiceDesign 组（如果有）---
        if voice_design_group:
            await self._send_progress(task_id, 12, "正在生成角色音色")
            character_seed_state, sample_rate = await self._design_seeds_via_voice_design(
                characters=voice_design_group,
                voice_cfg=voice_cfg,
                task_id=task_id,
                weight_name=design_weight,
                seed_state=character_seed_state,
                sample_rate=sample_rate,
            )
            if character_seed_state is None:
                return  # cancelled mid-way

        # --- Phase 1b: CustomVoice 组（如果有）。会触发 LRU evict，释放 VoiceDesign。---
        if custom_voice_group:
            await self._send_progress(task_id, 28, "正在生成预置角色音色")
            character_seed_state, sample_rate = await self._design_seeds_via_custom_voice(
                characters=custom_voice_group,
                voice_cfg=voice_cfg,
                task_id=task_id,
                weight_name=custom_voice_weight,
                seed_state=character_seed_state,
                sample_rate=sample_rate,
            )
            if character_seed_state is None:
                return

        # --- Phase 2: 切到 Base，做 voice_clone_prompt + 跑 dialogue clone ---
        if self.check_cancelled and await self.check_cancelled("before drama base load"):
            return
        await self._send_progress(task_id, 45, "正在准备对白合成")
        self.logger.info(
            "[qwen-tts][drama] task_id=%s designed %d characters; swapping to clone base weight=%r",
            task_id,
            len(character_seed_state),
            clone_base_weight,
        )
        base_bundle = await self._bundle_loader("tts", load_name=clone_base_weight)
        caps = base_bundle.get("capabilities") or {}
        if not (caps.get("voice_clone") and caps.get("create_voice_clone_prompt")):
            raise RuntimeError(
                f"qwen-tts drama Base weight at {base_bundle.get('model_ref')!r} does not expose "
                "voice_clone / create_voice_clone_prompt. 请确认 clone_base_load_name 指向 "
                "Qwen3-TTS-12Hz-*-Base 权重；如果 qwen-tts 服务被 pin 到不支持的权重，"
                "请去掉 fixed_model 或换 pin。"
            )
        if self._is_streaming_variant_bundle(base_bundle):
            raise RuntimeError(
                "qwen-tts drama 暂不支持 nano_vllm streaming 变体的 Base 权重。"
            )
        base_model = base_bundle["model"]
        if not hasattr(base_model, "create_voice_clone_prompt"):
            raise RuntimeError(
                "qwen-tts drama requires Base weight exposing create_voice_clone_prompt(); "
                f"got base_model_ref={base_bundle.get('model_ref')!r}"
            )

        # --- Phase 3: 用 Base 给每个 character 建 voice_clone_prompt（一次，下面整段都复用）---
        character_clone_state: Dict[str, Dict[str, Any]] = {}
        clone_total = max(1, len(character_seed_state))
        for clone_index, (character_id, seed) in enumerate(character_seed_state.items(), start=1):
            if self.check_cancelled and await self.check_cancelled(
                f"before drama character clone-prompt ({character_id})"
            ):
                return
            prompt_items = await asyncio.to_thread(
                base_model.create_voice_clone_prompt,
                ref_audio=(seed["seed_audio"], seed["seed_sr"]),
                ref_text=seed["seed_text"],
            )
            character_clone_state[character_id] = {
                "prompt": prompt_items,
                "language": seed["language"],
            }
            clone_progress = 45 + int(round((clone_index / clone_total) * 10))
            await self._send_progress(task_id, clone_progress, "正在准备角色提示")

        # --- Phase 4: 把 dialogue 切成"同一 character 连续段"的 block，按 block 批量克隆 ---
        dialogue_blocks = self._build_dialogue_blocks(dialogues)
        self.logger.info(
            "[qwen-tts][drama] task_id=%s dialogue_lines=%d blocks=%d",
            task_id,
            len(dialogues),
            len(dialogue_blocks),
        )

        pieces: List[np.ndarray] = []
        output_sr: Optional[int] = None
        block_total = max(1, len(dialogue_blocks))
        for block_index, block in enumerate(dialogue_blocks, start=1):
            if self.check_cancelled and await self.check_cancelled("before drama block clone"):
                return
            character_id = block["speaker_id"]
            character = characters.get(character_id)
            clone_state = character_clone_state.get(character_id)
            if character is None or clone_state is None:
                raise ValueError(f"unknown drama speaker_id: {character_id}")
            line_texts = [line["text"] for line in block["lines"]]
            line_languages = [
                normalize_qwen_language(
                    line.get("language") or character.get("language") or voice_cfg.language,
                    supported_languages=base_bundle.get("supported_languages"),
                )
                for line in block["lines"]
            ]
            clone_kwargs = self._build_common_generation_kwargs(voice_cfg)
            clone_kwargs.update(
                text=line_texts if len(line_texts) > 1 else line_texts[0],
                language=line_languages if len(line_languages) > 1 else line_languages[0],
                voice_clone_prompt=clone_state["prompt"],
            )
            wavs, sr = await asyncio.to_thread(base_model.generate_voice_clone, **clone_kwargs)
            sr = int(sr or sample_rate or 24000)
            if output_sr is None:
                output_sr = sr
            elif sr != output_sr:
                raise RuntimeError(
                    f"qwen-tts drama line sample_rate mismatch: {sr} != {output_sr}"
                )

            audios = self._collect_clone_outputs(wavs, expected=len(line_texts))
            for audio in audios:
                pieces.append(self._match_loudness(self._trim_silence(audio)))
            pause_ms = self._block_pause_after_ms(block)
            if pause_ms > 0:
                pieces.append(
                    np.zeros(int((output_sr or sr) * pause_ms / 1000), dtype=np.float32)
                )
            block_progress = 55 + int(round((block_index / block_total) * 35))
            await self._send_progress(task_id, block_progress, "正在合成对白")

        final_audio = (
            np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=np.float32)
        )
        final_sr = int(output_sr or sample_rate or 24000)
        if final_audio.size == 0:
            raise RuntimeError("qwen-tts drama generation returned no audio")

        params.file_type = "wav"
        await self._send_progress(task_id, 95, "正在保存对白音频")
        await self.result_handler.process_single_result(
            file_data=_audio_bytes(final_audio, final_sr),
            request_params=params,
            generate_time=time.time() - started_at,
            service_id=self.service_id,
            index=0,
            total=1,
        )

    async def _design_seeds_via_voice_design(
        self,
        *,
        characters: List[tuple[str, Dict[str, Any]]],
        voice_cfg: VoiceConfig,
        task_id: str,
        weight_name: str,
        seed_state: Dict[str, Dict[str, Any]],
        sample_rate: int,
    ) -> tuple[Optional[Dict[str, Dict[str, Any]]], int]:
        """加载 VoiceDesign 权重 → 给"voice_design 组"每个 character 产 seed audio。

        函数返回时 design bundle 的强引用已被丢弃；下一次 ``bundle_loader`` 拿不同
        ``load_name`` 时，``LRU=1`` 会立刻 ``release_fn(bundle)`` 把权重 VRAM 收回。
        """
        bundle = await self._bundle_loader("tts", load_name=weight_name)
        caps = bundle.get("capabilities") or {}
        if not caps.get("voice_design"):
            raise RuntimeError(
                f"qwen-tts drama voice_design 阶段加载到的权重 {bundle.get('model_ref')!r} 不暴露 "
                f"voice_design（caps={caps}）。请把 load_name 指向 VoiceDesign 权重，"
                "或如果 qwen-tts 服务被 pin 到 Base/CustomVoice，请去掉 fixed_model。"
            )
        if self._is_streaming_variant_bundle(bundle):
            raise RuntimeError(
                "qwen-tts drama 暂不支持 nano_vllm streaming 变体；"
                "请把 audio runtime.backend 切到 transformers 后再合成 drama。"
            )
        model = bundle["model"]
        supported_languages = bundle.get("supported_languages")
        sample_rate = int(bundle.get("sample_rate") or sample_rate or 24000)

        for character_id, character in characters:
            if self.check_cancelled and await self.check_cancelled(
                f"before drama voice_design seed ({character_id})"
            ):
                model = None
                bundle = None
                return None, sample_rate
            seed_text = self._seed_text_for_character(character)
            seed_instruct = str(character.get("instruct") or "").strip()
            if not seed_instruct:
                # 走到 voice_design 组但又没 instruct，多半是分组分错。明确报错让上层
                # 修正，不再像旧实现那样从 speaker meta 派生 instruct。
                raise ValueError(
                    f"character {character_id} 被分到 voice_design 组但没有 instruct；"
                    "voice_design 角色必须显式给 instruct"
                )
            seed_language = normalize_qwen_language(
                character.get("language") or voice_cfg.language,
                supported_languages=supported_languages,
            )
            self.logger.info(
                "[qwen-tts][drama][voice_design] task_id=%s character=%s language=%s instruct_len=%s seed_len=%s",
                task_id,
                character_id,
                seed_language,
                len(seed_instruct),
                len(seed_text),
            )
            kwargs = self._build_common_generation_kwargs(voice_cfg)
            kwargs.update(
                text=seed_text,
                language=seed_language,
                instruct=seed_instruct,
            )
            wavs, sr = await asyncio.to_thread(model.generate_voice_design, **kwargs)
            ref_audio = self._extract_first_audio(wavs)
            sr = int(sr or sample_rate or 24000)
            if ref_audio.size == 0:
                raise RuntimeError(
                    f"qwen-tts drama voice_design 阶段对 character {character_id} 产出空音频"
                )
            seed_state[character_id] = {
                "seed_text": seed_text,
                "seed_audio": ref_audio,
                "seed_sr": sr,
                "language": seed_language,
            }
            sample_rate = int(sr or sample_rate)

        del model
        del bundle
        return seed_state, sample_rate

    async def _design_seeds_via_custom_voice(
        self,
        *,
        characters: List[tuple[str, Dict[str, Any]]],
        voice_cfg: VoiceConfig,
        task_id: str,
        weight_name: str,
        seed_state: Dict[str, Dict[str, Any]],
        sample_rate: int,
    ) -> tuple[Optional[Dict[str, Dict[str, Any]]], int]:
        """加载 CustomVoice 权重 → 给"custom_voice 组"每个 character 用预置 speaker 产 seed。

        与 voice_design 路径不同的是，这里不动用 ``instruct``：用户既然已经选了
        Vivian/Serena 等预置音色，就保留预置原貌，不再从 speaker meta 派生 instruct
        丢回 design 步骤。``character.instruct`` 仍允许作为 tone tweak 透传给底层
        ``generate_custom_voice``，但不是必须的。
        """
        bundle = await self._bundle_loader("tts", load_name=weight_name)
        caps = bundle.get("capabilities") or {}
        if not caps.get("custom_voice"):
            raise RuntimeError(
                f"qwen-tts drama custom_voice 阶段加载到的权重 {bundle.get('model_ref')!r} 不暴露 "
                f"custom_voice（caps={caps}）。请确保 qwen-tts 服务能解析 "
                f"{weight_name!r}；如果服务被 pin 到 Base/VoiceDesign，请去掉 fixed_model。"
            )
        if self._is_streaming_variant_bundle(bundle):
            raise RuntimeError(
                "qwen-tts drama 暂不支持 nano_vllm streaming 变体；"
                "请把 audio runtime.backend 切到 transformers 后再合成 drama。"
            )
        model = bundle["model"]
        supported_languages = bundle.get("supported_languages")
        supported_speakers = bundle.get("supported_speakers")
        sample_rate = int(bundle.get("sample_rate") or sample_rate or 24000)

        for character_id, character in characters:
            if self.check_cancelled and await self.check_cancelled(
                f"before drama custom_voice seed ({character_id})"
            ):
                model = None
                bundle = None
                return None, sample_rate
            speaker_input = str(character.get("speaker_name") or "").strip()
            if not speaker_input:
                raise ValueError(
                    f"character {character_id} 被分到 custom_voice 组但没有 speaker_name"
                )
            speaker = resolve_qwen_custom_speaker(
                speaker_input,
                supported_speakers=supported_speakers,
            )
            if not speaker:
                raise ValueError(
                    f"character {character_id} 的 speaker_name={speaker_input!r} 不在 qwen 预置 "
                    f"({sorted(supported_speakers or [])}) 内"
                )
            seed_text = self._seed_text_for_character(character)
            seed_language = normalize_qwen_language(
                character.get("language") or voice_cfg.language,
                supported_languages=supported_languages,
            )
            self.logger.info(
                "[qwen-tts][drama][custom_voice] task_id=%s character=%s speaker=%s language=%s seed_len=%s",
                task_id,
                character_id,
                speaker,
                seed_language,
                len(seed_text),
            )
            kwargs = self._build_common_generation_kwargs(voice_cfg)
            kwargs.update(
                text=seed_text,
                language=seed_language,
                speaker=speaker,
            )
            instruct = str(character.get("instruct") or "").strip()
            if instruct:
                kwargs["instruct"] = instruct
            wavs, sr = await asyncio.to_thread(model.generate_custom_voice, **kwargs)
            ref_audio = self._extract_first_audio(wavs)
            sr = int(sr or sample_rate or 24000)
            if ref_audio.size == 0:
                raise RuntimeError(
                    f"qwen-tts drama custom_voice 阶段对 character {character_id} 产出空音频"
                )
            seed_state[character_id] = {
                "seed_text": seed_text,
                "seed_audio": ref_audio,
                "seed_sr": sr,
                "language": seed_language,
            }
            sample_rate = int(sr or sample_rate)

        del model
        del bundle
        return seed_state, sample_rate

    @staticmethod
    def _classify_drama_character(character: Dict[str, Any]) -> str:
        """决定 character 走 ``voice_design`` 还是 ``custom_voice`` 链路。

        - 显式声明 ``voice_mode='voice_design'`` → ``voice_design``
        - 显式声明 ``voice_mode='custom_voice'`` → ``custom_voice``
        - 未声明 voice_mode：``instruct`` 非空 → ``voice_design``，
          否则 ``speaker_name`` 非空 → ``custom_voice``，
          都没有 → 空串（调用方据此报错）

        ``voice_mode='voice_design'`` 的角色稍后会被进一步要求必须有 ``instruct``；
        ``voice_mode='custom_voice'`` 的角色必须有可解析的 ``speaker_name``。
        """
        voice_mode = str(character.get("voice_mode") or "").strip().lower()
        if voice_mode == "voice_design":
            return "voice_design"
        if voice_mode == "custom_voice":
            return "custom_voice"
        if str(character.get("instruct") or "").strip():
            return "voice_design"
        if str(character.get("speaker_name") or "").strip():
            return "custom_voice"
        return ""

    # ---- drama helpers ----

    @staticmethod
    def _normalize_drama_characters(raw: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw, list):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "").strip()
            if not cid:
                continue
            out[cid] = dict(item)
        return out

    @staticmethod
    def _normalize_drama_dialogues(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            speaker_id = str(item.get("speaker_id") or "").strip()
            text = str(item.get("text") or "").strip()
            if speaker_id and text:
                next_item = dict(item)
                next_item["speaker_id"] = speaker_id
                next_item["text"] = text
                out.append(next_item)
        return out

    @staticmethod
    def _seed_text_for_character(character: Dict[str, Any]) -> str:
        seed_text = str(character.get("seed_text") or "").strip()
        if seed_text:
            return seed_text
        name = str(character.get("name") or character.get("id") or "这个角色").strip()
        instruct = str(character.get("instruct") or "").strip()
        if instruct:
            return f"我是{name}。我的声音设定是：{instruct}。这段声音将作为我的角色音色基准。"
        return f"我是{name}。这段声音将作为我的角色音色基准，请保持稳定、清晰、自然。"

    @classmethod
    def _build_dialogue_blocks(
        cls,
        dialogues: List[Dict[str, Any]],
        *,
        max_lines_per_block: int = 4,
        merge_pause_threshold_ms: int = 800,
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for line in dialogues:
            if not isinstance(line, dict):
                continue
            speaker_id = str(line.get("speaker_id") or "").strip()
            text = str(line.get("text") or "").strip()
            if not speaker_id or not text:
                continue
            if blocks and cls._can_merge_into_block(
                blocks[-1],
                line,
                max_lines_per_block=max_lines_per_block,
                merge_pause_threshold_ms=merge_pause_threshold_ms,
            ):
                blocks[-1]["lines"].append(line)
            else:
                blocks.append({"speaker_id": speaker_id, "lines": [line]})
        return blocks

    @classmethod
    def _can_merge_into_block(
        cls,
        block: Dict[str, Any],
        line: Dict[str, Any],
        *,
        max_lines_per_block: int,
        merge_pause_threshold_ms: int,
    ) -> bool:
        lines = block.get("lines")
        if not isinstance(lines, list) or len(lines) >= max_lines_per_block:
            return False
        if str(block.get("speaker_id") or "").strip() != str(line.get("speaker_id") or "").strip():
            return False
        previous = lines[-1] if lines else {}
        return cls._pause_after_ms(previous) < merge_pause_threshold_ms

    @classmethod
    def _block_pause_after_ms(cls, block: Dict[str, Any]) -> int:
        lines = block.get("lines")
        if not isinstance(lines, list) or not lines:
            return 450
        return cls._pause_after_ms(lines[-1])

    @staticmethod
    def _pause_after_ms(line: Dict[str, Any]) -> int:
        raw = line.get("pause_after_ms")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 450
        return max(0, min(3000, value))

    def _build_common_generation_kwargs(self, voice_cfg: VoiceConfig) -> Dict[str, Any]:
        # 复用 engine 端的同名实现，避免两份 generation 参数白名单脱钩。
        return self._engine._build_common_generation_kwargs(voice_cfg)

    def _extract_first_audio(self, wavs: Any) -> np.ndarray:
        return self._engine._extract_first_audio(wavs)

    def _collect_clone_outputs(self, wavs: Any, *, expected: int) -> List[np.ndarray]:
        """把 generate_voice_clone 的输出整理为 ``expected`` 条单声道 1D float32。

        - 单条输入时模型常返回单 array / 单 list，保持向后兼容；
        - batch 输入时返回 ``[ndarray, ndarray, ...]``，长度需与 ``expected`` 对齐；
          长度不一致会抛错，避免静默串音 / 漏段。
        """
        if expected <= 0:
            return []
        if expected == 1:
            audio = self._extract_first_audio(wavs)
            return [audio]

        if isinstance(wavs, (list, tuple)):
            audios = [normalize_audio_array(item) for item in wavs]
        else:
            raw = wavs.detach().cpu().float().numpy() if hasattr(wavs, "detach") else np.asarray(wavs)
            if raw.ndim == 2 and raw.shape[0] == expected:
                audios = [normalize_audio_array(raw[i]) for i in range(expected)]
            else:
                audios = [normalize_audio_array(raw)]

        audios = [audio for audio in audios if audio.size > 0]
        if len(audios) != expected:
            raise RuntimeError(
                f"qwen-tts drama clone batch returned {len(audios)} audio segments; expected {expected}"
            )
        return audios

    @staticmethod
    def _trim_silence(audio: np.ndarray, *, threshold: float = 0.008, pad: int = 960) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return arr
        idx = np.flatnonzero(np.abs(arr) > threshold)
        if idx.size == 0:
            return arr
        start = max(0, int(idx[0]) - pad)
        end = min(arr.size, int(idx[-1]) + pad)
        return arr[start:end]

    @staticmethod
    def _match_loudness(audio: np.ndarray, *, target_rms: float = 0.08) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return arr
        rms = float(np.sqrt(np.mean(np.square(arr))) + 1e-8)
        if rms <= 1e-6:
            return arr
        gain = min(4.0, max(0.25, target_rms / rms))
        return np.clip(arr * gain, -0.98, 0.98).astype(np.float32, copy=False)

    @staticmethod
    def _is_streaming_variant_bundle(bundle: Dict[str, Any]) -> bool:
        return bool(bundle.get("streaming_variant"))

    # =====================  通用 helpers  =====================

    async def _cancel_check(self) -> bool:
        if not self.check_cancelled:
            return False
        return await self.check_cancelled("streaming qwen-tts")

    async def _send_progress(self, task_id: str, progress: int, message: str) -> None:
        if not self.status_sender:
            return
        try:
            await self.status_sender(
                task_id,
                "processing",
                progress=progress,
                message=message,
            )
        except Exception:
            self.logger.debug("[qwen-tts] progress update failed task_id=%s", task_id, exc_info=True)

    async def _emit_event(
        self,
        *,
        task_id: str,
        event_type: str,
        sequence: int,
        sample_rate: int,
        is_final: bool,
        data: Optional[str] = None,
    ) -> None:
        # 流式 chunk 协议：mime 为 ``audio/pcm;rate=N``，``data`` 是 base64(裸 Int16 LE PCM)。
        # 早期版本曾发 ``audio/wav`` + 每段独立 WAV header，导致前端按段独立解码出现段间
        # gap（说着说着卡一下的"破音"症结）。协议文档 chat_ws_protocol.md 里 audio_delta
        # 的 BINARY 帧明确要求"裸 PCM16 LE"，这里和协议对齐。
        payload: Dict[str, Any] = {
            "type": event_type,
            "task_id": task_id,
            "service_id": self.service_id,
            "audio_mode": self.audio_mode,
            "mime_type": f"audio/pcm;rate={int(sample_rate)}",
            "sequence": sequence,
            "is_final": is_final,
            "sample_rate": sample_rate,
        }
        if data is not None:
            payload["data"] = data
        sent = await self.stream_sender(payload)
        if not sent:
            raise RuntimeError("stream transport disconnected, audio task aborted")


def _audio_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32), sample_rate, format="WAV")
    return buffer.getvalue()
