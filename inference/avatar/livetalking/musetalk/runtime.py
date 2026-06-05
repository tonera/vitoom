"""``MuseTalkRuntime`` —— 替代上游 ``BaseAvatar`` + ``MuseReal``。

为什么不复用上游 ``BaseAvatar``：

1. ``BaseAvatar.__init__`` 强制 ``importlib.import_module('tts.edge')`` 等
   TTS plugin（vitoom 不需要内置 TTS）
2. ``BaseAvatar.__init__`` 强制 ``importlib.import_module('streamout.webrtc')``
   实例化 WebRTC HumanPlayer（vitoom sidecar 自己组 aiortc）
3. ``BaseAvatar`` 持有 ``self.asr = WhisperASR(opt, self, ...)``，``WhisperASR``
   反过来依赖 ``parent.custom_audiotype`` 等字段，循环依赖

本 runtime 只保留 MuseTalk 的本质：

    PCM → FeatureBuffer (whisper feat) → UNet diffusion → VAE decode →
    paste-back → BGR uint8 frame

线程模型（与上游 ``BaseAvatar.render()`` 等价但去掉 TTS/Output）：

* ``audio_thread``      —— 调 ``feature_buffer.step()`` 推进特征提取
* ``inference_thread``  —— 取 feat batch → ``inference_batch`` → 推 res 队列
* ``process_thread``    —— 取 res frame → ``paste_back_frame`` → 推 video 输出
* ``next_video_frame()``—— 外部（aiortc VideoStreamTrack）拉 BGR uint8 帧
* ``next_audio_frame()``—— 方案 A 音视频对齐时启用：aiortc AudioStreamTrack 拉
  16k mono float32 PCM（与 video 帧严格 1:N 对应，N=2，每条 320 samples=20ms）

所有线程在 ``stop()`` 时通过 ``threading.Event`` 优雅退出。

音视频对齐契约（方案 A）::

    set_audio_out_active(True)  # AudioTrack 启动时调
        ↓ _process_loop 每出一帧 video，紧接着把对应的 2 个 320-sample
          audio chunk push 到 _audio_out_queue（FIFO）
        ↓ AudioTrack.recv() 按 20ms wall-clock pacing 取一条 chunk
        ↓ aiortc 用同一 PeerConnection 的 RTP 时间戳让浏览器自动对齐 AV

video 帧与 audio chunk 的 1:2 比例由 ``_inference_loop`` 的固定切片
（``audio_frames[i*2:i*2+2]``）保证；本对齐契约的关键是：**每次 push video
都同时 push 对应的 audio**，不允许中途某次 skip。
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from common.logger import get_logger  # type: ignore[import-not-found]
from numpy.typing import NDArray

from .blending import get_image_blending
from .feature_buffer import (
    FRAME_TYPE_SILENCE,
    AudioFrame,
    FeatureBuffer,
    drain_queue as _drain_queue,
)
from .load import load_avatar, load_models, mirror_index

logger = get_logger(__name__)


class MuseTalkRuntimeError(RuntimeError):
    """模型加载 / 推理初始化失败。"""


class MuseTalkRuntime:
    """轻量 MuseTalk 推理内核（单 avatar / 单 session）。

    使用方式::

        runtime = MuseTalkRuntime(avatar_id="musetalk_avatar1", fps=25)
        runtime.start()
        # 持续喂 PCM
        runtime.push_pcm(pcm_float32)
        # 渲染线程在另一边按 fps 拉视频帧
        frame = runtime.next_video_frame(timeout=0.1)
        # 收尾
        runtime.stop()
    """

    def __init__(self, *, avatar_id: str, fps: int = 25, batch_size: int = 8):
        if fps != 25:
            # 上游 chunk_samples / feature_idx_multiplier 都按 25fps hardcode；
            # 强行改 fps 会跑偏特征对齐窗口，先严格只支持 25。
            raise MuseTalkRuntimeError(
                f"MuseTalk currently only supports fps=25 (got {fps})"
            )
        self.avatar_id = avatar_id
        self.fps = fps
        self.batch_size = batch_size

        # 模型 + avatar 资源（同步加载，启动时间较长）
        try:
            self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = load_models()
        except Exception as exc:  # noqa: BLE001
            raise MuseTalkRuntimeError(f"failed to load MuseTalk models: {exc}") from exc
        try:
            (
                self.frame_list_cycle,
                self.mask_list_cycle,
                self.coord_list_cycle,
                self.mask_coords_list_cycle,
                self.input_latent_list_cycle,
                self._idle_cycle_indices,
            ) = load_avatar(avatar_id)
        except Exception as exc:  # noqa: BLE001
            raise MuseTalkRuntimeError(
                f"failed to load avatar '{avatar_id}': {exc}"
            ) from exc

        # 流式特征提取
        self.feature_buffer = FeatureBuffer(
            self.audio_processor, fps=fps, batch_size=batch_size,
        )
        self.feature_buffer.warm_up()

        # 推理 → 渲染中间队列：与上游 res_frame_queue 对齐 batch_size*2
        self._res_frame_queue: "queue.Queue[Tuple[Optional[np.ndarray], List[AudioFrame], int]]" = queue.Queue(maxsize=batch_size * 2)
        # 视频输出队列：1 秒缓冲（fps 帧），超过 backpressure 让 process 线程 sleep
        self._video_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=fps)
        # 音频输出队列：方案 A 启用时由 _process_loop push 16k mono float32 PCM
        # chunk（每条 320 samples = 20ms）。1 video frame 对应 2 chunk → 容量
        # = fps * 2 * 1s ≈ 50 条，1 秒缓冲与 video_queue 对齐。
        # 默认 inactive：没有 AudioTrack 注册时不消耗 CPU，保持 D 方案行为。
        self._audio_out_active: bool = False
        self._audio_out_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=fps * 2)

        # 方案 A 的 AV 同步关键：让 audio_track / video_track 共享同一个
        # wall-clock 起点，否则 aiortc 的 audio sender / video sender
        # 各自第一次 recv() 时各自抓 time.time() → T0_a ≠ T0_v →
        # RTCP SR 把 audio/video 的 NTP 基准错开 → 浏览器 lip-sync 失败
        # （观测上是音频比视频快或慢几十~几百毫秒，用户主观感受"对不上"）
        # 我们用 _process_loop 第一次成功 push 一对 (video, audio) 的时刻
        # 作为 T0，audio_track + video_track 都从这个 T0 开始计 PTS。
        self._av_t0_wall: Optional[float] = None
        self._av_t0_event: threading.Event = threading.Event()

        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._speaking = False  # is_speaking() 调用方观测
        # 配置了 avator_info.json idle_cycle_frames 时，待机全图只轮播这些 0-based 下标
        self._idle_sub_index: int = 0

        logger.info(
            "MuseTalkRuntime ready avatar_id=%s fps=%d batch=%d frames=%d idle_subset=%s",
            avatar_id,
            fps,
            batch_size,
            len(self.frame_list_cycle),
            len(self._idle_cycle_indices)
            if self._idle_cycle_indices is not None
            else "full",
        )

    # ---------- public API ----------
    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._audio_loop, name="musetalk-audio", daemon=True),
            threading.Thread(target=self._inference_loop, name="musetalk-infer", daemon=True),
            threading.Thread(target=self._process_loop, name="musetalk-render", daemon=True),
        ]
        for t in self._threads:
            t.start()
        logger.info("MuseTalkRuntime started avatar_id=%s", self.avatar_id)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        logger.info("MuseTalkRuntime stopped avatar_id=%s", self.avatar_id)

    def push_pcm(self, pcm_float32: NDArray[np.float32]) -> int:
        """喂 16k mono float32 PCM；返回入队 chunk 数。线程安全。"""
        return self.feature_buffer.push_pcm(pcm_float32)

    def flush(self) -> None:
        """中断/段结束：清掉未消费的输入和已计算特征，但保留视频输出
        队列（让前端把已渲染好的最后几帧播完，体感上更自然）。

        D 方案语义：仅清 feature_buffer。aligned 方案要硬清音视频请用
        ``flush_av()``——否则会出现"声音断了但嘴还在动"。
        """
        self.feature_buffer.flush()
        self._idle_sub_index = 0

    def flush_av(self) -> None:
        """方案 A 中断专用：清掉 feature_buffer + 视频队列 + 音频队列。

        被 ``AvatarSession.interrupt`` 在 ``audio_out_active=True`` 时调用：
        如果只清音频不清视频，前端会出现"远端音频立刻停了，但 video track 还
        在播 200~500ms 旧帧"，对齐感崩坏；必须双队列同步 drain。
        """
        self.feature_buffer.flush()
        _drain_queue(self._video_queue)
        _drain_queue(self._audio_out_queue)
        self._idle_sub_index = 0

    def set_audio_out_active(self, active: bool) -> None:
        """AudioTrack init/stop 时调用。``False`` 时不再 push 音频，节省 CPU；
        切换瞬间清掉积压的音频帧避免下次 active 时灌出旧声。"""
        self._audio_out_active = bool(active)
        if not self._audio_out_active:
            _drain_queue(self._audio_out_queue)

    @property
    def audio_out_active(self) -> bool:
        return self._audio_out_active

    def is_speaking(self) -> bool:
        return self._speaking

    def next_video_frame(self, *, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """阻塞拉一帧 BGR uint8 视频帧；空队列 + 超时返回 ``None``。

        典型调用方是 aiortc ``VideoStreamTrack.recv()``，按 fps 节奏取帧。
        """
        try:
            return self._video_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def wait_av_t0(self, timeout: Optional[float] = None) -> Optional[float]:
        """阻塞等到 ``_process_loop`` 第一次 push 出一对 (video, audio) 的
        wall-clock 时刻；audio_track / video_track 用它做共享 PTS 起点。

        Returns:
            wall-clock float（``time.time()`` 时间基），timeout 时返回 None。

        线程安全：``threading.Event`` 多线程并发 wait 都能正确返回。即便
        track 启动时 T0 已经 set 过了，``Event.wait`` 也会立即返回 True。

        失败兜底（返回 None）由调用方决策——一般是 fallback 到自己当前
        ``time.time()``，行为退化为 "track 各自起点"，至少不会卡死。
        """
        if self._av_t0_event.wait(timeout=timeout):
            return self._av_t0_wall
        return None

    def reset_av_t0(self) -> None:
        """用于测试/重连：清掉共享 T0，让下一对 push 重新打基准。

        生产路径上 sidecar 只起一次 PC 就不再回收，调用 reset 不是必要的；
        但 PeerConnection 重新协商时（用户拔了又开数字人）需要清，否则两个
        新 track 会用旧 T0，PTS 全部偏移到 +N 秒。``avatar_session.cancel``
        / ``flush_av`` 路径会带上这个 reset。
        """
        self._av_t0_event.clear()
        self._av_t0_wall = None

    def next_audio_frame(self, *, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """阻塞拉一条 16k mono float32 PCM chunk（320 samples = 20ms）。

        方案 A 的 ``MuseTalkAudioTrack.recv()`` 调用入口。``_audio_out_active``
        为 False（D 方案 / 没有 AudioTrack 注册）时永远空 → 返回 ``None``，调用
        方需要用静音兜底。
        """
        try:
            return self._audio_out_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    # ---------- thread loops ----------
    def _audio_loop(self) -> None:
        """推进 FeatureBuffer 步进 + 简单的视频输出 backpressure。"""
        period = 1.0 / self.fps  # 一帧 40ms @ 25fps
        # 一个 step 出 batch_size 帧 → step 间隔 = batch_size * period
        step_interval = self.batch_size * period
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self.feature_buffer.step()
            except Exception as exc:  # noqa: BLE001
                logger.error("feature_buffer.step failed: %s", exc, exc_info=True)
                # avoid hot loop on persistent error
                self._stop_event.wait(0.5)
                continue
            # backpressure: 输出 buffer 多就 sleep 长一点
            qsize = self._video_queue.qsize()
            extra = 0.0 if qsize < 5 else min(0.5, period * (qsize - 5))
            elapsed = time.perf_counter() - t0
            sleep_for = max(0.0, step_interval - elapsed) + extra
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
        logger.debug("audio_loop exit")

    def _inference_loop(self) -> None:
        """从 FeatureBuffer 取 feat batch + audio frames → diffusion → res frame."""
        index = 0
        while not self._stop_event.is_set():
            try:
                audiofeat_batch = self.feature_buffer.get_feat_batch(timeout=0.5)
            except queue.Empty:
                continue

            # 同步取 batch_size*2 个 audio frame（speak/silence 标记用）
            audio_frames: List[AudioFrame] = []
            is_all_silence = True
            for _ in range(self.batch_size * 2):
                af = self.feature_buffer.get_audio_frame()
                if af.type != FRAME_TYPE_SILENCE:
                    is_all_silence = False
                audio_frames.append(af)

            if is_all_silence:
                # 静音段：不跑 diffusion，直接出 None 让 process_loop 用 full image
                idle_list = self._idle_cycle_indices
                if idle_list is not None:
                    ilen = len(idle_list)
                    for i in range(self.batch_size):
                        sub = mirror_index(ilen, self._idle_sub_index + i)
                        full_idx = idle_list[sub]
                        self._res_frame_queue.put((
                            None,
                            audio_frames[i * 2:i * 2 + 2],
                            full_idx,
                        ))
                    self._idle_sub_index += self.batch_size
                else:
                    length = len(self.frame_list_cycle)
                    for i in range(self.batch_size):
                        idx = mirror_index(length, index + i)
                        self._res_frame_queue.put((
                            None,
                            audio_frames[i * 2:i * 2 + 2],
                            idx,
                        ))
                index += self.batch_size
                continue

            try:
                pred_batch = self._inference_batch(index, audiofeat_batch)
            except Exception as exc:  # noqa: BLE001
                logger.error("inference_batch failed: %s", exc, exc_info=True)
                continue
            length = len(self.frame_list_cycle)
            for i, res_frame in enumerate(pred_batch):
                self._res_frame_queue.put((
                    res_frame,
                    audio_frames[i * 2:i * 2 + 2],
                    mirror_index(length, index),
                ))
                index += 1
        logger.debug("inference_loop exit")

    def _process_loop(self) -> None:
        """从 res_frame_queue 取 → paste-back → push 到 video 输出。

        Silence 段必须照常 push：``frame_list_cycle[idx]`` 的镜像循环索引
        给 avatar 提供"呼吸 / 眨眼 / 轻微姿态"的待机动画——不 push 视频会
        让 video element 卡在同一帧（用户主观感受是"数字人僵住了"）。
        Audio silence chunk 也照常 push，让 audio queue / video queue 的
        backlog 节奏一致；浏览器 jitter buffer 自己处理 silence。
        """
        # T0 = process_loop 启动瞬间。两个 track 第一次 recv() 立刻拿到
        # 共享 T0 不阻塞；之后 PTS 按 wall-clock 推进。这个改动跟 silence
        # push 行为无关，但它是 audio_track / video_track 共享 NTP 基准的
        # 必要前置（参见 audio_track.py 的 wait_av_t0 调用）。
        if self._av_t0_wall is None:
            self._av_t0_wall = time.time()
            self._av_t0_event.set()

        while not self._stop_event.is_set():
            try:
                res_frame, audio_frames, idx = self._res_frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            current_speaking = not all(af.type != 0 for af in audio_frames if af is not None)
            self._speaking = current_speaking

            if res_frame is None:
                # 静音段直接出 full image（avatar 待机动循环帧）
                target_frame = self.frame_list_cycle[idx]
                combine_frame = target_frame.copy()
            else:
                try:
                    combine_frame = self._paste_back_frame(res_frame, idx)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("paste_back_frame error idx=%d: %s", idx, exc)
                    continue

            # 跟上游 BaseAvatar 一样烙个小水印（左上角灰字），方便调试
            # 调试期间显式标记，正式上线可以删
            cv2.putText(
                combine_frame, "LiveTalking",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1,
            )

            # backpressure：满了就丢最早一帧避免无限堆积（更新比延迟重要）
            try:
                self._video_queue.put(combine_frame, block=False)
            except queue.Full:
                try:
                    self._video_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._video_queue.put_nowait(combine_frame)
                except queue.Full:
                    pass

            # 方案 A：紧跟 video frame push 对应的 audio chunks（每个 video
            # frame 严格对应 2 条 320-sample 音频，由 _inference_loop 切片保证）。
            # 没有 AudioTrack 注册时 _audio_out_active=False，直接跳过零开销。
            if self._audio_out_active:
                self._push_audio_chunks_for_frame(audio_frames)
        logger.debug("process_loop exit")

    def _push_audio_chunks_for_frame(self, audio_frames: List[Optional[AudioFrame]]) -> None:
        """把当前 video frame 对应的 2 条 16k mono PCM chunk push 到音频输出队列。

        backpressure 与 video 对称：满了 drop oldest 避免无限堆积，但不强行
        与 video drop 联动——AudioTrack 缺席场景已通过 ``audio_out_active=False``
        前置过滤；正常场景两个队列消费速率相同，自然保持 1:N 对齐。

        Defense in depth：方法内再次校验 ``_audio_out_active``，让本方法即便
        被外部直接调用也守住"未启用 → 不入队"的不变量。
        """
        if not self._audio_out_active:
            return
        chunk_samples = self.feature_buffer.chunk_samples  # 320 @ 25fps
        for af in audio_frames:
            if af is None:
                chunk = np.zeros(chunk_samples, dtype=np.float32)
            else:
                chunk = af.data
                if chunk.dtype != np.float32:
                    chunk = chunk.astype(np.float32, copy=False)
                # 极端情况下 chunk 长度异常 → 截/padding 到固定长度，
                # 否则 AudioTrack 端无法保持 20ms 步进 PTS。
                if chunk.size != chunk_samples:
                    if chunk.size < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - chunk.size))
                    else:
                        chunk = chunk[:chunk_samples]
            try:
                self._audio_out_queue.put(chunk, block=False)
            except queue.Full:
                try:
                    self._audio_out_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._audio_out_queue.put_nowait(chunk)
                except queue.Full:
                    pass

    # ---------- vendored from upstream MuseReal (unchanged) ----------
    @torch.no_grad()
    def _inference_batch(self, index: int, audiofeat_batch):
        length = len(self.input_latent_list_cycle)
        whisper_batch = np.stack(audiofeat_batch)
        latent_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            latent = self.input_latent_list_cycle[idx]
            latent_batch.append(latent)
        latent_batch = torch.cat(latent_batch, dim=0)

        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(
            device=self.unet.device, dtype=self.unet.model.dtype,
        )
        audio_feature_batch = self.pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

        pred_latents = self.unet.model(
            latent_batch, self.timesteps,
            encoder_hidden_states=audio_feature_batch,
        ).sample
        pred = self.vae.decode_latents(pred_latents)
        return pred

    def _paste_back_frame(self, pred_frame, idx: int):
        bbox = self.coord_list_cycle[idx]
        ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
        x1, y1, x2, y2 = bbox

        res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        mask = self.mask_list_cycle[idx]
        mask_crop_box = self.mask_coords_list_cycle[idx]

        return get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)


__all__ = ["MuseTalkRuntime", "MuseTalkRuntimeError"]
