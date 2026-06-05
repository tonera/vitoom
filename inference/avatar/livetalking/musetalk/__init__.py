"""vitoom 内嵌的 MuseTalk 推理内核。

设计目标（区别于上游 ``LiveTalking`` 仓库）：

* **不依赖 LiveTalking 的 BaseAvatar / TTS / Output 模块**：上游 ``BaseAvatar``
  是一个 "完整应用" 抽象，强制持有 TTS plugin + WebRTC HumanPlayer，不适合
  当库类用。本内核只保留 MuseTalk 的 **模型加载 / 音频特征 / Diffusion 推理 /
  paste-back 合成** 这几块，不涉及任何 TTS / 输出协议。
* **接口契约**：上层 sidecar 通过 ``MuseTalkRuntime.put_pcm()`` 喂 16k mono
  float32 PCM；通过 ``MuseTalkRuntime.next_video_frame()`` / 视频生成器拿 BGR
  uint8 帧。其它一切（WebRTC、aiortc、协议）由 sidecar 的 ``server.py`` 处理。
* **代码来源**：``vae.py`` / ``unet.py`` / ``audio2feature.py`` 都是直接 vendor
  自上游 LiveTalking（``avatars/musetalk/``），仅去掉 ``__main__`` / 训练相关
  代码。``inference/third_party/livetalking/`` 仅用于 vendor 时的对照参考，
  端到端验证完即可整目录删除。

文件清单：

* ``paths.py`` —— resources/models/livetalking 集中路径常量
* ``blending.py`` —— vendored ``get_image_blending``
* ``vae.py`` / ``unet.py`` —— vendored MuseTalk 模型 wrapper
* ``audio2feature.py`` —— vendored Whisper feature extractor
* ``load.py`` —— ``load_all_model`` + ``load_avatar`` 集中入口
* ``feature_buffer.py`` —— 替代上游 ``WhisperASR`` 的流式特征 buffer
* ``runtime.py`` —— ``MuseTalkRuntime``（替代 ``MuseReal`` + ``BaseAvatar``）

注意：本 ``__init__`` 故意不在顶层 import ``runtime``。``runtime`` 拉
torch / diffusers 等重依赖，dev 环境（如 macOS 没装 torch）单 import 本
package 就会崩。需要 ``MuseTalkRuntime`` 的地方显式
``from .runtime import MuseTalkRuntime``。
"""
