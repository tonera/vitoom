# Audio Inference

`inference/audio` 提供独立的音频推理入口，启动方式与 `image` / `video` 保持一致：

```bash
python inference/audio/main.py <service_id>
```

VoxCPM 短名配置：`inference/config/voxcpm.yaml`，启动示例：`python inference/audio/main.py voxcpm`。

## 已接入模型

下表中的 `model_name` 指任务请求里实际传入的值，即数据库里的 `models.full_name`。

| `model_name` | 上游来源 | `model_class` | `audio_mode` | 当前能力 | 当前生效输入参数 | 输出 / 备注 |
|------|------|------|------|------|------|------|
| `VibeVoice-1.5B` | `microsoft/VibeVoice-1.5B` | `VibeVoice` | `tts` | 高质量长文本 TTS；支持参考音色；可选流式音频 chunk | 必填：`prompt`。可选：`prompt_wav_path`，或 `voice_preset` / `speaker_name`（从 demo voices 匹配参考音色），`stream` | 最终输出 `.wav`；`stream=true` 时额外通过 WS 发送 `audio_stream_chunk` |
| `VibeVoice-Realtime-0.5B` | `microsoft/VibeVoice-Realtime-0.5B` | `VibeVoice` | `realtime_tts` | 低延迟流式 TTS；基于缓存好的 prompt `.pt` 做实时合成 | 必填：`prompt`、`stream=true`。可选：`voice_preset` / `speaker_name` / `prompt_wav_path`（用于定位 `.pt` prompt cache），`response_format` | 必定走流式；`response_format=audio_file|both` 时结束后落最终 `.wav` |
| `VibeVoice-ASR` | `microsoft/VibeVoice-ASR` | `VibeVoice` | `asr` | 语音转文本；支持流式文本增量与结构化转写片段 | 必填：`input_audio_url` 或 `prompt_wav_path`。可选：`prompt_text`（作为转写上下文提示）、`stream` | 最终输出 `.txt`；`stream=true` 时额外发送 `text_stream_delta` / `transcript_segment` |
| `VoxCPM2` | `openbmb/VoxCPM2` | `Voxcpm` | `tts` | 普通 TTS、参考音频克隆、带 `prompt_text` 的高保真克隆 | 必填：`prompt`。可选：`prompt_wav_path`、`prompt_text`、`guidance_scale`、`num_inference_steps`、`stream` | 无 `prompt_wav_path` 时走普通 `text -> audio`；有 `prompt_wav_path` 且无 `prompt_text` 时映射为 `reference_wav_path`；有两者时映射为 `prompt_wav_path + prompt_text + reference_wav_path` |
| `VoxCPM2` | `openbmb/VoxCPM2` | `Voxcpm` | `realtime_tts` | 流式语音合成；当前与 `tts + stream=true` 共用同一推理实现 | 必填：`prompt`、`stream=true`。可选：`prompt_wav_path`、`prompt_text`、`guidance_scale`、`num_inference_steps`、`response_format` | 若 `response_format=audio_file|both` 会落最终 `.wav`；当前不支持 `asr` |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `Qwen-tts` | `tts` | 预置 speaker 的多语言 TTS；支持自然语言风格指令 | 必填：`prompt`。可选：`speaker_name` / `voice_preset`、`language`、`instruct`（未传时兼容回退到 `prompt_text`）、`stream` | 最终输出 `.wav`；`stream=true` 时按生成后的完整音频切片发送 `audio_stream_chunk`；当前不支持 `asr` / `realtime_tts` |
| `Qwen3-ASR-1.7B` / `Qwen3-ASR-0.6B` | `Qwen/Qwen3-ASR-1.7B` / `Qwen/Qwen3-ASR-0.6B` | `Qwen-asr` | `asr` | 多语言语音转文本；支持语言识别；可选 forced aligner 时间戳 | 必填：`input_audio_url` 或 `prompt_wav_path`。可选：`language`、`prompt_text`（映射为 `context`）、`timestamps`、`stream` | 最终输出 `.txt`；`timestamps=true` 时默认只从本地加载 `Qwen3-ForcedAligner-0.6B` 生成 `transcript_segment`；`stream=true` 当前为整段结果透传，不是真流式 |

## 任务模式

- `audio_mode=tts`: 文本转语音，最终输出音频文件；若 `stream=true`，额外透传音频 chunk
- `audio_mode=asr`: 音频转文本，最终输出 `.txt` 文本文件；若 `stream=true`，额外透传文本增量与结构化片段
- `audio_mode=realtime_tts`: 低延迟语音合成，必须 `stream=true`；若 `response_format=audio_file|both`，结束后落最终 `.wav`

## 流式约定

- 流式消息只通过 WebSocket 转发，不落库、不落文件
- 推理过程中若流式消息发送失败，当前音频任务会立即终止

## 模型选择约定

- `/v1/tasks` 收到任务后统一按 `model_name == models.full_name` 查库，回填完整模型信息后再透传给音频推理器
- `audio` 推理器内部**不再查库**
- `runtime` 只根据 `model_class` 选择：
  - `model_class=VibeVoice` -> `vibevoice`
  - `model_class=Voxcpm` -> `voxcpm`
  - `model_class=Qwen-tts` -> `qwen_tts`
  - `model_class=Qwen-asr` -> `qwen_asr`
- `local_path` 仅保留为模型管理元数据，不参与任务选模与推理路径解析
- 音频模型目录只按以下顺序解析：
  - `{models_dir}/{model_name}`
  - `{weights_dir}/{model_name}`

## VibeVoice 代码来源

运行时直接使用 vendored 副本：

- `inference/third_party/vibevoice`
- `inference/third_party/vibevoice_demo/voices`

这些目录来自上游 `vibevoice-community`，通过 `cp` 复制到仓库内后再做本地集成。

## 当前实现说明

- `tts` 优先使用 `prompt_wav_path` 作为参考音色；若未提供，会尝试从 `voice_preset` / `speaker_name` 匹配 demo 语音
- `realtime_tts` 使用 `demo/voices/streaming_model/*.pt` 作为预置音色缓存
- `asr` 最终结果统一写入 `.txt` 文件，继续复用既有 `result/files` 链路
- `VoxCPM2` 当前支持 `tts` / `realtime_tts`，不支持 `asr`
- `Qwen-tts` 当前只接入 `CustomVoice` 路径，对应 `Qwen3TTSModel.generate_custom_voice(...)`
- `Qwen-tts` 使用 `speaker_name` / `voice_preset` 选择 speaker，默认回退到 `Vivian`；`language` 未传时默认按 `Auto` 处理
- `Qwen-tts` 的 `instruct` 为首选风格控制字段；为兼容现有任务协议，若 `instruct` 为空会回退读取 `prompt_text`
- `Qwen-tts` 的 `stream=true` 当前是“协议兼容流式”：先完整生成，再按固定 chunk 切片通过 WS 发送；不是边推理边出包
- `Qwen-tts` 依赖 `qwen-tts` Python 包；`inference/requirements.txt` 与 `inference/audio/requirements.txt` 均需包含该依赖
- `Qwen-asr` 当前只接入 transformers backend，对应 `Qwen3ASRModel.from_pretrained(...).transcribe(...)`
- `Qwen-asr` 当前只支持 `audio_mode=asr`；不支持 `tts` / `realtime_tts`
- `Qwen-asr` 的 `language` 支持传常见语言代码（如 `en` / `zh` / `ja`），运行时会归一化为官方 API 期望的语言名；未传时走自动语种识别
- `Qwen-asr` 的 `prompt_text` 映射为 `transcribe(..., context=...)`
- `Qwen-asr` 的 `timestamps=true` 默认只会查找本地 `{models_dir|weights_dir}/Qwen3-ForcedAligner-0.6B`
- 若本地不存在 forced aligner，会直接报清晰错误，不再默认回退到 Hugging Face 下载
- 若确实希望远端下载 forced aligner，可显式传 `model_cfg.runtime.forced_aligner=Qwen/Qwen3-ForcedAligner-0.6B`，并保持 `allow_remote_assets=true`
- `Qwen-asr` 当前不支持说话人分离；`speaker_diarization` 会被忽略
- `Qwen-asr` 的 `stream=true` 当前同样是“协议兼容流式”：识别完成后发送整段 `text_stream_delta`，若有时间戳再补发 `transcript_segment`
- `Qwen-asr` 依赖 `qwen-asr` Python 包；`inference/requirements.txt` 与 `inference/audio/requirements.txt` 均需包含该依赖
- 表格中的“当前生效输入参数”只列运行时实际消费的字段；请求模型里保留的一些兼容字段不代表当前 handler 一定会使用
- `VoxCPM2` 严格按模型卡示例调用：
  - 无 `prompt_wav_path`：普通 `text -> audio`
  - 有 `prompt_wav_path` 且无 `prompt_text`：映射为 `reference_wav_path`
  - 有 `prompt_wav_path` 且有 `prompt_text`：映射为 `prompt_wav_path + prompt_text + reference_wav_path`
- `VoxCPM2` 依赖固定为 `voxcpm==2.0.2`，与当前官方文档/模型卡保持一致
- 由于音频服务在线程中执行推理，`VoxCPM2` 运行时固定使用 `optimize=False`，避免 `torch.compile`/CUDA Graphs 与多线程冲突
- **VoxCPM 推理后端**（`config.runtime.backend`）：`transformers` 走官方 `voxcpm` 包；`nano_vllm_voxcpm` 走 [Nano-vLLM-VoxCPM](https://github.com/a710128/nanovllm-voxcpm)（需 CUDA，与 Qwen 的 `nano_vllm` 不是同一依赖）。可选依赖见 `inference/audio/requirement-voxcpm-nano.txt`。Nano 路径下 `inference_timesteps` 等在**加载模型时**由 `runtime` 字典传入，任务里的 `num_inference_steps` 不会热切换引擎配置。


参考链接
https://huggingface.co/microsoft/VibeVoice-1.5B
https://huggingface.co/openbmb/VoxCPM2
https://huggingface.co/k2-fsa/OmniVoice
https://huggingface.co/hexgrad/Kokoro-82M
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
https://huggingface.co/Qwen/Qwen3-ASR-1.7B