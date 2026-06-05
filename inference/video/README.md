## 视频推理器（`inference/video`）

### 当前实现（简述）

- **Wan2 系列（Wan2.2/变体）视频推理器**：按 `job_type` 路由到对应 handler 执行推理。
- 复用 `inference/common` 的 WS 框架：连接/重连、消息入队、任务消费、取消、结果回传。

### 启动方式

```bash
python inference/video/main.py <service_id>
```

依赖配置：

- `inference/config/<service_id>.yaml`：需包含 `service_type: "video"`
- `inference/config/inference.yaml`：**`models_dir` / `outputs_dir` / ws/api 等全局配置**

### 支持的 job_type（Wan2 handlers）

- **MKV**：视频生成（T2V/I2V/TI2V/VICV/IVV2V 自动分支）
- **S2V**：语音生视频（长视频支持增量回传：`status=processing` + 递增 `progress`，最终 `completed`，同一输出文件覆盖更新）
- **INP**：首尾帧视频补全（尾帧可选）
- **CCV**：镜头控制（`direction`/`speed`）

### TurboDiffusion 接入（MKV：T2V/I2V）

本项目已支持将 **TurboDiffusion** 作为 MKV 的“极速推理后端”（先接入 T2V/I2V 两类）。

- **开关**：使用既有字段 `InferenceRequestParams.fast_mode`
  - 当 `fast_mode=true` 且 `model_name` 以 `Turbo` 开头（或包含 `TurboWan`）时，MKV 将路由到 TurboDiffusion；
  - 其他情况仍走现有 Wan2 handler（不影响旧模型）。
- **fps**：当前 TurboDiffusion 分支先固定为 **16 fps**（后续跑通后再评估 15~30fps 的可行性与质量/兼容性）。
- **分辨率**：前端使用 `resolution + aspect_ratio` 计算得到 `width/height` 并提交；后端与推理侧会透传 `resolution/aspect_ratio` 以便 TurboDiffusion 复用其原生分辨率表。

#### 模型目录约定（自包含）

TurboDiffusion 与本项目保持一致：**一个 `model_name` 对应一个本地目录（或绝对路径），目录内包含全部组件，不共享任何组件**。

目录内文件命名支持“宽松匹配”：
- **VAE**：匹配 `*VAE*.pth`（或 `*vae*.pth`）
- **Text Encoder（umT5）**：优先使用量化 `.pth`（更省显存），并按系统能力自动选择：
  - 若支持 FP8：`models_t5_umt5-xxl-enc-fp8.pth` → `models_t5_umt5-xxl-enc-int8.pth` → `models_t5_umt5-xxl-enc-bf16.pth`
  - 若不支持 FP8：`models_t5_umt5-xxl-enc-int8.pth` → `models_t5_umt5-xxl-enc-bf16.pth`
  - 最后兜底：`*umt5*enc*.pth` / `*umt5*.pth` / `*t5*.pth`（以及打包为 `*.safetensors` 的情况）
- **T2V DiT**：目录内最大的 `.pth`（排除 VAE/umt5）
- **I2V DiT**：分别匹配文件名包含 `high` 与 `low` 的 `.pth`（同样排除 VAE/umt5）

#### 权重文件共享池（TurboDiffusion 与 Wan2 统一规则）

为减少重复（umt5/vae/tokenizer），组件查找按以下顺序做三段兜底：

1. `{models_dir}/{model_name}/...`（模型根目录，优先）
2. `{models_dir}/WanVideo/...`
3. `{weights_dir}/WanVideo/...`

若 VAE/umt5 仅提供 `*.safetensors`，会自动转换为 `*.pth` 并缓存到临时目录（`/tmp/vitoom_turbo_cache`），避免每次重复转换。

### 参数约定（关键）

- **只认 `model_name`**：用于本地模型目录名或绝对路径；不支持 `model_id`
- **纯离线本地加载**：所有组件均从 `model_name` 对应目录内查找并用 `ModelConfig(path=...)` 加载；缺文件直接报错，不触发下载
- **`models_dir` 不可通过命令行覆盖**：统一从 `inference/config/inference.yaml` 读取

### Pipeline 缓存与回收（Wan2 + TurboDiffusion 统一）

- **缓存实现**：统一使用 `common.pipeline_cache.PipelineCache`（**LRU=1 + TTL**），由 `VideoInferrer` 持有并管理；各 handler 通过 `acquire/release_use` 复用 pipeline/models。
- **TTL 配置**：`inference/config/inference.yaml: pipeline_cache_ttl_seconds`
  - `0`：关闭缓存（每次请求创建，结束即释放）
  - `>0`：启用缓存（空闲超过 TTL 自动驱逐）
- **切模型自动释放**：当缓存 key 变化（如 `model_name` / `low_vram(force_offload)` / TurboDiffusion 模型路径与关键参数变化）时，会**立即驱逐旧对象**并走强释放（显存 + 内存 best-effort）。

补充说明：
- **Wan2**：`runtime/wan2_pipeline_factory.py` 只负责“计算 key + 创建新 pipeline”（无全局缓存）；缓存/TTL/驱逐释放由 `VideoInferrer.pipeline_cache` 统一管理。
- **TurboDiffusion**：MKV handler 同样接入 `PipelineCache`，缓存对象为 TurboDiffusion 的 `TurboModels`；key 由 `engine.build_models_cache_key()` 生成（handler 中会再做 hash 缩短便于日志）。

### 低显存模式（low_vram / force_offload）

- **触发规则**：
  - `force_offload = bool(params.model_cfg.get("force_offload"))` 为 true 时：直接启用低显存模式推理
  - **当请求带 LoRA（`loras` 或 prompt `<lora:...>`）时**：为保证 LoRA 走可逆的 hotload（避免污染缓存权重），handler 会 **自动启用 low_vram/force_offload**
  - 常规推理若发生显存 OOM：执行一次 best-effort 清理后，**自动回退到低显存模式重试一次**
- **实现方式**：
  - 在 `runtime/wan2_pipeline_factory.py` 内为每个 `ModelConfig` 注入 `vram_config`（offload/onload/preparing/computation 的 device/dtype）；`low_vram/force_offload` 会纳入缓存 key，避免与常规 pipeline 混用缓存
  - **low_vram 与 FP8 T5**：启用 offload 时优先选用 `int8`/`bf16` 的 `models_t5_umt5-xxl-enc-*.pth`，避免 `fp8` 权重（`Float8Tensor`）在 `AutoWrappedLinear.cast_to` 中触发 `aten.empty_like` 未实现错误；若共享目录仅有 fp8，需在 `WanVideo/` 下补充 int8 或 bf16 版本
- **TurboDiffusion（MKV：T2V/I2V）**：
  - `model_cfg.force_offload=true` 时启用“降峰值”策略（会牺牲一定速度）；
  - `model_cfg.turbo_offload_level`：
    - `balanced`（默认）：UMT5/VAE 在关键阶段使用 GPU 计算，阶段结束立即 offload 回 CPU，以降低显存峰值但保持 GPU 利用率；
    - `max`：UMT5 常驻 CPU 且 VAE decode 走 CPU（最省显存，速度最慢）。
- **说明**：
  - S2V（长视频分段回传）为避免重复回传造成混乱：仅在“尚未回传任何片段”时允许 OOM 自动重试一次

### FPS（默认值与覆盖）

- **默认 fps**：统一为 **24**
- **覆盖方式**：可通过 `model_cfg.fps` 覆盖（范围 1~60）
- **帧数计算**：
  - MKV/INP/CCV：`num_frames = duration * fps + 1`
  - S2V：在上式基础上做 `4n+1` 约束（Wan2.2 S2V 推荐）

### LoRA（视频侧 / Wan2）

- **来源**（两种方式可同时使用，参数会覆盖 prompt 同名）：
  - prompt 内标签：`<lora:name:0.8>`
  - 请求参数 `loras`：支持 JSON 字符串 / list / dict（例如：`[{"name":"xxx.safetensors","weight":0.8,"target":"dit2"}]`）
- **加载策略**：
  - 使用 diffsynth `BasePipeline.load_lora(module, hotload=True)` 对 `pipe.dit/pipe.dit2` 进行热加载
  - **为避免污染 pipeline 缓存权重**：仅允许在 **low_vram（vram_management_enabled）** 下启用 LoRA；当请求带 LoRA 时，handler 会自动启用 `force_offload/low_vram`
  - 推理结束后会调用 `pipe.clear_lora(verbose=0)` 做 best-effort 清理

### CLI（不走 WS，直接调用推理）

```bash
python test/tools/run_video_infer_cli.py \
  --job_type MKV \
  --prompt "a cat" \
  --duration 5 --width 832 --height 480 \
  --model_name "Wan2.2-TI2V-5B-FP8"
```

常用示例：

```bash
# 低显存模式（force_offload）
python test/tools/run_video_infer_cli.py --job_type MKV --prompt "a cat" --duration 5 --width 832 --height 480 --force-offload

# 更高 FPS（默认 24；可显式指定）
python test/tools/run_video_infer_cli.py --job_type MKV --prompt "a cat" --duration 5 --width 832 --height 480 --fps 24

# 视频 LoRA（可多次）：name[:weight][@target]，target 可选 dit/dit2
python test/tools/run_video_infer_cli.py --job_type MKV --url "https://xxx/a.jpg" --prompt "" --duration 5 --width 832 --height 480 \
  --lora my_high_noise.safetensors:1@dit \
  --lora my_low_noise.safetensors:1@dit2
```

### 系统架构设计（高层）

- **入口**：WebSocket Server 推送 `type=task` → `common/ws_client.py` 接收
- **解耦**：任务写入 `common/message_queue.py`（`queue.Queue`），I/O 与推理解耦
- **消费**：`common/task_processor.py` 从队列取任务并调用 `VideoInferrer.inference_callback()`
- **推理线程**：`common/base_inferrer.py` 提供 `run_blocking()`，将重型推理放入单线程 `ThreadPoolExecutor`（避免阻塞 event loop）
- **结果处理**：`common/result_handler.py` 保存/上传输出并通过 WS 回传 `result`；`VideoInferrer` 负责发送最终 `completed/failed`

### 分支定义（签名说明）

### 四种业务类型参数对照表

说明：

- **通用输入字段**（四类都会用到的常见字段）：`model_name`、`duration`、`width`、`height`、`prompt`、`negative_prompt`、`seed`、`num_inference_steps`、`guidance_scale`、`model_cfg`（如 `fps`/`force_offload`）、`loras`
- **输出**：当前 Wan2 handlers 统一强制输出 `mp4`（`req.file_type = "mp4"`）

| job_type | 子模式（仅 MKV） | 必填字段 | 可选字段 | 备注 / 字段映射 |
|---|---|---|---|---|
| **MKV** | **TI2V** | `prompt` + `url` | `negative_prompt`、`seed`、`num_inference_steps`、`guidance_scale`、`model_cfg.fps`、`model_cfg.force_offload`、`loras` | 由 `Wan2MkvHandler.resolve_mode()` 自动判定；`url` 作为 `input_image` |
| **MKV** | **IVV2V（Animate）** | `url` + `ref_video` + `face_video` | `prompt`（允许为空）及通用可选字段 | `ref_video`→`animate_pose_video`，`face_video`→`animate_face_video` |
| **MKV** | **VICV（Control）** | `url` + `ref_video` | `prompt`（允许为空）及通用可选字段 | `ref_video`→`control_video`，`url`→`reference_image` |
| **MKV** | **T2V** | `prompt` | 通用可选字段 | 纯文生视频：不传 `url` |
| **MKV** | **I2V** | `url` | `prompt`（允许为空）及通用可选字段 | `url`→`input_image`；若模型支持可注入 `switch_DiT_boundary`（调用前会按 `pipe.__call__` 签名过滤） |
| **S2V** | - | `url` + `prompt_wav_path` | `prompt`（建议提供） 、`ref_video`（复用为 pose_video）及通用可选字段 | `prompt_wav_path` 下载为音频；`ref_video`→`s2v_pose_video`（可选） |
| **INP** | - | `url` | `image_file2`、`prompt`（允许为空）及通用可选字段 | `url`→`input_image`；`image_file2`→`end_image` |
| **CCV** | - | `url` + `direction` + `speed` | `prompt`（允许为空）及通用可选字段 | `direction`→`camera_control_direction`；`speed`→`camera_control_speed` |

#### job_type = MKV（Make Video）

自动路由（优先级从上到下）：

- **TI2V（图文生视频）**：`prompt` 非空 且 `url` 有值  
- **IVV2V（图+pose视频+face视频）**：`url` + `ref_video` + `face_video`
- **VICV（图+控制视频）**：`url` + `ref_video`（且 `face_video` 为空）
- **T2V（文生视频）**：`prompt` 非空 且 `url` 为空
- **I2V（图生视频）**：`url` 有值 且 `ref_video` 为空（`prompt` 允许为空）

#### job_type = S2V（Speech to Video）

- **必填**：`url`（参考图）、`prompt_wav_path`（音频）
- **可选**：`ref_video`（复用为 pose_video）
- **输出行为**：短视频一次回传；长视频分段生成并**增量回传**（覆盖同一 mp4，`status=processing` + 递增 `progress`，最终 `completed`）

#### job_type = INP（Inpainting / 首尾帧补全）

- **必填**：`url`（首帧）
- **可选**：`image_file2`（尾帧）、`prompt`（允许为空）

#### job_type = CCV（Camera Control Video）

- **必填**：`url`（参考图）、`direction`、`speed`
- **可选**：`prompt`（允许为空）

#### 通用约定

- **时长**：`duration`（秒）；内部按 `fps` 计算 `num_frames = duration * fps + 1`（`fps` 可由 `model_cfg.fps` 覆盖）
- **离线模型选择**：只使用 `model_name`（目录名或绝对路径），从 `inference.yaml: models_dir` 解析模型根目录

### 目录结构（关键文件）

```text
inference/video/
  main.py                     # 入口：读取 service_id，启动 VideoInferrer.run()
  inferrer.py                  # 推理入口：inference_callback()，按 job_type 分发到 handler
  README.md                    # 本文档
  handlers/
    video_handler.py           # handler 基类/占位（定义接口与通用约束）
    wan2/                      # Wan2 系列实现（当前已实现）
      mkv_handler.py           # MKV：T2V/I2V/TI2V/VICV/IVV2V 路由 + 推理 + 回传
      s2v_handler.py           # S2V：短/长视频（长视频增量回传）
      inp_handler.py           # INP：首尾帧补全
      ccv_handler.py           # CCV：镜头控制
      types.py                 # Wan2 类型/枚举（如 MKV mode）
  runtime/
    io_utils.py                # 下载 URL → 临时文件、加载图片等 I/O 工具
    wan2_pipeline_factory.py   # Wan2 pipeline 构建（纯离线：ModelConfig(path=...)；缓存由 VideoInferrer + PipelineCache 管理）
    wan2_lora_manager.py       # Wan2 LoRA 解析/加载/卸载（避免污染缓存权重）
```

```text
test/tools/
  run_video_infer_cli.py       # 直跑推理的 CLI（不走 WS），用于本地/服务器快速验证
```

### 附录：Wan2 系列模型分辨率支持汇总

#### 按分辨率分组（快速查询）

| 分辨率 | 版本 | 模型名称 | job_type | 模型类型 | 说明 |
|--------|------|----------|----------|----------|------|
| **720P** | Wan2.2 | T2V-A14B | MKV | Text-to-Video MoE | 支持 480P & 720P |
| **720P** | Wan2.2 | I2V-A14B | MKV | Image-to-Video MoE | 支持 480P & 720P |
| **720P** | Wan2.2 | TI2V-5B | MKV | High-compression VAE, T2V+I2V | 仅支持 720P |
| **720P** | Wan2.2 | S2V-14B | S2V | Speech-to-Video | 支持 480P & 720P |
| **720P** | Wan2.1 | T2V-14B | MKV | Text-to-Video | 支持 480P & 720P |
| **720P** | Wan2.1 | I2V-14B-720P | MKV | Image-to-Video | 仅支持 720P |
| **480P** | Wan2.2 | T2V-A14B | MKV | Text-to-Video MoE | 支持 480P & 720P |
| **480P** | Wan2.2 | I2V-A14B | MKV | Image-to-Video MoE | 支持 480P & 720P |
| **480P** | Wan2.2 | S2V-14B | S2V | Speech-to-Video | 支持 480P & 720P |
| **480P** | Wan2.1 | T2V-14B | MKV | Text-to-Video | 支持 480P & 720P |
| **480P** | Wan2.1 | I2V-14B-480P | MKV | Image-to-Video | 仅支持 480P |
| **480P** | Wan2.1 | T2V-1.3B | MKV | Text-to-Video | 仅支持 480P |

#### 快速参考（按分辨率）

**720P 支持的模型：**
- Wan2.2: `T2V-A14B` (MKV)、`I2V-A14B` (MKV)、`TI2V-5B` (MKV)、`S2V-14B` (S2V)
- Wan2.1: `T2V-14B` (MKV)、`I2V-14B-720P` (MKV)

**480P 支持的模型：**
- Wan2.2: `T2V-A14B` (MKV)、`I2V-A14B` (MKV)、`S2V-14B` (S2V)
- Wan2.1: `T2V-14B` (MKV)、`I2V-14B-480P` (MKV)、`T2V-1.3B` (MKV)

**注意：**
- `Animate-14B`（Wan2.2）的分辨率支持信息未明确说明，使用时请参考官方文档
- 部分模型同时支持 480P 和 720P，可根据需求选择

### 附录：Wan2 系列模型清单（按业务类型）

来源：扫描 `DiffSynth-Studio/examples/wanvideo/model_inference_low_vram/` 下所有 `Wan2.1*.py` / `Wan2.2*.py` 脚本得到的 `model_id`。

约定：

- **本项目选择模型用 `model_name`（本地目录名）**；通常可直接使用 `model_id` 的最后一段作为目录名（例如 `Wan-AI/Wan2.1-T2V-14B` → `Wan2.1-T2V-14B`）

#### job_type = MKV

- **T2V（文生视频）**
  - Wan2.2：`Wan2.2-T2V-A14B`、`Wan2.2-TI2V-5B`（TI2V 模型也可用于纯 T2V）、`Wan2.2-TI2V-5B-Unlimited`
  - Wan2.1：`Wan2.1-T2V-1.3B`、`Wan2.1-T2V-14B`
- **I2V（图生视频）**
  - Wan2.2：`Wan2.2-I2V-A14B`、`Wan2.2-TI2V-5B`（TI2V 模型也可用于纯 I2V）、`Wan2.2-TI2V-5B-Unlimited`
  - Wan2.1：`Wan2.1-I2V-14B-480P`、`Wan2.1-I2V-14B-720P`
- **TI2V（图文生视频）**
  - Wan2.2：`Wan2.2-TI2V-5B`、`Wan2.2-TI2V-5B-Unlimited`
  - Wan2.1：暂无（示例脚本目录中未发现对应模型）
- **VICV（图 + 控制视频 / Fun-Control）**
  - Wan2.2：`Wan2.2-Fun-A14B-Control`
  - Wan2.1：`Wan2.1-Fun-1.3B-Control`、`Wan2.1-Fun-14B-Control`、`Wan2.1-Fun-V1.1-1.3B-Control`、`Wan2.1-Fun-V1.1-14B-Control`
- **IVV2V（图 + pose视频 + face视频 / Animate）**
  - Wan2.2：`Wan2.2-Animate-14B`
  - Wan2.1：暂无（示例脚本目录中未发现对应模型）

#### job_type = S2V

- Wan2.2：`Wan2.2-S2V-14B`（示例脚本包含 `multi_clips` 变体，模型相同）
- Wan2.1：暂无（示例脚本目录中未发现对应模型）

#### job_type = INP

- Wan2.2：`Wan2.2-Fun-A14B-InP`
- Wan2.1：`Wan2.1-Fun-1.3B-InP`、`Wan2.1-Fun-14B-InP`、`Wan2.1-Fun-V1.1-1.3B-InP`、`Wan2.1-Fun-V1.1-14B-InP`

#### job_type = CCV

- Wan2.2：`Wan2.2-Fun-A14B-Control-Camera`
- Wan2.1：`Wan2.1-Fun-V1.1-1.3B-Control-Camera`、`Wan2.1-Fun-V1.1-14B-Control-Camera`

#### 备注：示例脚本里存在但本项目暂未接入的任务

- Wan2.2：`Wan2.2-VACE-Fun-A14B`（VACE）
- Wan2.1：`Wan2.1-FLF2V-14B-720P`（First-Last Frame to Video）
- Wan2.1：`Wan2.1-VACE-1.3B` / `Wan2.1-VACE-14B` / `VACE-Wan2.1-1.3B-Preview`（VACE）
- Wan2.1：`Wan2.1-1.3b-speedcontrol-v1`（SpeedControl 变体）
