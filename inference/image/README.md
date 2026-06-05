# 图片推理器文档（现行实现）

## 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [目录结构](#目录结构)
- [核心模块](#核心模块)
- [业务流程与 job_type 说明](#业务流程与-job_type-说明核心)
- [完整推理流程](#完整推理流程)
- [使用方式](#使用方式)
- [配置说明](#配置说明)
- [扩展指南](#扩展指南)
- [API参考](#api参考)
- [常见问题](#常见问题)
- [相关文档](#相关文档)
- [更新日志](#更新日志)

## 概述

图片推理器（Image Inferrer）是一个基于 `BaseInferrer` 的图片推理服务，支持多种 diffusers 模型（SD15/SDXL/Flux/Flux2/Qwen/ZImage…）以及多种“无 pipeline”的图片处理任务（去背景/超分/换脸）。

> 注意：本文件描述的是当前代码（`image/inferrer.py` + `handlers/*` + `runtime/*`）的**现行实现**。早期的 `pipeline_router.py` / `pipelines/*` 方案已不再使用。

### 主要特性（现行）

- ✅ **Handler 模式**：按 `job_type` 分发到对应的 Handler（DiffusionHandler/IdHandler/PoseHandler/RbgSrFsHandler）
- ✅ **多模型支持**：通过 `ModelCatalog(model_families) + PipelineDetector` 自动选择 diffusers pipeline（含 img2img 自动切换/回退；Pipeline 类为延迟解析）
- ✅ **多业务支持**：MK/ED/SED/RBG/SR/FS/ID/POSE
- ✅ **逐张回传**：每生成/处理一张图片立即通过 WS 回传（`ResultHandler.process_single_result`）
- ✅ **Pipeline 缓存**：LRU=1 + TTL 的 Pipeline 缓存，避免重复加载权重
- ✅ **fast_mode 加速**：sd15/sdxl/flux 启用 fbcache（nunchaku adapter）
- ✅ **显存策略**：优先 `pipe.to(device)`，OOM 时自动回退到 cpu offload
- ✅ **任务取消**：支持任务取消检查，在多个阶段检查并中止推理
- ✅ **模型版本解析**：自动解析 `model_class` → `model_version`（canonical），支持自动侦测
- ✅ **后处理链**：upscale/face_enhance + remove_bg（最后执行，强制 png）

## 架构设计

图片推理器采用"编排器 + Handler + 运行时组件"的分层结构：

```
┌─────────────────────────────────────────┐
│         ImageInferrer (主类)            │
│  继承 BaseInferrer，实现推理流程编排      │
│  - 参数预处理                            │
│  - 模型版本解析                           │
│  - Handler 分发                          │
│  - 任务生命周期管理                        │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┬───────────────────────────────┐
       │                │                               │
┌──────▼──────────┐  ┌─▼──────────────┐      ┌────────▼─────────┐
│   Handlers      │  │ 运行时组件       │      │ ResultHandler     │
│                 │  │                 │      │ (逐张存储&回传)    │
│ - Diffusion     │  │ - PipelineLifecycle│   └───────────────────┘
│ - IdHandler     │  │ - DevicePlanner │
│ - PoseHandler   │  │ - ModelLocator  │
│ - RbgSrFsHandler│  │ - PipelineCache │
└──────┬──────────┘  │ - SeedManager   │
       │             └──────────────────┘
       │
┌──────▼──────────────────────────────────┐
│  PipelineDetector + diffusers Pipeline  │
│  (text2img/img2img/edit)                │
└──────────────────────────────────────────┘
```

### 核心组件（现行）

1. **`image/inferrer.py`**：主编排器
   - 参数预处理（如 RBG 强制 png）
   - 模型版本解析与归一化（`model_class` → `model_version`）
   - 根据 `job_type` 分发到对应的 Handler
   - 任务生命周期管理（取消检查、状态上报、缓存清理）

2. **`image/handlers/`**：任务处理器（按 job_type 分发）
   - **`diffusion_handler.py`**：处理扩散类任务（MK/ED/SED），使用 diffusers pipeline
   - **`id_handler.py`**：处理 ID 任务（PuLID identity），使用 PuLIDFluxPipeline
   - **`pose_handler.py`**：处理 POSE 任务（姿态控制）
   - **`rbg_sr_fs_handler.py`**：处理无 pipeline 任务（RBG/SR/FS）

3. **`common/model_catalog/*` + `common/model_families/*`**：模型家族单一事实源（SSOT）
   - family aliases / model_index 规则 / PipelineRef（延迟解析 diffusers pipeline 类）

4. **`common/pipeline_detector.py`**：选择 diffusers pipeline 类，并构建 pipeline 参数（含 nunchaku 量化模块注入）

5. **`image/runtime/params_preprocessor.py`**：推理前参数预处理（扁平入口）
   - url 回退 / model_version 归一化 / prompt&LoRA / RBG png / 尺寸约束等

6. **`image/runtime/pipeline_service.py`**：Pipeline 装配门面（创建/缓存 + LoRA load + inference_kwargs build）

7. **`image/runtime/pipeline_lifecycle.py`**：Pipeline 生命周期管理
   - 复用 `PipelineService` 获取 pipeline + base inference params
   - 设备迁移与 OOM 回退（含 offload）
   - 释放与缓存使用权归还

8. **`image/runtime/device_planner.py`**：规划 device/dtype

9. **`image/runtime/model_locator.py`**：模型路径定位

10. **`inference/common/pipeline_cache.py`**：Pipeline 缓存（LRU=1，TTL 控制）

11. **`image/runtime/seed_manager.py`**：种子管理

12. **`image/inference_params_builder.py`**：把 `InferenceRequestParams` 映射成 diffusers 调用参数

13. **`common/result_handler.py`**：逐张保存/上传并发送 WS result

14. **`image/runtime/postprocess_pipeline.py`**：后处理链（upscale/face_enhance/remove_bg）

## 目录结构（现行）

```
inference/image/
├── __init__.py                    # 模块导出
├── inferrer.py                   # 图片推理器主类（编排器）
├── main.py                        # 主程序入口
├── inference_params_builder.py    # diffusers 调用参数构建
├── inference_param_specs.py       # 参数规范定义
├── inference_param_utils.py       # 参数工具函数
├── handlers/                      # 任务处理器（按 job_type 分发）
│   ├── __init__.py
│   ├── diffusion_handler.py      # 扩散类任务（MK/ED/SED）
│   ├── id_handler.py             # ID 任务（PuLID）
│   ├── pose_handler.py           # POSE 任务
│   ├── pose_backends.py          # POSE 后端实现
│   ├── pose_assets.py             # POSE 资源管理
│   └── rbg_sr_fs_handler.py      # RBG/SR/FS 任务
├── runtime/                       # 运行时组件
│   ├── __init__.py
│   ├── device_planner.py         # 设备规划
│   ├── model_locator.py          # 模型定位
│   # (moved) inference/common/pipeline_cache.py  # Pipeline 缓存
│   ├── params_preprocessor.py    # 推理前参数预处理（扁平入口）
│   ├── pipeline_service.py       # Pipeline 装配门面（创建/缓存/LoRA/kwargs）
│   ├── pipeline_lifecycle.py     # Pipeline 生命周期管理
│   ├── seed_manager.py           # 种子管理
│   ├── scheduler_loader.py       # Scheduler 加载
│   ├── lora_manager.py           # LoRA 管理
│   ├── prompt_utils.py            # Prompt 工具
│   ├── postprocess_pipeline.py   # 后处理链
│   ├── vae_dtype_fixer.py        # VAE dtype 修复
│   ├── controlnet_image_builder.py # ControlNet 图像构建
│   ├── pulid_flux_pipeline_with_paths.py # PuLID Flux Pipeline
│   └── briarmbg.py               # BriaRMBG 实现
└── controlnet/                   # ControlNet 相关
    ├── controlnet_union.py
    ├── pipeline_controlnet_union_sd_xl.py
    └── eva_clip/                  # EVA-CLIP 实现
```

## 核心模块

### 1. ImageInferrer（主推理器类）

**文件**：`inferrer.py`

**职责**：
- 继承 `BaseInferrer`，实现图片推理的完整流程编排
- 参数预处理与模型版本解析
- 根据 `job_type` 分发到对应的 Handler
- 任务生命周期管理（取消检查、状态上报、缓存清理）

**主要方法**：

```python
class ImageInferrer(BaseInferrer):
    def __init__(self, service_id: str):
        # 初始化组件：detector, device_planner, model_locator, 
        # seed_manager, pipeline_cache 等
    
    async def initialize(self):
        # 初始化推理器（加载配置、初始化组件）
    
    async def cleanup(self):
        # 清理资源（信号处理器回调）
    
    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        """
        推理回调函数（任务入口）
        
        流程：
        1. 检查任务是否已取消
        2. 进入任务上下文（_task_context）
        3. 推理前参数预处理（`runtime/params_preprocessor.py`：包含 RBG png、model_version 归一化、prompt/LoRA 等）
        4. 根据 job_type 分发到对应的 Handler
        5. Handler 执行推理并处理结果
        6. 更新任务状态
        """
```

### 2. Handler 处理器（按 job_type 分发）

#### 2.1 DiffusionHandler（扩散类任务）

**文件**：`handlers/diffusion_handler.py`

**职责**：
- 处理需要 diffusers pipeline 的任务（MK/ED/SED）
- 管理 Pipeline 生命周期
- 驱动迭代生成
- 应用后处理链

**主要方法**：

```python
class DiffusionHandler:
    async def run(self, params: InferenceRequestParams, *, task_id: str):
        # 1. 通过 PipelineLifecycle 创建 pipeline
        # 2. 配置 scheduler 和 fast_mode
        # 3. 迁移到目标设备
        # 4. 构建迭代规格（IterationSpec）
        # 5. 驱动迭代生成
        # 6. 逐张应用后处理并回传结果
```

#### 2.2 IdHandler（ID 任务）

**文件**：`handlers/id_handler.py`

**职责**：
- 处理 PuLID identity 任务
- 使用 PuLIDFluxPipeline
- 复用 PipelineCache 避免重复加载权重

**主要方法**：

```python
class IdHandler:
    async def run(self, params: InferenceRequestParams, *, task_id: str):
        # 1. 解析 PuLID 资源路径
        # 2. 从缓存获取或创建 PuLIDFluxPipeline
        # 3. 加载 LoRA（如需要）
        # 4. 执行推理
        # 5. 应用后处理并回传结果
```

#### 2.3 PoseHandler（POSE 任务）

**文件**：`handlers/pose_handler.py`

**职责**：
- 处理姿态控制任务
- 支持多种姿态后端（OpenPose、MediaPipe 等）

#### 2.4 RbgSrFsHandler（无 pipeline 任务）

**文件**：`handlers/rbg_sr_fs_handler.py`

**职责**：
- 处理不需要 diffusers pipeline 的任务
- RBG：去背景（使用 BriaRMBG）
- SR：超分/人脸增强（使用 RealESRGAN/GFPGAN）
- FS：换脸（使用 InsightFace/InSwapper）

**主要方法**：

```python
class RbgSrFsHandler:
    async def run(self, params: InferenceRequestParams):
        # 根据 job_type 分发：
        # - RBG: 逐张去背景，强制输出 png
        # - SR: 逐张超分/人脸增强
        # - FS: 换脸（url=源脸，tpl_list=目标列表）
```

### 3. PipelineLifecycle（Pipeline 生命周期管理）

**文件**：`runtime/pipeline_lifecycle.py`

**职责**：
- Pipeline 创建与缓存管理
- 设备迁移与 OOM 回退
- LoRA 加载/卸载
- Scheduler 配置
- Fast mode（fbcache）应用

**主要方法**：

```python
class PipelineLifecycle:
    async def create_pipeline(self, params: InferenceRequestParams):
        # 创建或从缓存获取 pipeline
        # 返回 (pipe, inference_params, device_plan)
    
    def apply_fast_mode_cache(self, pipe, params):
        # 应用 fast_mode（fbcache）
    
    def move_to_device(self, pipe, device_plan, params):
        # 迁移到目标设备，OOM 时回退到 cpu offload
    
    async def release_pipeline_twice_async(self, pipe, ...):
        # 二次释放 pipeline（确保显存释放）
```

### 4. PipelineCache（Pipeline 缓存）

**文件**：`inference/common/pipeline_cache.py`

**职责**：
- LRU=1 的 Pipeline 缓存
- TTL 控制（由 `inference.yaml` 配置）
- 缓存驱逐与资源释放

### 5. DevicePlanner（设备规划）

**文件**：`runtime/device_planner.py`

**职责**：
- 规划 device（cuda/cpu）
- 规划 dtype（float16/bfloat16/float32）
- 根据模型和配置选择最优设备策略

## 业务流程与 job_type 说明（核心）

### 1) MK（文生图/图生图）
- **文生图**：默认走 text2img pipeline（prompt）
- **图生图**：当 `url` 非空且图片可成功加载时，自动切换到对应 img2img pipeline，并注入 `image + strength`
- **回退策略**：`url` 不可加载时会清空 `url` 回退为文生图（避免选错 img2img pipeline）

### 2) ED（图片编辑-生成）
- 输入来自 `tpl_list`：会加载为 `List[PIL.Image]` 并以 `image=<list>` 传给 edit pipeline（例如 Qwen/Flux/Flux2 的 edit 类 pipeline）

### 3) SED（图片编辑-批量处理）
- 输入来自 `tpl_list`：逐张加载、逐张调用 pipeline、逐张回传
- `total` 固定为 `len(tpl_list)`（不会随跳过/失败减少）

### 4) RBG（去背景）
- 不加载 diffusers pipeline
- 输入来自 `tpl_list`：每张图走 `BriaRMBG` 推理并输出 RGBA
- 输出格式强制为 png（alpha）

### 5) SR（超分/人脸增强）
- 不加载 diffusers pipeline
- 输入来自 `tpl_list`
- `upscale` 仅当值为 2/4 才执行超分；`face_enhance=True` 时启用人脸增强（默认 GFPGAN，可通过环境变量切换为 CodeFormer）

### 6) FS（换脸）
- 不加载 diffusers pipeline
- **输入约定**：`url=源脸`，`tpl_list=目标人物列表`
- 对所有目标图执行换脸并逐张回传

### 7) 生成后的通用后处理（MK/ED/SED）
- `upscale`：仅 2/4 生效（0/1 表示不超分）
- `face_enhance`：启用人脸增强（默认 GFPGAN，可切换 CodeFormer；可与 upscale 组合）
- `remove_bg`：必须最后执行，并强制输出 png

> 注意：以上"job_type 流程"为现行实际流程，通过 Handler 模式实现。

## 完整推理流程

```
1. 任务接收
   ↓ WebSocket接收任务消息
   ↓ inference_callback(params) 被调用
2. 取消检查
   ↓ _check_cancelled() - 检查任务是否已取消
3. 推理前参数预处理（扁平入口）
   ↓ runtime/params_preprocessor.py::preprocess_inference_params()
   ↓ - url 回退（避免误选 img2img pipeline）
   ↓ - model_version 归一化（model_class/model_version(raw) → canonical；必要时触发 PipelineDetector 自动侦测）
   ↓ - RBG 强制 png、prompt/LoRA/尺寸约束等
5. Handler 分发（根据 job_type）
   ↓
   ├─ JT_ID → IdHandler
   │   ↓ 使用 PuLIDFluxPipeline
   │   ↓ 加载 LoRA（如需要）
   │   ↓ 执行推理
   │
   ├─ JT_POSE → PoseHandler
   │   ↓ 姿态检测与处理
   │   ↓ 执行推理
   │
   ├─ JT_RBG/JT_SR/JT_FS → RbgSrFsHandler
   │   ↓ 不加载 diffusers pipeline
   │   ↓ 直接处理图片（去背景/超分/换脸）
   │
   └─ 其他（MK/ED/SED） → DiffusionHandler
       ↓ 通过 PipelineLifecycle 创建 pipeline
       ↓ 配置 scheduler（load_scheduler_from_pipe）和 fast_mode
       ↓ 迁移到目标设备（OOM 时回退）
       ↓ 构建迭代规格（IterationSpec）
       ↓ 驱动迭代生成
6. 结果处理（每个 Handler 内部）
   ↓ ResultHandler.process_single_result()
   ↓ - 应用后处理（upscale/face_enhance/remove_bg）
   ↓ - 保存图片到本地
   ↓ - 生成缩略图
   ↓ - 更新files表
   ↓ - 逐张发送WebSocket消息
7. 任务生命周期管理
   ↓ _task_context() - 统一的任务生命周期包装
   ↓ - 清理缓存（_cleanup_cache）
   ↓ - 更新任务状态（_send_task_status）
   ↓ status = "completed" 或 "failed" 或 "cancelled"
8. 流程结束
```

### 取消任务流程

```
用户取消任务
   ↓ DELETE /v1/tasks/{task_id}
后端发送cancel消息
   ↓ WebSocket发送cancel消息
推理器接收cancel消息
   ↓ TaskProcessor 标记任务为已取消
检查当前任务
   ↓ _check_cancelled(task_id, stage) - 在多个阶段检查
   ↓ 如果正在处理该任务：
   ↓   - Handler 通过 check_cancelled 回调检查
   ↓   - 中止推理循环
   ↓   - 发送 "cancelled" 状态
更新任务状态
   ↓ status = "cancelled"
   ↓ 清理缓存（_cleanup_cache）
```

## 使用方式

### 启动推理器

```bash
# 从命令行启动
python inference/image/main.py <service_id>

# 示例
python inference/image/main.py service_123
```

### 启动配置

推理器需要从 `config/{service_id}.yaml` 读取配置，配置格式参考 `config/example.yaml`。

**必需配置项**：
- `service_id`：服务ID
- `service_type`：必须为 `"image"`
- `api_base_url`：API后端地址（用于上报启动信息等；见 `inference/config/inference.yaml`）

**消息通道配置（二选一）**：
- **WS（默认，历史兼容）**：在 `inference/config/inference.yaml` 配置 `ws_url`，且不配置 `transport`（或 `transport.ingresses/egresses` 选择 `ws`）。
- **Redis List 队列（兼容对方 RPOP 模型）**：在 `inference/config/inference.yaml` 配置 `transport.ingresses/egresses` 为 `redis_list`，并填写 `redis.host/port/pwd/channel/reschannle`。

### 依赖安装

```bash
pip install -r requirements.txt
```

### Redis List 队列接入（对接 RPOP 系统）

如果你希望推理器从 Redis list 获取任务、并将 `result` / `task_status` 写回 Redis list：

1. 安装推理侧依赖（确保包含 `redis` 包）：

```bash
pip install -r inference/requirements.txt
```

2. 修改 `inference/config/inference.yaml`，将 `transport.ingresses/egresses` 配置为 `redis_list`，并填写 `host/port/pwd/channel/reschannle`。

3. 可用脚本做快速联调（需要本地有 Redis 服务）：

```bash
python3 test/manual_redis_queue_inference_client.py --req atz.req --res atz.res --prompt "a cute cat"
```

### 模型目录约定（`inference_config.models_dir`，默认 `resources/models`）

- **RBG（去背景）**（从 `models_dir` 解析，不再使用 `weights_dir`）：
  - **RMBG-2.0（推荐，transformers+safetensors）**：
    - `resources/models/RMBG-2.0/`（本地模型目录，内容与 HuggingFace `briaai/RMBG-2.0` 仓库一致）
    - 说明：本项目会优先检测并使用 RMBG-2.0；无需 onnxruntime/cuda12 额外运行库，用户安装更简单。
  - **RMBG-1.4（兼容，briarmbg）**：
    - `resources/models/RMBG-1.4/`（本地模型目录）
- **SR (RealESRGAN + FaceEnhancer)**：
  - `RealESRGAN_x2plus.pth` / `RealESRGAN_x4plus.pth` / `RealESRGAN_x4plus_anime_6B.pth` / `realesr-animevideov3.pth`
  - GFPGAN（默认后端）：`GFPGANv1.4.pth` 或 `RestoreFormer.pth`
  - CodeFormer（可选后端）：`codeformer.pth`（放在 `{models_dir}/roop/`）
- **FS (insightface + inswapper)**：
  - 目录：`{models_dir}/roop/`
  - `buffalo_l/`
  - `inswapper_128.onnx`
  - （可选）`GFPGANv1.4.pth`、`detection_Resnet50_Final.pth`、`parsing_parsenet.pth`
  - 说明：FS 内部增强默认关闭（避免与统一后处理链重复增强）；如需开启，设置 `VITOOM_FS_INTERNAL_ENHANCE=1`

## 配置说明

### 启动配置文件示例

```yaml
# inference/config/{service_id}.yaml

service_id: "service_123"
name: "Image Generation Service"
type: "diffusers"  # 推理器类型
service_type: "image"  # 必须为 "image"

# API后端配置
api_host: "127.0.0.1"
api_port: 8888
api_base_url: "http://127.0.0.1:8888/api/inference/services"

# WebSocket Server配置
ws_host: "127.0.0.1"
ws_port: 8888
ws_url: "ws://127.0.0.1:8888/ws/inference"
```

### 环境变量

推理器支持通过环境变量覆盖配置：

- `DATABASE_URL`：数据库URL
- `API_BASE_URL`：API后端地址
- `WS_URL`：WebSocket Server地址
- `VITOOM_FACE_ENHANCER`：人脸增强后端（`codeformer`/`gfpgan`，默认 `gfpgan`）
- `VITOOM_FACE_ENHANCER_STRICT`：严格模式（`1` 时后端初始化失败直接报错；默认 best-effort 回退）
- `VITOOM_CODEFORMER_CKPT`：CodeFormer 权重文件名（默认 `codeformer.pth`，从 `{models_dir}/roop` 查找）
- `VITOOM_CODEFORMER_W`：CodeFormer fidelity weight（默认 `0.5`）
- `VITOOM_FS_INTERNAL_ENHANCE`：FS 内部增强开关（`1` 开启；默认关闭以避免重复增强）

> 权重自动下载：当 `face_enhance=true` 且选择的后端为 CodeFormer 时，如果 `{models_dir}/roop/codeformer.pth` 不存在，推理器会尝试从 GitHub Release 下载到该路径；下载失败则按回退策略（GFPGAN/禁用增强）继续执行。
> 许可证提醒：CodeFormer 上游为 **S-Lab License 1.0（非商用再分发许可）**。若你的场景涉及商业用途，请先确认合规或取得授权。

## 扩展指南

### 添加新的 job_type Handler

如果需要添加新的任务类型，可以创建新的 Handler：

1. **创建 Handler 类**：在 `handlers/` 目录下创建新的 Handler 文件

```python
# handlers/my_handler.py
class MyHandler:
    def __init__(
        self,
        *,
        inference_config: Any,
        result_handler: Any,
        service_id: str,
        logger: Any,
        # ... 其他依赖
    ):
        self.inference_config = inference_config
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
    
    async def run(self, params: InferenceRequestParams, *, task_id: str) -> None:
        # 实现推理逻辑
        # 使用 result_handler.process_single_result() 回传结果
        pass
```

2. **在 ImageInferrer 中注册**：在 `inference_callback()` 中添加分发逻辑

```python
# inferrer.py
from image.handlers.my_handler import MyHandler

async def inference_callback(self, params: InferenceRequestParams) -> Any:
    # ...
    if params.job_type == "MY_TYPE":
        handler = MyHandler(
            inference_config=self.inference_config,
            result_handler=self.result_handler,
            service_id=self.service_id,
            logger=logger,
        )
        await handler.run(params, task_id=task_id)
```

3. **添加 job_type 常量**：在 `common/Constant.py` 中添加新的 job_type 常量

### 添加新的模型类型支持

新增模型家族/类型建议按“单一事实源”维护（避免多处登记遗漏）：

1. **新增/修改家族 spec**：在 `inference/common/model_families/*.py` 添加 `SPEC = ModelFamilySpec(...)`
   - aliases、model_index.json `_class_name` 规则、默认 pipeline（可选，单文件识别用）
2. **按需补充 PipelineDetector**：
   - 单文件权重识别：补 `_detect_model_type_from_file_keys`
   - 需要特殊组件注入（nunchaku transformer/unet 等）：补 `build_pipeline_params`
3. **按需补充参数 spec**：在 `inference/image/inference_param_specs.py` 新增 `InferenceParamSpec` 子类（自动发现，无需手工列表）
4. **如需新增通用预处理规则**：统一改 `image/runtime/params_preprocessor.py`（扁平入口）

### 自定义结果处理

如果需要自定义结果处理逻辑，可以：

1. **继承 ResultHandler**：创建自定义的结果处理器
2. **重写方法**：重写 `process_single_result()` 或其他方法
3. **在 ImageInferrer 中使用**：在 `initialize()` 中初始化自定义的结果处理器

```python
class CustomResultHandler(ResultHandler):
    async def process_single_result(self, ...):
        # 自定义处理逻辑
        pass

class ImageInferrer(BaseInferrer):
    async def initialize(self):
        await super().initialize()
        self.result_handler = CustomResultHandler(
            ws_client=self.ws_client,
            storage_base_path=self.inference_config.outputs_dir,
        )
```

### 自定义 Pipeline 缓存策略

Pipeline 缓存由 `PipelineCache` 管理，可以通过配置调整：

1. **配置 TTL**：在 `inference.yaml` 中设置 `pipeline_cache_ttl_seconds`
2. **调整 LRU 大小**：修改 `PipelineCache` 的 `max_size` 参数（当前为 1）
3. **自定义释放逻辑**：重写 `_release_cached_pipeline()` 方法

## API参考

### ImageInferrer类

#### `__init__(service_id: str)`

初始化图片推理器。

**参数**：
- `service_id`：服务ID

**初始化组件**：
- `detector`：PipelineDetector 实例
- `device_planner`：DevicePlanner 实例
- `model_locator`：ModelLocator 实例
- `seed_manager`：SeedManager 实例
- `pipeline_cache`：PipelineCache 实例（LRU=1，TTL 控制）

#### `async initialize()`

初始化推理器（加载配置、初始化组件）。

**流程**：
1. 调用 `super().initialize()`
2. 初始化 `ResultHandler`
3. 启动 `pipeline_cache` 的驱逐循环（若 TTL > 0）

#### `async cleanup()`

清理资源（信号处理器回调）。

**流程**：
1. 停止 `pipeline_cache` 的驱逐循环
2. 调用 `super().cleanup()`

#### `async inference_callback(params: InferenceRequestParams) -> Any`

推理回调函数，实现完整的图片推理流程。

**参数**：
- `params`：推理请求参数（`InferenceRequestParams`）

**流程**：
1. 检查任务是否已取消（`_check_cancelled`）
2. 进入任务上下文（`_task_context`）
3. 推理前参数预处理（`image/runtime/params_preprocessor.py::preprocess_inference_params`）
5. 根据 `job_type` 分发到对应的 Handler：
   - `JT_ID` → `IdHandler`
   - `JT_POSE` → `PoseHandler`
   - `JT_RBG/JT_SR/JT_FS` → `RbgSrFsHandler`
   - 其他 → `DiffusionHandler`
6. Handler 执行推理并处理结果
7. 更新任务状态（在 `_task_context` 中自动处理）

#### 推理前参数预处理（统一入口）

统一在 `image/runtime/params_preprocessor.py::preprocess_inference_params` 中完成（扁平入口，避免分散/调用链变长），主要包含：
- `url` 不可加载时回退为文生图（清空 `url`）
- `model_version` 归一化（model_class/model_version(raw) → canonical；必要时触发 `PipelineDetector` 自动侦测）
- RBG 强制 `file_type="png"`
- prompt/LoRA：解析并缓存 `parsed_loras`，拼接 trigger_words，再 sanitize（会移除 `<lora...>` 标签）
- 尺寸约束：总像素超过 1280×1280 则等比缩小（`common/image_utils.constrain_size`）

#### `async _check_cancelled(task_id: str, stage: str) -> bool`

检查任务是否已取消。

**参数**：
- `task_id`：任务ID
- `stage`：检查阶段描述

**返回**：
- `True`：任务已取消，已发送 "cancelled" 状态
- `False`：任务未取消

#### `async _task_context(task_id: str)`

统一的任务生命周期上下文管理器。

**功能**：
- 异常处理：捕获异常并发送 "failed" 状态
- 成功处理：发送 "completed" 状态
- 清理缓存：调用 `_cleanup_cache`

### Handler 接口

所有 Handler 都实现 `async run(params: InferenceRequestParams, *, task_id: str) -> None` 方法。

#### DiffusionHandler

处理扩散类任务（MK/ED/SED）。

**依赖**：
- `lifecycle`：PipelineLifecycle 实例
- `seed_manager`：SeedManager 实例
- `result_handler`：ResultHandler 实例
- `check_cancelled`：取消检查回调

#### IdHandler

处理 ID 任务（PuLID identity）。

**依赖**：
- `inference_config`：推理配置
- `device_planner`：DevicePlanner 实例
- `pipeline_cache`：PipelineCache 实例（可选）
- `oom_helper`：OOM 辅助器（PipelineLifecycle）

#### PoseHandler

处理 POSE 任务（姿态控制）。

**依赖**：
- `inference_config`：推理配置
- `device_planner`：DevicePlanner 实例
- `check_cancelled`：取消检查回调
- `pipeline_cache`：PipelineCache 实例（可选）
- `oom_helper`：OOM 辅助器

#### RbgSrFsHandler

处理无 pipeline 任务（RBG/SR/FS）。

**依赖**：
- `inference_config`：推理配置
- `result_handler`：ResultHandler 实例

## 常见问题

### Q: 如何添加新的模型支持？

A: 新的模型类型通常通过 `PipelineDetector` 自动检测。参考[扩展指南](#扩展指南)中的"添加新的模型类型支持"部分。

### Q: 推理结果存储在哪里？

A: 默认存储在 `{inference_config.outputs_dir}/{user_id}/{task_id}/` 目录下（通常为 `resources/outputs/{user_id}/{task_id}/`）。

### Q: 如何自定义存储路径？

A: 在创建 `ResultHandler` 时指定 `storage_base_path` 参数（在 `initialize()` 方法中）。

### Q: 支持哪些图片格式？

A: 支持 PNG、JPEG、WebP 等格式，由 `InferenceRequestParams.file_type` 指定。注意：RBG 任务强制使用 PNG（因为需要 alpha 通道）。

### Q: 如何实现批量生成？

A: 通过 `InferenceRequestParams.generate_num` 参数指定生成数量。Handler 会根据此参数构建多个 `IterationSpec`，逐张生成并回传。

### Q: 任务取消后会发生什么？

A: 如果任务在推理过程中被取消：
1. `_check_cancelled()` 会检测到取消状态
2. Handler 通过 `check_cancelled` 回调检查并中止推理循环
3. 任务状态更新为 `cancelled`
4. 已生成的结果会正常回传，但后续迭代会中止

### Q: Pipeline 缓存如何工作？

A: Pipeline 缓存采用 LRU=1 策略，配合 TTL 控制：
- 缓存最近使用的一个 pipeline
- TTL 到期后自动驱逐并释放显存
- 缓存命中时复用 pipeline，避免重复加载权重

### Q: OOM 时如何处理？

A: 当发生 OOM 时：
1. `PipelineLifecycle.move_to_device()` 会捕获异常
2. 自动回退到 CPU offload 模式
3. 重新尝试推理
4. 如果仍然失败，任务状态更新为 `failed`

### Q: 如何添加新的后处理步骤？

A: 在 `runtime/postprocess_pipeline.py` 的 `apply_postprocess()` 函数中添加新的后处理逻辑。当前支持：
- `upscale`：超分（仅 2/4 生效）
- `face_enhance`：人脸增强（默认 GFPGAN，可切换 CodeFormer）
- `remove_bg`：去背景（必须最后执行，强制 png）

## 相关文档

- [推理服务公共模块文档](../common/README.md)
- [推理服务架构设计文档](../../docs/推理服务架构设计文档.md)
- [InferenceRequestParams](../schemas.py)

## 更新日志

### v3.0.0 (当前版本)

- ✅ 重构为 Handler 模式：按 `job_type` 分发到对应的 Handler
- ✅ 新增 `IdHandler`：支持 PuLID identity 任务
- ✅ 新增 `PoseHandler`：支持姿态控制任务
- ✅ 新增 `RbgSrFsHandler`：统一处理无 pipeline 任务（RBG/SR/FS）
- ✅ 重构 `DiffusionHandler`：从 `inferrer.py` 抽离扩散类任务逻辑
- ✅ 新增 `PipelineLifecycle`：统一管理 Pipeline 生命周期
- ✅ 新增 `PipelineCache`：LRU=1 + TTL 的 Pipeline 缓存
- ✅ 新增 `SeedManager`：统一管理种子生成
- ✅ 新增 `ModelLocator`：统一管理模型路径定位
- ✅ 支持模型版本自动解析：`model_class` → `model_version`（canonical）
- ✅ 支持任务取消检查：在多个阶段检查任务是否已取消
- ✅ 支持任务生命周期管理：统一的任务上下文管理器

### v2.0.0 (已废弃)

- ✅ 现行编排器架构：`ImageInferrer + ModelCatalog(model_families) + PipelineDetector + DevicePlanner`
- ✅ 支持 MK/ED/SED/RBG/SR/FS
- ✅ 支持 fast_mode(fbcache) 与 OOM 回退
- ✅ 支持生成后后处理：upscale/face_enhance/remove_bg(最后强制 png)

> 注意：v2.0.0 中提到的 `PipelineRouter` 和 `pipelines/*` 方案已不再使用。
