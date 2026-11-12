# MiniMax-H3 视频推理器接入开发文档

## 1. 目标与边界

在视频推理器（`inference/video`）中原生支持 MiniMax-H3：文生视频、首帧 / 首尾帧生视频，后续扩展参考生视频。输出带立体声音轨的 mp4，走现有 WebSocket 任务、取消和结果回传。

- 直接使用本地 `Comfy-Org/MiniMax-H3` 的量化单文件，不换格式、不下载官方 bf16 权重。
- **不引用任何 ComfyUI 代码**，不起 ComfyUI 进程。只依赖 diffusers 和 `comfy-kitchen`（两者都是 Apache-2.0），交付物可以放进商业许可的版本。
- 运行环境为 `vitoom-infer`（torch 2.14.1+cu130，diffusers 0.42.0.dev0，transformers 5.19），单张 RTX 5090（32 GB）。

明确不做：2K 重生成、Context-IR、latent upscaler。Fun-ControlNet 放到最后一个阶段（第 5 阶段）单独评估。

## 2. 技术路线

| 组件 | 骨架（Apache-2.0） | 我们写的适配 |
|---|---|---|
| DiT | diffusers `MiniMaxH3Transformer3DModel` | 量化层替换、pruned 时间嵌入、键名映射 |
| 文本编码器 | transformers `Qwen3VLTextModel`（只建 50 层） | nvfp4 / AWQ 量化层替换、int8 词嵌入、去掉最终 norm |
| 视频 VAE | diffusers `AutoencoderKLMiniMaxH3` | 键名映射、int8 convrot 量化层、qkv 与 SwiGLU 重排 |
| 音频 VAE | diffusers `AutoencoderKLMiniMaxH3Audio` | 把折叠的 weight_norm 拆回 `weight_g / weight_v` |
| 量化内核 | `comfy-kitchen` 的 `QuantizedTensor` 和各布局类 | `QuantLinear` 包装，支持旁路 LoRA |
| 采样与条件构造 | diffusers `MiniMaxH3Blocks`（modular pipeline）、`MiniMaxH3Scheduler` | 拆掉其中的文本编码步骤，传入预先算好的 `prompt_embeds` |
| Turbo LoRA | 无 | Comfy 命名的 LoRA 以旁路方式挂到 `QuantLinear` 上 |

原型代码在开发机的 `test/h3_exp/`（该目录被 `.gitignore` 忽略，不在仓库中），已在本机跑通端到端，见第 9 节。

## 3. 权重格式与已验证的映射

每个量化层都附带 `comfy_quant` 张量（UTF-8 JSON），按其中的 `format` 选择 `comfy-kitchen` 的布局类：

| format | 布局类 | 文件中的张量 |
|---|---|---|
| `w6a8_int8`（`group_size` 32，带 convrot，旋转块大小 256） | `AsymW4A8Int8Layout` | `weight` I8 `[N, 3K/4]`、`weight_s_rel` F8 `[N, K/32]`、`weight_s_channel` F32 `[N]` |
| `int8_tensorwise`（可带 convrot） | `TensorWiseINT8Layout` | `weight` I8 `[N, K]`、`weight_scale` F32 `[N, 1]` |
| `nvfp4` | `TensorCoreNVFP4Layout` | `weight` U8 `[N, K/2]`、`weight_scale` F8（块缩放）、`weight_scale_2` F32 标量；部分层另有 `pre_quant_scale` |

量化都是按输出行分组的，convrot 只旋转输入维，所以**按行切分或重排量化层是无损的**。qkv 拆分、SwiGLU 换序都依赖这一点。

以下映射都经实验确认，实现时必须照此处理：

| 位置 | 事实 | 依据 |
|---|---|---|
| DiT 时间嵌入 | `adaln_t_table [1025, 8]` 是 t∈[0,1] 上的秩 8 系数表，**第 0 行对应 t=0（噪声端）**，格点间线性插值。每个 block 的 `adaln_proj.linear [96768, 8]` 和 `final_layer.adaln_proj.linear [10752, 8]` 直接以这 8 维系数为输入，**不经过 SiLU** | 纯噪声输入时速度与噪声的相关度测试；端到端成片正常 |
| DiT 时间约定 | 沿用 diffusers：`t = 1 − sigma`，t=1 为干净，速度 `v = x0 − noise` | 同上 |
| DiT `attn.qkv_proj` | 整块的 `[q; k; v]` 行排列，按行切成 `to_q/to_k/to_v` | 端到端 |
| DiT `mlp.fc1` | 文件中为 `[gate; value]`，diffusers 的 SwiGLU 要求 `[value; gate]`，需交换两半 | 端到端 |
| DiT 其余键名 | 与 diffusers 的 `convert_minimax_h3_transformer_checkpoint_to_diffusers` 中的重命名表一致；`rope.inv_freq` 与 diffusers 计算值相同 | 键覆盖检查：缺失 0，多余 0 |
| 视频 VAE `to_qkv` | **按注意力头交错排列**，`[head0: q k v, head1: q k v, …]`，需先重排为整块 `[q; k; v]` 再切分 | 往返 PSNR：8.5 dB 升到 27.75 dB |
| 视频 VAE `ff.w1` | `[gate; value]`，需交换两半 | 同上 |
| 视频 VAE 其余 | `encoder.down.i.block.j` 对应 `encoder.down_blocks.i.resnets.j`，`nin_shortcut` 对应 `conv_shortcut`，`downsample` 对应 `downsamplers.0`，`x_embedder` 对应 `proj_in`，`to_out` 对应 `to_out.0`，`ff.w2` 对应 `ff.net.2`。`mask_token` 全为 0，对应 diffusers 的全零附加 token，可以丢弃。非量化张量一律转为 fp32（文件里编码器的卷积权重是 fp16、偏置是 fp32）。`latents_mean/std` 取自文件元数据 `minimax_h3_video_vae`（diffusers 默认配置是全 0 和全 1，不能用） | 键覆盖检查 |
| 音频 VAE | 只需把 weight_norm 拆回：`weight_g = ‖w‖`（在 g 形状为 1 的维度上求范数），`weight_v = w`。`latents_mean/std` 取自元数据 `minimax_h3_audio_vae` | 键覆盖检查；往返测试 |
| 文本编码器 | 文件只有 50 层（`model.layers.0-49`）且没有 `model.norm`。构建 50 层，把 `norm` 换成 `Identity`，`last_hidden_state` 即等于完整模型的 `hidden_states[50]`。配置：hidden 5120，64 个注意力头，8 个 KV 头，head_dim 128，intermediate 25600，rope_theta 5e6，mrope 交错，分段 `[24, 20, 20]`，vocab 151936 | 语义聚类测试 |
| 文本编码器 AWQ | `pre_quant_scale` 只出现在 `o_proj` 和 `down_proj` 上，约定为 **乘在输入上**（`x * pre_quant_scale`）。用除法会让数值发散 | 同上 |
| 文本编码器词嵌入 | `int8_tensorwise`，按行缩放，加载时反量化为 bf16（约 1.5 GB） | — |
| 文本编码器视觉塔 | `visual.*` 没有量化（bf16，351 个张量）。配置同 Qwen3-VL-8B，只有 `out_hidden_size=5120`：27 层，hidden 1152，16 头，patch 16，merge 2，deepstack 取第 8/16/24 层。按 `Qwen3VLVisionModel` 装载后和 50 层语言模型拼成 `Qwen3VLModel`。不能用 diffusers 的 `get_qwen3vl_prompt_embeds`：它要求层数大于 50 | 键全部对齐；首尾帧端到端 |
| 分词 | 文生视频为原样分词、不加特殊 token；首尾帧使用 Qwen-VL 标准视觉 token。Qwen3-VL 系列的分词器可以通用 | 端到端 |
| Turbo LoRA | Comfy 命名（`diffusion_model.blocks.N.{attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2}` 以及 `token_refiner`），每个 block 4 个 LoRA，共 208 个，`lora_A/lora_B/alpha`。`qkv_proj` 的秩为 384（q、k、v 三个秩 128 拼接），B 矩阵按行切三份并共享 A；`fc1` 的 B 矩阵交换 gate 和 value 两半；缩放系数为 `alpha / rank` | 端到端 |

## 4. 代码结构

新增 `inference/video/handlers/minimax_h3/`，由原型 `test/h3_exp/` 整理而来：

```
handlers/minimax_h3/
├── quant.py          # parse_quant / build_qtensor / split_rows / QuantLinear（含旁路 LoRA、pre_quant_scale）
├── dit.py            # load_pruned_dit：骨架、量化层替换、PrunedTimeBasis / PrunedModulation / PrunedNormOut
├── text_encoder.py   # load_text_encoder（nvfp4，50 层，可带视觉塔）、build_presentation / build_reference_presentation、encode_prompt
├── vae.py            # load_video_vae / load_audio_vae
├── lora.py           # attach_comfy_lora / detach_lora
├── checkpoints.py    # 模型根目录内各文件的路径解析、分词器目录
├── request.py        # 入参校验与映射：工作流、画布、帧数、步数与偏移、参考来源
├── references.py     # Ref2VA 参考素材的下载与解码（imageio-ffmpeg，不依赖 PyAV / torchaudio）
├── runner.py         # 分阶段执行：条件编码 → DiT 采样 → VAE 解码；取消与进度回调；序列长度预检
├── media.py          # mp4 写出与合流、产物校验
├── mkv_handler.py    # MiniMaxH3MkvHandler：加载关键帧 / 参考素材、调用 runner、ResultHandler 回传
├── cli.py            # 命令行工具与自动验收
└── release.py        # 缓存驱逐时的释放
```

### 4.1 模型识别与路径解析

**是否为 H3 只看 `params.family == "minimax_h3"`**：

- `params.family` 由 `InferenceRequestParams.from_task_dict` 从 `task_dict["model"]["family"]` 读出。这个值是后端在 `send_task_to_inference_service` 里放进去的模型目录（model_catalog）记录，不读任务 `params` 里的 `family`。
- 视频推理器不做 family 归一化，`from_task_dict` 也只做 `strip()`、不转小写。所以后端登记时固定写小写的 `minimax_h3`，推理侧做精确比较。
- 不看目录内容、`load_name` 字符串或 `fast_mode`。`family` 为空时也不做自动侦测，保持现有路由。

确定是 H3 之后：

- `load_name` 按现有约定解析为模型根目录 `{models_dir}/{load_name}`，参考 TurboDiffusion 的 `_resolve_model_root`。
- 在根目录内按子目录加 `minimax_h3_*` 前缀查找各组件文件：
  - `diffusion_models/`：`fl2va` 或 `ref2va` 的 DiT
  - `text_encoders/`：nvfp4 文本编码器
  - `vae/`：视频 VAE 和音频 VAE
  - `loras/`：Turbo LoRA
  - `model_patches/`：ControlNet

  文件名不要写死。缺少必需文件时，报错里列出缺失的文件。
- 分词器和 processor：把 Qwen3-VL 的 `tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json` 等小文件放进模型根目录的 `tokenizer/` 子目录，作为模型包的一部分交付。`tokenizer.json` 和 `preprocessor_config.json` 缺一不可：分词和关键帧的视觉预处理都通过 `AutoProcessor` 加载。原型里用的是 `/home/tonera/models/Qwen3-VL-8B-Instruct`。

### 4.2 路由

在 `VideoInferrer.inference_callback` 的**最前面**加一个窄分支，位置在 `job_type` 分支和 TurboDiffusion 的 `fast_mode` 判断之前。现有代码先看 `MKV/S2V/INP/CCV`，Turbo 只在 `MKV` 里用 `fast_mode` 加 `load_name` 前缀判断。如果 H3 走现有分支，会被送进 Wan2。

- `family == "minimax_h3"` 且 `job_type` 为 `MKV` 或 `INP`：走 `MiniMaxH3MkvHandler`。
- `family == "minimax_h3"` 且是其他 `job_type`：直接失败，提示“H3 不支持该任务类型”。
- 其他情况：一行不改，仍走现有分支。

这一期**不**整理成完整的 family registry。分支越窄，“其他 family 行为不变”这条验收越容易保证。

### 4.3 缓存与释放

- `PipelineCache` 中存放的 `H3Models` **只有分词器和元信息**。文本编码器、DiT（含 Turbo LoRA）、VAE 都由 `runner` 按阶段加载，用完即丢，任务结束时卡上不留 H3 权重。原因：文本编码器和 DiT 不能同时在卡上（见 4.4），让 DiT 跨任务常驻就只能在任务内把它搬到 CPU 再搬回来，而对 `QuantizedTensor` 做 `.to("cpu")` 没有验证过，每任务来回搬 16 GiB 也不比从磁盘重新加载（约 2 s）快。
- H3 仍然经过 `PipelineCache.acquire`：key 变化时它会先驱逐常驻的 Wan2 / TurboDiffusion pipeline，否则 H3 会和它们抢显存。缓存键为 `(model_root, 分词器目录)`。
- **TTL 大于 0**：分词器跨任务复用；任务结束时在 `finally` 里调用 `release_use`，让 TTL 驱逐能正常生效。
- **TTL 为 0**：`PipelineCache.acquire` 只调用 `create_fn`，不保存、不驱逐。handler 在 `finally` 里自己丢弃 `H3Models`。
- 每个阶段的加载函数都在自己的 `finally` 里断开局部引用并执行 `gc.collect()`、`torch.cuda.empty_cache()`。异常时 traceback 仍引用栈帧，不显式断开的话，显存要等异常处理完才会还回来。
- `_release_cached_pipeline` 中 H3 的分支要**独立放在最前面**：用 `isinstance(obj, H3Models)` 判断，命中就调用 H3 自己的释放函数，然后 `return`。不要套进现有 Turbo 那段 `try/except: pass`，因为那段一旦失败，会落到 Wan2 的 `release_wan2_pipeline_twice_async(aggressive_cpu=True)`，也就是对 `QuantizedTensor` 做尚未验证的 `.to("cpu")`。H3 的释放函数内部也不调用 `.to("cpu")`，只丢弃引用。

### 4.4 显存调度（单张 32 GB）

实测（768×1344、124 帧、4 步）：

| 阶段 | 常驻 | 峰值 | 耗时 |
|---|---|---|---|
| 文本编码器（加载 + 编码） | 14.3 GiB | 19.3 GiB | 加载约 2 s，编码约 0.3 s |
| DiT + Turbo LoRA | 16.7 GiB | 23.1 GiB | 加载 2.1 s，采样 4 步共 51 s |
| 视频 VAE + 音频 VAE 解码 | 3.8 GiB | 12.4 GiB | 17 s |

按阶段串行调度后的整任务峰值（`torch.cuda.max_memory_allocated`；未注明时为 768 短边）。480p 下，5 秒 t2va Turbo 每步 3.2 s、总耗时 33 s；5 秒 ref2va（一张参考图）总耗时 37 s；两者峰值都是 20.82 GiB，此时峰值在文本编码阶段，与分辨率无关：

| 场景 | 峰值 | 采样耗时 | 总耗时 |
|---|---|---|---|
| t2va 5 秒 16:9，Turbo 4 步 | 20.82 GiB | 约 50 s | 81 s |
| t2va 5 秒 16:9，非 Turbo 30 步 | 20.82 GiB | 6 分 20 秒（每步 12.7 s） | 6 分 48 秒 |
| fl2va 5 秒首尾帧，Turbo 4 步（省显存改写之前） | 21.71 GiB | 约 62 s | 约 96 s |
| fl2va 15 秒首尾帧，Turbo 4 步 | 25.26 GiB（改写前 27.51） | 6 分 21 秒（每步约 95 s） | 约 7 分 |
| ref2va 5 秒，一张参考图，Turbo 4 步 | 20.89 GiB | — | 2 分 18 秒 |
| ref2va 5 秒，图 + 带音轨视频 + 音频（约 9.5 万行），Turbo 4 步 | 23.97 GiB | — | 5 分 44 秒 |

15 秒非 Turbo 30 步按每步 95 s 估算，采样约 48 分钟，没有实测。

DiT 采样的激活显存由 `dit.py` 里的几处等价改写压下来：
- `_lean_transformer_forward`：各模态嵌入拷进打包序列后立即释放。原实现一直持有 float32 的 `video_embeds`，约两份隐藏状态。
- `_lean_block_forward`：AdaLN 调制按行分段计算，残差原地累加。原实现会把逐 token 的 scale、shift、gate 物化成整段，残差还会新建一份。
- `_ChunkedRopeProcessor`：qk-norm 和 RoPE 分段原地计算；注意力算完立即释放 q、k、v。
- `PrunedNormOut`：分段直接写 float32。

这几处改写与 diffusers 原式逐位一致，由 `inference/tests/test_minimax_h3_dit.py` 在 CPU 上用缩小的模型验证。5 秒时，一次 DiT 前向内的激活峰值从 +3.56 GiB 降到 +2.81 GiB，剩下的基本是注意力必需的部分：隐藏状态、调制结果、q、k、v 和输出。15 秒首尾帧不再出现 `expandable_segments: memory mapping failed` 警告，耗时不变。

规则如下：

1. 文本编码器和 DiT 不能同时在卡上。顺序为：加载文本编码器、编码、释放，然后再让 DiT 上卡。
2. VAE 默认在解码阶段才加载，加载只需几秒。首尾帧需要先编码关键帧，顺序见第 6 节。
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 必须在进程启动时就设置好。** CUDA 显存分配器在进程第一次使用 CUDA 时就读取这个变量，之后再改不生效。同一个视频进程里的 Wan2 任务可能早已初始化过 CUDA，所以写在 H3 handler 里太晚了。做法有两种，都要做：
   - 在 `inference/video/main.py` 最顶部、任何 `import` 之前执行 `os.environ.setdefault(...)`，本地和 Docker 都能覆盖；
   - Docker 的 `supervisord.conf` 中 `[program:video]` 加上 `environment=`。

   这个设置对同进程的 Wan2 和 TurboDiffusion 同样生效，第 2 阶段要回归它们。不设的话，LoRA 旁路的临时张量会造成显存碎片，进而 OOM。
4. LoRA 旁路必须用原地 `y.addmm_(x @ A.T, B.T, alpha=scale)`，不能用 `y = y + ...`，否则 `fc1` 上会多分配约 2 GiB 的临时张量。

### 4.5 取消与进度

- 每步 DiT 前向检查一次 `is_task_cancelled()`，命中时抛出 `TaskCancelledError`。优先用 diffusers 的步进回调；如果 modular pipeline 不支持，就在 `PrunedTimeBasis.forward` 里埋点。
- 文本编码前、采样前、解码前、合成前、上传前各检查一次。
- 取消的生效时机（也是验收口径）：
  - 采样期间，最多再等一步（4 步 Turbo 下单步约 13 秒）；
  - VAE 解码期间，在解码结束后生效（5 秒视频的解码约 17 秒，15 秒视频会更长）。不在解码内部打断。
  - 取消之后，显存要回落到“无 H3 任务”时的水平（TTL=0 和 TTL>0 相同）。

### 4.6 产物

- **只支持 mp4**。`file_type` 与 Wan2 一样强制为 mp4，`webm`、`avi`、`mov` 直接报错。
- 画面用现有的 imageio 写出（与 Wan2 的 `save_video` 相同，使用 imageio-ffmpeg 自带的二进制），帧率固定 24。
- 立体声 32 kHz 音轨先写成 wav，再用 `imageio_ffmpeg.get_ffmpeg_exe()` 拿到可执行文件，把 wav 编码成 AAC 并与视频合流（`-c:v copy -c:a aac -shortest`）。**不要复用 Wan2 的 `merge_video_audio`**，原因有二：
  - 它调用 PATH 上的 `ffmpeg`，不能假设部署环境里有；
  - 它出错时只 `print` 后返回、不抛异常，会悄悄产出一条没有音轨的视频。

  H3 的合流失败必须让任务失败。
- 用 `ResultHandler.process_single_result(file_data=mp4_bytes, ...)` 回传，`file_seed` 填实际使用的种子。回传的 `duration` 直接取 `request_params.duration`（1–60 的整数，没有单独的覆盖参数），所以 handler 在开始时就要把校验后的整数秒写回 `request_params.duration`。对齐到 `17n+5` 后的精确帧数和秒数只记日志。例如 5 秒对应 124 帧（约 5.17 秒），15 秒对应 362 帧（约 15.08 秒）。

## 5. 入参映射（`InferenceRequestParams` 到推理参数）

| 入参字段 | H3 推理参数 | 规则 |
|---|---|---|
| `family` | 是否走 H3 | 等于 `minimax_h3` 才走 H3，见 4.1、4.2 |
| `job_type` | 工作流 | `MKV`、`INP`：没有图为 `t2va`；有 `url` 或 `image_file2` 为 `fl2va`；`MKV` 带 `tpl_list` / `ref_video` / `ref_audio` 为 `ref2va`（不需要新增 job_type）。参考素材与首尾帧不能同时出现，`INP` 不能带参考素材。其他值报错 |
| `load_name` | 模型根目录 | 见 4.1 |
| `prompt` | `prompt` | `MKV` 必须非空。`INP` 只有带首帧或尾帧时允许为空，没有图时与文生视频相同、必须非空。提示用户：画面、镜头和音频（对白、音效、音乐）都写在同一段提示词里 |
| `prompt` 中的 `<lora:...>`、`loras` | — | **第一期直接失败**。Wan2 会自己从 prompt 里解析 LoRA 标签，H3 如果原样送进文本编码器，标签会变成提示词的一部分 |
| `negative_prompt` | — | 忽略，记 warning。权重做过 CFG 蒸馏，没有负向分支 |
| `guidance_scale` | — | 忽略，原因同上 |
| `url` | `image`（首帧） | 有值即为首帧。`INP` 下可以为空（只有尾帧），这一点与 Wan2 的 INP 不同（Wan2 要求首帧必填）。handler 按原图比例下载（长边超过 2048 先等比缩小），缩放和裁剪交给 diffusers 的 `MiniMaxH3ResizeStep`：第一张关键帧拉伸到画布，跟随的那张 cover-crop |
| `image_file2` | `last_image`（尾帧） | 有值即为尾帧，可以单独出现 |
| `duration` | `num_frames` | **超出 5–15 秒直接失败**，不截断，否则用户会以为生成了 60 秒。在范围内时：`num_frames = round(duration × 24)`，再向上取到 `17n+5` 交给 pipeline，`request_params.duration` 保持整数秒（见 4.6） |
| `runtime_config.fps` | — | 忽略，记 warning。后端会把任务里的 `fps` 写进 `model.runtime_config`，Wan2 会用它改帧率；H3 固定 24 fps，不能用它来算 `num_frames` |
| `aspect_ratio`、`width`、`height`、`resolution` | `height`、`width` | 宽高比的确定顺序：①有 `aspect_ratio` 就用它；②有首帧就用首帧比例；③`width × height` 不等于默认值 1024×1024 时，用这对数值的比例；④其余情况用 16:9。原因是 schema 和 `from_task_dict` 会把缺省的宽高补成 1024×1024，无法区分“没传”和“要 1:1”，所以确实要 1:1 的用户应传 `aspect_ratio="1:1"`。比例确定后调用 diffusers 的 `resolve_canvas_size`：短边取 `resolution`，面积上限 768×1344，两边都取 32 的倍数，宽高比限制在 1:4 到 4:1，超出则失败。`resolution` 是短边像素，与前端约定一致（`480`、`"720"`、`"544p"` 都可以），范围 352–768，缺省为官方的 768；短边不是 32 的倍数时取最接近的倍数，恰在中间时取大。16:9 下：480 得 864×480，544 得 960×544（与 ComfyUI 的 0.4 MP、0.5 MP 相同），720 得 1312×736，768 得 1344×768 |
| `fast_mode` | 是否挂 Turbo LoRA | `True`（默认）：挂 `loras/` 下的 Turbo LoRA。步数和视频偏移必须与蒸馏时一致，由 `turbo_schedule` 从文件名读出：`4step` 即 4 次前向；`768p` 版视频偏移 6，544p 版 12（ModelTC 发布说明）。`False`：不挂 LoRA，视频偏移 12，步数取 `num_inference_steps` |
| `num_inference_steps` | DiT 前向次数 | 仅在 `fast_mode=False` 时生效。默认 30（schema 默认值；为 0 时也按 30），范围 1–50。口径与 ModelTC / SGLang 一致：N 步 = N 次前向；传给 diffusers 时加 1，因为 diffusers 数的是含终点 0 的网格点。官方底模示例用 50 |
| `seed` | `generator` | 为 `None` 或小于 0 时随机。实际使用的种子写入结果。用 CPU 上的 `torch.Generator` |
| `generate_num` | — | 循环生成，第 i 条用 `seed + i`。DiT 只加载一次，文本 embedding 复用 |
| `schedulerName` | — | 忽略，固定使用 `MiniMaxH3Scheduler`（视频偏移见 `fast_mode`，音频偏移 3） |
| `file_type` | 容器 | 只支持 mp4，其他值失败，见 4.6 |
| `low_vram` | — | 第一期忽略 |
| `tpl_list` | 参考图 | 图片 URL 或本地路径的列表，最多 9 张，空字符串忽略。按 ComfyUI 默认的 `match` 方式缩放：保持比例、只缩不放，缩到生成画面的像素量，两边取 32 的倍数，LANCZOS 插值。没有采用 diffusers 的做法（放大到短边 2048）：那样一张 16:9 参考图约 7200 行序列，480p 生成时采样会慢好几倍 |
| `ref_video` | 参考视频 | URL 或本地路径（下载上限 500 MB）。只解码前 16 秒，ffmpeg 按旋转元数据摆正；diffusers 再重采样到 24 fps、截到生成时长、放到它自身比例的 768p 画布上。自带音轨时一并作为条件 |
| `ref_audio` | 参考音频 | URL 或本地路径，也可以是带音轨的视频。只解码前 16 秒，ffmpeg 直接转成 32 kHz 双声道。不能单独使用，至少还要有一张参考图或一段参考视频 |

`InferenceRequestParams` 第一期不需要新增字段。

## 6. 推理流程

```
handler.run(params)
  ├─ （已由 inferrer 按 family == "minimax_h3" 路由到这里）
  ├─ reference_sources：参考来源的数量和组合校验（不下载）
  ├─ 下载 url / image_file2 对应的关键帧，以及 tpl_list / ref_video / ref_audio 对应的参考素材（有才下载）
  ├─ 校验：file_type、duration 范围、prompt（没有首尾帧时必填）、LoRA（loras 或 <lora:> 标签存在即失败）
  ├─ 解析模型根目录、工作流（t2va / fl2va / ref2va）、画布（aspect_ratio → 第一张关键帧 → 宽高 → 16:9）、帧数、种子、是否 Turbo；
  │     把整数秒写回 request_params.duration
  ├─ check_cancelled("before encode")
  ├─ [run_blocking] runner.run_h3
  │  ├─ fl2va：MiniMaxH3ResizeStep 把关键帧放到画布上，得到 keyframes 和 keyframe_anchors
  │  ├─ 文本编码：build_presentation 拼出与 diffusers 逐 token 一致的序列
  │  │     t2va：原样的 prompt，不加特殊 token
  │  │     fl2va：每张关键帧一个 "<Picture i>: " 标签加视觉块，再接 prompt；文本编码器带上视觉塔
  │  │     → 立刻释放文本编码器（含视觉塔）
  │  ├─ fl2va：加载视频 VAE，用 MiniMaxH3KeyframeVaeEncoderStep 编码关键帧
  │  │     （ImageNet 归一化、固定种子 42 采样后验；加噪到 t=0.999 由去噪块完成）→ 释放 VAE
  │  ├─ ref2va：MiniMaxH3Ref2VASetupStep 规整参考 → build_reference_presentation → 估算序列行数，超过 17 万直接失败
  │  │     → 文本编码（带视觉塔）→ 释放 → MiniMaxH3Ref2VAReferenceEncoderStep 用视频 / 音频 VAE 编码参考 → 释放
  │  ├─ 加载 DiT（fast_mode 时挂 Turbo LoRA）；按工作流用 Core / FL2VACore / Ref2VACore 去噪块（ref2va 的 DiT 挂在 transformer_ref 上）
  │  │     pipe(prompt_embeds, text_token_tags, condition_latents?, keyframe_anchors?, height, width, num_frames,
  │  │          num_inference_steps=steps + 1, generator, output=["latents", "audio_latents"])
  │     每一步检查是否取消
  │     → 丢弃 DiT
  ├─ [run_blocking] 加载 VAE，运行 MiniMaxH3DecodeStep，得到 videos (T,H,W,3) 和 audio (2,S)，采样率 32 kHz
  ├─ 合成 mp4（imageio 写出画面 + imageio-ffmpeg 合流 AAC，失败则任务失败）
  │     → 丢弃 VAE
  ├─ ResultHandler.process_single_result → 发送 completed 状态
  └─ finally：TTL>0 时 release_use；TTL=0 时丢弃 H3Models（只有分词器）
```

## 7. 分阶段计划与验收

每个阶段验收通过后再进入下一阶段。

**开始之前**

- `.gitignore` 第 49 行忽略了整个 `/test` 目录。所以原型 `test/h3_exp/` 和本文档都**不在仓库里**，只存在于开发机 `/home/tonera/project/vitoom/test/`。第 1 阶段开始前要确认这些文件还在，并把本文档移到会被版本管理的位置（例如 `inference/video/handlers/minimax_h3/README.md`，或者 `docs/`），以免丢失。
- 依赖：`inference/requirements.txt` 目前写的是 `diffusers>=0.42.0`、`transformers`（无上限）和 `comfy-kitchen==0.2.41`。原型是在 `vitoom-infer` 的 diffusers `0.42.0.dev0` 上跑通的，而开发版在 PyPI 上无法固定。所以要先选一个包含 H3 模块的正式版（例如本机 `vitoom-comfy` 里的 `0.42.1`），用它重跑原型的实验 4 到 6，通过后再把 `requirements.txt` 固定为 `diffusers==<该版本>`。transformers 也固定到验证过的版本。

**第 1 阶段：模块化与命令行工具（文生视频）**

- 把 `test/h3_exp/` 整理进 `inference/video/handlers/minimax_h3/`，再加一个命令行入口，可以指定提示词、时长、宽高比、种子、是否 Turbo、TTL 模式。
- 在 `vitoom-infer` 中安装 `comfy-kitchen==0.2.41`，删除 `test/h3_exp/_vendor` 软链接。
- 在 `inference/video/main.py` 顶部设置 `PYTORCH_CUDA_ALLOC_CONF`（见 4.4）。
- 验收，以下检查项都由命令行工具自动判定：
  - 输出的分辨率等于画布尺寸，帧数等于对齐后的 `17n+5`，帧率为 24；
  - 音轨存在且非静音，为双声道 32 kHz，时长与视频相差不超过一个音频块（约 25 ms）；
  - 峰值显存：5 秒 ≤ 24 GiB；15 秒 ≤ 31 GiB。15 秒约 11 万个 token，**需要实测**，必要时开启 VAE 分块，或降低 DiT 注意力的显存峰值；
  - 同种子两次运行的结果，在 latent 上的相对误差和解码后的 PSNR 都在预设阈值内。不要求逐字节一致，因为量化内核和 GPU 矩阵乘通常做不到；
  - TTL=0 和 TTL>0 两种模式下，进程内连续跑 3 个任务，任务之间的显存回落到同一水平，不持续增长；
  - 5 秒和 15 秒、16:9 和 9:16 各出一条，人工确认画面和音轨与提示词相符。

**第 2 阶段：接入 `VideoInferrer`（只做文生视频）**

- 实现 4.2 的窄路由、4.3 的缓存与释放、4.5 的取消与进度、4.6 的 mp4 回传，以及第 5 节的全部校验。
- 后端在 model_catalog 中为 H3 登记 `family = "minimax_h3"`（小写）。
- 带图的 `MKV` / `INP` 请求在这一阶段先明确报“暂不支持”。
- 验收：
  - 通过 WebSocket 提交 `family=minimax_h3` 的 MKV 文生视频任务成功，前端能播放带音轨的视频，结果消息里的 `duration` 是整数秒；
  - `family` 为空或为其他值的视频任务（Wan2、TurboDiffusion），路由和产物与改动前完全一致。因为 `PYTORCH_CUDA_ALLOC_CONF` 对整个进程生效，这几类任务要回归；
  - 以下情况都会失败，并给出清楚的错误信息：时长超出 5–15 秒、`file_type` 不是 mp4、带 LoRA、带 `<lora:>` 标签、MKV 空提示词；
  - 按 4.5 的口径验证取消；
  - 与 Wan2 模型交替提交任务不会 OOM（TTL=0 和 TTL>0 都要测）。

**第 3 阶段：首帧、首尾帧（视觉塔）**

- 文本编码器加上 Qwen3-VL 视觉塔（文件里的 `visual.*` 没有量化）。提示词拼装照搬 diffusers 的 `MiniMaxH3FL2VATextEncoderStep`；关键帧缩放和 VAE 编码直接用 diffusers 的 `MiniMaxH3ResizeStep`、`MiniMaxH3KeyframeVaeEncoderStep`。
- 按第 6 节的顺序调度：视觉编码后立即释放文本编码器，再编码关键帧并释放 VAE，最后让 DiT 上卡。实测显存已补进 4.4。
- 覆盖只有首帧、只有尾帧、首尾都有三种情况；`INP` 有关键帧时允许空提示词。
- 验收：三种情况都能出片，且首帧、尾帧与输入图一致（在画布尺寸下 PSNR 达到阈值）；峰值显存不超过 31 GiB。
- **已完成**（`cli.py --check --image ... --last-image ...` 自动判定，关键帧 PSNR 阈值 22 dB）：
  - 5 秒：首尾帧 32.63 / 33.04 dB，只有首帧 32.29 dB，只有尾帧（INP，空提示词）33.22 dB；
  - 15 秒首尾帧：32.18 / 32.68 dB，峰值 27.51 GiB（后续省显存改写后为 25.26 GiB）；
  - 采样期间 q/k/v 留到注意力输出投影时会多占 3 GiB 以上，15 秒首尾帧因此 OOM。已在 `_ChunkedRopeProcessor` 里，注意力算完就释放 q/k/v。

**第 3 阶段之后：非 Turbo 模式**

- 定出 `fast_mode=False` 时的推荐步数和耗时，写进模型配置。这一项单独做，不和视觉塔放在同一阶段，避免同时验证两条没测过的路径。
- **已完成**：
  - 默认 30 步，沿用 schema 的 `num_inference_steps` 默认值，所以后端不需要额外配置。允许范围 1–50，官方底模示例用 50。
  - 视频偏移 12、音频偏移 3。
  - 5 秒 t2va 30 步实测：每步 12.7 s，总耗时 6 分 48 秒，峰值 20.82 GiB，画面正常。
- 同时修正了 Turbo 的视频偏移：768p 4-step LoRA 蒸馏时用的是 6，不是 12（ModelTC 的 `DIFFUSERS_SETUP_AND_INFERENCE.md` 示例显式传 `--video-shift 6`）。Turbo 的步数和偏移改为从 LoRA 文件名读出，换成 8-step 版也能自动对上。

**第 4 阶段：参考生视频（Ref2VA）**

- DiT 换成 `minimax_h3_ref2va_pruned_w6a8`，LoRA 换成 `minimax_h3_ref2v_turbo_4step_*`。两者结构相同，加载器直接复用。
- 输入约定需要和后端商定。建议新增 `job_type`（例如 `R2V`）：`tpl_list` 为参考图（最多 9 张），`ref_video` 为参考视频（自带音轨的话一并作为条件），`ref_audio` 为参考音频；总数最多 12 个，且不能只有音频参考。这些限制来自 diffusers 的 `MiniMaxH3Blocks` 说明。
- 参考的顺序会影响结果，前端需要保证顺序稳定。
- **已完成**：
  - **输入约定**：沿用 `MKV`，有参考素材即走 ref2va，不新增 job_type。打包顺序固定为：`tpl_list` 里的图片（按列表顺序）、`ref_video`、`ref_audio`。
  - **提示词里的标签**：按模态分别编号。图片依次是 `<Picture 1>`、`<Picture 2>`……；参考视频是 `<Video 1>`。音频的编号要特别注意：**参考视频自带音轨时，这条音轨占用 `<Audio 1>`，`ref_audio` 就变成 `<Audio 2>`**；参考视频没有音轨时，`ref_audio` 才是 `<Audio 1>`。前端或用户写提示词时要按这个规则引用。
  - **Turbo**：Ref2VA Turbo LoRA（v0.1，544p 版）是 4 步、视频偏移 12，由 `turbo_schedule` 从文件名读出。
  - **序列长度预检**：`runner.estimate_sequence_rows` 在加载任何模型之前估算 DiT 序列行数，超过 17 万行（约 30 GiB）就直接失败，并给出可操作的错误信息。参考图按 `match` 方式缩到生成画面的像素量，768p 生成时一张约 1000 行，480p 时约 400 行；参考视频按 24 fps 和 768p 计，与同样时长的生成视频行数相当。所以“15 秒生成配 15 秒参考视频”这类请求会被拒绝。
  - **参考视频的视觉 token**：打开了 Qwen3-VL 视频处理器的逐帧像素上限（`cap_pixels_per_frame`）。这是 qwen-vl-utils 的做法，也是 transformers 5.22 起的默认；5 秒参考视频从约 6000 个 token 降到约 4300 个。
  - **验收**（`cli.py --check --ref-image ... --ref-video ... --ref-audio ...`）：只有一张参考图、以及“图 + 带音轨视频 + 音频”两种组合都能出片，人物和动作与参考一致，峰值显存见 4.4。

**第 5 阶段：Fun-ControlNet（单独评估）**

- diffusers 中没有现成实现，需要自己写 patch 逻辑，权重是 `model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors`（int8 convrot）。
- 先评估工作量和实现方式，再决定是否做。

## 8. 风险与待验证项

| 项 | 说明 | 处理 |
|---|---|---|
| 与 ComfyUI 的数值一致性 | 原型用的是 diffusers 的 Euler 调度，ComfyUI 工作流用的是 `res_multistep` + `simple`。成片质量正常，但不会和 ComfyUI 像素级一致 | 默认用 diffusers 调度；只有业务要求对齐 ComfyUI 时，才补实现 `res_multistep` |
| diffusers 版本 | 原型跑在开发版 `0.42.0.dev0` 上，而 `requirements.txt` 只写了 `>=0.42.0`。H3 模块较新，接口可能变动 | 开始前选定正式版并重跑实验，然后固定版本（见第 7 节“开始之前”） |
| `comfy-kitchen` 版本 | 布局类的构造参数可能随版本变化 | 已固定为 `0.2.41`；升级时重跑实验 1 |
| 文档和原型未纳入版本管理 | `/test` 整个目录被 `.gitignore` 忽略 | 第 1 阶段开始前迁移 |
| `PYTORCH_CUDA_ALLOC_CONF` 的影响面 | 该设置对整个视频进程生效，Wan2 和 TurboDiffusion 也会受影响 | 第 2 阶段回归这两类任务 |
| 音频 VAE 质量 | 往返测试只用了合成信号；端到端音轨听感正常 | 第 1 阶段用真实语音和音乐素材做往返测试，并人工试听 |
| 视觉塔显存 | 峰值仍在 DiT 阶段，见 4.4 | 已实测 |
| 15 秒显存余量 | 首尾帧 15 秒峰值 25.26 GiB，不再出现分配器报警 | 已完成省显存改写，见 4.4 |
| 非 Turbo 耗时 | 5 秒 30 步约 7 分钟；15 秒估算约 50 分钟 | 前端需要提示耗时，或者对 15 秒的非 Turbo 请求限制步数 |
| Turbo 偏移 | 之前按 12 跑，与 768p LoRA 蒸馏时的 6 不一致 | 已修正；`turbo_schedule` 从文件名读出步数和偏移 |
| `family` 登记 | 推理侧只认小写的 `minimax_h3`，后端漏登记或大小写不一致时，任务会落到 Wan2 路由，报出的错误难以理解 | 后端登记时校验 |
| Ref2VA Turbo 的分辨率 | Ref2VA Turbo LoRA v0.1 按 544p 训练，fl2va 的 Turbo LoRA 按 768p 训练。现在分辨率可选，但默认是 768 | 后端或前端可以考虑在 Ref2VA 时默认选 544 |
| 参考图的缩放方式 | 第 4 阶段的实测（2 分 18 秒、5 分 44 秒两条）用的是 diffusers 的 2048 放大，后来改成了 ComfyUI 默认的 `match` | 若在意人物的一致性，可以评估改回放大（相当于 ComfyUI 的 `max`），代价是慢好几倍 |
| 参考视频的逐帧像素上限 | 从 diffusers 的代码看不出 MiniMax 参考实现是否加上限；目前选择加上限（`cap_pixels_per_frame=True`） | 若与官方效果有差异，先关掉它再对比 |
| 复用 diffusers 的内部方法 | `build_reference_presentation` 直接调用 `MiniMaxH3Ref2VATextEncoderStep` 的 `_gather_vision_features` / `_build_presentation`，以保证逐 token 一致；它们不是公开接口 | 升级 diffusers 时回归 ref2va |
| 权重许可 | MiniMax-H3 模型本身的许可条款没有核对 | 交付前由法务确认 |

## 9. 原型与实验记录（`test/h3_exp/`）

| 文件 | 内容 | 结论 |
|---|---|---|
| `exp1_quant_layers.py` | 用 `comfy-kitchen` 跑四种量化层，比较内核与“反量化 + bf16 矩阵乘” | 内核可用，快 2.6–3.3 倍，相对误差约 1%（来自激活值的 int8 量化） |
| `exp2_dit_load.py` | DiT 装进 diffusers 骨架，检查键覆盖，单步前向 | 键全部覆盖；768p、5 秒单步约 12.5 s，峰值 21.3 GiB |
| `exp3_time_direction.py` | 判定时间表方向 | 第 0 行对应 t=0（噪声端），不需要翻转 |
| `exp4_vae_roundtrip.py` | 视频和音频 VAE 往返重建 | 视频 qkv 交错 + SwiGLU 交换后 PSNR 27.75 dB；音频解码正常 |
| `exp5_text_encoder.py` | 文本编码器装载，判定 AWQ 约定 | `pre_quant_scale` 乘在输入上；语义聚类正常 |
| `exp6_t2va_e2e.py` | 端到端文生视频带音轨 | 768×1344、124 帧，总耗时约 80 s，峰值 23.1 GiB，成片和音轨都与提示词相符 |
| `h3_dit.py`、`h3_te.py`、`h3_vae.py` | 装载器原型 | 第 1 阶段整理进 `handlers/minimax_h3/` |

原型中有两处与本文档的正式方案不同，整理时要改掉：mp4 合流用的是 PATH 上的 `ffmpeg`（应改为 4.6 的做法）；`PYTORCH_CUDA_ALLOC_CONF` 写在命令行上（应改为 4.4 的做法）。

复现方式：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=test/h3_exp/_vendor:test/h3_exp \
  /home/tonera/miniconda3/envs/vitoom-infer/bin/python test/h3_exp/exp6_t2va_e2e.py --seconds 5
```
