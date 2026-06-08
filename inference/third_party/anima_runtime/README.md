### Anima Runtime（临时后端）集成说明

本目录提供 **Anima 预览版（非 diffusers 格式）** 的推理运行时，并通过 `AnimaPipeline` 适配到项目现有的图片推理器链路（`ImageInferrer` → `DiffusionHandler`）。

设计目标是 **长远可平滑切换**：当官方推出 diffusers 正式版本后，你通常只需要更新 `common/model_families/anima.py` 的映射即可，无需返工主流程。

---

### 1) 目录结构建议（Model Bundle）

推荐把 Anima 的权重与 tokenizer/config 组织成一个 bundle 目录，例如：

```
models/Anima/
├── anima_paths.json
└── split_files/
    ├── diffusion_models/anima-preview.safetensors
    ├── vae/qwen_image_vae.safetensors
    └── text_encoders/qwen_3_06b_base.safetensors
sd-scripts/
└── configs/
    ├── qwen3_06b/...
    └── t5_old/...
```

其中 `anima_paths.json`（或 `anima.json`）用于自动识别并装配 runtime。

---

### 2) `anima_paths.json` 示例（推荐）

`anima_paths.json` 内容建议如下（路径支持相对 bundle 目录，也支持绝对路径）：

```json
{
  "anima_paths": {
    "dit_path": "split_files/diffusion_models/anima-preview.safetensors",
    "vae_path": "split_files/vae/qwen_image_vae.safetensors",
    "qwen3": {
      "model_or_weights_path": "split_files/text_encoders/qwen_3_06b_base.safetensors",
      "config_dir": "sd-scripts/configs/qwen3_06b",
      "tokenizer_dir": "sd-scripts/configs/qwen3_06b"
    },
    "t5_tokenizer_dir": "sd-scripts/configs/t5_old"
  }
}
```

当 `PipelineDetector` 发现模型目录中存在 `anima_paths.json`/`anima.json` 时，会把该模型识别为 **family=`anima`** 并自动选择 runtime pipeline（除非显式强制走 diffusers）。

---

### 3) 请求侧开关（灰度/回滚）

你可以通过 `request.model_config.anima.backend` 控制后端：

- **`"auto"`（默认）**：优先 diffusers（若模型目录存在 `model_index.json` 且 catalog 可识别），否则回退 runtime
- **`"runtime"`**：强制使用本目录的 `AnimaPipeline`（需要 `anima_paths` 或目录 manifest）
- **`"diffusers"`**：强制使用 diffusers pipeline（要求模型目录有 `model_index.json` 且能被 catalog 命中）

示例：

```json
{
  "model_config": {
    "anima": {
      "backend": "runtime"
    }
  }
}
```

---

### 4) runtime 专用参数（可选）

可以在 `model_config.anima` 里提供（用于加载策略/性能/显存）：

- **`dit_loading_device`**：例如 `"cpu"` / `"cuda"`
- **`qwen3_loading_device`**：例如 `"cpu"` / `"cuda"`
- **`text_device`**：文本侧设备（默认同主 device）
- **`text_dtype`**：文本侧 dtype（默认随实现选择）
- **`attn_mode`**：默认 `"torch"`
- **`split_attn`**
- **`enable_block_swap`**
- **`vae_spatial_chunk_size`**
- **`vae_disable_cache`**
- **`pretouch_cpu_tensors_before_to_cuda`**

生成侧可调参数（每次请求可变，走 `inference_param_specs` 透传）：

- **`flow_shift`**
- **`qwen3_max_len`**
- **`t5_max_len`**

示例：

```json
{
  "model_config": {
    "anima": {
      "backend": "runtime",
      "flow_shift": 3.0,
      "qwen3_max_len": 512,
      "t5_max_len": 512,
      "dit_loading_device": "cuda",
      "qwen3_loading_device": "cuda"
    }
  }
}
```

---

### 5) 能力范围与限制

- **当前仅支持 text2img（MK）**；不支持 `url/img2img`，也不支持 `ED/SED` 编辑模式。
- **LoRA**：runtime 后端不支持 diffusers adapters，因此会自动跳过；未来若切换到官方 diffusers 后端，LoRA 将按 diffusers 逻辑加载（若官方 pipeline 支持）。

---

### 6) 依赖提示

本 runtime 依赖运行环境提供：

- **`torch`**
- **`PIL`/Pillow**
- **`numpy`**（仅用于把 tensor 转成 PIL 图片时需要；缺失会抛出明确错误）

---

### 7) 未来切换到官方 diffusers 版怎么做（最小改动）

当官方推出 diffusers 版本后，建议将模型目录切换为标准 diffusers 格式（含 `model_index.json`）。

你通常只需要：

- 在 `inference/common/model_families/anima.py` 中新增/更新 `model_index_rules`：
  - 把官方 `model_index.json` 里的 `_class_name` 映射到 `PipelineRef("diffusers", "<OfficialPipelineClassName>")`

随后：

- `backend="auto"` 会自动优先走 diffusers（无需改推理主流程）
- 如需灰度：先用 `backend="diffusers"` 强制走官方，出问题可回滚 `backend="runtime"`

## anima_runtime（可复制推理模块）

目标：把 `anima_runtime/` 整个目录复制到新项目，配好依赖与本地模型路径，就能直接推理出图。

### 你需要提供的本地路径

- **DiT**：`anima-preview.safetensors`（支持带 `net.` 前缀的权重）
- **Qwen3 文本编码器**
  - 传目录：HF 标准目录（包含 config/tokenizer/权重）
  - 或传单个 `.safetensors`：同时提供 `config_dir` 和 `tokenizer_dir`
- **T5 tokenizer 目录**：本地 tokenizer 目录（只用 tokenizer，不加载 T5 权重）
- **VAE**：`qwen_image_vae.safetensors`（支持 ComfyUI key → 官方 key 映射）

### 依赖（新项目需要安装）

- `torch`
- `transformers`
- `safetensors`
- `einops`
- `Pillow`
- `numpy`（**仅在把结果转成 PIL 图片/保存时需要**；你也可以后续让我改成完全不依赖 numpy）

### 启动加载加速（可选）

`AnimaInferencer` 额外提供几个参数用于加速“进程启动时的模型加载”：

- `dit_loading_device="cuda"`：DiT 权重直接读到 CUDA（减少 CPU→GPU 迁移耗时；但会增加加载时 GPU 峰值占用）
- `qwen3_loading_device="cuda"`：同理，用于 Qwen3 safetensors
- `pretouch_cpu_tensors_before_to_cuda=True`：在 `.to("cuda")` 前先预触达 CPU 上的参数/缓冲区页，可缓解 safetensors mmap 导致的缺页风暴
  - **代价**：会增加少量 CPU 预处理时间，不建议无脑开启

### 最小用法示例

```python
# 在本项目内使用（推荐）：确保 `inference/` 在 PYTHONPATH 中
from third_party.anima_runtime import AnimaInferencer, AnimaPaths, AnimaRunConfig
from third_party.anima_runtime.tokenizers import Qwen3LocalPaths

# 若你把本目录整体拷贝到新项目，并作为顶层包名 `anima_runtime/` 使用，则可改成：
# from anima_runtime import AnimaInferencer, AnimaPaths, AnimaRunConfig
# from anima_runtime.tokenizers import Qwen3LocalPaths

paths = AnimaPaths(
    dit_path="/abs/path/to/anima-preview.safetensors",
    vae_path="/abs/path/to/qwen_image_vae.safetensors",
    qwen3=Qwen3LocalPaths(
        model_or_weights_path="/abs/path/to/qwen_3_06b_base.safetensors",
        config_dir="/abs/path/to/configs/qwen3_06b",
        tokenizer_dir="/abs/path/to/configs/qwen3_06b",
    ),
    t5_tokenizer_dir="/abs/path/to/configs/t5_old",
)

inf = AnimaInferencer(paths, device="cuda", dtype="bf16")
img = inf.generate(
    "a cute cat, anime style",
    negative_prompt="blurry, low quality",
    config=AnimaRunConfig(height=1024, width=1024, steps=40, cfg=4.5, seed=42),
)
img.save("out.png")
```

