# Mini 推理服务（小模型工具集）

## 定位

Mini 服务是系统里的**第五类**推理服务，和 image/video/text/audio 并列。它的定位不是"某一种模态"，而是一个**小模型工具集**——承载所有具备以下共同特征的模型：

- **单个模型体积小、显存占用低**（通常 < 3GB）
- **调用稀疏、但多种类**：一个系统里可能同时需要 OCR / Rerank / Embedding / Layout / Detector…
- **在任一时刻只需要一个被激活**：用 LRU=1 缓存 + TTL 超时驱逐保证显存始终被"当前在用"的那个模型独占

这类模型如果分别塞到 image/video/text/audio，会污染主服务的逻辑；如果每个都起一个常驻进程，又浪费显存、管理成本高。Mini 服务就是为它们准备的共同宿主。

## 首发能力：GLM-OCR

- **模型**：`zai-org/GLM-OCR`（0.9B encoder-decoder VLM）
- **Runtime**：vLLM（同步 LLM 引擎；OCR 没有流式需求）
- **输入**：图片 / PDF（PDF 在 handler 里用 PyMuPDF 逐页渲染后喂给模型）
- **四种任务模式**（通过 `extract.task` 切换）：
  - `text` — **图文混排 Markdown**（默认）。先用 `doclayout-yolo` 做版面切分，再对文本/表格/公式块分别用 GLM-OCR 识别，插图原图裁剪保留；最终打包为单个 zip 返回
  - `table` — 表格识别（HTML/Markdown 风格）
  - `formula` — 公式识别
  - `extract` — 信息抽取（配合 `extract.schema`）

### text 模式返回包结构

`file_type="zip"`，result 消息的 `url` 指向该 zip；zip 内部：

```
document.md           # 按阅读顺序拼接的图文混排 Markdown
images/
  fig_p000_00.png     # 版面检测到的插图，按 "页号_块号" 命名
  fig_p000_01.png
  ...
meta.json             # 页数、每页块统计、耗时、模型名、layout backend
```

result 消息里同时通过 `content` 字段内联 `document.md` 的全文（保留 `![](images/xxx.png)` 占位），方便 agent / 前端免二次下载直接展示。

### 回退到纯文字 md

请求侧显式传 `file_type: "md"` → 走旧的纯文字 Markdown 输出（`file_type="md"`，文本内容在 `content` 字段）。

layout 模型加载失败（例如权重文件没放好）时也会自动降级到这条路径并记 warn 日志，不影响请求整体成功。

## 对外协议

和其他推理服务完全一致，走统一的 `/v1/tasks`：

```json
POST /v1/tasks
{
  "task_type":  "mini",
  "job_type":   "OCR",
  "model_name": "GLM-OCR",
  "tpl_list":   ["https://.../invoice.pdf", "resources/uploads/note.jpg"],
  "extract": {
    "task": "text"
  }
}
```

> `model_name` 必填。backend 按 `models` 表查库、回填 `model_class` 等字段后派发到 mini 推理服务；
> mini 推理服务按 `model_class` 路由到对应 handler（详见下"内部结构 · 分发表"）。

信息抽取示例：

```json
{
  "task_type":  "mini",
  "job_type":   "OCR",
  "model_name": "GLM-OCR",
  "tpl_list":   ["https://.../idcard.jpg"],
  "extract": {
    "task": "extract",
    "schema": {
      "id_number": "",
      "last_name": "",
      "first_name": ""
    }
  }
}
```

### 结果回传

每个输入文件对应一条 `result` WS 消息，**同时**：

1. **落盘**：
   - `extract.task == "text"` 默认 → `.zip`（图文混排，见上节）；显式 `file_type="md"` 则为 `.md`
   - `extract.task in {table, formula}` → `.md`
   - `extract.task == "extract"` → `.json`
   - 均走现有 `ResultHandler` 管道
2. **内联**：result 消息里额外附带 `content` 字段，携带完整的 OCR 文本（text-zip 模式下即 `document.md` 全文），方便下游（agent / 前端）免去二次下载

## 内部结构

```
inference/mini/
├── main.py            # 入口，和 image/main.py 风格一致
├── inferrer.py        # MiniInferrer：继承 BaseInferrer；维护 bundle_cache；按 model_class 分发
├── handlers/
│   ├── __init__.py
│   └── ocr_handler.py # 首发 handler
├── runtime/
│   ├── __init__.py
│   ├── ocr_runtime.py       # OCR runtime 统一接口（OcrBundleLike.generate_from_messages / shutdown）
│   ├── runtime_resolver.py  # 合并 config.runtime 公共项 + transformers / vllm 分块
│   ├── hf_ocr_bridge.py     # transformers 薄封装（默认 runtime，冷启动快、低显存稳态）
│   └── vllm_ocr_bridge.py   # vLLM 薄封装（高吞吐、常驻 KV 池；按 service_mini.yaml 切换）
├── pipeline/
│   ├── __init__.py
│   ├── layout_bridge.py    # 版面分析薄封装（doclayout-yolo，可扩展）
│   └── doc_pipeline.py     # 图文混排：layout → 分派 GLM-OCR → 组装 md → 打 zip
└── README.md
```

### 分发表（`inferrer._HANDLER_REGISTRY`）

| model_class | handler |
|---|---|
| `GLM-OCR` | `OcrHandler`（支持 transformers / vLLM 双 runtime，见下节） |

### OCR runtime 选择

在 `inference/config/service_mini.yaml` 里切换 `config.runtime.backend`，并把各后端专属参数写在
`runtime.transformers` / `runtime.vllm`；与 `backend` 同级的为两后端共用的公共项（如 `trust_remote_code`、`temperature`、`dtype` 等）。

```yaml
config:
  runtime:
    backend: "transformers"   # 默认：AutoProcessor + AutoModelForImageTextToText，冷启动快
    # backend: "vllm"          # 可选：高吞吐场景，但冷启动慢、常驻 KV 池
```

- `transformers`（默认）：显存稳态 ≈ 权重 + 单请求 KV；冷启动秒级；实现贴近官方示例。
- `vllm`：启动慢（10–30s）、常驻 `gpu_memory_utilization` × 总显存的 KV 池；适合高并发。

两个后端都实现了 `OcrBundleLike.generate_from_messages`，`OcrHandler` / `doc_pipeline`
共用同一条调用路径，切换时无需改业务代码。

> 新模型加入时在此表增加一行即可；handler 侧的 bridge 选择由 handler 自行收敛。

### 缓存策略

- 使用 `common/pipeline_cache.py` 的 `PipelineCache(LRU=1 + TTL)`
- `cache_key` 由"模型路径 + policy 指纹"共同组成；切换模型或更改 policy 自动触发旧 bundle shutdown
- TTL 由 `inference.yaml` 的 `pipeline_cache_ttl_seconds` 控制（默认 300s；mini 服务复用）

### 扩展新的小模型

1. 在管理后台把模型登记到 `models` 表；约定：
   - `model_class` 使用**稳定唯一标识**（例如 `GLM-OCR`、`BGE-rerank`）
   - `is_local_model = true`（除非接入云端 API）
   - `model_type` 按你们前端约定填写
2. （如需要）在 `handlers/` 下新建 `<name>_handler.py`，实现 `async def handle(self, params)` 方法
3. 在 `MiniInferrer._HANDLER_REGISTRY` 增加一行：`"<model_class>": "<handler_id>"`
4. 在 `MiniInferrer.initialize` 的 `self._handlers` 字典里实例化对应 handler
5. 如果运行时和 OCR 不同（比如纯 transformers / onnx），在 `runtime/` 下并列新增一个 bridge
6. （可选）在 `Constant.py` 加一个 `JT_XXX` 常量给任务**语义标签**用，但**不再参与** handler 分发
7. 请求端只需填对应 `model_name` + `extract` 里的专属参数——**协议层零改动**

## 运行

```bash
python inference/mini/main.py service_mini
```

前提：

- `inference/config/service_mini.yaml` 存在且 `service_type: "mini"`
- `models` 表里已登记至少一个 mini 支持的模型（首发需登记 **`GLM-OCR`**，`model_class=GLM-OCR`，`is_local_model=true`，`full_name` 与请求里 `model_name` 一致）
- GPU 可用（OCR 默认 `gpu_memory_utilization=0.35`）

## 和其他服务的边界

| 关注点 | image/video/text/audio | mini |
|---|---|---|
| 模型常驻 | 是（text/audio 长驻） | **否**（TTL 驱逐） |
| 显存占用策略 | 为一类模态预留 | 给当前一个小模型即用即还 |
| 模型来源 | `models` 表 | **同**（`models` 表，model_class 决定 handler） |
| 任务派发 | `task_type == service_type` | **同规则**（`task_type="mini"`） |
| 适用模型规模 | 2B+ | ≤ 3B 级 |

## 版面模型权重位置（约定）

text 模式的图文混排路径依赖 `DocLayout_YOLO_DocStructBench_imgsz1280_2501` 权重，位置与 audio/image 推理器的约定一致（依次在 `models_dir` / `weights_dir` 下查找）：

```
{models_dir or weights_dir}/DocLayout_YOLO_DocStructBench_imgsz1280_2501/doclayout_yolo_docstructbench_imgsz1280_2501.pt
```

参数（置信度阈值 / IoU / imgsz / pdf_dpi / max_pages / title_font_ratio 等）在代码里写死（见 `inference/mini/pipeline/layout_bridge.py` 和 `inference/mini/pipeline/doc_pipeline.py`），不通过 YAML 暴露——需要调整时直接改源码。

## 未来工作（不在本迭代）

- 接入 GLM-OCR 官方 SDK（含 PP-DocLayoutV3）作为 layout backend 备选（在 `layout_bridge.py` 增加一个实现类即可）
- Rerank / Embedding handler（给 RAG）
