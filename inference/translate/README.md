# Translate 推理服务

与 `text` 并列的专用翻译服务（`service_type=translate`），首发支持 **TranslateGemma**（`AutoModelForImageTextToText` + 专用 chat template）。

## 部署方式

**与 text 共用同一 Docker 镜像与 Python 环境**（`inference/text/requirements.txt`），由 supervisord 以**独立进程**运行，与 text 互不干涉：

```ini
[program:text]      → inference/text/main.py text
[program:translate] → inference/translate/main.py translate
```

本地开发可单独启动：

```bash
python inference/translate/main.py translate
```

前提：

- `inference/config/translate.yaml` 中 `service_type: "translate"`
- `config/default.yaml` 中 `agents.tools.translate.default_model_name`（默认 `translategemma-4b-it`）
- `models` 表登记同名模型，`family=TranslateGemma`（`load_name` 可省略，会回退到默认配置）

## 任务 API

```json
POST /v1/tasks
{
  "task_type": "translate",
  "job_type": "TRANSLATE",
  "prompt": "待翻译文本",
  "extract": {
    "source_lang": "zh",
    "target_lang": "en"
  }
}
```

`load_name` / `family` 可省略：分别回退到 `config/default.yaml` 里的
`agents.tools.translate.default_model_name` 与 `default_family`。

图片翻译：

```json
{
  "tpl_list": ["https://example.com/sign.jpg"],
  "extract": { "source_lang": "cs", "target_lang": "de-DE" }
}
```

## 配置

见 `inference/config/ex_translate.yaml`。
