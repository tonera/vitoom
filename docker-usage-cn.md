# Vitoom Docker 用户使用指南

[English](docker-usage-en.md) | **中文** | [日本語](docker-usage-jp.md)

## 1. 准备环境

需要安装：

- Docker
- Docker Compose

如果要启动推理服务，还需要：

- NVIDIA GPU
- **支持 CUDA 13.0 的 NVIDIA 驱动**（与 `cu130` 推理镜像一致）
- NVIDIA Container Toolkit

确认 Docker 可以访问 GPU，且 CUDA 13.0 运行时可用：

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

Windows 建议使用 Docker Desktop + WSL2，并确认模型目录所在磁盘已允许文件共享。

## 2. 准备配置

生成 `.env` **二选一**（不要两种都做）：

### 方法 A：安装向导（推荐）

```bash
python scripts/setup_vitoom.py
```

交互式选择组件；自动检测架构、写入镜像 tag、生成密钥；末尾可选择获取 Docker 镜像。

### 方法 B：手动编辑

```bash
cp .env.example .env
```

Windows PowerShell：`copy .env.example .env`

编辑 `.env`，填写全部部署必填项（漏项会导致启动失败或拉错镜像）：

```env
VITOOM_TARGET_ARCH=x86_64
VITOOM_INFERENCE_UPLOAD_AUTH_SECRET=请填写一个随机长字符串
VITOOM_SERVER_PORT=8888
VITOOM_BACKEND_URL=http://BACKEND_IP:8888
VITOOM_WS_URL=ws://BACKEND_IP:8888
```

镜像 tag（须与 Docker Hub / 离线 tar 一致；aarch64 见 `scripts/vitoom_setup/constants.py`）：

```env
VITOOM_BACKEND_IMAGE=tonera/vitoom-backend:latest-x86_64
VITOOM_VISUAL_IMAGE=tonera/vitoom-inference-visual:experimental-cu130-torch2.11-x86_64
VITOOM_TEXT_IMAGE=tonera/vitoom-inference-text:experimental-cu130-torch2.11-x86_64
VITOOM_AUDIO_IMAGE=tonera/vitoom-inference-audio:experimental-cu130-torch2.9.1-x86_64
VITOOM_MINI_IMAGE=tonera/vitoom-inference-mini:experimental-cu130-torch2.11-x86_64
VITOOM_DOWNLOAD_IMAGE=tonera/vitoom-inference-download:experimental-x86_64
```

推理节点另填与 Backend 相同的密钥，及本机已部署服务的 Supervisor URL（未部署留空）：

```env
VITOOM_VISUAL_SUPERVISOR_URL=http://INFERENCE_IP:9001
VITOOM_TEXT_SUPERVISOR_URL=http://INFERENCE_IP:9002
VITOOM_AUDIO_SUPERVISOR_URL=http://INFERENCE_IP:9003
VITOOM_DOWNLOAD_SUPERVISOR_URL=http://INFERENCE_IP:9004
VITOOM_MINI_SUPERVISOR_URL=http://INFERENCE_IP:9005
```

Backend 地址用局域网 IP，不用 `127.0.0.1` 或容器名；改端口时 `VITOOM_SERVER_PORT`、`VITOOM_BACKEND_URL`、`VITOOM_WS_URL` 一起改。

---

`.env` 就绪后，继续 [§3 获取镜像](#3-获取镜像)（方法 A 若在向导中已获取镜像可跳过）。批量下载常用模型见 [§6 初始模型下载（可选）](#6-初始模型下载可选)。

## 3. 获取镜像

```bash
python scripts/load_vitoom_images.py
```

优先 `docker load` 项目目录 `images/<VITOOM_TARGET_ARCH>/` 下的 tar，不存在则 `docker pull`。

只获取部分服务：

```bash
python scripts/load_vitoom_images.py --components backend,visual
```

强制重新加载：

```bash
python scripts/load_vitoom_images.py --force
```

## 4. 启动 Backend

启动：

```bash
docker compose up -d backend
```

查看状态和日志：

```bash
docker compose ps
docker compose logs -f backend
```

健康检查：

```bash
curl http://127.0.0.1:8888/api/health
```

浏览器访问：

```text
http://127.0.0.1:8888
```

如果你修改了 `VITOOM_SERVER_PORT`，把上面的端口同步替换。

## 5. 启动推理服务

启动 Visual（图片和视频生成服务，首次运行较慢）：

```bash
docker compose -f docker-compose.inference.release.yml --profile visual up -d
```

启动 Text（文本大模型服务，首次运行较慢，在DGX Spark上约需5分钟）：

```bash
docker compose -f docker-compose.inference.release.yml --profile text up -d
```

启动 Audio（音频生成服务）：

```bash
docker compose -f docker-compose.inference.release.yml --profile audio up -d
```

启动 Download（模型下载服务）：

```bash
docker compose -f docker-compose.inference.release.yml --profile download up -d
```

启动 Mini（小模型服务）：

```bash
docker compose -f docker-compose.inference.release.yml --profile mini up -d
```

查看状态：

```bash
docker compose -f docker-compose.inference.release.yml ps
```

查看日志：

```bash
docker compose -f docker-compose.inference.release.yml logs -f visual
docker compose -f docker-compose.inference.release.yml logs -f text
docker compose -f docker-compose.inference.release.yml logs -f audio
docker compose -f docker-compose.inference.release.yml logs -f mini
docker compose -f docker-compose.inference.release.yml logs -f download
```

查看容器内 supervisor 状态：

```bash
docker exec -it vitoom-inference-visual supervisorctl -s unix:///tmp/supervisor.sock status
docker exec -it vitoom-inference-text supervisorctl -s unix:///tmp/supervisor.sock status
docker exec -it vitoom-inference-audio supervisorctl -s unix:///tmp/supervisor.sock status
docker exec -it vitoom-inference-mini supervisorctl -s unix:///tmp/supervisor.sock status
```

## 6. 初始模型下载（可选）

完成 `python scripts/setup_vitoom.py` 并生成 `.env` 后，可运行初始模型下载脚本，便于安装后快速体验，总大小合计约 **100G+**。


在部署目录（含 `.env` 的仓库根目录）执行：

```bash
python scripts/download_initial_models.py
```

## 7. 资源目录

推理服务默认挂载：

```text
resources/models
resources/weights
resources/loras
resources/outputs
```

如果模型、权重、LoRA 或输出目录在其他位置，在 `.env` 中修改：

```env
VITOOM_MODELS_HOST_DIR=/data/vitoom/models
VITOOM_WEIGHTS_HOST_DIR=/data/vitoom/weights
VITOOM_LORAS_HOST_DIR=/data/vitoom/loras
VITOOM_OUTPUTS_HOST_DIR=/data/vitoom/outputs
```

Windows 路径使用正斜杠：

```env
VITOOM_MODELS_HOST_DIR=C:/vitoom/models
VITOOM_WEIGHTS_HOST_DIR=C:/vitoom/weights
VITOOM_LORAS_HOST_DIR=C:/vitoom/loras
VITOOM_OUTPUTS_HOST_DIR=C:/vitoom/outputs
```

## 8. 数据目录

Backend 数据保存在部署目录的 `data/` 下：

```text
data/config             用户配置
data/inference/config   推理服务配置
data/resources          SQLite 数据库、输出文件、知识库、内置 ES 数据
data/logs               Backend 和 ES 日志
data/logs/inference     推理服务日志
data/inference/cache    推理服务编译缓存和加速缓存
```

升级或重启时不要删除 `data/`。


### 查看 Backend 日志

应用日志在 `data/logs/app.log`：`docker compose exec backend tail -f /app/logs/app.log`。内置 Elasticsearch 日志在 `data/logs/elasticsearch/`。


修改 `.env` 中的 `VITOOM_BACKEND_URL` / `VITOOM_WS_URL` 后，重启推理容器即可；entrypoint 会把 `data/inference/config/inference.yaml` 里的 `api_base_url` / `ws_url` 同步为 `.env` 的值。

如需**整文件**按 entrypoint 模板重写（含 `storage`、各服务 yaml 等），临时设置：

```env
VITOOM_OVERWRITE_CONFIG=1
```

然后重启对应推理服务：

```bash
docker compose -f docker-compose.inference.release.yml --profile visual up -d --force-recreate
```

确认配置已更新后，把 `.env` 中的 `VITOOM_OVERWRITE_CONFIG` 改回：

```env
VITOOM_OVERWRITE_CONFIG=0
```

## 9. 分布式部署约定

Backend 是控制面；Visual、Text、Audio、Mini、Download 都可以是独立推理节点，可以分别部署在不同 GPU 服务器上。

推理机器上的 `.env` 至少要能访问 Backend：

```env
VITOOM_BACKEND_URL=http://BACKEND_IP:8888
VITOOM_WS_URL=ws://BACKEND_IP:8888
VITOOM_INFERENCE_UPLOAD_AUTH_SECRET=和 Backend 一致的密钥
```


## 10. 停止和升级

停止 Backend：

```bash
docker compose down
```

停止推理服务：

```bash
docker compose -f docker-compose.inference.release.yml --profile visual down
docker compose -f docker-compose.inference.release.yml --profile text down
docker compose -f docker-compose.inference.release.yml --profile audio down
docker compose -f docker-compose.inference.release.yml --profile mini down
```

升级 Backend：

```bash
python scripts/load_vitoom_images.py --components backend --force
docker compose up -d backend
```

升级推理服务：

```bash
python scripts/load_vitoom_images.py --components visual --force
docker compose -f docker-compose.inference.release.yml --profile visual up -d --force-recreate
```

升级前建议备份：

```text
data/
resources/
```

## 11. 常见排查

Backend 是否正常：

```bash
curl http://127.0.0.1:8888/api/health
docker compose logs --tail=200 backend
```

推理服务是否启动：

```bash
docker compose -f docker-compose.inference.release.yml ps
docker compose -f docker-compose.inference.release.yml logs --tail=200 visual
```

确认推理配置是否仍是旧地址：

```bash
cat data/inference/config/inference.yaml
```

如果仍是旧地址，设置 `VITOOM_OVERWRITE_CONFIG=1` 后重启推理容器。

确认镜像 tag 是否和 `.env` 一致：

```bash
docker images | grep vitoom-inference
```

## 12. 使用小提示

以下路径以 Docker 部署为准：Backend 配置在 `data/config/`（首次启动会从镜像复制 `default.yaml`、`tts_speakers.json` 等默认文件，可再建 `app.yaml` 覆盖）；推理配置在 `data/inference/config/`。宿主机直接跑 Backend 时，对应项目根目录的 `config/`。

修改 YAML 后需**重启对应服务**；改 `.env` 后需 **`docker compose up -d` 重建容器**。推理全局/服务配置若被 entrypoint 写过且未生效，可临时设 `VITOOM_OVERWRITE_CONFIG=1` 后重启推理容器（见 §9）。

### 13.1 调节文本大模型显存（`gpu_memory_utilization`）

该参数控制 **vLLM 文本服务**预占 GPU 显存比例，取值 `(0, 1]`，越大占用越多。仅对 `config.runtime.backend: vllm` 的文本服务有效。

**Docker（推荐改持久化文件）**

编辑文本服务配置（首次启动 Text 后生成）：

```text
data/inference/config/text.yaml
```

在 `config.runtime.vllm` 下调整，例如：

```yaml
config:
  runtime:
    vllm:
      gpu_memory_utilization: 0.75
```

- 显存紧张：适当**调低**（如 `0.5`～`0.7`）。
- 换更大上下文或更大权重：可能需要**调高**，并同步检查 `max_model_len`。
- 首次生成 `text.yaml` 时，entrypoint 会按「约 14GiB / 当前 GPU 总显存」自动算一个比例；**之后不会自动改**，换卡或换模型需手动改。

改完后重启 Text 推理容器：

```bash
docker compose -f docker-compose.inference.release.yml --profile text restart
```

**Web 管理端**：登录管理员 → 推理服务管理 → 选中文本服务（如 `text`）→ 服务配置，可改 `config.runtime.vllm.gpu_memory_utilization`（保存后按页面提示重启服务）。

**宿主机开发**：改 `inference/config/ex_text.yaml` 或本地 `inference/config/text.yaml` 中同一路径，重启文本推理进程。

### 13.2 将存储改为 S3

Backend 任务/上传产物的落盘方式由 **`storage.default`** 决定（`server` | `s3` | `oss`）。

**1）Backend**

在 `data/config/app.yaml`（没有则新建）写入，例如：

```yaml
storage:
  default: s3
  s3:
    endpoint: "https://s3.amazonaws.com"   # 或 MinIO / 兼容 S3 的 endpoint
    region: "ap-southeast-1"
    bucket: "your-bucket"
    access_key_id: "YOUR_ACCESS_KEY"
    secret_access_key: "YOUR_SECRET_KEY"
    public_base_url: "https://your-bucket.s3.ap-southeast-1.amazonaws.com"
```

`public_base_url` 用于生成对外可访问的文件 URL，需与桶的公网访问方式一致。

重启 Backend：

```bash
docker compose up -d backend
```

**2）推理侧**

还需编辑 `data/inference/config/inference.yaml` 的 `storage` 段（`default: s3` 及 `storage.s3` 密钥与 Backend 侧一致或按桶策略单独配置），并重启对应推理容器。


### 13.3 更换默认文本大模型


**更换对话默认模型**

在 `data/config/app.yaml` 中设置（且 Text 推理服务已启动、权重在 `resources/models` 下可用）：

```yaml
agents:
  default_model: "你的模型名"
```

修改后重启 Backend。新会话或新任务即生效；已打开的会话若绑定了旧 `load_name`，需在 Web 里换模型或新建会话。

**固定文本推理服务只跑一个模型**

在 `text.yaml` 中取消注释并填写：

```yaml
config:
  fixed_model: "你的模型名"
```
具体可参考：inference/config/ex_gemma_text.yaml和inference/config/ex_text.yaml

保存后重启 Text 容器；同时建议按 §12.1 重新评估 `gpu_memory_utilization`。

### 13.4 更换默认视频生成模型


在 `data/config/app.yaml` 中设置：

```yaml
agents:
  tools:
    video_generator:
      default_model_name: "TurboWan2.1-T2V-1.3B-480P"
```

模型名须与系统中已注册的视频模型 **模型名** 一致，且 Visual/Video 推理服务已部署、权重路径正确（默认在 `resources/models`）。

### 13.5 支持模型范围
视频模型：Wan系列、TurboWan系列
音频模型：Qwen-tts、Qwen-asr、VoxCPM
图片模型：SDXL、Qwen-Image、Z-Image、Flux、Flux.2等所有主流图片生成模型
语言模型：Qwen系列

### 13.6 启用实时联网搜索
在　https://www.tavily.com/　申请一个api key（额度内免费），然后修改.env中的TAVILY_API_KEY

### 13.7 缓存模型以加速推理
新增或修改data/inference/config/{image/video/text/qwen_asr/qwen_tts}.yaml，将pipeline_cache_ttl_seconds修改为大于0的值，可以让对应推理服务缓存模型，从而大大加速下一次推理。（注意：缓存这些模型将会占用显存而不会释放，直到超时才会释放显存。）

你也可以修改data/inference/config/inference.yaml里这个配置项的值，这将对所有推理服务生效。

```yaml
pipeline_cache_ttl_seconds: 1800
```