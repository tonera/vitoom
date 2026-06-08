# Vitoom

[English](README.md) | **中文** | [日本語](README_JP.md)

Vitoom 是一套本地部署的 **AIGC 应用平台**：通过浏览器访问，可在本地电脑（DGX Spark/RTX Spark/RTX30,40,50系列显卡的个人电脑）上运行文本、图像、音频、视频等推理能力，并内置 **AI Agent** 统一调度写作、翻译、文档处理、知识库检索与多模态生成等任务。适合个人创作与局域网团队共享。如果你的工作内容要求私密性非常强，你的资料绝对不能公开，也不能使用云端大模型，那么本应用就是为你量身定做。

![Vitoom 应用界面截图](assets/shot.jpg)

## 主要用途

| 场景 | 说明 |
| --- | --- |
| 写作与办公 | 撰写文档、工作报告、总结；辅助文案构思；将对话内容导出为 Markdown / PDF |
| 知识库 | 归档 PDF、Word、PPT 等资料，语义检索与问答，持续积累专属知识库 |
| 语音与有声内容 | 文字转语音（多音色、声线设计、声音克隆）；多角色对白 / 广播剧式配音；语音听写转文字 |
| 图像与视频 | 文生图（支持主流开源模型）、图生图编辑；图片理解问答；文生视频 / 图生视频 |
| 文档与 OCR | 网页 / PDF / Office 链接总结与转换；扫描件 OCR（含表格、公式）；表格导出 Excel |
| 翻译 | 长文本多语言翻译；支持图片内文字翻译 |
| 智能检索 | 可选联网搜索（需配置 Tavily API Key） |


## 环境要求

- **Docker** 与 **Docker Compose**（`docker compose` 子命令）
- **推理环境**：**NVIDIA GPU**、**支持 CUDA 13.0 的 NVIDIA 驱动**（与 `cu130` 推理镜像一致；可用 `nvidia-smi` 查看驱动/CUDA 版本）及 **NVIDIA Container Toolkit**（Linux 原生或 Windows 下 Docker Desktop + WSL2 后端）
- 运行安装脚本时需 **Python 3.10+**（仅用于 `scripts/` 下的配置与模型下载，不要求本机安装完整推理环境）
- 拉取 Docker 镜像 / 模型权重时需能访问对应源（安装向导选 **中国大陆** 时会优先使用国内镜像与 ModelScope；选 **其他地区** 则主要使用 Docker Hub / Hugging Face）

**平台说明**

| 平台 | 说明 |
| --- | --- |
| Linux | 推荐原生 Docker；在仓库根目录执行下列命令即可 |
| Windows | 使用 [Docker Desktop](https://www.docker.com/products/docker-desktop/) + **WSL2**，并在 Docker 设置中开启 GPU、允许项目所在磁盘 **File Sharing**；**Python 脚本与 `docker compose` 请在同一环境执行**（全程用 WSL2 终端，或全程用 PowerShell，避免混用导致路径不一致） |

验证 GPU 与 CUDA 13.0 运行时（可选，需已安装 Docker 且 GPU 透传可用）：

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

## 快速安装

在**仓库根目录**依次执行（交互式向导会生成 `.env`、检测 CPU 架构 `x86_64` / `aarch64`，并为推理服务写入局域网地址；**勿将 `VITOOM_BACKEND_URL` 设为 `127.0.0.1`**，否则容器内推理无法连上 Backend）：

**1. 配置环境**

```bash
python scripts/setup_vitoom.py
```

Windows 若未将 `python` 加入 PATH，可在 PowerShell 使用：`py -3 scripts/setup_vitoom.py`（其余脚本同理）。

**2. 获取镜像**

```bash
python scripts/load_vitoom_images.py
```

优先从 `images/<架构>/` 离线加载 tar，不存在则从 Docker Hub 拉取。仅获取部分组件示例：

```bash
python scripts/load_vitoom_images.py --components backend,visual,text
```

**3. 启动服务**

须**先启动 Backend**（会创建 Docker 网络 `vitoom-net`），再启动推理容器：

```bash
docker compose up -d backend
```

按安装向导勾选的组件启动推理 profile（下面为常用全套，**请写成一行**，避免 Windows CMD 不支持 `\` 续行）：

```bash
docker compose -f docker-compose.inference.release.yml --profile visual --profile text --profile audio --profile mini --profile download up -d
```

仅启动部分服务时，保留需要的 `--profile` 即可，例如只要图像与文本：

```bash
docker compose -f docker-compose.inference.release.yml --profile visual --profile text up -d
```

浏览器访问：`http://<本机局域网IP>:8888`（IP 与端口以 `.env` 中 `VITOOM_BACKEND_URL` / `VITOOM_SERVER_PORT` 为准；本机调试时浏览器可用 `127.0.0.1`，但 `.env` 里仍应使用局域网 IP）。

**4. 下载模型（可选，体积较大）**

```bash
python scripts/download_initial_models.py
```

也可稍后在 Web 端 **模型管理** 页面按需下载（需已启动 `download` profile）。首次体验建议至少启动 Backend 与 Text，并下载对应大语言模型。

更完整的 Docker 部署说明见 [`docker-usage-cn.md`](docker-usage-cn.md)（[English](docker-usage-en.md) / [日本語](docker-usage-jp.md)）。

## 使用方法

1. **登录**：在浏览器打开上述地址；首次部署后默认管理员为 `admin@vitoom.ai`，密码见 `.env` 中的 `DEFAULT_ADMIN_PASSWORD`（运行 `setup_vitoom.py` 时若仍为示例占位符 `admin123456`，向导会**自动生成 10 位随机密码**并写入 `.env`）。另可在宿主机执行  
   `python scripts/create_user.py -e you@example.com -p your_password`  
   或由管理员在 Web 端用户管理中添加用户。
2. **智能助手**：进入 Agent 对话，用自然语言完成写作、翻译、文档转换、知识库查询、生成图片/音频/视频等（系统自动选择工具）。
3. **专业工作台**：通过首页进入 **图像生成**、**视频生成**、**音频**（ASR/TTS）、**翻译** 等页面，使用表单提交任务。
4. **模型管理**：在模型列表中下载、激活本地权重；需已启动 `download` 推理 profile 或完成步骤 4 的脚本下载。
5. **知识库**：将文件或对话归档入库后，在 Agent 中提问即可检索已入库资料。
6. **联网搜索（可选）**：在 `.env` 中配置 `TAVILY_API_KEY` 后，Agent 可检索公开网页信息（参见 [Tavily](https://www.tavily.com/)）。

推理服务首次启动可能较慢（加载权重）。查看日志：

```bash
docker compose logs -f backend
docker compose -f docker-compose.inference.release.yml logs -f visual
```

## 相关文档

| 文档 | 说明 |
| --- | --- |
| [`docker-usage-cn.md`](docker-usage-cn.md) | Docker 部署、profile、数据目录与排错 |
| [`docker-usage-en.md`](docker-usage-en.md) | 英文版 Docker 指南 |
| [`docker-usage-jp.md`](docker-usage-jp.md) | 日文版 Docker 指南 |

## 特别感谢

- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) — 视频推理极速方案
- [Nunchaku](https://github.com/nunchaku-ai/nunchaku) — 图片推理加速
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — 语音合成
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-asr) — 语音识别
- [VoxCPM](https://voxcpm.readthedocs.io/) — 高速语音合成
- [vLLM](https://github.com/vllm-project/vllm) — 文本推理引擎
- [RMBG-2.0](https://github.com/Bria-AI/RMBG-2.0) — 图片去背景
- [MeanCache](https://github.com/UnicomAI/MeanCache) — 图片推理加速


## 许可证

本项目采用 [GNU Affero General Public License v3.0](LICENSE)（AGPL-3.0）。商业授权说明见 [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)（如有）。

各推理模型与第三方组件遵循其上游许可证，使用前请自行确认合规性。
