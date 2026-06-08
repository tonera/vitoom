# 推理器公共模块

本目录包含所有推理器（image、text、video、audio）共用的功能模块。

## 模块说明

### 1. `logger.py` - 日志模块
提供统一的日志配置，支持控制台和文件输出。

```python
from common.logger import get_logger

logger = get_logger(__name__, service_id="service_123")
logger.info("Hello, world!")
```

### 2. `config_loader.py` - 配置加载模块
读取 `inference/config/{service_id}.yaml` 启动配置，并读取 `inference/config/inference.yaml` 全局配置。
两者会进行合并使用，且 `{service_id}.yaml` 会覆盖 `inference.yaml` 的同名配置（dict 递归合并）。

```python
from common.config_loader import load_startup_config

config = load_startup_config("service_123")
print(config.service_id, config.service_type, config.port)
```

### 3. `system_monitor.py` - 系统监控模块
获取系统信息：GPU内存、系统负载、内存使用等。

```python
from common.system_monitor import SystemMonitor

monitor = SystemMonitor()
info = monitor.get_all_info()
print(info["gpu"], info["memory"], info["system_load"])
```

### 4. `api_client.py` - API客户端模块
调用后端API上报推理器状态。

```python
from common.api_client import APIClient

api_client = APIClient("http://127.0.0.1:8000")
await api_client.notify_start(
    service_id="service_123",
    host="127.0.0.1",
    port=8001,
    config={"gpu_available_memory": 1000000000}
)
```

### 5. `ws_client.py` - WebSocket 客户端模块（WS 传输实现）
连接 WS Server，发送心跳，接收消息（`task/cancel/ping`），并发送 `result/task_status`。

```python
from common.ws_client import WebSocketClient
from common.message_queue import MessageQueue
from common.message_cache import MessageCache

message_queue = MessageQueue()
message_cache = MessageCache("resources/cache/messages")
ws_client = WebSocketClient(
    ws_url="ws://127.0.0.1:8000",
    message_queue=message_queue,
    service_id="service_123",
    message_cache=message_cache
)
await ws_client.connect()
```

**消息协议（task/cancel/result/task_status）**详见下方“消息协议”章节。

> 提示：当前项目已支持 `Redis List` 队列作为另一种传输方式（见下文“传输层（Ingress/Egress）”）。

### 6. `message_queue.py` - 消息队列模块
基于 `queue.Queue` 的消息队列，存储来自各类 Ingress（WS/Redis…）的消息。

```python
from common.message_queue import MessageQueue

message_queue = MessageQueue(maxsize=1000)
message_queue.put_task_message("task_123", {"id": "task_123", "type": "image", "params": {"job_type": "MK"}})
message = message_queue.get(timeout=1.0)  # {"type":"task","task_id":...,"task_data":...}
```

### 7. `db_client.py` - 数据库访问模块（已废弃）
推理器侧**不再直连数据库**。任务与文件记录由上游/后端根据 `result/task_status` 消息创建与更新。

### 8. `task_processor.py` - 任务处理模块
从消息队列读取任务（全量数据），验证 `task.type` 与推理器 `service_type`，发送 `task_status`，并将任务映射为 `InferenceRequestParams` 后调用 `inference_callback`。

```python
from common.task_processor import TaskProcessor

async def inference_callback(params):
    # 执行实际推理
    pass

task_processor = TaskProcessor(
    message_queue=message_queue,
    message_cache=message_cache,
    ws_client=ws_client,  # 这里的形状是 egress：需实现 send_task_status(...)
    service_type="image",
    inference_callback=inference_callback
)
await task_processor.process_loop()
```

### 9. `signal_handler.py` - 信号处理模块
实现优雅退出逻辑，确保推理器停止时调用 stop API。

```python
from common.signal_handler import SignalHandler

async def cleanup():
    await inferrer.stop()

signal_handler = SignalHandler(cleanup_callback=cleanup)
signal_handler.register()
```

### 10. `base_inferrer.py` - 推理器基类
抽象所有公共功能，子类只需实现 `inference_callback` 方法。

```python
from common.base_inferrer import BaseInferrer
from schemas import InferenceRequestParams

class MyInferrer(BaseInferrer):
    async def inference_callback(self, params: InferenceRequestParams):
        # 实现推理逻辑
        pass

inferrer = MyInferrer(service_id="service_123")
await inferrer.run()
```

### 11. `result_handler.py` - 通用结果处理模块
处理图片、视频、音频等文件的存储、缩略图生成、上传（local/s3/oss/server），以及发送 `result` 消息（通过 egress）。

```python
from common.result_handler import ResultHandler
from schemas import InferenceRequestParams

result_handler = ResultHandler(
    ws_client=ws_client,  # 这里的形状是 egress：需实现 send_result(...)
    storage_base_path="resources/outputs"
)

# 逐张回传（多图/长任务推荐）
# response_params = await result_handler.process_single_result(...)
```

**主要功能**：
- **文件存储**：支持图片、视频、音频等多种文件类型的本地存储
- **缩略图生成**：
  - 图片：自动生成缩略图（原文件名后加 `_s`）
  - 视频：提取第10帧或使用参考URL生成缩略图
  - 音频：不生成缩略图
- **消息回传**：自动发送 `result` 消息到上游（WS / Redis list 等）

**待实现功能**：
- `generate_video_thumbnail()` 中的参考URL下载逻辑（TODO）

## 使用示例

完整的使用示例请参考 `main_template.py`。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 传输层（Ingress/Egress）

当前推理器将“消息从哪里来/发到哪里去”抽象为可配置的 Ingress/Egress（默认仍为 WS）。

配置位置：`inference/config/{service_id}.yaml`（字段名：`transport`）

- **默认（WS）**：不配置 `transport`，或显式配置 `ws`
- **Redis List 队列**：配置 `redis_list`，使用 `BRPOP channel` 拉取、`LPUSH/RPUSH reschannle` 回传

示例（Redis List）：

```yaml
transport:
  ingresses:
    - type: "redis_list"
      redis:
        host: "127.0.0.1"
        port: 6379
        pwd: ""
        channel: "atz.req"
        brpop_timeout: 5
  egresses:
    - type: "redis_list"
      redis:
        host: "127.0.0.1"
        port: 6379
        pwd: ""
        reschannle: "atz.res"
        push: "lpush"
```

> 说明：Redis List 模式是“最多一次”语义（无 ack，允许进程崩溃导致任务丢失），以兼容对方项目。

## 消息协议（WS/Redis List 通用）

无论采用 WS 还是 Redis List，核心业务消息（`task/cancel/result/task_status`）均为 JSON 格式，结构一致。

差异点：
- **WS**：支持 `ping/pong/heartbeat`，以及断线时的本地消息缓存重发
- **Redis List**：不使用心跳帧；本项目当前不对 Redis 回传做落盘缓存（失败只记录日志）

### 消息方向说明

- **接收消息**：推理器从 WebSocket Server 接收的消息
- **发送消息**：推理器发送到 WebSocket Server 的消息

---

### 1. 接收消息（推理器 ← WebSocket Server）

#### 1.1 任务消息 (`type: "task"`)

WebSocket Server 向推理器推送新任务，包含全量任务数据和模型信息。

**消息格式**：
```json
{
    "type": "task",
    "task_id": "task_123",
    "task_data": {
        "id": "task_123",
        "user_id": "user_456",
        "type": "image",
        "status": "pending",
        "prompt": "A beautiful landscape",
        "params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "seed": 42
        },
        "progress": 0,
        "error": null,
        "priority": 5,
        "model_name": "stable-diffusion-xl",
        "storage": "local",
        "created_at": "2025-01-15T10:30:00Z",
        "started_at": null,
        "completed_at": null,
        "model": {
            "name": "stable-diffusion-xl",
            "type": "image",
            "load_name": "stable-diffusion-xl",
            "family": "sdxl",
            "runtime_config": {},
            "status": "active"
        }
    }
}
```

**字段说明**：
- `type`: 固定为 `"task"`
- `task_id`: 任务ID
- `task_data`: 全量任务数据，包含：
  - 任务基本信息（id, user_id, type, status等）
  - 任务参数（prompt, params等）
  - 模型信息（使用 model_name 标识模型；model 对象为可选附加信息）

**处理流程**：
1. 将消息放入消息队列
2. 异步写入缓存文件（`task_{task_id}_{timestamp}.json`）
3. 任务处理器从队列读取并处理

---

#### 1.2 取消消息 (`type: "cancel"`)

WebSocket Server 通知推理器取消某个任务。

**消息格式**：
```json
{
    "type": "cancel",
    "task_id": "task_123",
    "timestamp": "2025-01-15T10:35:00Z"
}
```

**字段说明**：
- `type`: 固定为 `"cancel"`
- `task_id`: 要取消的任务ID
- `timestamp`: 取消时间戳

**处理流程**：
1. 标记任务为已取消
2. 删除对应的缓存文件
3. 如果任务正在处理，推理函数会检查取消状态并中止

---

#### 1.3 心跳消息 (`type: "ping"`)

WebSocket Server 发送心跳，推理器需要回复 `pong`。

**消息格式**：
```json
{
    "type": "ping",
    "timestamp": "2025-01-15T10:30:00Z"
}
```

**响应**：推理器自动回复 `pong` 消息（见下方发送消息章节）

---

### 2. 发送消息（推理器 → WebSocket Server）

#### 2.1 推理结果消息 (`type: "result"`)

推理器完成任务后，发送推理结果和文件信息。

**消息格式**：
```json
{
    "type": "result",
    "queue_length": 0,
    "service_id": "service_123",
    "task_id": "task_123",
    "user_id": "user_456",
    "task_type":"image",
    "job_type": "MK",
    "status": "completed",
    "progress": 100,
    "storage": "local",
    "seed": 42,
    "model_name": "stable-diffusion-xl",
    "reference_id": "",
    "duration": 0,
    "generate_time": 10.5,
    "upload_time": 0.2,
    "used_time": 10.7,
    "total": 2,
    "files": [
        {
            "file_id": "file_uuid_1",
            "storage_path": "resources/outputs/images/2025/01/15/task_123_0.png",
            "file_name": "task_123_0.png",
            "file_size": 1048576,
            "mime_type": "image/png",
            "seed": 42,
            "index": 0,
            "thumbnail_path": "resources/outputs/images/2025/01/15/task_123_0_s.png",
            "width": 1024,
            "height": 1024
        },
        {
            "file_id": "file_uuid_2",
            "storage_path": "resources/outputs/images/2025/01/15/task_123_1.png",
            "file_name": "task_123_1.png",
            "file_size": 1048576,
            "mime_type": "image/png",
            "seed": 1234567890,
            "index": 1,
            "thumbnail_path": "resources/outputs/images/2025/01/15/task_123_1_s.png",
            "width": 1024,
            "height": 1024
        }
    ]
}
```

**字段说明**：
- `type`: 固定为 `"result"`
- `queue_length`: 推理器当前队列长度
- `service_id`: 推理器服务ID
- `task_id`: 任务ID
- `user_id`: 用户ID
- `job_type`: 任务类型（image/video/audio/text）
- `status`: 任务状态（通常为 `"completed"`）
- `progress`: 进度（0-100）
- `storage`: 任务产物存储目标（local/server/s3/oss）
- `seed`: 随机种子
- `model_name`: 模型名称
- `reference_id`: 参考ID
- `duration`: 音视频时长（仅video/audio）
- `generate_time`: 文件生成耗时（秒）
- `upload_time`: 文件存储耗时（秒）
- `used_time`: 总耗时（秒）
- `total`: 文件总数（当前任务生成的文件数量）
- `files`: 文件信息数组，每个文件包含：
  - `file_id`: 文件ID（UUID）
  - `storage_path`: 存储路径（相对路径）
  - `file_name`: 文件名
  - `file_size`: 文件大小（字节）
  - `mime_type`: MIME类型
  - `seed`: 当前文件使用的随机种子（可选，多文件时每个文件可能有不同的seed）
  - `index`: 当前文件在任务中的序号（从0开始，用于前端UI展示）
  - `thumbnail_path`: 缩略图路径（可选）
  - `http_url`: HTTP URL（可选）
  - `width`: 图片/视频宽度（可选）
  - `height`: 图片/视频高度（可选）

**后端处理**：
- WebSocket Server 收到此消息后，会根据 `files` 数组创建数据库文件记录
- 更新任务状态为 `completed`

---

#### 2.2 任务状态更新消息 (`type: "task_status"`)

推理器发送任务状态更新（processing/completed/failed/cancelled）。

**消息格式**：
```json
{
    "type": "task_status",
    "task_id": "task_123",
    "status": "processing",
    "timestamp": "2025-01-15T10:30:00Z",
    "started_at": "2025-01-15T10:30:00Z"
}
```

**状态类型**：
- `queued`: 任务已入队列
- `processing`: 任务开始处理
- `completed`: 任务完成（通常与 `result` 消息一起发送）
- `failed`: 任务失败
- `cancelled`: 任务已取消

**不同状态的字段**：

**processing 状态**：
```json
{
    "type": "task_status",
    "task_id": "task_123",
    "status": "processing",
    "timestamp": "2025-01-15T10:30:00Z",
    "started_at": "2025-01-15T10:30:00Z"
}
```

**completed 状态**：
```json
{
    "type": "task_status",
    "task_id": "task_123",
    "status": "completed",
    "timestamp": "2025-01-15T10:35:00Z",
    "completed_at": "2025-01-15T10:35:00Z"
}
```

**failed 状态**：
```json
{
    "type": "task_status",
    "task_id": "task_123",
    "status": "failed",
    "timestamp": "2025-01-15T10:32:00Z",
    "error": "Model loading failed: ..."
}
```

**cancelled 状态**：
```json
{
    "type": "task_status",
    "task_id": "task_123",
    "status": "cancelled",
    "timestamp": "2025-01-15T10:33:00Z"
}
```

**字段说明**：
- `type`: 固定为 `"task_status"`
- `task_id`: 任务ID
- `status`: 任务状态
- `timestamp`: 时间戳
- `started_at`: 开始时间（processing状态）
- `completed_at`: 完成时间（completed状态）
- `error`: 错误信息（failed状态）

**后端处理**：
- WebSocket Server 收到此消息后，会更新数据库中的任务状态

---

#### 2.3 心跳消息 (`type: "heartbeat"`)

推理器主动发送心跳，保持连接活跃。

**消息格式**：
```json
{
    "type": "heartbeat",
    "timestamp": "2025-01-15T10:30:00Z"
}
```

**字段说明**：
- `type`: 固定为 `"heartbeat"`
- `timestamp`: 当前时间戳

**发送频率**：每30秒发送一次

---

#### 2.4 心跳响应 (`type: "pong"`)

推理器响应服务器的 `ping` 消息。

**消息格式**：
```json
{
    "type": "pong",
    "timestamp": "2025-01-15T10:30:00Z"
}
```

**字段说明**：
- `type`: 固定为 `"pong"`
- `timestamp`: 当前时间戳

---

### 3. 消息缓存机制

当 **WS 连接断开** 时，消息会被缓存到本地文件系统（用于 WS 断线重发）：

**缓存文件命名规则**：
- 任务消息：`task_{task_id}_{unix_timestamp}.json`
- 状态结果：`res_{task_id}_{status}_{unix_timestamp}.json`

**缓存目录**：`resources/cache/messages/`

**恢复机制**：
- 任务处理器在队列无消息时，每30秒扫描缓存目录
- 发现缓存文件后，自动处理并删除文件
- 状态结果文件会尝试重新发送到 WebSocket Server

---

### 4. 消息发送失败处理

当 **WS 连接断开或发送失败** 时：

1. **推理结果消息**：
   - 写入状态结果缓存文件：`res_{task_id}_completed_{timestamp}.json`
   - 后续扫描时重新发送

2. **任务状态更新消息**：
   - 写入状态结果缓存文件：`res_{task_id}_{status}_{timestamp}.json`
   - 后续扫描时重新发送

---

### 5. 消息流程图

```
Ingress（WS/Redis）                推理器
     |                                |
     |---- task (全量数据) -------->  |
     |                                | 1. 放入消息队列
     |                                | 2. （WS）可选写入缓存文件
     |                                |
     |<--- task_status (queued) ------|
     |<--- task_status (processing) --|
     |                                |
     |                                | 3. 处理任务
     |                                | 4. 执行推理
     |                                |
     |<--- result (文件信息) ---------|
     |<--- task_status (completed) ---|
     |                                |
     |---- cancel ------------------> |
     |                                | 5. 标记取消
```

---

## 启动配置文件格式

启动配置文件位于 `inference/config/{service_id}.yaml`，格式请参考 `inference/config/example.yaml`。

