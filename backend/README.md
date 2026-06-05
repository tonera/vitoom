# Backend API README

本文档面向前端/调用方，描述本项目后端（FastAPI）提供的 **HTTP API / WebSocket / 静态资源** 访问方式与统一规范。

---

## API 统一规范

### 基础约定

- **Base URL**：开发环境通常为 `http://127.0.0.1:8888`
- **认证**：多数接口需要 Bearer Token
- **统一返回结构（成功/失败）**：

```json
{
  "code": 1,
  "data": {},
  "msg": "ok"
}
```

- **成功 code**：`code = 1`
- **失败 code**：`code != 1`，通常为 `backend/core/error_codes.py::ErrorCode` 的值（例如 1000+）

### 认证方式

HTTP Header：

```http
Authorization: Bearer <access_token>
```

### 错误返回示例

所有异常（包含 404/401/422/500 等）会被统一包装为 `{code,data,msg}`：

```json
{
  "code": 3000,
  "data": {
    "type": "user_error",
    "details": {
      "validation_errors": []
    },
    "http_status": 422
  },
  "msg": "Validation error for body.xxx: field required"
}
```

### 分页约定（Models 列表）

`GET /api/models` 使用 `limit/offset` 查询参数，并在响应顶层返回 `meta`：

```json
{
  "code": 1,
  "data": [],
  "msg": "ok",
  "meta": {
    "current_page": 1,
    "from": 1,
    "last_page": 3,
    "per_page": 24,
    "to": 24,
    "total": 68
  }
}
```

> 注意：`meta.total` 是在当前筛选条件下的 **总数**，不是当前页数量。

---

## API 列表（HTTP）

### System

#### GET `/api/health`

- **说明**：健康检查
- **认证**：不需要

返回示例：

```json
{
  "code": 1,
  "data": {
    "status": "healthy",
    "service": "Vitoom API",
    "version": "1.0.0"
  },
  "msg": "ok"
}
```

---

### Authentication（`/api/auth`）

#### POST `/api/auth/register`

- **认证**：不需要
- **Body**：
  - `email` (string, required)
  - `password` (string, required)
  - `nickname` (string, optional)

返回示例：

```json
{
  "code": 1,
  "data": {
    "id": "uuid",
    "email": "user@example.com",
    "nickname": "nick",
    "status": "active",
    "is_admin": false,
    "created_at": "2026-01-01T00:00:00"
  },
  "msg": "registered"
}
```

#### POST `/api/auth/login`

- **认证**：不需要
- **Body**：
  - `email` (string, required)
  - `password` (string, required)

返回示例：

```json
{
  "code": 1,
  "data": {
    "access_token": "jwt",
    "refresh_token": "jwt",
    "token_type": "bearer"
  },
  "msg": "ok"
}
```

#### POST `/api/auth/refresh`

- **认证**：不需要
- **Body**：
  - `refresh_token` (string, required)

返回示例：

```json
{
  "code": 1,
  "data": {
    "access_token": "jwt",
    "token_type": "bearer"
  },
  "msg": "ok"
}
```

#### GET `/api/auth/me`

- **认证**：需要

返回示例：

```json
{
  "code": 1,
  "data": {
    "id": "uuid",
    "email": "user@example.com",
    "nickname": "nick",
    "status": "active",
    "is_admin": false,
    "created_at": "2026-01-01T00:00:00"
  },
  "msg": "ok"
}
```

#### POST `/api/auth/logout`

- **认证**：需要

返回示例：

```json
{
  "code": 1,
  "data": {
    "user_id": "uuid"
  },
  "msg": "logged_out"
}
```

---

### Models（`/api/models`）

#### GET `/api/models`

- **认证**：需要
- **Query**：
  - `type` (string, optional)：`image|video|audio|text`
  - `storage_type` (string, optional)：`local|cloud`
  - `status` (string, optional)：`active|inactive`
  - `limit` (int, optional)：默认 100
  - `offset` (int, optional)：默认 0

返回示例（含分页 meta）：

```json
{
  "code": 1,
  "data": [
    {
      "id": "model_id",
      "name": "NoobAI-XL-Vpred",
      "type": "image",
      "storage_type": "local",
      "status": "active",
      "model_class": "sdxl",
      "ck_type": "checkpoint",
      "thumb": "outputs/models/xxx.png",
      "is_local_model": true
    }
  ],
  "msg": "ok",
  "meta": {
    "current_page": 1,
    "from": 1,
    "last_page": 2,
    "per_page": 10,
    "to": 10,
    "total": 15
  }
}
```

#### GET `/api/models/{model_id}`

- **认证**：需要

返回示例：

```json
{
  "code": 1,
  "data": {
    "id": "model_id",
    "name": "xxx",
    "thumb": "outputs/models/xxx.png",
    "ck_type": "checkpoint"
  },
  "msg": "ok"
}
```

#### POST `/api/models`

- **认证**：需要
- **Body**（常用字段）：
  - `name` (string, required)
  - `type` (string, required)：`image|video|audio|text`
  - `storage_type` (string, required)：`local|cloud`
  - `is_local_model` (bool, required)：是否本地模型（用于推理器）
  - `model_class` (string, optional)
  - `ck_type` (string, optional)：`checkpoint|lora`，默认 `checkpoint`
  - `thumb` (string, optional)：缩略图路径/URL（建议使用 `/outputs/...` 或 `outputs/...`）

返回示例：

```json
{
  "code": 1,
  "data": { "id": "model_id" },
  "msg": "created"
}
```

#### PUT `/api/models/{model_id}`

- **认证**：需要
- **Body**：可更新字段（示例）
  - `name` (string, optional)
  - `version` (string, optional)
  - `description` (string, optional)
  - `thumb` (string, optional)
  - `ck_type` (string, optional)
  - `model_config` (object, optional)

返回示例：

```json
{
  "code": 1,
  "data": { "id": "model_id", "thumb": "outputs/models/xxx.png" },
  "msg": "updated"
}
```

#### PUT `/api/models/{model_id}/activate`

- **认证**：需要

返回示例：

```json
{ "code": 1, "data": { "id": "model_id", "status": "active" }, "msg": "activated" }
```

#### PUT `/api/models/{model_id}/deactivate`

- **认证**：需要

返回示例：

```json
{ "code": 1, "data": { "id": "model_id", "status": "inactive" }, "msg": "deactivated" }
```

#### DELETE `/api/models/{model_id}`

- **认证**：需要

返回示例：

```json
{ "code": 1, "data": { "model_id": "model_id" }, "msg": "deleted" }
```

#### POST `/api/models/{model_id}/download`

- **认证**：需要
- **Body**：
  - `download_url` (string, required)
  - `model_name` (string, required)
  - `model_type` (string, required)
  - `expected_size` (int, optional)
  - `expected_hash` (string, optional)

返回示例：

```json
{ "code": 1, "data": { "success": true }, "msg": "ok" }
```

---

### Tasks（`/v1`）

#### POST `/v1/tasks`

- **认证**：需要
- **Body**：统一任务创建请求（常用字段）
  - `task_type` (string, required)：`image|video|audio|text`
  - `prompt` (string, optional)
  - `model_id` (string, image 必需)
  - `storage` (string, optional)：默认 `local`
  - 其他：见 `backend/api/tasks/routes.py::TaskCreateRequest`

返回示例：

```json
{
  "code": 1,
  "data": { "task_id": "uuid", "status": "pending", "message": "Task created successfully" },
  "msg": "created"
}
```

#### GET `/v1/tasks/{task_id}`

- **认证**：需要

返回示例：

```json
{ "code": 1, "data": { "task_id": "uuid", "status": "running", "progress": 10 }, "msg": "ok" }
```

#### GET `/v1/tasks`

- **认证**：需要
- **Query**：
  - `status` (string, optional)
  - `limit` (int, optional)
  - `offset` (int, optional)

返回示例：

```json
{ "code": 1, "data": { "tasks": [], "total": 0 }, "msg": "ok" }
```

#### DELETE `/v1/tasks/{task_id}`

- **认证**：需要

返回示例：

```json
{ "code": 1, "data": { "task_id": "uuid" }, "msg": "cancelled" }
```

---

### Inference Services（`/api/inference`）

#### CRUD & 状态同步：`/api/inference/services`

- `POST /api/inference/services`：创建服务
- `GET /api/inference/services`：列出服务
- `GET /api/inference/services/{service_id}`：获取服务
- `PUT /api/inference/services/{service_id}`：更新服务
- `POST /api/inference/services/{service_id}/start`：推理器同步启动
- `POST /api/inference/services/{service_id}/stop`：推理器同步停止
- `DELETE /api/inference/services/{service_id}`：删除服务

均返回 `{code,data,msg}`。

#### 上传：POST `/api/inference/upload`

用于推理侧直传文件（multipart），返回 `{code,data,msg}`，其中 `data.key` 是相对存储 key。

---

### Agents（`/v1/agents`）

Agent 模板 + 一次性 Run 的调试接口。正式的多轮对话请使用下文 [Conversations](#conversationsv1conversations)。`scripts/agent_chatbox.py` 在 `--raw` 模式下使用这套接口。

#### GET `/v1/agents`

- **认证**：需要
- **说明**：列出可用的 Agent（含预设 preset 与用户创建）
- **Query**：
  - `status` (string, optional)
  - `is_preset` (bool, optional)

返回示例：

```json
{
  "code": 1,
  "data": {
    "agents": [
      {
        "id": "preset-master-agent",
        "name": "Master",
        "type": "general",
        "is_preset": true,
        "status": "active",
        "description": "Master routing agent"
      }
    ],
    "total": 1
  },
  "msg": "ok"
}
```

#### GET `/v1/agents/{agent_id}`

- **认证**：需要

返回单个 agent 的完整配置（含 role / tools / routing 等）。

#### POST `/v1/agents/runs`

- **认证**：需要
- **说明**：创建一次性 agent run（不绑定到会话）。状态异步流转，需要通过下面的 GET 接口轮询。
- **Body**：
  - `agent_id` (string, required)：agent 模板 ID（如 `preset-master-agent`）
  - `message` (string, required)：用户输入
  - `source_type` (string, optional)：`web|cli|conversation|channel_ingress`，默认 `web`
  - `source_ref` (string, optional)：来源侧引用 ID
  - `attachments` (array, optional)：附件列表（图片/文件等）
  - `context` (object, optional)：上下文字段（`conversation_id`、`history_turns` 等会被 Master 消费）
  - `runtime_config` (object, optional)：运行期覆盖配置

返回示例：

```json
{
  "code": 1,
  "data": {
    "run_id": "uuid",
    "task_id": "uuid",
    "status": "pending"
  },
  "msg": "created"
}
```

#### GET `/v1/agents/runs/{run_id}`

- **认证**：需要

返回示例：

```json
{
  "code": 1,
  "data": {
    "id": "run_uuid",
    "agent_id": "preset-master-agent",
    "task_id": "task_uuid",
    "status": "completed",
    "result": {
      "final_answer": "...",
      "steps": []
    },
    "created_at": "2026-01-01T00:00:00",
    "completed_at": "2026-01-01T00:00:05"
  },
  "msg": "ok"
}
```

`status` 可能取值：`created | queued | running | waiting_input | completed | failed | cancelled`。其中 `waiting_input` 为 Human-in-the-loop 预留状态，Worker 当前未实现挂起/恢复逻辑。

#### GET `/v1/agents/runs/{run_id}/events`

按时间顺序返回一次 AgentRun 的**执行步骤事件流**（`agent_run_events` 表），用于前端渲染 Agent "思考过程" 卡片 / 问题复盘 / 审计。

- **认证**：需要（权限会校验 run 属于当前用户）
- **Query**：
  - `limit` (int, optional, 默认 500，上限 2000)
  - `offset` (int, optional, 默认 0)
  - `order` (string, optional, 默认 `asc`；传 `desc` 则按 sequence 倒序)
  - `after_sequence` (int, optional)：仅返回 `sequence > after_sequence` 的事件，用于**增量轮询**。前端每次请求时把上一批响应里的 `last_sequence` 作为下次的 `after_sequence` 即可，避免重复传输已渲染的事件

常见 `event_type`：

| event_type | 说明 | 关键 payload |
|---|---|---|
| `run_started` | Worker 开始处理一次 run | `agent_id / source_type / conversation_id / message / runtime_config` |
| `tool_selected` | ToolSelectionService 完成候选筛选 | `declared / selected / pool / preferred / max_tools` |
| `tool_call_started` | 工具开始执行（由 ToolResolver 统一包装） | `provider / target_tool_name / args_preview` |
| `tool_call_completed` | 工具成功返回 | `provider / duration_ms / output_len / output_preview` |
| `tool_call_failed` | 工具执行异常 | `provider / duration_ms`；`content` 字段存错误消息 |
| `crew_tool_invoked` | Master Agent 调用了一个 Crew-as-Tool（除 tool_call_started/completed 外的专有语义事件） | `preset_id / child_run_id / query` |
| `run_completed` | run 正常结束 | `output_len / usage_metrics` |
| `run_failed` | run 异常结束 | `content` 为错误消息 |
| `run_cancelled` | 用户 cancel | - |

返回示例：

```json
{
  "code": 1,
  "data": {
    "run_id": "run_uuid",
    "events": [
      {
        "id": "evt_uuid",
        "agent_run_id": "run_uuid",
        "sequence": 1,
        "event_type": "run_started",
        "tool_name": null,
        "content": null,
        "payload": {
          "agent_id": "preset-master-agent",
          "source_type": "conversation",
          "conversation_id": "conv_uuid",
          "message": "帮我规划东京三日游",
          "runtime_config": { "priority": 5, "process": "sequential" }
        },
        "started_at": "2026-01-01T00:00:00",
        "completed_at": null,
        "created_at": "2026-01-01T00:00:00"
      },
      {
        "sequence": 2,
        "event_type": "tool_selected",
        "payload": {
          "declared": ["tavily_search", "travel_planner"],
          "selected": ["travel_planner"],
          "pool": "global",
          "preferred": [],
          "max_tools": 3
        }
      },
      {
        "sequence": 3,
        "event_type": "tool_call_started",
        "tool_name": "travel_planner",
        "payload": {
          "provider": "crew",
          "target_tool_name": "preset-travel-planner-agent",
          "args_preview": "规划东京三日游 ..."
        }
      },
      {
        "sequence": 4,
        "event_type": "crew_tool_invoked",
        "tool_name": "travel_planner",
        "payload": {
          "preset_id": "preset-travel-planner-agent",
          "child_run_id": "run_child_uuid",
          "query": "规划东京三日游"
        }
      },
      {
        "sequence": 5,
        "event_type": "tool_call_completed",
        "tool_name": "travel_planner",
        "payload": {
          "provider": "crew",
          "duration_ms": 8421,
          "output_len": 2048,
          "output_preview": "..."
        }
      },
      {
        "sequence": 6,
        "event_type": "run_completed",
        "payload": {
          "output_len": 2048,
          "usage_metrics": { "total_tokens": 1024, "tokens_per_second": 80.5 }
        }
      }
    ],
    "total": 6,
    "last_sequence": 6
  },
  "msg": "ok"
}
```

> 使用姿势（增量轮询 pseudocode）：
>
> ```js
> let cursor = 0;
> while (!terminalStatus) {
>   const { events, last_sequence } = await GET(`/v1/agents/runs/${run_id}/events?after_sequence=${cursor}`);
>   render(events);
>   if (last_sequence != null) cursor = last_sequence;
>   await sleep(500);
> }
> ```
>
> 同时轮询 `GET /v1/agents/runs/{run_id}` 判断终态（`status in {completed, failed, cancelled}`）。子 run（crew-as-tool 派生）有独立事件流，通过 `crew_tool_invoked.payload.child_run_id` 导航进入；子 run 的 `run_completed` 只会出现在子 run 自己的 `/events` 里，不会混进父 run。

#### GET `/v1/agents/runs`

- **认证**：需要
- **Query**：
  - `status` (string, optional)
  - `limit` (int, optional, 默认 50)
  - `offset` (int, optional, 默认 0)

返回：`{ code:1, data: { runs: [...], total: N }, msg: "ok" }`。

#### POST `/v1/agents/runs/{run_id}/cancel`

- **认证**：需要

返回示例：

```json
{
  "code": 1,
  "data": { "run_id": "uuid", "task_id": "uuid", "status": "cancelled" },
  "msg": "cancelled"
}
```

---

### Chat Sessions（`/v1/chat/sessions`）

统一会话主入口。会话默认挂在 **Master Agent** 上，由 LLM 自行决定路由、工具调用与专家委派；`scripts/agent_chatbox.py` 默认使用这套接口。

#### POST `/v1/chat/sessions`

- **认证**：需要
- **Body**：
  - `agent_id` (string, optional)：[debug] 显式绑定某个 agent
  - `title` (string, optional)：会话标题
  - `input_mode` (string, optional)：默认 `text`
  - `output_mode` (string, optional)：默认 `text_stream`
  - `load_name` (string, optional)：本会话默认模型加载名
  - `metadata` (object, optional)

返回示例：

```json
{
  "code": 1,
  "data": {
    "id": "session_uuid",
    "user_id": "user_uuid",
    "agent_id": "preset-master-agent",
    "title": null,
    "status": "active",
    "metadata": {
      "input_mode": "text",
      "output_mode": "text_stream",
      "load_name": "Qwen/Qwen3-8B"
    },
    "created_at": "2026-01-01T00:00:00",
    "updated_at": "2026-01-01T00:00:00"
  },
  "msg": "created"
}
```

#### GET `/v1/chat/sessions/{session_id}`

- **认证**：需要

返回会话基础信息。

#### GET `/v1/chat/sessions/{session_id}/messages`

- **认证**：需要
- **Query**：`limit` (int, 默认 200)，`offset` (int, 默认 0)

返回：`{ code:1, data: { items: [...], count: N }, msg: "ok" }`。

消息 `role` 可能取值：`user | assistant | system | tool`。

---

### Uploads（`/v1/uploads`）

供用户前端直传图片/视频/音频/pdf/doc/docx/txt，供后续 task、conversation 或 session 作为 attachment / 多模态输入引用。

#### POST `/v1/uploads`

- **认证**：需要
- **Content-Type**：`multipart/form-data`，字段名 `file`
- **限制**：默认 50MB（由 `upload.max_size` 控制）

返回示例：

```json
{
  "code": 1,
  "data": {
    "id": "upload_uuid",
    "storage_path": "uploads/202601/xxx.jpeg",
    "url": "http://127.0.0.1:8888/outputs/uploads/202601/xxx.jpeg",
    "http_url": "http://127.0.0.1:8888/outputs/uploads/202601/xxx.jpeg",
    "file_name": "demo.jpg",
    "mime_type": "image/jpeg",
    "file_size": 12345
  },
  "msg": "uploaded"
}
```

#### GET `/v1/uploads`

- **认证**：需要
- **Query**：
  - `keyword` (string, optional)：按原始文件名模糊搜索
  - `limit` (int, 1~200, 默认 60)
  - `offset` (int, 默认 0)

返回：`{ code:1, data: { items: [...], total: N }, msg: "ok" }`。

---

### OpenAI Compatible（`/v1`）

OpenAI 兼容层，直接复用平台上已接入的 `text` 类模型。主要用于把第三方工具（如 IDE 插件、ChatBox 客户端、curl）接到自家文本推理。

- `GET /v1/models`：列出 OpenAI 风格的模型清单（仅展示 `type=text` 且 `status=active` 的模型）
- `GET /v1/models/{model_id}`：查询单个模型
- `POST /v1/chat/completions`：OpenAI `chat/completions` 完全兼容接口，支持 `stream=true` 的 SSE 流式输出、`tools` / `tool_choice`、多模态 `content parts` 等

> 认证同样使用 `Authorization: Bearer <access_token>`。请求体、响应体、错误码格式均与 OpenAI 官方协议对齐，不会被包装成统一的 `{code,data,msg}` 结构。

---

### Channel Ingress（`/v1/channel-ingress`）

> 内部/外部渠道（如 IM、消息队列）把消息投递到 Master Agent 的入口。

#### POST `/v1/channel-ingress`

- **认证**：需要（通常由网关/转发器持有系统 token）
- **说明**：接收一条渠道消息并异步触发一次 Master agent run，返回 `202 Accepted`。

---

## WebSocket

### `/ws/task/{task_id}`

- **用途**：用户前端接收任务状态/结果推送
- **Query**：`token=<jwt>`（必需）

消息示例：

```json
{
  "type": "task_status",
  "task_id": "uuid",
  "status": "running",
  "progress": 10,
  "timestamp": "2026-01-01T00:00:00"
}
```

### `/ws/chat/{session_id}`

- **用途**：统一会话实时双向通道（配合 `POST /v1/chat/sessions` 使用）
- **Query**：`token=<jwt>`（必需）
- **消息格式**：均为 JSON 文本帧，主事件模型见 `backend/websocket/chat_ws_protocol.md`

典型流程：

1. 客户端连接后，服务端主动下发 `session_ready`
2. 客户端发送 `user_message`（文本）或 `audio_chunk` + `session_commit`（音频）
3. 服务端推送 `message_started` / `message_delta` / `message_completed`
4. 若触发工具或派生任务，还会推送 `tool_call_*`、`status_changed`、`artifact_created`
5. 客户端可发送 `interrupt` 打断当前轮次，或发送 `session_close` 结束整个会话

常见服务端事件：

- `session_ready`：会话进入可交互状态
- `message_started`：assistant 开始输出
- `message_delta`：文本流式增量
- `message_completed`：本轮回答完成（含最终文本、附件、usage）
- `status_changed`：运行状态变化，或派生任务状态投影
- `artifact_created`：本轮生成了图片/视频/音频等产物

### `/ws/model/{model_id}`

- **用途**：前端订阅模型下载进度/日志
- **Query**：`token=<jwt>`（必需）
- **消息**：`download_status`（状态/进度文本）、`download_log`（实时日志）

### `/ws/inference/{service_id}`

- **用途**：推理器连接，接收任务、回传状态/结果
- **保活**：服务端定期 ping，推理器需回复 pong

---

## 静态资源（Outputs）

后端会将 `storage.local.base_path`（默认 `resources/outputs`）挂载为静态服务：

- **URL 前缀**：`/outputs`
- **访问示例**：`GET /outputs/models/5014037e....png`

> 前端开发模式建议通过 Vite 代理 `/outputs/*` 到后端（已支持）。


