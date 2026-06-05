# `/ws/chat/{session_id}` 统一会话 WS 协议 v1

本文件是 Phase 1 设计产物（P1-2），是 `/ws/chat/{session_id}` 唯一权威契约；Review 通过后才允许进入 Phase 2 实现。

> 来源映射：
> - 协议方向与事件名收敛：`.cursor/plans/统一会话重构_63361fab.plan.md` §4
> - 事件名映射表：主文档 §4.2
> - 状态机与 interrupt 语义：以 `backend/services/chat/session.py` 的当前实现为准

## 0. 契约边界

- 本协议是**会话主入口**：今天的 `conversation + agent_run + poll run` 与旧实时会话 WS 路径的全部职责，统一收敛到 `/ws/chat/{session_id}`。
- `/ws/task/{task_id}`、`/ws/inference/{service_id}`、`/ws/model/{model_id}` 不在本文件覆盖范围；按主文档保留为"离线纯任务通道 / 推理器回连 / 模型下载订阅"。
- **推理侧协议不改名**（主文档 §B.3）：`/ws/inference` 入口继续收 `session_audio_chunk / session_text_input / session_interrupt / session_open / session_close` 与发 `llm_text_delta / transcript_partial / transcript_final / audio_stream_* / result`；这些由后端映射/聚合为本协议的事件，**客户端绝不直接看到推理侧事件名**。

## 1. 握手与鉴权

### 1.1 Endpoint

```
WS /ws/chat/{session_id}?token=<JWT>
```

### 1.2 鉴权流程

1. 服务端提取 `session_id`，查库确认会话存在（`conversations.id`）。
2. 校验 `token` 合法并解出 `user_id`；`conversations.user_id != user_id` → 关闭连接 `close(1008, "Permission denied")`。
3. 通过后立即 `accept`，进入 `opening` 状态，然后发 `session_ready`（见 §3.1）。

### 1.3 关闭码

| code | reason | 触发条件 |
|---|---|---|
| 1000 | normal closure | 客户端发 `session_close` 或服务端 `status=closed` 清理完成 |
| 1008 | policy violation | 鉴权失败 / 会话不存在 / 用户越权 |
| 1011 | internal error | 服务端未捕获异常 |

## 2. 通用消息结构

### 2.1 客户端 → 服务端（通用字段）

```jsonc
{
  "type": "<message-type>",   // 见 §2.3 白名单
  "session_id": "<uuid>",     // 必填；与 URL path 一致
  "turn_id": "<uuid>",        // 可选；客户端不传则服务端自动分配
  "payload": { ... },         // 随 type 变化；见 §4
  "client_ts": "2026-04-20T12:34:56.789Z"  // 可选；客户端时间戳，仅用于排障
}
```

- 未列出的顶层字段一律忽略。
- 未知 `type` → 服务端回 `error{code=unsupported_message_type}`，连接不关闭。

### 2.2 服务端 → 客户端（通用字段）

```jsonc
{
  "type": "<event-type>",     // 见 §2.4 白名单
  "session_id": "<uuid>",
  "turn_id": "<uuid|null>",   // Turn 级事件必填；会话级事件（如 session_ready）可空
  "run_id":  "<uuid|null>",   // SessionRun 级事件必填（对齐 agent_runs.id）
  "sequence": 123,            // 可选；若事件源自 agent_run_events，则透传其 sequence，便于客户端排序与增量恢复
  "server_ts": "2026-04-20T12:34:56.789Z",
  "payload": { ... }
}
```

### 2.2.1 Binary frame v1（音频 FIFO 配对）

自本版本起，下列四类事件**不再**在 JSON 内携带 `audio_b64`；改为：

1. **先发一帧 WebSocket TEXT**：JSON 与往常一样，但 `payload` 内使用 `bytes_len: int` 描述紧随其后的 PCM 字节数（`0` 表示本 meta 后无 binary 帧，例如仅 `is_final=true` 的收尾包）。
2. **再发一帧 WebSocket BINARY**（当 `bytes_len > 0`）：裸 **PCM16 LE mono** 字节流，无自定义 wire header。

同一条连接上帧顺序保序，接收端用 FIFO：`audio_delta` / `audio_chunk` / `session.asr.chunk` / `session.audio.chunk` 的 meta 与下一帧 binary 一一配对。未来若引入 Opus 等压缩，通过 `mime` 字段区分编码。

**部署注意（硬切）**：旧客户端若仍只解析 JSON、不读 binary，将听不到声音但不会崩溃；升级需 **backend + inference + web** 同版本发布。

### 2.3 客户端 → 服务端 消息白名单

| type | 允许的状态（见状态机 §D.1） | 载荷必填字段 |
|---|---|---|
| `session_open` | `opening` | 无 |
| `user_message` | `ready / turn_buffering` | `payload.text: string` |
| `audio_chunk` | `ready / turn_buffering / reasoning / tool_running / streaming_output / waiting_task` | `payload.bytes_len: int`（>0）；可选 `payload.mime`、`payload.seq`；**MUST** 紧跟一帧 BINARY（PCM16 LE）。输出态仅用于后端 VAD barge-in 监听 |
| `session_commit` | `turn_buffering` | 无（显式完结一次音频或流式文本轮次） |
| `user_message_end` | `turn_buffering`（仅 `input_mode=text_stream`） | 无（**首版不开放**，Phase 2 预留） |
| `interrupt` | `turn_buffering / reasoning / tool_running / streaming_output / waiting_task` | 无 |
| `session_close` | 任意非终态 | 无 |
| `ack` | 任意 | `payload.ack_of: sequence` |

**语义说明**：
- `session_open`：第一次连上时客户端需发此消息以声明 `input_mode`（见 §4.1）；首版**客户端可省略**，服务端默认按 `input_mode=text` 进入 `ready`。
- `session_commit`：显式完结一次 Turn（`audio_once / audio_stream` 的首版完成判据之一，详见状态机 §D.2）。
- `interrupt`：打断当前 Run，不关闭会话；详见 §4.6 与状态机 §D.3。
- `session_close`：关闭整个会话，进入终态。

### 2.4 服务端 → 客户端 事件白名单

| type | 触发节点 | 映射来源 |
|---|---|---|
| `session_ready` | 连接完成 + 会话初始化结束 | 新增；替代旧实时会话路径里的初始化 ready 事件 |
| `status_changed` | 每次状态机状态切换 | 新增；替代裸 JSON 的 `send_progress / send_task_update` |
| `message_started` | Master Agent 开始产出 assistant 消息 | 收敛 `llm_text_delta` 首包 / `text_stream_delta` 首包 / `transcript_partial` 首段 |
| `message_delta` | assistant 文本增量 / transcript 增量 | 收敛 `llm_text_delta / text_stream_delta / transcript_partial` |
| `message_completed` | assistant 消息收尾 | 收敛 `llm_text_delta` 尾包 / `text_stream_delta` 尾包 / `transcript_final` / 旧 `run` 的 `result_summary` |
| `audio_delta` | TTS / 流式音频输出增量 | 收敛 `audio_stream_start / audio_stream_chunk / audio_stream_end` |
| `transcript_delta` | ASR 局部/完整转写（`role=transcript`） | 收敛 `transcript_partial / transcript_final / transcript_segment` |
| `transcript_canceled` | 行为指令：本 turn 用户输入被丢弃（环境噪声 / 空 / ASR 凑词），请清掉 in-flight transcript 气泡 | `payload: { reason: "empty" \| "noise", text: string }`。同时后端会配套 emit `error{empty_transcript}` 提供事件卡片可见信息。 |
| `tool_call_started` | 工具执行开始 | 镜像 `agent_run_events.tool_call_started`（字段不变） |
| `tool_call_completed` | 工具执行完成 | 镜像 `agent_run_events.tool_call_completed` |
| `tool_call_failed` | 工具执行失败 | 镜像 `agent_run_events.tool_call_failed` |
| `artifact_created` | 派生 Task 产出文件 / 非流式多模态结果 | 收敛 `inference/common/result_handler.py` 的 `result` 消息、Task 完成后的 `File` 记录 |
| `error` | 任意错误 | 收敛 `session_error` + `task_status(status=failed)` |
| `session_closed` | 会话进入 `closed` | 复用旧名，语义收敛 |

## 3. 会话级事件（非 Turn 内）

### 3.1 `session_ready`

连接 accept 后立即下发，标志进入 `ready`（或在需要预检的场景下先发一个 `status_changed` 进入 `opening`，再发 `session_ready`）。

```json
{
  "type": "session_ready",
  "session_id": "6b2c2b90-7b01-4f89-9da3-0a9efc4f7c01",
  "server_ts": "2026-04-20T12:34:56.000Z",
  "payload": {
    "mode": "chat",
    "input_mode": "text",
    "output_mode": "text_stream",
    "capabilities": {
      "supports_audio_input": true,
      "supports_audio_output": true,
      "supports_tool_artifacts": true
    },
    "agent_id": "preset-master-agent",
    "conversation_id": "6b2c2b90-7b01-4f89-9da3-0a9efc4f7c01"
  }
}
```

- **不带旧场景字段**。`capabilities` 只表示"该 session 是否允许某类输入/输出"，与推理服务 `capabilities` 字段（已删除）无关。
- `conversation_id == session_id`（统一会话里两者是同一主键）。

### 3.2 `status_changed`

```json
{
  "type": "status_changed",
  "session_id": "6b2c...",
  "turn_id": null,
  "run_id": null,
  "server_ts": "2026-04-20T12:34:57.000Z",
  "payload": { "state": "ready", "prev": "opening" }
}
```

`state` 取值与状态机 §D.1 表格一致：`opening / ready / turn_buffering / reasoning / tool_running / streaming_output / waiting_task / completed / interrupted / failed / closed`。

### 3.3 `error`

```json
{
  "type": "error",
  "session_id": "6b2c...",
  "turn_id": "...",
  "server_ts": "2026-04-20T12:34:58.000Z",
  "payload": {
    "code": "model_not_available",
    "message": "No inference service is serving model_name=Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "recoverable": true
  }
}
```

**错误 `code` 冻结清单（首版）**：

| code | 含义 | recoverable |
|---|---|---|
| `invalid_payload` | JSON 解析失败 / 必填字段缺失 | true |
| `unsupported_message_type` | 未知 type | true |
| `busy` | 当前状态不接收输入（见状态机 §D.4） | true |
| `interrupt_not_allowed` | 当前状态不允许 `interrupt` | true |
| `model_not_available` | `ModelNameRouter` 匹配不到 running 推理服务 | false |
| `tool_execution_failed` | 工具层抛异常（镜像 `tool_call_failed`） | true |
| `internal_error` | 未捕获异常 | false |

### 3.4 `session_closed`

```json
{
  "type": "session_closed",
  "session_id": "6b2c...",
  "server_ts": "2026-04-20T12:35:10.000Z",
  "payload": { "reason": "client_requested" }
}
```

`reason ∈ { client_requested, server_timeout, failed, admin_closed }`。

## 4. Turn 内事件流（主链路）

### 4.1 输入模态声明

`session_open` 携带：

```json
{
  "type": "session_open",
  "session_id": "6b2c...",
  "payload": {
    "input_mode": "text",
    "output_mode": "text_stream"
  }
}
```

- `input_mode ∈ { text, text_stream, audio_once, audio_stream, mixed }`（见主文档 §5）。首版实现 `text / audio_once / audio_stream`，其余保留定义不进入首版。
- `output_mode ∈ { text_once, text_stream, audio_once, audio_stream, multimodal_result }`。由系统根据输入与 Master Agent 决策**动态决定**，客户端在 `session_open` 携带的值仅作为偏好提示。

### 4.2 `user_message`（文字输入）

```json
{
  "type": "user_message",
  "session_id": "6b2c...",
  "turn_id": null,
  "payload": {
    "text": "帮我生成一张山水画"
  }
}
```

- 客户端**不必传 `turn_id`**；服务端在第一次用户输入到达时分配并在后续事件里回填。
- 服务端收到后按状态机 §D.2 判据完成 Turn，立即进入 `reasoning`。

### 4.3 `audio_chunk` / `session_commit`（音频输入）

`audio_chunk` 的 JSON **不含** PCM 本体；`payload.bytes_len` 必须与紧随的 **BINARY** 帧字节数一致。

```json
{
  "type": "audio_chunk",
  "session_id": "6b2c...",
  "turn_id": null,
  "payload": {
    "bytes_len": 640,
    "mime": "audio/pcm;rate=16000",
    "seq": 12
  }
}
```

→ 紧跟 **BINARY**：640 字节的 Int16 LE PCM mono（示例为 20ms @ 16kHz）。

音频流基于后端 VAD 官方端点或显式 `session_commit` 完成 Turn。普通自动端点使用 FunASR `max_end_silence_time`（默认 2500ms，可通过会话 metadata 的 `vad.max_end_silence_time` 调整），避免把句中短暂停顿误判为整轮结束：

```json
{
  "type": "session_commit",
  "session_id": "6b2c...",
  "turn_id": "<assigned>"
}
```

### 4.4 `message_started / message_delta / message_completed`

单次 Turn 的 assistant 输出序列：

```json
{ "type": "message_started",
  "session_id": "6b2c...",
  "turn_id": "t-001", "run_id": "r-001",
  "sequence": 42,
  "server_ts": "...",
  "payload": { "role": "assistant", "content_type": "text" }
}
```

```json
{ "type": "message_delta",
  "session_id": "6b2c...",
  "turn_id": "t-001", "run_id": "r-001",
  "sequence": 43,
  "payload": { "role": "assistant", "delta": "好的，我来帮你生成" }
}
```

```json
{ "type": "message_completed",
  "session_id": "6b2c...",
  "turn_id": "t-001", "run_id": "r-001",
  "sequence": 99,
  "payload": {
    "role": "assistant",
    "content_type": "text",
    "content": "好的，我来帮你生成一张山水画...",
    "interrupt_reason": null,
    "usage_metrics": { "total_tokens": 123 }
  }
}
```

- `role`：`assistant / transcript / tool`（`tool` 仅在需要把工具原始输出作为消息展示时使用；否则只发 `tool_call_*`）。
- `interrupt_reason` 非空表示该消息因 `interrupt` 中断收尾（状态机 §D.3）。

### 4.5 `audio_delta` / `transcript_delta` / `transcript_canceled`

`transcript_canceled` 是后端 → 前端的行为指令事件，**不**是给最终用户看的信息。语义："本 turn 的 ASR 输入应当被丢弃，请清掉对应的 in-flight transcript 气泡"。前端必须按事件类型无条件路由（清气泡），不应对 `payload.reason` / `text` 做显示判断。后端在 emit 这个事件的同时，会再 emit 一个 `error{code: "empty_transcript", recoverable: true}`，承载用户/开发者可见的事件卡片描述（empty / noise + 原文）。

```json
{ "type": "transcript_canceled",
  "session_id": "6b2c...",
  "turn_id": "t-001", "run_id": null,
  "payload": {
    "reason": "noise",
    "text": "嗯。"
  }
}
```



`audio_delta`：JSON `payload` 含 `bytes_len`、`mime`、`is_final`、`sample_rate`（可选）；当 `bytes_len > 0` 时 **MUST** 紧跟一帧 BINARY（裸 PCM16 LE）。

```json
{ "type": "audio_delta",
  "session_id": "6b2c...",
  "turn_id": "t-001", "run_id": "r-001",
  "payload": {
    "bytes_len": 9600,
    "mime": "audio/pcm;rate=24000",
    "is_final": false,
    "sample_rate": 24000
  }
}
```

→ 紧跟 **BINARY**：9600 字节的 PCM16 LE（示例）。

```json
{ "type": "transcript_delta",
  "session_id": "6b2c...",
  "turn_id": "t-001", "run_id": "r-001",
  "payload": {
    "text": "帮我生成一张山水",
    "is_final": false
  }
}
```

`is_final=true` 时表示该模态本段已结束；整个 Turn 的收尾仍以 `message_completed` 为准。

### 4.6 `tool_call_started / tool_call_completed / tool_call_failed`

镜像 `agent_run_events` 三类事件，字段不做重命名：

```json
{ "type": "tool_call_started",
  "session_id": "6b2c...",
  "turn_id": "t-001", "run_id": "r-001",
  "sequence": 56,
  "payload": {
    "tool_name": "image_generator",
    "provider": "builtin",
    "target_tool_name": "image_generator",
    "args_preview": { "prompt": "...", "model_name": "waiIllustrious..." }
  }
}
```

`tool_call_completed / tool_call_failed` 字段结构与 `backend/services/agent/events.py` 一致（`duration_ms / output_preview / output_len / error`）。

### 4.7 `artifact_created`

派生 Task 产出文件、或非流式的多模态结果（图片、视频、静态音频文件）都统一走此事件；**不再让客户端订阅 `/ws/task/{task_id}`**。

```json
{ "type": "artifact_created",
  "session_id": "6b2c...",
  "turn_id": "t-001", "run_id": "r-001",
  "payload": {
    "file_id": "f-abc",
    "category": "image",
    "mime": "image/png",
    "url": "/api/files/f-abc/raw",
    "file_name": "xxx.png",
    "file_size": 123456,
    "derived_task_id": "task-xyz"
  }
}
```

### 4.8 `interrupt`

```json
{ "type": "interrupt", "session_id": "6b2c...", "turn_id": "t-001" }
```

服务端响应：
1. 立即 `status_changed{state=interrupted}`。
2. 推理侧发送 `session_interrupt`（推理侧协议名保持不变）。
3. 当前 partial assistant 消息以 `message_completed{interrupt_reason="user_interrupt"}` 收尾并落 `conversation_messages`（`status` 元数据写 `interrupted`）。
4. `SessionRun.status=interrupted`；状态回到 `ready` 自动下发 `status_changed{state=ready}`。

### 4.9 后端 VAD barge-in

实时语音会话中，客户端可在助手输出期间继续发送 `audio_chunk`。服务端在 `reasoning / tool_running / streaming_output / waiting_task` 下将这些 PCM 送入 FunASR VAD：

1. VAD 未确认用户开口时，PCM 只用于后端监听，不进入 ASR，也不会创建新 Turn。
2. VAD 触发 `speech_start` 后，服务端复用 §4.8 interrupt 语义收尾旧 assistant，并创建新的 audio Turn；这里不等待 `speech_end`，保证低延迟打断。
3. 服务端把 VAD pre-roll 与后续 PCM 转发给 ASR，进入 `turn_buffering`。
4. VAD 按 `max_end_silence_time` 触发 `speech_end` 后，服务端自动执行 `session.asr.commit`，随后按 `session.transcript.final` 启动下一轮 run。

该能力不要求前端实现 VAD；前端只需持续上传 PCM。

## 5. 完整示例流（文字问答 + 工具）

```text
C→S  session_open  {input_mode:text}
S→C  session_ready
S→C  status_changed {state:ready}
C→S  user_message  {text:"帮我画张山水"}
S→C  status_changed {state:reasoning, turn_id:t1, run_id:r1}
S→C  tool_call_started {tool_name:"image_generator", ...}
S→C  status_changed {state:tool_running}
S→C  status_changed {state:waiting_task}    # 派生 Task 后
S→C  artifact_created {file_id:f1, ...}
S→C  tool_call_completed {tool_name:"image_generator", duration_ms:...}
S→C  status_changed {state:reasoning}       # 工具结果回注 LLM
S→C  message_started {role:assistant}
S→C  message_delta   {delta:"这是你要的山水画"}
S→C  message_delta   {delta:"，希望你喜欢"}
S→C  message_completed
S→C  status_changed {state:completed}
S→C  status_changed {state:ready}           # 自动回 ready
```

## 6. 完整示例流（音频输入 + 流式音频输出）

```text
C→S  session_open      {input_mode:audio_stream, output_mode:audio_stream}
S→C  session_ready
S→C  status_changed    {state:ready}
C→S  audio_chunk       TEXT{seq:1, bytes_len:N1} + BINARY pcm
S→C  status_changed    {state:turn_buffering, turn_id:t1}
S→C  transcript_delta  {text:"帮我", is_final:false}
C→S  audio_chunk       TEXT{seq:2, bytes_len:N2} + BINARY pcm
S→C  transcript_delta  {text:"帮我讲个笑话", is_final:false}
# VAD 按 max_end_silence_time 判定端点后自动 commit，或客户端发 session_commit
S→C  transcript_delta  {text:"帮我讲个笑话", is_final:true}
S→C  status_changed    {state:reasoning, run_id:r1}
S→C  message_started   {role:assistant, content_type:"text"}
S→C  message_delta     {delta:"好的"}
S→C  audio_delta       TEXT{bytes_len:..., mime:...} + BINARY pcm
S→C  audio_delta       TEXT{..., is_final:false} + BINARY pcm
S→C  audio_delta       TEXT{bytes_len:0, is_final:true}
S→C  message_completed
S→C  status_changed    {state:completed}
S→C  status_changed    {state:ready}
```

## 7. 映射关系一览（权威）

本表是 §4.2 映射表在协议层的字段级落地。Phase 2 后端在 `/ws/inference` 收到左列事件时，按右列逻辑转出到 `/ws/chat/{session_id}`。

| 推理侧事件（后端入） | 统一 chat 事件（后端出） | 转换逻辑 |
|---|---|---|
| `llm_text_delta` 首包 | `message_started` + `message_delta` | 本 Run 首次收到 → 先发 `message_started`；后续仅 `message_delta` |
| `llm_text_delta` 非首包 | `message_delta` | 透传 `delta` 字段 |
| `llm_text_delta` `is_final=true` | `message_completed` | 聚合本 Run 全量文本作为 `payload.content` |
| `text_stream_delta`（任务型流式文本） | 同 `llm_text_delta` 三条规则 | 同上 |
| `transcript_partial` | `transcript_delta{is_final:false}` | — |
| `transcript_final` | `transcript_delta{is_final:true}` | — |
| `transcript_segment` | `transcript_delta` | `is_final` 按段落属性映射 |
| `audio_stream_start` | `message_started{role:assistant,content_type:"audio"}` 或 前置 `audio_delta` | 若本 Turn 已发过 `message_started(text)`，只发首个 `audio_delta` |
| `audio_stream_chunk` | `audio_delta{is_final:false}` | — |
| `audio_stream_end` | `audio_delta{is_final:true}` + `message_completed` | — |
| `result`（图片/视频/静态音频） | `artifact_created` + `message_completed` | 先发每个文件的 `artifact_created`，再发 Turn 级的 `message_completed` |
| `session_ready`（推理侧） | **丢弃** | 客户端看到的 `session_ready` 由后端自行合成 |
| `session_error` | `error{code=...}` | — |
| `session_closed` | `session_closed` | 仅作为推理侧状态参考，不直接转发 |
| `tool_call_started / completed / failed`（agent_run_events） | 同名事件 | 字段透传 |
| `task_status`（`/ws/task` 语义） | `status_changed` + `artifact_created`（如有） | 完成时额外发 `artifact_created` |

## 8. 客户端 → 服务端 消息的 `type` 收敛对照

| 旧消息 type（旧 `/ws/session`） | 新 chat type |
|---|---|
| `session_open` | `session_open`（保留） |
| 旧文本输入帧 | `user_message` |
| 旧音频分片输入帧 | `audio_chunk` |
| `interrupt` | `interrupt`（保留） |
| `session_close` | `session_close`（保留） |
| — | `session_commit`（新增） |
| — | `user_message_end`（预留，首版不开） |
| — | `ack`（预留） |

## 9. 约定与风险

### 9.1 顺序保证

- 服务端在同一 WS 上按发送顺序保证 `sequence` 单调递增（对 `agent_run_events` 相关事件透传其 sequence；非事件类消息自分配）。
- 客户端必须能容忍 `message_delta` 乱序**不出现**（服务端已保证有序），但应容忍 `tool_call_*` 与 `message_delta` 交错。

### 9.2 背压与丢帧

- 首版不做客户端 `ack` 窗口；`ack` 预留但不强制。若后续出现 WS 写入积压，服务端可对高频增量（`audio_delta`）做合并，但不得丢失 `message_completed / status_changed / artifact_created` 这三类关键事件。

### 9.3 与旧 `/ws/session` 的兼容

- **不做兼容层**。Phase 5 硬切后，旧 endpoint 直接下线。
- 迁移期 `test/text_ws_experiment.py` 与新脚本并行；新协议稳定后老脚本归档。

### 9.4 与 `/ws/task/{task_id}` 的关系

- 会话派生的子 Task**不再**让前端订阅 `/ws/task`；派生任务进度通过 `status_changed(state=waiting_task) + artifact_created + tool_call_completed` 组合表达。
- `/ws/task` 作为**离线纯任务通道**保留，不承担会话。

## 10. 验收清单

Phase 1 本文件 Review 通过的判定标准：

- [ ] 客户端/服务端所有 `type` 白名单冻结（§2.3 / §2.4）。
- [ ] 每个客户端 `type` 对应哪个状态机状态允许接收有明确映射（§2.3 与状态机 §D.1）。
- [ ] 每个服务端事件都给出至少一个完整示例（§3 / §4）。
- [ ] 映射表 §7 覆盖主文档 §4.2 列举的所有来源事件。
- [ ] `error.code` 清单冻结（§3.3）。
- [ ] 不出现旧场景 / 旧绑定服务字段 / `capabilities` 等旧语义字段。
- [ ] 与状态机文档 `state_machine.md` 交叉引用一致，无字段冲突。
