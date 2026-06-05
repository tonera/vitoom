"""MasterAgentRuntime —— 统一会话下的 Master Agent 执行器。

对应重构计划 §3 / §5 的 "统一 Master Agent 入口"：

    - ``/ws/chat/{session_id}`` 每个 Turn 的推理/工具调用都由此模块接管；
    - 复用 ``AgentWorker`` 里那套"AgentSpec → ToolSelectionService →
      ToolResolver → CrewFactory → FlowRunner"管线，Master Agent 的工具决策
      能力全量继承过来；
    - 开启 CrewAI 的 ``LLM(stream=True)``，并通过 ``crewai_event_bus``
      订阅 ``LLMStreamChunkEvent``，把 token 级 delta 转译为统一会话
      协议的 ``message_delta`` 推给前端；
    - 工具开始 / 结束 / 失败映射为 ``tool_call_started /
      tool_call_completed / tool_call_failed``。

设计上有两个取舍：

1. CrewAI ``kickoff`` 同步阻塞，这里放到 ``asyncio.to_thread`` 里跑；事件
   回调发生在工作线程，通过 ``asyncio.run_coroutine_threadsafe`` 调回主
   loop 上的 ``SessionRuntime`` emit 方法，保证协议事件顺序与线程安全。
2. ``crewai_event_bus.scoped_handlers()`` 作为 per-run 订阅作用域，避免
   多个并发 Turn 的监听器互相污染。
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# CrewAI 的 telemetry 会在 crew.kickoff 线程里注册 SIGINT/SIGTERM，
# 非主线程下会抛 "signal only works in main thread" 警告。我们在
# chat ws 路径用 asyncio.to_thread 跑 kickoff，必然踩到这条告警。
# 在这里统一关掉遥测即可消除噪音；CrewAI 本身对这个变量有判断。
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from backend.core.logger import get_app_logger
from backend.services.chat.error_summary import summarize_chat_run_error
from backend.database import Agent, AgentRun, Conversation, Task
from backend.services.agent.crews import CrewFactory
from backend.services.agent.crew_tools.bridge import is_nested_tool_event_scope_active
from backend.services.agent.events import (
    record_run_completed,
    record_run_failed,
    record_run_started,
    record_tool_selected,
)
from backend.services.agent.no_tool_runner import run_no_tool_completion
from backend.services.agent.presets import ensure_default_agent_presets
from backend.services.agent.settings import get_master_preset_agent_id
from backend.services.agent.specs import AgentSpec, GLOBAL_TOOL_POOL, TaskSpec
from backend.services.agent.tool_resolver import ToolResolver
from backend.services.agent.tool_selection import ToolSelectionService
from backend.services.agent.types import AgentCommand
from backend.services.chat.artifacts import (
    build_chat_files_from_tool_result,
    default_category_for_tool,
    try_parse_tool_result_payload,
)
from backend.services.chat.media_context import build_conditional_context
from backend.services.conversation import build_prompt_with_history
from backend.services.chat.router import LoadNameRouter
from backend.services.chat.session import SessionRuntime, Turn
from backend.services.chat.task_event_adapter import handle_task_event
from backend.services.chat.tool_call_stream_renderers import ToolCallStreamRendererRegistry
from backend.services.chat.voice_reply import (
    VoiceReplyStream,
    should_emit_voice_reply,
    synthesize_voice_reply,
)

logger = get_app_logger(__name__)


# 单轮 Master Agent 执行的最大墙钟时间（兜底防挂）
_DEFAULT_RUN_TIMEOUT_SECONDS = 300.0


def _should_forward_llm_stream_chunk(
    *,
    chunk: str,
    call_type: Any = None,
    tool_call: Any = None,
) -> bool:
    if not chunk:
        return False
    normalized_call_type = str(getattr(call_type, "value", call_type) or "").strip().lower()
    if normalized_call_type == "tool_call":
        return False
    if tool_call is not None:
        return False
    return True


class MasterAgentRuntime:
    """同步执行一次 Run：构建 Master Agent Crew → 流式跑完 → 回写 SessionRuntime。"""

    def __init__(
        self,
        *,
        ws_manager: Any,
        router: Optional[LoadNameRouter] = None,
        default_load_name: str = "",
    ) -> None:
        # ws_manager / router / default_load_name 保留为构造参数兼容
        # 上游（backend.websocket.chat_routes）的现有调用点；当前实现以
        # CrewAI + Vitoom 内部 /v1/chat/completions 为主路径，不再直接
        # 占用推理服务 WS。留着这些字段也方便未来扩展（比如 audio 路径）。
        self._ws = ws_manager
        self._router = router
        self._default_load_name = default_load_name

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def run(self, runtime: SessionRuntime, turn: Turn) -> None:
        """由 ``SessionRuntime`` 注入的 ``MasterRun`` 回调。"""
        user_text = str(turn.user_text or "").strip()
        if not user_text:
            await runtime.fail_run(code="invalid_payload", message="empty user message")
            return

        if await _maybe_handle_slash_command(runtime, turn, user_text):
            return

        # 解析 Master Agent：优先会话自带 agent_id，缺省落回 Master preset。
        agent_id = (runtime.agent_id or "").strip() or get_master_preset_agent_id()
        try:
            ensure_default_agent_presets()
            agent_record = Agent.get_by_id(agent_id)
        except Exception as exc:
            await runtime.fail_run(
                code="internal_error", message=f"load agent failed: {exc}"
            )
            return
        if not agent_record:
            await runtime.fail_run(
                code="internal_error", message=f"agent not found: {agent_id}"
            )
            return

        # 把最近若干轮历史拼进 prompt，让 Master Agent 拥有连贯上下文。
        # - command.message 使用"含历史"的完整文本，传给 CrewAI task description
        #   的 {message} 占位符，LLM 每轮都能看到 user/assistant 交替；
        # - context.original_user_message 保留纯本轮输入，供 ToolSelection
        #   打分，避免历史噪音干扰工具相关性。
        try:
            prompt_with_history = build_prompt_with_history(
                conversation_id=runtime.session_id,
                new_message=user_text,
            )
        except Exception as exc:
            logger.warning(
                "build_prompt_with_history failed session=%s err=%s; fallback to bare message",
                runtime.session_id,
                exc,
            )
            prompt_with_history = user_text

        command_context = build_conditional_context(prompt_with_history)
        command_context["original_user_message"] = user_text
        command_context["turn_id"] = turn.turn_id

        command = AgentCommand(
            user_id=runtime.user_id,
            agent_id=agent_id,
            message=prompt_with_history,
            source_type="chat_ws",
            source_ref=runtime.session_id,
            conversation_id=runtime.session_id,
            runtime_config=self._resolve_runtime_config(runtime, agent_record),
            context=command_context,
        )

        loop = asyncio.get_running_loop()
        await runtime.enter_reasoning()
        current_run_id = str(turn.run_id or "").strip()

        def _run_is_current() -> bool:
            active_turn = runtime.current_turn
            if not active_turn:
                return False
            return str(active_turn.run_id or "").strip() == current_run_id

        # 共享的 emit 包装：把 SessionRuntime 的协程封进线程安全调度器，
        # 让 CrewAI 事件线程也能把 delta 推回主 loop。
        #
        # 两个独立 flag：
        # - message_started_emitted：是否已向 WS 发出 message_started
        #   （只要 LLM 开始调用或第一次 chunk 来都应发一次）；
        # - any_delta_emitted：是否真正发过 message_delta
        #   （决定后续要不要走 fake-stream 兜底）。
        message_started_emitted: Dict[str, bool] = {"flag": False}
        any_delta_emitted: Dict[str, bool] = {"flag": False}
        tool_failure_message: Dict[str, str] = {"text": ""}
        tool_seq_by_id: Dict[str, int] = {}

        voice_reply_stream = VoiceReplyStream(
            runtime=runtime,
            turn=turn,
            logger=logger,
            run_timeout_seconds=_DEFAULT_RUN_TIMEOUT_SECONDS,
        )

        async def _emit_and_buffer(chunk: str) -> None:
            """emit message_delta + 喂入 TTS 句级缓冲。"""
            if not chunk:
                return
            await runtime.emit_message_delta(chunk)
            await voice_reply_stream.push_chunk(chunk)

        async def _emit_text_only(chunk: str) -> None:
            """emit message_delta without feeding session-level voice reply TTS."""
            if not chunk:
                return
            await runtime.emit_message_delta(chunk)

        tool_call_stream_renderers = ToolCallStreamRendererRegistry()

        async def _ensure_message_started() -> None:
            if not _run_is_current():
                return
            if not message_started_emitted["flag"]:
                await runtime.enter_streaming_output()
                message_started_emitted["flag"] = True
                await runtime.emit_message_started()

        async def _on_llm_started() -> None:
            if not _run_is_current():
                return
            # 注意：这里只负责 emit message_started，不能把"已发 delta"的
            # flag 置 True，否则非流式 CrewAI 下 fake-stream 兜底会被跳过。
            await _ensure_message_started()

        async def _on_llm_chunk(
            chunk: str,
            *,
            call_type: Any = None,
            tool_call: Any = None,
        ) -> None:
            if not _run_is_current():
                return
            rendered_tool_chunks = tool_call_stream_renderers.feed_delta(
                chunk=chunk,
                call_type=call_type,
                tool_call=tool_call,
            )
            if rendered_tool_chunks:
                await _ensure_message_started()
                any_delta_emitted["flag"] = True
                for rendered_chunk in rendered_tool_chunks:
                    await _emit_text_only(rendered_chunk)
                return
            if not _should_forward_llm_stream_chunk(
                chunk=chunk,
                call_type=call_type,
                tool_call=tool_call,
            ):
                return
            if tool_call_stream_renderers.active:
                return
            await _ensure_message_started()
            any_delta_emitted["flag"] = True
            await _emit_and_buffer(chunk)

        async def _on_tool_started(name: str, args: Any, event_id: str) -> None:
            if not _run_is_current():
                return
            await runtime.enter_tool_running()
            tool_seq_by_id[event_id] = tool_seq_by_id.get(event_id, 0) + 1
            await runtime.emit_tool_call_started(
                payload={
                    "tool_name": name,
                    "tool_call_id": event_id,
                    "arguments": _safe_stringify(args),
                    "args_preview": _args_preview_for_ui(args),
                }
            )
            rendered_tool_chunks = tool_call_stream_renderers.feed_complete_args(tool_name=name, args=args)
            if rendered_tool_chunks:
                await _ensure_message_started()
                any_delta_emitted["flag"] = True
                for rendered_chunk in rendered_tool_chunks:
                    await _emit_text_only(rendered_chunk)

        async def _on_tool_finished(name: str, output: Any, event_id: str) -> None:
            if not _run_is_current():
                return
            await runtime.enter_reasoning()
            parsed_output = try_parse_tool_result_payload(output)
            if parsed_output:
                parsed_status = str(parsed_output.get("status") or "").strip().lower()
                parsed_error = str(parsed_output.get("error") or "").strip()
                if parsed_status == "failed" and parsed_error and not tool_failure_message["text"]:
                    tool_failure_message["text"] = parsed_error
                _bind_task_ids_to_turn(turn, parsed_output)
                for file_info in build_chat_files_from_tool_result(
                    parsed_output,
                    default_category=default_category_for_tool(name),
                    source_tool=name,
                ):
                    await runtime.emit_artifact_created(payload=file_info)
                for rendered_chunk in tool_call_stream_renderers.render_completed_note(
                    tool_name=name,
                    status=parsed_status,
                ):
                    await _ensure_message_started()
                    any_delta_emitted["flag"] = True
                    await _emit_text_only(rendered_chunk)
                    await voice_reply_stream.push_chunk(rendered_chunk)
            await runtime.emit_tool_call_completed(
                payload={
                    "tool_name": name,
                    "tool_call_id": event_id,
                    "output": _safe_stringify(output),
                }
            )

        async def _on_tool_failed(name: str, error: str, event_id: str) -> None:
            if not _run_is_current():
                return
            if error and not tool_failure_message["text"]:
                tool_failure_message["text"] = str(error).strip()
            await runtime.emit_tool_call_failed(
                payload={
                    "tool_name": name,
                    "tool_call_id": event_id,
                    "error": error,
                }
            )

        async def _emit_nested_tool_started(
            name: str, args: Any, event_id: str, parent_crew_tool: str
        ) -> None:
            if not _run_is_current():
                return
            safe_eid = str(event_id or "").strip() or "na"
            await runtime.enter_tool_running()
            await runtime.emit_tool_call_started(
                payload={
                    "tool_name": name,
                    "tool_call_id": f"sub:{parent_crew_tool}:{safe_eid}",
                    "arguments": _safe_stringify(args),
                    "args_preview": _args_preview_for_ui(args),
                    "parent_crew_tool": parent_crew_tool,
                }
            )

        async def _emit_nested_tool_finished(
            name: str, output: Any, event_id: str, parent_crew_tool: str
        ) -> None:
            if not _run_is_current():
                return
            safe_eid = str(event_id or "").strip() or "na"
            await runtime.enter_reasoning()
            parsed_output = try_parse_tool_result_payload(output)
            if parsed_output:
                _bind_task_ids_to_turn(turn, parsed_output)
                for file_info in build_chat_files_from_tool_result(
                    parsed_output,
                    default_category=default_category_for_tool(name),
                    source_tool=name,
                ):
                    await runtime.emit_artifact_created(payload=file_info)
            out_s = _safe_stringify(output)
            if len(out_s) > 2000:
                out_s = out_s[:1997] + "..."
            await runtime.emit_tool_call_completed(
                payload={
                    "tool_name": name,
                    "tool_call_id": f"sub:{parent_crew_tool}:{safe_eid}",
                    "output": out_s,
                    "parent_crew_tool": parent_crew_tool,
                }
            )

        async def _emit_nested_tool_failed(
            name: str, error: str, event_id: str, parent_crew_tool: str
        ) -> None:
            if not _run_is_current():
                return
            safe_eid = str(event_id or "").strip() or "na"
            await runtime.emit_tool_call_failed(
                payload={
                    "tool_name": name,
                    "tool_call_id": f"sub:{parent_crew_tool}:{safe_eid}",
                    "error": str(error or "")[:4000],
                    "parent_crew_tool": parent_crew_tool,
                }
            )

        session_nested_tool_hooks: Dict[str, Any] = {
            "emit_tool_started": lambda name, args, eid, parent: _schedule(
                _emit_nested_tool_started(name, args, eid, parent)
            ),
            "emit_tool_finished": lambda name, output, eid, parent: _schedule(
                _emit_nested_tool_finished(name, output, eid, parent)
            ),
            "emit_tool_failed": lambda name, err, eid, parent: _schedule(
                _emit_nested_tool_failed(name, err, eid, parent)
            ),
        }

        # 收集所有被调度到主 loop 的回调 future，保证 complete_run
        # 之前全部执行完毕，避免 message_delta 被排在 message_completed
        # 之后（WS 客户端会把它们视作乱序/丢失）。
        scheduled_futures: List["asyncio.Future[Any]"] = []

        def _schedule(coro) -> None:
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                scheduled_futures.append(fut)
            except Exception as exc:
                logger.warning("schedule crew event callback failed: %s", exc)

        async def _on_task_event(event: Dict[str, Any]) -> None:
            await handle_task_event(runtime=runtime, turn=turn, event=event)

        def _task_event_callback(event: Dict[str, Any]) -> None:
            _schedule(_on_task_event(dict(event or {})))

        async def _drain_scheduled_callbacks() -> None:
            if not scheduled_futures:
                return
            pending: List["asyncio.Future[Any]"] = [
                asyncio.wrap_future(f) for f in scheduled_futures if not f.done()
            ]
            if not pending:
                return
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[master-run] drain scheduled callbacks timed out, pending=%d",
                    len(pending),
                )

        # ---------- 在工作线程里跑 crew，主线程监听事件 ----------

        started_at = datetime.utcnow()
        record_run_started(
            turn.run_id or "",
            agent_id=agent_id,
            source_type=command.source_type,
            conversation_id=command.conversation_id,
            message=command.message,
            runtime_config=command.runtime_config,
            started_at=started_at,
        )

        try:
            raw_result = await asyncio.wait_for(
                asyncio.to_thread(
                    _run_crew_blocking,
                    agent_record=agent_record,
                    command=command,
                    agent_run_id=turn.run_id or "",
                    schedule=_schedule,
                    on_llm_started=_on_llm_started,
                    on_llm_chunk=_on_llm_chunk,
                    on_tool_started=_on_tool_started,
                    on_tool_finished=_on_tool_finished,
                    on_tool_failed=_on_tool_failed,
                    task_event_callback=_task_event_callback,
                    session_nested_tool_hooks=session_nested_tool_hooks,
                    session_runtime=runtime,
                ),
                timeout=_DEFAULT_RUN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            await voice_reply_stream.teardown()
            await runtime.fail_run(
                code="internal_error",
                message=f"master agent run timed out after {_DEFAULT_RUN_TIMEOUT_SECONDS}s",
            )
            if turn.run_id:
                record_run_failed(turn.run_id, error="timeout", completed_at=datetime.utcnow())
            return
        except asyncio.CancelledError:
            # interrupt 路径：先清 consumer，再把取消向上冒
            await voice_reply_stream.teardown()
            raise
        except Exception as exc:
            log_msg, user_msg, _, _ = summarize_chat_run_error(exc)
            logger.error("master crew failed: %s", log_msg)
            await voice_reply_stream.teardown()
            await runtime.fail_run(code="internal_error", message=user_msg)
            if turn.run_id:
                record_run_failed(turn.run_id, error=log_msg, completed_at=datetime.utcnow())
            return

        # 等待所有被 run_coroutine_threadsafe 调度的事件回调完成，
        # 否则 message_delta 可能排在本方法后续发出的 message_completed
        # 之后，导致客户端看到"started→completed"而丢掉中间的 delta。
        await _drain_scheduled_callbacks()

        if not _run_is_current():
            logger.info(
                "[master-run] stale run ignored after callback drain run_id=%s",
                current_run_id,
            )
            await voice_reply_stream.teardown()
            return

        output_text = _result_to_text(raw_result)
        if not output_text and tool_failure_message["text"]:
            output_text = tool_failure_message["text"]
        if tool_call_stream_renderers.active and turn.assistant_text().strip():
            output_text = turn.assistant_text()
        completed_at = datetime.utcnow()
        elapsed = max(0.0, (completed_at - started_at).total_seconds())
        usage_metrics = _extract_usage_metrics(raw_result, elapsed_seconds=elapsed)
        logger.info(
            "[master-run] run_id=%s chars=%d started=%s any_delta=%s preview=%r",
            turn.run_id,
            len(output_text or ""),
            message_started_emitted["flag"],
            any_delta_emitted["flag"],
            (output_text or "")[:200],
        )

        # 若 LLM 流里没任何真正的 chunk 回来（例如 provider/中间层未产出 stream event，
        # 或返回工具调用后直接结束），兜底把最终 output_text 补发给前端。
        # 默认关闭 fake stream，优先保持“能真流式就真流式”；只有显式打开环境变量时，
        # 才把最终文本切块模拟打字机效果。
        if not any_delta_emitted["flag"] and output_text:
            await _ensure_message_started()
            any_delta_emitted["flag"] = True
            fake_stream = os.environ.get("VITOOM_MASTER_FAKE_STREAM", "0").strip() in {"1", "true", "True"}
            if fake_stream:
                chunk_size = max(1, int(os.environ.get("VITOOM_MASTER_FAKE_STREAM_CHARS", "8")))
                delay_ms = max(0, int(os.environ.get("VITOOM_MASTER_FAKE_STREAM_DELAY_MS", "15")))
                for i in range(0, len(output_text), chunk_size):
                    await _emit_and_buffer(output_text[i : i + chunk_size])
                    if delay_ms:
                        await asyncio.sleep(delay_ms / 1000.0)
            else:
                await _emit_and_buffer(output_text)

        if turn.run_id:
            record_run_completed(
                turn.run_id,
                output_text=output_text,
                usage_metrics=usage_metrics,
                completed_at=completed_at,
            )
            try:
                AgentRun.update(
                    turn.run_id,
                    status="completed",
                    result_summary=output_text[: 64 * 1024],
                    usage_metrics=usage_metrics,
                    completed_at=completed_at,
                    error_message=None,
                )
            except Exception:
                # SessionRuntime.complete_run 之后还会再写一次，失败不致命
                pass

        if voice_reply_stream.enabled:
            await voice_reply_stream.drain()
        elif should_emit_voice_reply(runtime, turn, output_text):
            try:
                await synthesize_voice_reply(
                    runtime=runtime,
                    turn=turn,
                    assistant_text=output_text,
                    logger=logger,
                )
            except Exception as exc:
                logger.warning(
                    "[master-run] voice reply failed run_id=%s err=%s",
                    turn.run_id,
                    exc,
                    exc_info=True,
                )
                await runtime.emit_error_event(
                    code="tts_failed",
                    message=f"TTS failed: {exc}",
                    recoverable=True,
                )

        await runtime.complete_run(
            assistant_text=output_text,
            usage_metrics=usage_metrics or None,
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _resolve_runtime_config(
        self,
        runtime: SessionRuntime,
        agent_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """聚合 Agent 运行配置。

        合并优先级（后者覆盖前者）：

        1. Agent preset 的 ``config.runtime_defaults``
           —— 比如 Master preset 的 ``max_tools_per_run: 3`` /
           ``process: sequential`` / ``priority: 5``。这一层与离线
           ``agent_worker`` 的 ``_get_runtime_defaults`` 对齐，保证
           chat ws 与 offline worker 在 ToolSelection / Crew 行为上一致。
        2. Conversation.metadata 里的会话级覆盖（目前只接 ``load_name``）。
        """
        cfg: Dict[str, Any] = {}

        config = dict(agent_record.get("config") or {}) if agent_record else {}
        runtime_defaults = config.get("runtime_defaults")
        if isinstance(runtime_defaults, dict):
            cfg.update(runtime_defaults)

        try:
            conv = Conversation.get_by_id(runtime.session_id)
        except Exception:
            conv = None
        if conv and isinstance(conv.get("metadata"), dict):
            meta = conv["metadata"]
            load_name = str(meta.get("load_name") or "").strip()
            if load_name:
                cfg["load_name"] = load_name
        elif self._default_load_name and not cfg.get("load_name"):
            cfg["load_name"] = self._default_load_name
        return cfg

# ---------------------------------------------------------------------------
# 在工作线程里跑 crew（阻塞），通过 crewai_event_bus 注册事件转发
# ---------------------------------------------------------------------------


def _run_crew_blocking(
    *,
    agent_record: Dict[str, Any],
    command: AgentCommand,
    agent_run_id: str,
    schedule,
    on_llm_started,
    on_llm_chunk,
    on_tool_started,
    on_tool_finished,
    on_tool_failed,
    task_event_callback=None,
    session_nested_tool_hooks: Optional[Dict[str, Any]] = None,
    session_runtime: Optional["SessionRuntime"] = None,
) -> Any:
    """同步执行：ToolSelection → CrewFactory → kickoff，把事件桥接回主 loop。"""

    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.llm_events import (
        LLMCallStartedEvent,
        LLMStreamChunkEvent,
    )
    from crewai.events.types.tool_usage_events import (
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )

    agent_specs: List[AgentSpec] = AgentSpec.list_from_agent_record(agent_record)
    task_specs: List[TaskSpec] = TaskSpec.list_from_agent_record(agent_record)

    unique_tool_names: List[str] = []
    seen_tool_names: set = set()
    preferred_tool_names: List[str] = []
    for spec in agent_specs:
        for name in spec.tools:
            if name not in seen_tool_names:
                seen_tool_names.add(name)
                unique_tool_names.append(name)
        for name in spec.preferred_tool_names:
            if name not in preferred_tool_names:
                preferred_tool_names.append(name)

    uses_global_pool = any(spec.tool_pool == GLOBAL_TOOL_POOL for spec in agent_specs)

    tool_selection_started = time.perf_counter()
    selected_tool_names = ToolSelectionService().select_tool_names(
        unique_tool_names,
        command=command,
        task_specs=task_specs,
        runtime_allowlist=command.runtime_config.get("tool_allowlist"),
        max_tools=command.runtime_config.get("max_tools_per_run"),
        pool="global" if uses_global_pool else "declared",
        preferred_tool_names=preferred_tool_names,
    )
    tool_selection_elapsed_ms = (time.perf_counter() - tool_selection_started) * 1000
    logger.info(
        "[master-run] tool_selection elapsed_ms=%.2f run_id=%s turn_id=%s pool=%s declared=%d preferred=%d selected=%d selected_names=%s",
        tool_selection_elapsed_ms,
        agent_run_id,
        command.context.get("turn_id") if isinstance(command.context, dict) else None,
        "global" if uses_global_pool else "declared",
        len(unique_tool_names),
        len(preferred_tool_names),
        len(selected_tool_names),
        list(selected_tool_names),
    )
    record_tool_selected(
        "",  # run_id 在 master_runtime 外层已记录 run_started；这里仅补选工具事件
        declared=unique_tool_names,
        selected=list(selected_tool_names),
        pool="global" if uses_global_pool else "declared",
        preferred=preferred_tool_names,
        max_tools=command.runtime_config.get("max_tools_per_run"),
    )

    # 默认真流式优先；若上游/中间层未稳定产出 chunk，再回退到外层的非流式兜底逻辑。
    # 如需临时禁用真实流式，可显式设置 VITOOM_MASTER_STREAM=0。
    use_stream = os.environ.get("VITOOM_MASTER_STREAM", "1").strip() in {"1", "true", "True"}
    if not selected_tool_names:
        logger.info(
            "[master-run] no tools selected; using direct LLM fast path run_id=%s",
            agent_run_id,
        )
        return run_no_tool_completion(
            agent_specs=agent_specs,
            task_specs=task_specs,
            command=command,
            stream=use_stream,
            on_llm_started=lambda: schedule(on_llm_started()),
            on_llm_chunk=lambda chunk: schedule(on_llm_chunk(chunk)),
        )

    tool_resolver = ToolResolver()
    resolved_tools = tool_resolver.resolve_tools(
        selected_tool_names,
        agent_run_id=agent_run_id,
        runtime_allowlist=command.runtime_config.get("tool_allowlist"),
        crew_tool_context={
            "user_id": command.user_id,
            "agent_run_id": agent_run_id,
            "turn_id": command.context.get("turn_id") if isinstance(command.context, dict) else None,
            "task_event_callback": task_event_callback if callable(task_event_callback) else None,
            "conversation_id": command.conversation_id,
            "session_runtime": session_runtime,
            "runtime_config": command.runtime_config,
            "source_type": command.source_type or "chat-ws",
            "session_nested_tool_hooks": session_nested_tool_hooks,
        },
    )
    tools_by_name = {name: tool for name, tool in zip(selected_tool_names, resolved_tools)}

    for spec in agent_specs:
        if spec.tool_pool != GLOBAL_TOOL_POOL:
            continue
        merged: list = []
        seen: set = set()
        for name in spec.tools + list(selected_tool_names):
            norm = str(name or "").strip()
            if norm and norm not in seen:
                seen.add(norm)
                merged.append(norm)
        spec.tools = merged

    crew, inputs = CrewFactory().build(
        agent_specs=agent_specs,
        task_specs=task_specs,
        command=command,
        tools_by_name=tools_by_name,
        process_name="sequential",
        stream=use_stream,
    )

    # ---- 在 scoped 内订阅 LLM / Tool 事件 ----
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMCallStartedEvent)
        def _on_call_started(source, event):  # noqa: ARG001
            schedule(on_llm_started())

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def _on_chunk(source, event):  # noqa: ARG001
            schedule(
                on_llm_chunk(
                    str(getattr(event, "chunk", "") or ""),
                    call_type=getattr(event, "call_type", None),
                    tool_call=getattr(event, "tool_call", None),
                )
            )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _on_tool_start(source, event):  # noqa: ARG001
            if is_nested_tool_event_scope_active():
                return
            schedule(
                on_tool_started(
                    str(getattr(event, "tool_name", "") or ""),
                    getattr(event, "tool_args", None),
                    str(getattr(event, "event_id", "") or ""),
                )
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _on_tool_done(source, event):  # noqa: ARG001
            if is_nested_tool_event_scope_active():
                return
            schedule(
                on_tool_finished(
                    str(getattr(event, "tool_name", "") or ""),
                    getattr(event, "output", None),
                    str(getattr(event, "event_id", "") or ""),
                )
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def _on_tool_err(source, event):  # noqa: ARG001
            if is_nested_tool_event_scope_active():
                return
            schedule(
                on_tool_failed(
                    str(getattr(event, "tool_name", "") or ""),
                    str(getattr(event, "error", "") or ""),
                    str(getattr(event, "event_id", "") or ""),
                )
            )

        raw_result = crew.kickoff(inputs=inputs)

    return raw_result


# ---------------------------------------------------------------------------
# 工具函数（复刻 agent_worker 的轻量版本）
# ---------------------------------------------------------------------------


def _result_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    # CrewAI CrewOutput 有 .raw 字段
    raw = getattr(value, "raw", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    try:
        import json

        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _safe_stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _args_preview_for_ui(raw_args: Any, *, max_chars: int = 400) -> Any:
    """供 WS payload.args_preview：短预览，避免 detail 恒为空。"""
    if raw_args is None:
        return None
    try:
        s = _safe_stringify(raw_args)
    except Exception:
        return None
    if not s:
        return None
    if len(s) <= max_chars:
        try:
            import json

            return json.loads(s)
        except Exception:
            return s
    return s[: max_chars - 3] + "..."


def _collect_task_ids_from_payload(payload: Dict[str, Any]) -> List[str]:
    if not isinstance(payload, dict):
        return []

    results: List[str] = []
    seen = set()

    def _append(raw_task_id: Any) -> None:
        normalized = str(raw_task_id or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        results.append(normalized)

    _append(payload.get("task_id"))
    for item in payload.get("items") or []:
        if isinstance(item, dict):
            _append(item.get("task_id"))
    return results


def _bind_task_ids_to_turn(turn: Turn, payload: Dict[str, Any]) -> None:
    task_ids = _collect_task_ids_from_payload(payload)
    if not task_ids:
        return

    for task_id in task_ids:
        turn.bind_task_id(task_id)

    if turn.run_id and len(task_ids) == 1:
        try:
            AgentRun.update(turn.run_id, task_id=task_ids[0])
        except Exception:
            logger.debug("failed to bind task to agent run (ignored)")

    if turn.run_id:
        for task_id in task_ids:
            try:
                Task.update(task_id, agent_run_id=turn.run_id)
            except Exception:
                logger.debug("failed to backfill task.agent_run_id task=%s", task_id)


def _extract_usage_metrics(raw_result: Any, *, elapsed_seconds: Optional[float]) -> Dict[str, Any]:
    """从 CrewOutput.token_usage 抽 prompt/completion/total tokens，并算 tok/s。"""

    def _coerce_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    usage_obj = getattr(raw_result, "token_usage", None)
    if usage_obj is None and isinstance(raw_result, dict):
        usage_obj = raw_result.get("token_usage")

    if usage_obj is None:
        return {
            "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        }

    if isinstance(usage_obj, dict):
        usage = dict(usage_obj)
    else:
        usage = {}
        for method in ("model_dump", "dict"):
            dumper = getattr(usage_obj, method, None)
            if callable(dumper):
                try:
                    usage = dumper()
                    break
                except Exception:
                    continue
        if not usage:
            for key in (
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
                "successful_requests",
                "cached_prompt_tokens",
            ):
                usage[key] = getattr(usage_obj, key, None)

    def _first(*keys: str) -> Any:
        for key in keys:
            v = usage.get(key)
            if v is not None:
                return v
        return None

    total = _coerce_int(_first("total_tokens", "totalTokens"))
    prompt = _coerce_int(_first("prompt_tokens", "promptTokens"))
    completion = _coerce_int(_first("completion_tokens", "completionTokens", "output_tokens"))
    requests = _coerce_int(_first("successful_requests", "successfulRequests"))
    cached = _coerce_int(_first("cached_prompt_tokens", "cachedPromptTokens"))

    metrics: Dict[str, Any] = {
        "prompt_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        "successful_requests": requests,
        "cached_prompt_tokens": cached,
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
    }
    if elapsed_seconds and elapsed_seconds > 0:
        divisor = completion if completion else total
        if divisor:
            metrics["tok_s_total"] = round(divisor / elapsed_seconds, 2)
    return metrics


__all__ = ["MasterAgentRuntime"]


async def _maybe_handle_slash_command(
    runtime: SessionRuntime,
    turn: Turn,
    user_text: str,
) -> bool:
    from backend.services.chat.slash_commands import (
        ensure_slash_commands_registered,
        try_dispatch,
    )

    ensure_slash_commands_registered()
    result = await try_dispatch(user_text, user_id=runtime.user_id)
    if not result.handled:
        return False

    unique_task_ids: List[str] = []
    seen = set()
    for task_id in result.task_ids:
        normalized = str(task_id or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_task_ids.append(normalized)

    if turn.run_id and len(unique_task_ids) == 1:
        try:
            AgentRun.update(turn.run_id, task_id=unique_task_ids[0])
        except Exception:
            logger.debug("failed to bind slash task to agent run (ignored)")

    if turn.run_id:
        for task_id in unique_task_ids:
            turn.bind_task_id(task_id)
            try:
                Task.update(task_id, agent_run_id=turn.run_id)
            except Exception:
                logger.debug("failed to backfill task.agent_run_id for slash task=%s", task_id)

    await runtime.enter_streaming_output()
    await runtime.emit_message_started()

    for artifact in result.artifacts:
        if not isinstance(artifact, dict):
            continue
        await runtime.emit_artifact_created(payload=artifact)

    assistant_text = str(result.assistant_text or "").strip()
    if not assistant_text:
        if result.status == "usage":
            assistant_text = "命令帮助信息不可为空。"
        elif result.error:
            assistant_text = f"命令执行失败：{result.error}"
        elif not result.artifacts:
            assistant_text = "命令已执行完成。"

    if assistant_text:
        await runtime.emit_message_delta(assistant_text)
    await runtime.complete_run(assistant_text=assistant_text)
    return True
