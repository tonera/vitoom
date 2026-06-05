"""`image_generator` / `video_generator` 等媒体类工具的共享实现。

这里抽取的是"把一个已经填好的 `TaskCreateRequest` 交给后端 + 订阅 WS +
聚合 `result` 文件 + 超时 DB 兜底"的通用流程。image/video 仅在以下三处不同：

1. `TaskCreateRequest.task_type`（image vs video）
2. 日志 / tool 名（用于 log 与结果返回的 `tool` 字段）
3. 收集到的每个 file 字典里需要保留的字段（image 有 seed / width / height，
   video 额外有 duration / fps / resolution / aspect_ratio）

把这些可变部分抽成参数，主流程就可以完全复用。
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _format_submit_error(exc: Exception, *, task_kind_label: str, request_obj: Any) -> tuple[str, bool]:
    """把建任务阶段的后端错误整理成给 LLM 看的工具错误。

    Invalid model_name 通常是服务端模型表/配置问题。若把原始错误直接回给 LLM，
    它容易自行编造新 model_name 再试；这里显式标记为不可重试。
    """
    raw_detail = getattr(exc, "detail", None)
    detail = str(raw_detail if raw_detail not in (None, "") else exc).strip()
    if "Invalid model_name" in detail:
        model_name = str(getattr(request_obj, "model_name", "") or "").strip()
        model_label = f" '{model_name}'" if model_name else ""
        return (
            "non-retryable configuration error: "
            f"failed to create {task_kind_label} task because the configured model_name{model_label} "
            "is not available in the server model registry. Do not invent or switch model_name; "
            "ask the operator to sync the model registry/configuration and retry later."
        ), False
    return f"failed to create {task_kind_label} task: {exc}", True


# 常见的 file 字段映射（image / 通用媒体）：对 LLM 只暴露 ``url``（原资源可访问地址）与
# ``thumb_url``（图/视频缩略图，有则出现）；不再输出 ``http_url`` 以免与 ``url`` 混淆。
_DEFAULT_FILE_FIELDS = (
    "file_id",
    "url",
    "thumb_url",
    "storage_path",
    "file_name",
    "width",
    "height",
    "seed",
    "index",
    "thumbnail_path",
)

# 视频增量字段（除了默认字段，还会多拿这些）
_VIDEO_EXTRA_FILE_FIELDS = (
    "duration",
    "fps",
    "resolution",
    "aspect_ratio",
    "bitrate",
    "mime_type",
)

# 音频增量字段（TTS 输出的音频文件，ASR 输出的文本文件）
_AUDIO_EXTRA_FILE_FIELDS = (
    "duration",
    "sample_rate",
    "mime_type",
    "file_size",
    "bitrate",
)


def _extract_file_info(info: Dict[str, Any], fields: tuple) -> Dict[str, Any]:
    """聚合工具 ``files[]``：仅输出 ``fields`` 中的键。遗留数据若只有 ``http_url``，并入 ``url``。"""
    merged = dict(info)
    if merged.get("http_url") and not merged.get("url"):
        merged["url"] = merged["http_url"]

    result: Dict[str, Any] = {}
    for key in fields:
        if key in merged:
            result[key] = merged.get(key)
    if "url" in fields and not result.get("url") and merged.get("http_url"):
        result["url"] = merged["http_url"]
    result["file_id"] = str(info.get("file_id") or "").strip()
    return result


def _dispatch_capability_for_tool(tool_name: str) -> str:
    normalized = str(tool_name or "").strip().lower()
    if normalized == "audio_asr":
        return "asr"
    if normalized in {"audio_tts", "audio_drama_tts"}:
        return "tts"
    return ""


def submit_and_collect(
    *,
    tool_name: str,
    user_id: str,
    request_obj: Any,
    expected_total: int,
    effective_timeout: float,
    task_kind_label: str,
    file_fields: Optional[tuple] = None,
    submit_timeout: float = 30.0,
    stream_text_types: Optional[tuple] = None,
    extra_result_fields: Optional[tuple] = None,
    task_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """image / video 工具共用的"提交任务 + 等 WS 消息 + DB 兜底"主循环。

    Args:
        tool_name: 返回结果里 `tool` 字段的值（例如 "image_generator"）。
        user_id: 由上层从 context 或 slash handler 绑定。
        request_obj: 已经构造好的 `TaskCreateRequest` 实例。
        expected_total: 期望生成的文件数（image 为 generate_num；video 通常为 1）。
        effective_timeout: 全局等待超时秒数。
        task_kind_label: 日志友好名（"image"/"video"/"audio-asr"/"audio-tts"）。
        file_fields: 从 result 消息里保留哪些字段，默认 image 字段集。
        stream_text_types: 要监听的"流式文本增量"消息类型集合（例如
            {"text_stream_delta"}）。命中时会把 `delta` 追加到结果 `text` 字段，
            保留完整的结构化增量到 `stream_deltas`。给 ASR 这种文本增量型
            任务使用。
        extra_result_fields: 从最终 `result` 消息中额外保留的字段（例如
            "language", "segments"），直接平铺到返回 dict 里。

    Returns:
        dict 永远包含 `tool / task_id / status / total / files / error?`。
        当 `stream_text_types` 非空时，额外含 `text / stream_deltas`。
    """
    from backend.api.tasks.routes import _create_task
    from backend.services.chat.router import DispatchSelectionError, DispatchSpec, get_dispatch_router
    from backend.services.chat.user_messages import unavailable_message
    from backend.websocket.manager import get_websocket_manager

    tool_result: Dict[str, Any] = {
        "tool": tool_name,
        "task_id": None,
        "status": "failed",
        "total": 0,
        "files": [],
    }

    ws_manager = get_websocket_manager()
    loop = ws_manager.get_event_loop()
    if loop is None:
        tool_result["error"] = (
            "WebSocketManager event loop not initialized; ensure FastAPI startup ran"
        )
        return tool_result

    fields = file_fields or _DEFAULT_FILE_FIELDS

    async def _precheck_service_availability() -> Optional[str]:
        task_type = str(getattr(request_obj, "task_type", "") or "").strip().lower()
        if not task_type:
            return None
        model_name = str(getattr(request_obj, "model_name", "") or "").strip()
        connected_service_ids = await ws_manager.get_connected_inference_service_ids()
        try:
            get_dispatch_router().pick_service(
                DispatchSpec(
                    service_type=task_type,
                    require_supports_task=True,
                    reason=f"task_type={task_type}",
                    load_name=model_name,
                    capability=_dispatch_capability_for_tool(tool_name),
                ),
                connected_service_ids=connected_service_ids,
            )
        except DispatchSelectionError as exc:
            logger.warning("%s precheck failed before _create_task: %s", tool_name, exc)
            return str(exc)
        return None

    try:
        unavailable_reason = asyncio.run_coroutine_threadsafe(
            _precheck_service_availability(), loop
        ).result(timeout=submit_timeout)
    except Exception as exc:
        logger.exception("%s: precheck service availability failed", tool_name)
        tool_result["error"] = f"failed to precheck {task_kind_label} service: {exc}"
        return tool_result
    if unavailable_reason:
        # 把 dispatch 的具体 reason 透传给 LLM/调用方，便于 self-correct（比如
        # 引导用户启动正确的推理服务）。user-facing 兜底文案保留作为前缀，
        # reason 附在括号里——LLM 同时看到 user-friendly 文本与 dispatch 细节。
        tool_result["error"] = f"{unavailable_message()} (reason: {unavailable_reason})"
        logger.warning(
            "%s precheck unavailable: %s",
            tool_name,
            unavailable_reason,
        )
        return tool_result

    # -------------- 建任务 + 订阅（必须在 loop 线程里保证 register 顺序）--------------
    async def _submit_and_subscribe() -> tuple[str, asyncio.Queue]:
        task_response = await _create_task(request_obj, user_id)
        task_id = task_response.task_id
        queue = await ws_manager.register_task_subscriber(task_id)
        return task_id, queue

    try:
        submit_future = asyncio.run_coroutine_threadsafe(_submit_and_subscribe(), loop)
        task_id, queue = submit_future.result(timeout=submit_timeout)
    except Exception as exc:
        logger.exception("%s: failed to submit task", tool_name)
        error_message, retryable = _format_submit_error(
            exc,
            task_kind_label=task_kind_label,
            request_obj=request_obj,
        )
        tool_result["error"] = error_message
        tool_result["retryable"] = retryable
        return tool_result

    tool_result["task_id"] = task_id
    if callable(task_event_callback):
        try:
            task_event_callback(
                {
                    "type": "task_bound",
                    "task_id": task_id,
                    "status": "pending",
                    "progress": 0,
                    "task_kind": task_kind_label,
                }
            )
        except Exception:
            logger.debug("%s: task_event_callback(task_bound) failed (ignored)", tool_name)
    logger.info(
        "%s task created: task_id=%s user=%s expected=%d",
        tool_name,
        task_id,
        user_id,
        expected_total,
    )

    collected: Dict[str, Dict[str, Any]] = {}
    running_expected = max(1, int(expected_total or 1))
    deadline = time.monotonic() + effective_timeout
    terminal_status: Optional[str] = None
    error_message: Optional[str] = None

    stream_types_set = set(stream_text_types or ())
    stream_text_buffer: List[str] = []
    stream_deltas: List[Dict[str, Any]] = []
    extra_result: Dict[str, Any] = {}

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminal_status = "timeout"
                break

            try:
                get_future = asyncio.run_coroutine_threadsafe(queue.get(), loop)
                message = get_future.result(timeout=remaining)
            except TimeoutError:
                get_future.cancel()
                terminal_status = "timeout"
                break
            except Exception as exc:
                logger.exception("%s: error while waiting for task message", tool_name)
                error_message = str(exc)
                terminal_status = "failed"
                break

            msg_type = str(message.get("type") or "").strip().lower()

            if msg_type == "task_status":
                status = str(message.get("status") or "").strip().lower()
                if callable(task_event_callback):
                    try:
                        task_event_callback(
                            {
                                "type": "task_status",
                                "task_id": task_id,
                                "status": status or "processing",
                                "progress": message.get("progress"),
                                "error": message.get("error"),
                                "task_kind": task_kind_label,
                            }
                        )
                    except Exception:
                        logger.debug("%s: task_event_callback(task_status) failed (ignored)", tool_name)
                if status == "failed":
                    terminal_status = "failed"
                    error_message = str(message.get("error") or "inference failed")
                    break
                if status == "cancelled":
                    terminal_status = "cancelled"
                    break
                continue

            if msg_type in {"audio_stream_start", "audio_stream_chunk", "audio_stream_end"}:
                if callable(task_event_callback):
                    try:
                        task_event_callback(dict(message))
                    except Exception:
                        logger.debug("%s: task_event_callback(%s) failed (ignored)", tool_name, msg_type)
                continue

            if msg_type == "result":
                files = message.get("files") or []
                for info in files:
                    file_id = str(info.get("file_id") or "").strip()
                    if not file_id or file_id in collected:
                        continue
                    collected[file_id] = _extract_file_info(info, fields)

                message_total = message.get("total")
                if isinstance(message_total, int) and message_total > 0:
                    running_expected = max(running_expected, message_total)

                if extra_result_fields:
                    for key in extra_result_fields:
                        if key in message and key not in extra_result:
                            extra_result[key] = message.get(key)

                result_status = str(message.get("status") or "").strip().lower()
                progress = message.get("progress")
                if callable(task_event_callback):
                    try:
                        task_event_callback(
                            {
                                "type": "task_result",
                                "task_id": task_id,
                                "status": result_status or "completed",
                                "progress": progress,
                                "total": message_total,
                                "files_count": len(collected),
                                "task_kind": task_kind_label,
                            }
                        )
                    except Exception:
                        logger.debug("%s: task_event_callback(task_result) failed (ignored)", tool_name)
                is_completed = (
                    result_status == "completed"
                    and isinstance(progress, int)
                    and progress >= 100
                )
                if is_completed or len(collected) >= running_expected:
                    terminal_status = "completed"
                    break
                continue

            if stream_types_set and msg_type in stream_types_set:
                delta_text = message.get("delta")
                if isinstance(delta_text, str) and delta_text:
                    stream_text_buffer.append(delta_text)
                # 原样保留每条增量（包含 sequence / is_final / segment 等），方便
                # 上层按需要做结构化渲染
                stream_deltas.append(
                    {k: v for k, v in message.items() if k not in {"task_id", "service_id"}}
                )
                continue

            # 其它消息类型（ping/pong 等）：忽略
    finally:
        try:
            asyncio.run_coroutine_threadsafe(
                ws_manager.unregister_task_subscriber(task_id, queue), loop
            ).result(timeout=5.0)
        except Exception:
            logger.debug("%s: failed to unregister task subscriber (ignored)", tool_name)

    # 超时兜底：再查一次 DB
    if terminal_status == "timeout" and not collected:
        try:
            from backend.database import File, Task

            db_files = File.list_by_task(task_id, limit=100, offset=0) or []
            for row in db_files:
                file_id = str(row.get("id") or "").strip()
                if not file_id or file_id in collected:
                    continue
                metadata = row.get("metadata") or {}
                merged_info = {
                    "file_id": file_id,
                    "url": row.get("http_url") or row.get("url"),
                    "storage_path": row.get("storage_path"),
                    "file_name": row.get("file_name"),
                    **{k: metadata.get(k) for k in fields if k not in {
                        "file_id", "url", "storage_path", "file_name"
                    }},
                }
                collected[file_id] = _extract_file_info(merged_info, fields)
            task_row = Task.get_by_id(task_id)
            if task_row and task_row.get("status") == "completed":
                terminal_status = "completed"
        except Exception:
            logger.debug("%s: DB fallback lookup failed (ignored)", tool_name)

    sorted_files = sorted(
        collected.values(),
        key=lambda item: (item.get("index") if isinstance(item.get("index"), int) else math.inf),
    )
    tool_result["files"] = sorted_files
    tool_result["total"] = len(sorted_files)
    tool_result["status"] = terminal_status or ("completed" if sorted_files else "failed")
    if error_message:
        tool_result["error"] = error_message
    elif terminal_status == "timeout":
        tool_result["error"] = (
            f"{task_kind_label} generation did not finish within {effective_timeout:.0f}s"
        )

    if stream_types_set:
        tool_result["text"] = "".join(stream_text_buffer)
        tool_result["stream_deltas"] = stream_deltas

    for key, value in extra_result.items():
        tool_result.setdefault(key, value)

    return tool_result


def video_file_fields() -> tuple:
    return _DEFAULT_FILE_FIELDS + _VIDEO_EXTRA_FILE_FIELDS


def audio_file_fields() -> tuple:
    """音频任务（TTS 的 .wav / ASR 的 .txt）在结果里保留的 file 字段集合。"""
    return _DEFAULT_FILE_FIELDS + _AUDIO_EXTRA_FILE_FIELDS
