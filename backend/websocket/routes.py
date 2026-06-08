"""
WebSocket路由
提供WebSocket端点和连接处理
"""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Optional

from backend.websocket.manager import WebSocketManager, get_websocket_manager
from backend.auth import get_optional_user_id
from backend.core.logger import get_app_logger
from backend.core.response import ok
from backend.database import Task, File
from backend.database import Model
from backend.services.inference.service import get_inference_service_manager
from backend.utils import utc_now
from backend.utils.artifact_storage import normalize_storage_label, resolve_artifact_public_url
from backend.models.download_text import upsert_download_block, clear_download_block
from backend.i18n.ws_messages import enrich_task_ws_message

logger = get_app_logger(__name__)

router = APIRouter()


def _inference_audio_binary_len(message: dict) -> int:
    """推理侧 ``session.audio.chunk``：JSON meta 后紧跟 binary PCM 时的 payload 字节数。"""
    if str(message.get("type") or "") != "session.audio.chunk":
        return 0
    try:
        return max(0, int(message.get("bytes_len") or 0))
    except (TypeError, ValueError):
        return 0


@router.websocket("/ws/task/{task_id}")
async def websocket_task_progress(
    websocket: WebSocket,
    task_id: str,
    token: Optional[str] = None,
    locale: Optional[str] = None,
):
    """
    WebSocket端点：用户前端任务进度推送
    
    - **task_id**: 任务ID
    - **token**: JWT Token（通过查询参数传递，必需）
    
    连接后会自动推送任务进度更新
    """
    manager = get_websocket_manager()
    
    # 验证任务是否存在
    task = Task.get_by_id(task_id)
    if not task:
        await websocket.close(code=1008, reason="Task not found")
        return
    
    # 验证用户权限（token必需）
    user_id = None
    if token:
        try:
            from backend.auth.jwt_utils import verify_token
            payload = verify_token(token)
            user_id = payload.get("sub")
            
            # 检查用户是否有权限访问该任务
            if task["user_id"] != user_id:
                await websocket.close(code=1008, reason="Permission denied")
                return
        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            await websocket.close(code=1008, reason="Invalid token")
            return
    else:
        await websocket.close(code=1008, reason="Token required")
        return
    
    # 建立用户前端连接
    await manager.connect_user(websocket, task_id, user_id)
    
    try:
        # 发送初始状态（使用 task_status 消息格式）
        initial_data = enrich_task_ws_message({
            "type": "task_status",
            "task_id": task_id,
            "status": task["status"],
            "progress": task.get("progress", 0),
            "timestamp": task.get("created_at") or utc_now().isoformat()
        })
        
        # 根据状态添加时间字段
        if task.get("started_at"):
            initial_data["started_at"] = task["started_at"]
        if task.get("completed_at"):
            initial_data["completed_at"] = task["completed_at"]
        
        # 如果有错误信息，添加它
        if task.get("error"):
            initial_data["error"] = task["error"]

        initial_data = enrich_task_ws_message(initial_data)

        await websocket.send_json(initial_data)
        
        # 保持连接，等待客户端消息或连接断开
        while True:
            try:
                # 接收客户端消息（心跳或控制消息）
                data = await websocket.receive_text()
                
                # 可以处理客户端消息（如心跳响应）
                # 这里简单忽略，保持连接活跃
                logger.debug(f"Received message from user client: {data}")
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                break
    
    except WebSocketDisconnect:
        logger.info(f"User WebSocket disconnected for task: {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # 断开连接
        await manager.disconnect(websocket)


@router.websocket("/ws/model/{model_key}")
async def websocket_model_download(
    websocket: WebSocket,
    model_key: str,
    token: Optional[str] = None,
    locale: Optional[str] = None,
):
    """
    WebSocket端点：模型下载进度/日志推送（前端订阅）

    - **model_key**: 模型稳定键
    - **token**: JWT Token（通过查询参数传递，必需）
    """
    manager = get_websocket_manager()

    # 验证模型是否存在
    model = Model.get_by_model_key(model_key)
    if not model:
        await websocket.close(code=1008, reason="Model not found")
        return

    # token 必需（复用 task ws 的校验方式）
    user_id = None
    if token:
        try:
            from backend.auth.jwt_utils import verify_token
            payload = verify_token(token)
            user_id = payload.get("sub") or "unknown"
        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            await websocket.close(code=1008, reason="Invalid token")
            return
    else:
        await websocket.close(code=1008, reason="Token required")
        return

    await manager.connect_model(websocket, model_key, str(user_id))

    try:
        # 发送初始状态（以 download_status 格式，便于前端统一处理）
        initial = {
            "type": "download_status",
            "model_key": model_key,
            "source": model.get("source") or {},
            "status": model.get("download_status") or "pending",
            "load_name": model.get("load_name") or "",
            "description": model.get("description") or "",
            "timestamp": utc_now().isoformat(),
        }
        await websocket.send_json(initial)

        # 保持连接
        while True:
            try:
                data = await websocket.receive_text()
                logger.debug(f"Received message from model client: {data}")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Model WebSocket error: {e}", exc_info=True)
                break
    finally:
        await manager.disconnect(websocket)


@router.websocket("/ws/inference/{service_id}")
async def websocket_inference_service(
    websocket: WebSocket,
    service_id: str
):
    """
    WebSocket端点：推理器连接
    
    - **service_id**: 推理服务ID
    
    推理器在启动时连接此端点，用于接收任务和发送推理结果
    
    保活机制：
    - 服务端每30秒发送一次ping消息
    - 推理器应回复pong消息
    - 推理器也可以主动发送heartbeat消息
    - 如果60秒内没有收到任何消息，连接将被关闭
    """
    import asyncio
    from datetime import datetime, timedelta
    
    manager = get_websocket_manager()
    
    # 验证服务是否存在（在accept之前检查）
    from backend.database import InferenceService
    service = InferenceService.get_by_id(service_id)
    if not service:
        logger.warning(f"Inference service not found: {service_id}, rejecting WebSocket connection")
        await websocket.close(code=1008, reason="Service not found")
        return
    
    # 建立推理器连接（这里会调用websocket.accept()）
    await manager.connect_inference_service(websocket, service_id)
    
    # 保活配置
    PING_INTERVAL = 30  # 每30秒发送一次ping
    TIMEOUT = 300  # 5 分钟无消息才断开（长 OCR/推理任务期间推理器可能较久不回包）
    last_message_time = datetime.utcnow()
    ping_task = None

    async def _safe_send_json(payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except RuntimeError:
            return False
        except Exception as e:
            logger.debug("Failed to send ws message to inference service %s: %s", service_id, e)
            return False
    
    async def send_ping():
        """定期发送ping消息"""
        nonlocal last_message_time
        while True:
            try:
                await asyncio.sleep(PING_INTERVAL)
                
                # 检查是否超时
                time_since_last_message = (datetime.utcnow() - last_message_time).total_seconds()
                if time_since_last_message > TIMEOUT:
                    logger.warning(
                        f"Inference service {service_id} timeout "
                        f"({time_since_last_message:.1f}s > {TIMEOUT}s), closing connection"
                    )
                    try:
                        await websocket.close(code=1000, reason="Timeout: no response")
                    except:
                        pass
                    break
                
                # 发送ping消息
                try:
                    ping_message = {
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    if not await _safe_send_json(ping_message):
                        break
                    logger.info(f"Sent ping to inference service {service_id}")  # 改为info级别以便调试
                except Exception as e:
                    logger.error(f"Error sending ping to inference service {service_id}: {e}")
                    break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping task for inference service {service_id}: {e}")
                break
    
    try:
        # 启动ping任务
        ping_task = asyncio.create_task(send_ping())
        
        # 保持连接，接收和转发消息
        while True:
            try:
                # 接收推理器消息（不设置超时，由ping任务处理超时）
                raw_in = await websocket.receive()
                if raw_in.get("type") == "websocket.disconnect":
                    break
                if raw_in.get("type") != "websocket.receive" or "text" not in raw_in:
                    logger.warning("unexpected inference ws frame keys=%s", list(raw_in.keys()))
                    continue
                data = raw_in["text"]

                # 更新最后消息时间
                last_message_time = datetime.utcnow()
                
                try:
                    message = json.loads(data)
                    message_type = message.get("type")
                    task_id = message.get("task_id")
                    session_id = message.get("session_id")

                    bin_len = _inference_audio_binary_len(message)
                    if bin_len > 0:
                        fr2 = await websocket.receive()
                        if fr2.get("type") != "websocket.receive" or "bytes" not in fr2:
                            logger.warning("inference session.audio.chunk: expected binary frame, got %s", fr2)
                            continue
                        blob = bytes(fr2["bytes"])
                        if len(blob) != bin_len:
                            logger.warning(
                                "inference session.audio.chunk bytes_len mismatch expected=%s actual=%s",
                                bin_len,
                                len(blob),
                            )
                        message["binary_bytes"] = blob

                    # 处理不同类型的消息
                    if message_type == "pong":
                        # 心跳响应
                        logger.debug(f"Received pong from inference service {service_id}")
                        # 更新最后消息时间（pong也算作活跃消息）
                        last_message_time = datetime.utcnow()
                        continue
                    elif message_type == "heartbeat":
                        # 主动心跳
                        logger.debug(f"Received heartbeat from inference service {service_id}")
                        try:
                            get_inference_service_manager().sync_service_heartbeat(service_id)
                        except Exception as e:
                            logger.warning(f"Failed to sync heartbeat for inference service {service_id}: {e}")
                        # 更新最后消息时间
                        last_message_time = datetime.utcnow()
                        # 回复pong
                        pong_message = {
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        if not await _safe_send_json(pong_message):
                            break
                        continue
                    elif message_type == "service_register":
                        try:
                            get_inference_service_manager().sync_service_registration(
                                service_id,
                                content_service_type=message.get("service_type") or service.get("service_type"),
                                supports_task=bool(message.get("supports_task", True)),
                                supported_models=message.get("supported_models"),
                                capabilities=message.get("capabilities"),
                                fixed_model=message.get("fixed_model"),
                                fixed_family=message.get("fixed_family"),
                            )
                            await websocket.send_json(
                                {
                                    "type": "service_registered",
                                    "service_id": service_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )
                            await manager.notify_inference_services_changed(
                                service_id=service_id,
                                reason="registered",
                            )
                        except Exception as e:
                            logger.error(f"Failed to sync service registration for {service_id}: {e}", exc_info=True)
                            await websocket.send_json(
                                {
                                    "type": "service_error",
                                    "service_id": service_id,
                                    "error": str(e),
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )
                        continue
                    elif message_type == "service_heartbeat":
                        try:
                            get_inference_service_manager().sync_service_heartbeat(service_id)
                        except Exception as e:
                            logger.warning(f"Failed to sync service heartbeat for {service_id}: {e}")
                        continue
                    elif message_type == "task_status" and task_id:
                        # 处理任务状态更新消息
                        # 1. 更新数据库中的任务状态
                        status = message.get("status")
                        update_data = {}
                        
                        if status:
                            update_data["status"] = status
                        
                        # 提取时间字段
                        if "started_at" in message:
                            update_data["started_at"] = message["started_at"]
                        if "completed_at" in message:
                            update_data["completed_at"] = message["completed_at"]
                        
                        # 提取错误信息
                        if "error" in message:
                            update_data["error"] = message["error"]
                        
                        # 更新数据库
                        try:
                            updated_task = Task.update(task_id, **update_data)
                            if updated_task:
                                logger.info(
                                    f"Task status updated in database: task_id={task_id}, "
                                    f"status={status}, update_fields={list(update_data.keys())}"
                                )
                            else:
                                logger.warning(f"Failed to update task status in database: task_id={task_id}")
                        except Exception as e:
                            logger.error(
                                f"Error updating task status in database: task_id={task_id}, error={e}",
                                exc_info=True
                            )
                        
                        # 2. 转发状态消息到用户前端
                        try:
                            conn_cnt = await manager.get_connection_count(task_id)
                        except Exception:
                            conn_cnt = "unknown"
                        logger.info(
                            f"Forwarding task status to task_id={task_id} "
                            f"connections={conn_cnt}, status={status}"
                        )
                        await manager.forward_inference_message(task_id, message)
                        if status in {"completed", "failed", "cancelled"}:
                            await manager.clear_task_dispatch_service(task_id)
                    
                    elif message_type == "download_status":
                        # 下载状态更新（下载服务 -> 后端）
                        model_key = message.get("model_key")
                        if not model_key:
                            logger.warning(f"download_status missing model_key: {message}")
                            continue

                        status = str(message.get("status") or "").strip() or "pending"
                        progress_text = str(message.get("progress_text") or "")
                        error_text = str(message.get("error_text") or "")
                        load_name = str(message.get("load_name") or "")

                        # 更新 model_catalog 表
                        try:
                            model_dict = Model.get_by_model_key(model_key)
                            if not model_dict:
                                logger.warning(f"Model not found for download_status: model_key={model_key}")
                            else:
                                desc = model_dict.get("description") or ""
                                # completed 时清空区段，否则 upsert
                                if status == "completed":
                                    new_desc = clear_download_block(desc)
                                else:
                                    new_desc = upsert_download_block(
                                        desc,
                                        status=status,
                                        progress=progress_text,
                                        error=error_text,
                                        worker=service_id,
                                        updated_at=datetime.utcnow().isoformat(),
                                    )

                                updates = {
                                    "download_status": status,
                                    "description": new_desc,
                                }
                                if status == "completed" and load_name:
                                    updates["load_name"] = load_name

                                if isinstance(message.get("source"), dict):
                                    current_source = dict(model_dict.get("source") or {})
                                    current_source.update(message["source"])
                                    updates["source"] = current_source

                                Model.update(model_key, **updates)
                        except Exception as e:
                            logger.error(f"Failed to update model download_status: {e}", exc_info=True)

                        # 转发给前端订阅者（原样转发 + 附带 service_id）
                        try:
                            message["service_id"] = service_id
                        except Exception:
                            pass
                        await manager.forward_model_message(model_key, message)

                    elif message_type == "download_log":
                        model_key = message.get("model_key")
                        if not model_key:
                            logger.warning(f"download_log missing model_key: {message}")
                            continue
                        try:
                            message["service_id"] = service_id
                        except Exception:
                            pass
                        await manager.forward_model_message(model_key, message)

                    elif (
                        (
                            message_type in {
                                "audio_stream_start",
                                "audio_stream_chunk",
                                "audio_stream_end",
                                "text_stream_delta",
                                "transcript_segment",
                            }
                        )
                        and task_id
                    ):
                        # task 通道：推理侧流事件按 task_id 转发到前端 /ws/task
                        try:
                            message["service_id"] = service_id
                        except Exception:
                            pass
                        await manager.forward_inference_message(task_id, message)

                    elif message_type.startswith("session.") and session_id:
                        # session 通道（新协议，PR2 起）：按前缀统一分流到 session
                        # 订阅者，由 /ws/chat 的 SessionRuntime 映射成统一 chat 协议。
                        # 推理侧 session_id 采用 `<chat_sid>:<role>` 形式（例如
                        # ``<chat_sid>:asr`` / ``<chat_sid>:tts``），这里顺便把
                        # role-scoped 事件 fan-out 到 `<chat_sid>` 父订阅者，
                        # 让 chat 层只需要订阅一次就能同时收到两个 role 的回流。
                        try:
                            message["service_id"] = service_id
                        except Exception:
                            pass
                        sess_bin = message.pop("binary_bytes", None)
                        await manager.publish_session_message(session_id, message, binary=sess_bin)
                        if ":" in session_id:
                            parent_sid = session_id.split(":", 1)[0].strip()
                            if parent_sid and parent_sid != session_id:
                                await manager.publish_session_message(parent_sid, message, binary=sess_bin)

                    elif message_type in {
                        "session_ready",
                        "llm_text_delta",
                        "audio_chunk",
                        "session_error",
                        "session_closed",
                    } and session_id:
                        # 遗留 session 事件（text session_runtime / OpenAI 兼容层
                        # 仍在使用的旧命名）：保持原有行为直到它们迁到 session.*
                        try:
                            message["service_id"] = service_id
                        except Exception:
                            pass
                        await manager.publish_session_message(session_id, message)

                    elif message_type == "result" and task_id:
                        # 处理推理结果消息
                        # 1. 创建文件记录到数据库
                        user_id = message.get("user_id")
                        storage = normalize_storage_label(message.get("storage", "local"))
                        files_list = message.get("files", [])
                        message_total = message.get("total")
                        
                        # 获取任务类型（task_type 或从数据库查询任务的 type）
                        task_type = message.get("task_type")
                        if not task_type:
                            # 如果消息中没有 task_type，从数据库查询任务的 type
                            try:
                                task_dict = Task.get_by_id(task_id)
                                if task_dict:
                                    task_type = task_dict.get("type")
                            except Exception as e:
                                logger.warning(f"Failed to get task type from database: {e}")
                        
                        # 如果没有找到 task_type，使用默认值
                        if not task_type:
                            logger.warning(f"Task type not found, using default 'image' for task_id={task_id}")
                            task_type = "image"
                        
                        # task_type 到 category 的映射（image/video/audio/text 直接对应）
                        category = task_type
                        
                        # 验证必需字段并创建文件记录
                        created_files = []
                        if not user_id:
                            logger.error(f"Result message missing user_id: task_id={task_id}")
                            logger.warning(
                                f"Skipped file record creation due to missing user_id: task_id={task_id}"
                            )
                        else:
                            
                            for file_info in files_list:
                                try:
                                    file_id = file_info.get("file_id")
                                    if not file_id:
                                        logger.warning(f"File info missing file_id, skipping: {file_info}")
                                        continue
                                    
                                    # 构建文件元数据（包含额外信息）
                                    metadata = {}
                                    if "seed" in file_info:
                                        metadata["seed"] = file_info["seed"]
                                    if "index" in file_info:
                                        metadata["index"] = file_info["index"]
                                    if "thumbnail_path" in file_info:
                                        metadata["thumbnail_path"] = file_info["thumbnail_path"]
                                    if "width" in file_info:
                                        metadata["width"] = file_info["width"]
                                    if "height" in file_info:
                                        metadata["height"] = file_info["height"]
                                    
                                    storage_path = file_info.get("storage_path", "") or ""
                                    http_url = resolve_artifact_public_url(storage, storage_path)

                                    file_dict = File.create(
                                        id=file_id,
                                        user_id=user_id,
                                        category=category,
                                        storage=storage,
                                        storage_path=storage_path,
                                        file_name=file_info.get("file_name"),
                                        file_size=file_info.get("file_size"),
                                        mime_type=file_info.get("mime_type"),
                                        http_url=http_url,
                                        task_id=task_id,
                                        metadata=metadata if metadata else None,
                                    )
                                    
                                    if file_dict:
                                        created_files.append(file_id)
                                        logger.debug(f"File record created: file_id={file_id}, task_id={task_id}")
                                    else:
                                        logger.warning(f"Failed to create file record: file_id={file_id}")
                                except Exception as e:
                                    logger.error(
                                        f"Error creating file record: file_id={file_info.get('file_id')}, "
                                        f"task_id={task_id}, error={e}",
                                        exc_info=True
                                    )
                            
                            logger.info(
                                f"Created {len(created_files)}/{len(files_list)} file records "
                                f"for task_id={task_id}"
                            )

                            # 为转发到前端的 result 补全可访问 URL（local 无 URL）
                            for fi in files_list:
                                if not isinstance(fi, dict):
                                    continue
                                sp = fi.get("storage_path") or ""
                                pub = resolve_artifact_public_url(storage, sp)
                                if pub:
                                    fi["url"] = pub
                                    fi["http_url"] = pub
                                else:
                                    fi.pop("url", None)
                                    fi.pop("http_url", None)
                                    fi.pop("thumb_url", None)
                                tk = fi.get("thumbnail_path")
                                if tk and pub:
                                    thumb_pub = resolve_artifact_public_url(storage, tk)
                                    if thumb_pub:
                                        fi["thumb_url"] = thumb_pub
                        
                        # 2. 更新任务状态为 completed
                        try:
                            # 计算该任务期望文件总数：
                            # 1) 优先使用推理器 result.total（协议字段）
                            # 2) 否则回退到 tasks.params.generate_num
                            # 3) 再回退到本条消息携带的 files 数量
                            expected_total = None
                            if isinstance(message_total, int) and message_total > 0:
                                expected_total = message_total
                            else:
                                try:
                                    task_dict = Task.get_by_id(task_id)
                                except Exception:
                                    task_dict = None
                                if task_dict:
                                    params = task_dict.get("params") or {}
                                    gen_num = params.get("generate_num")
                                    if isinstance(gen_num, int) and gen_num > 0:
                                        expected_total = gen_num
                            if not expected_total:
                                expected_total = max(1, len(files_list))

                            # 统计已落库文件数（多图模式会多次收到 result）
                            try:
                                existing_files = File.list_by_task(task_id, limit=1000, offset=0)
                                received_total = len(existing_files)
                            except Exception:
                                received_total = len(created_files)

                            is_completed = received_total >= expected_total
                            new_status = "completed" if is_completed else "processing"

                            # 进度：优先使用推理器给的 progress；否则按已完成张数估算
                            msg_progress = message.get("progress")
                            if isinstance(msg_progress, int):
                                new_progress = msg_progress
                            else:
                                new_progress = int(round((received_total / expected_total) * 100))

                            if is_completed:
                                new_progress = 100
                            else:
                                new_progress = max(0, min(99, new_progress))

                            update_data = {
                                "status": new_status,
                                "progress": new_progress,
                            }

                            # 仅在真正完成时写 completed_at
                            if is_completed:
                                if "timestamp" in message and message["timestamp"]:
                                    update_data["completed_at"] = message["timestamp"]
                                else:
                                    update_data["completed_at"] = utc_now().isoformat()

                            updated_task = Task.update(task_id, **update_data)
                            if updated_task:
                                logger.info(
                                    f"Task status updated to {new_status}: task_id={task_id}, "
                                    f"files_count={received_total}/{expected_total}"
                                )
                            else:
                                logger.warning(f"Failed to update task status: task_id={task_id}")
                        except Exception as e:
                            logger.error(
                                f"Error updating task status: task_id={task_id}, error={e}",
                                exc_info=True
                            )
                        
                        # 3. 转发消息到用户前端
                        try:
                            conn_cnt = await manager.get_connection_count(task_id)
                        except Exception:
                            conn_cnt = "unknown"

                        # 重要：推理器多图模式下会逐张发送 result。
                        # 如果本条消息的 status/progress 过早标记为 completed，前端可能会提前断开 ws。
                        # 因此按当前已接收数量重写 status/progress/total 后再转发。
                        try:
                            message["total"] = expected_total
                            message["status"] = new_status
                            message["progress"] = new_progress
                        except Exception:
                            pass

                        logger.info(
                            f"Forwarding inference result to task_id={task_id} "
                            f"connections={conn_cnt}, files_count={len(created_files)}"
                        )
                        await manager.forward_inference_message(task_id, message)
                        if new_status == "completed":
                            await manager.clear_task_dispatch_service(task_id)
                    else:
                        logger.debug(f"Received message from inference service {service_id}: {message_type}")
                
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from inference service {service_id}: {data}")
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                break
    
    except WebSocketDisconnect:
        logger.info(f"Inference service WebSocket disconnected: {service_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # 取消ping任务
        if ping_task and not ping_task.done():
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
        
        # 断开连接
        await manager.disconnect(websocket)
        
        # 更新数据库中的服务状态为stopped
        try:
            from backend.services.inference.service import InferenceServiceManager
            inference_manager = InferenceServiceManager()
            inference_manager.sync_service_stop(service_id)
            logger.info(f"Updated inference service {service_id} status to stopped")
        except Exception as e:
            logger.error(f"Failed to update service status to stopped: {e}", exc_info=True)


@router.get("/ws/stats")
async def websocket_stats(user_id: str = Depends(get_optional_user_id)):
    """
    获取WebSocket连接统计信息
    
    需要认证（可选）
    """
    manager = get_websocket_manager()
    
    total_connections = await manager.get_total_connections()
    
    return ok(
        data={"total_connections": total_connections},
        msg="ok",
    )

