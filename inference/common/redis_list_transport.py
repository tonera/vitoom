"""
Redis List 传输（兼容对方项目的 RPOP 队列模型）：
- Ingress：BRPOP 从请求队列拉取消息，投递到本项目 MessageQueue
- Egress：LPUSH/RPUSH 将 result / task_status 写入响应队列

设计目标：
- 不要求对方改生产端（无 ack，允许丢任务）
- 消息 JSON 与本项目“几乎一致”时，尽量做到宽松兼容
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from .logger import get_logger, summarize_tpl_list_for_log
from .message_queue import MessageQueue

logger = get_logger(__name__)

# 连接类异常：需要关闭旧连接并重建客户端
_REDIS_RECONNECT_ERRORS: Tuple[type, ...] = (
    RedisConnectionError,
    RedisTimeoutError,
    ConnectionError,
    TimeoutError,
    OSError,
    BrokenPipeError,
)

# 重连退避上限（秒）
_MAX_RECONNECT_BACKOFF = 30


def _is_redis_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, _REDIS_RECONNECT_ERRORS):
        return True
    # redis-py 有时会把底层 socket 错误包在其它异常里
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "connection reset",
            "connection refused",
            "connection closed",
            "broken pipe",
            "timed out",
            "timeout",
            "eof",
        )
    )


def create_redis_client(cfg: RedisListConfig, *, socket_timeout: float) -> redis.Redis:
    """
    创建带保活/超时/健康检查的 Redis 客户端。

    socket_timeout 必须略大于 brpop_timeout，否则正常空轮询也会被误判为超时。
    """
    return redis.Redis(
        host=cfg.host,
        port=cfg.port,
        password=cfg.password or None,
        db=cfg.db,
        decode_responses=True,
        socket_keepalive=True,
        socket_timeout=socket_timeout,
        socket_connect_timeout=10,
        retry_on_timeout=True,
        health_check_interval=30,
    )


async def close_redis_client(client: Optional[redis.Redis]) -> None:
    if client is None:
        return
    try:
        await client.aclose()
    except Exception:
        try:
            await client.close()
        except Exception:
            pass


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def normalize_incoming_message(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    将外部消息（对方项目队列中的 JSON）规范化为本项目 TaskProcessor 可消费的 message dict。

    目标格式：
    - task:   {"type":"task", "task_id": "...", "task_data": {...}}
    - cancel: {"type":"cancel", "task_id": "...", "timestamp": "..."}
    """

    if not isinstance(raw, dict):
        return None

    mtype = (raw.get("type") or "").strip().lower()

    # 1) 已是本项目标准格式（或与 WS 侧一致）
    if mtype in ("task", "cancel"):
        msg = dict(raw)
        if mtype == "task":
            task_id = msg.get("task_id") or msg.get("id")
            task_data = msg.get("task_data")
            # 兼容：对方可能直接把 task_data（全量任务字典）扔进队列
            if not isinstance(task_data, dict):
                task_data = raw.get("task_data") if isinstance(raw.get("task_data"), dict) else raw
            msg["task_id"] = task_id
            msg["task_data"] = task_data
            return msg if task_id else None

        # cancel
        task_id = msg.get("task_id") or msg.get("id")
        msg["task_id"] = task_id
        msg.setdefault("timestamp", _now_iso())
        return msg if task_id else None

    # 2) 未带 type：视为“全量 task_data”
    # 约定：task_id 优先 raw.task_id，其次 raw.id
    task_id = raw.get("task_id") or raw.get("id")
    if not task_id:
        return None
    return {"type": "task", "task_id": task_id, "task_data": raw}


@dataclass
class RedisListConfig:
    host: str
    port: int
    password: str = ""
    db: int = 0

    # ingress
    channel: str = ""

    # egress
    reschannel: str = ""

    # list push direction: "lpush" | "rpush"
    push: str = "lpush"

    # ingress brpop timeout seconds (short timeout allows graceful stop)
    brpop_timeout: int = 5


class RedisListEgress:
    """
    将 result / task_status 写回 Redis list。

    注意：该 egress 不实现“缓存落盘”逻辑；失败只记录日志并返回 False。
    """

    def __init__(self, cfg: RedisListConfig):
        if not cfg.reschannel:
            raise ValueError("redis.reschannel is required for RedisListEgress")
        self.cfg = cfg
        self._redis: redis.Redis = create_redis_client(cfg, socket_timeout=30.0)
        self._closed = False
        self._reconnect_attempts = 0

    def is_connected(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
        await close_redis_client(self._redis)

    async def _reset_redis(self) -> None:
        await close_redis_client(self._redis)
        self._redis = create_redis_client(self.cfg, socket_timeout=30.0)

    async def _push(self, payload: dict) -> bool:
        if self._closed:
            return False
        try:
            s = json.dumps(payload, ensure_ascii=False)
            if (self.cfg.push or "").lower() == "rpush":
                await self._redis.rpush(self.cfg.reschannel, s)
            else:
                await self._redis.lpush(self.cfg.reschannel, s)
            self._reconnect_attempts = 0
            return True
        except Exception as e:
            if _is_redis_connection_error(e):
                self._reconnect_attempts += 1
                backoff = min(2 ** self._reconnect_attempts, _MAX_RECONNECT_BACKOFF)
                logger.warning(
                    f"RedisListEgress connection error, reconnecting in {backoff}s: {e}"
                )
                try:
                    await self._reset_redis()
                except Exception as reset_err:
                    logger.error(f"RedisListEgress reconnect failed: {reset_err}", exc_info=True)
                await asyncio.sleep(backoff)
            else:
                logger.error(f"RedisListEgress push failed: {e}", exc_info=True)
            return False

    async def send_result(self, result_message: dict) -> bool:
        return await self._push(result_message)

    async def send_task_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        msg = {
            "type": "task_status",
            "task_id": task_id,
            "status": status,
            "timestamp": _now_iso(),
        }
        if error:
            msg["error"] = error
        msg.update(kwargs)
        return await self._push(msg)


class RedisListIngress:
    """
    从 Redis list 取任务消息并投递到 MessageQueue。
    """

    def __init__(
        self,
        cfg: RedisListConfig,
        *,
        message_queue: MessageQueue,
        egress: Optional[Any] = None,
        on_cancel_message: Optional[Any] = None,
    ):
        if not cfg.channel:
            raise ValueError("redis.channel is required for RedisListIngress")
        self.cfg = cfg
        self.message_queue = message_queue
        self.egress = egress  # 可选：用于发 queued status（保持与 WS 接入一致）
        self.on_cancel_message = on_cancel_message
        brpop_timeout = max(int(cfg.brpop_timeout or 5), 1)
        # 必须大于 BRPOP 服务端超时，避免正常空轮询触发 socket 超时
        self._socket_timeout = float(brpop_timeout + 10)
        self._redis: redis.Redis = create_redis_client(cfg, socket_timeout=self._socket_timeout)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._reconnect_attempts = 0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await close_redis_client(self._redis)

    async def _reset_redis(self) -> None:
        await close_redis_client(self._redis)
        self._redis = create_redis_client(self.cfg, socket_timeout=self._socket_timeout)

    async def _reconnect_after_error(self, err: Exception) -> None:
        self._reconnect_attempts += 1
        backoff = min(2 ** self._reconnect_attempts, _MAX_RECONNECT_BACKOFF)
        logger.warning(
            f"RedisListIngress connection error on channel={self.cfg.channel}, "
            f"reconnecting in {backoff}s: {err}"
        )
        try:
            await self._reset_redis()
        except Exception as reset_err:
            logger.error(f"RedisListIngress reconnect failed: {reset_err}", exc_info=True)
        await asyncio.sleep(backoff)

    async def _loop(self) -> None:
        logger.info(
            f"RedisListIngress started: channel={self.cfg.channel}, "
            f"brpop_timeout={self.cfg.brpop_timeout}, socket_timeout={self._socket_timeout}"
        )
        while self._running:
            try:
                item: Optional[Tuple[str, str]] = await self._redis.brpop(
                    self.cfg.channel,
                    timeout=self.cfg.brpop_timeout,
                )
                if not item:
                    continue

                self._reconnect_attempts = 0

                _, raw_str = item
                if not raw_str:
                    continue

                # 直接输出 Redis 收到的原始消息（用于定位 tpl_list 在链路中何处丢失）
                # data URI 可能数 MB，禁止把 payload 打进日志。
                try:
                    raw_s = raw_str if isinstance(raw_str, str) else str(raw_str)
                    if "data:" in raw_s[:4096]:
                        logger.info(
                            f"[RAW_REDIS_INGRESS] omitted data URI payload, {len(raw_s)} chars"
                        )
                    else:
                        max_len = 8000
                        if len(raw_s) <= max_len:
                            logger.info(f"[RAW_REDIS_INGRESS] {raw_s}")
                        else:
                            head = 4000
                            tail = 3500
                            logger.info(
                                f"[RAW_REDIS_INGRESS] {raw_s[:head]}...[truncated {len(raw_s) - head - tail} chars]...{raw_s[-tail:]}"
                            )
                except Exception:
                    logger.info("[RAW_REDIS_INGRESS] <unavailable>")
                try:
                    raw = json.loads(raw_str)
                except Exception as e:
                    logger.error(f"Failed to parse redis message as JSON: {e}, raw={raw_str[:200]}")
                    continue

                # 输出 raw 结构中的 tpl_list（未经过 normalize / from_task_dict）
                # 兼容两种形态：
                # 1) raw 本身就是 task_data（无 wrapper）
                # 2) raw 是 {"type":"task","task_id":...,"task_data":{...}} wrapper
                try:
                    raw_task_id = raw.get("task_id") or raw.get("id")
                    task_data0 = raw.get("task_data") if isinstance(raw.get("task_data"), dict) else raw
                    params0 = task_data0.get("params") if isinstance(task_data0, dict) else None
                    tpl0 = params0.get("tpl_list") if isinstance(params0, dict) else None
                    logger.info(
                        f"[REDIS_RAW_PARAMS] task_id={raw_task_id} "
                        f"tpl_list={summarize_tpl_list_for_log(tpl0)}"
                    )
                except Exception:
                    logger.info("[REDIS_RAW_PARAMS] tpl_list=<unavailable>")

                msg = normalize_incoming_message(raw)
                if not msg:
                    logger.warning(f"Ignoring invalid redis message (cannot normalize): keys={list(raw)[:20]}")
                    continue

                # 输出 normalize 后的 task_data.params.tpl_list（更接近入队后的形态）
                try:
                    if msg.get("type") == "task":
                        task_data = msg.get("task_data") if isinstance(msg.get("task_data"), dict) else None
                        params2 = task_data.get("params") if isinstance(task_data, dict) else None
                        tpl2 = params2.get("tpl_list") if isinstance(params2, dict) else None
                        logger.info(
                            f"[REDIS_NORMALIZED_TASK_DATA] task_id={msg.get('task_id')} "
                            f"tpl_list={summarize_tpl_list_for_log(tpl2)}"
                        )
                except Exception:
                    logger.info("[REDIS_NORMALIZED_TASK_DATA] tpl_list=<unavailable>")

                # 入队
                if msg.get("type") == "cancel" and callable(self.on_cancel_message):
                    try:
                        await self.on_cancel_message(msg)
                    except Exception as e:
                        logger.error(f"Immediate redis cancel handling failed: {e}", exc_info=True)
                        self.message_queue.put_message(msg)
                else:
                    self.message_queue.put_message(msg)

                # queued 状态（与 WS 接入保持一致：收到即 queued）
                if msg.get("type") == "task" and self.egress:
                    try:
                        await self.egress.send_task_status(task_id=msg.get("task_id"), status="queued")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                if _is_redis_connection_error(e):
                    await self._reconnect_after_error(e)
                else:
                    logger.error(f"Error in RedisListIngress loop: {e}", exc_info=True)
                    # 短暂停顿，避免异常时 tight loop
                    await asyncio.sleep(0.2)

