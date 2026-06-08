from __future__ import annotations

from threading import Event, Lock
from typing import Dict


class TaskCancelledError(RuntimeError):
    """任务被用户取消时抛出的协作式中断异常。"""

    def __init__(self, task_id: str, stage: str = ""):
        self.task_id = str(task_id or "").strip()
        self.stage = str(stage or "").strip()
        detail = f"task_id={self.task_id or '<unknown>'}"
        if self.stage:
            detail = f"{detail} stage={self.stage}"
        super().__init__(f"Task cancelled: {detail}")


class TaskCancellationRegistry:
    """线程安全的任务取消注册表。

    设计目标：
    - 事件循环线程可以随时标记某个 task 已取消；
    - 阻塞推理线程可以通过 ``Event`` 做低开销轮询；
    - 后续视频/音频等 handler 可以复用同一套取消原语。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._cancelled: set[str] = set()
        self._events: Dict[str, Event] = {}

    @staticmethod
    def _normalize(task_id: str) -> str:
        return str(task_id or "").strip()

    def mark_cancelled(self, task_id: str) -> bool:
        normalized = self._normalize(task_id)
        if not normalized:
            return False
        with self._lock:
            self._cancelled.add(normalized)
            event = self._events.get(normalized)
            if event is None:
                event = Event()
                self._events[normalized] = event
            event.set()
        return True

    def is_cancelled(self, task_id: str) -> bool:
        normalized = self._normalize(task_id)
        if not normalized:
            return False
        with self._lock:
            return normalized in self._cancelled

    def get_event(self, task_id: str) -> Event:
        normalized = self._normalize(task_id)
        event = Event()
        if not normalized:
            return event
        with self._lock:
            existing = self._events.get(normalized)
            if existing is None:
                existing = Event()
                if normalized in self._cancelled:
                    existing.set()
                self._events[normalized] = existing
            return existing

    def clear(self, task_id: str) -> None:
        normalized = self._normalize(task_id)
        if not normalized:
            return
        with self._lock:
            self._cancelled.discard(normalized)
            self._events.pop(normalized, None)

