from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# AgentRun 状态枚举（字符串常量）
#
# 运行流转：
#   created   -> queued -> running -> completed / failed / cancelled
#                           │
#                           └── waiting_input ──(resume)──> running
#
# `waiting_input` 当前为占位状态（Human-in-the-loop 预留），Worker 侧尚未
# 实现对应的挂起/恢复逻辑，仅保留枚举位置以便后续引入审批流时无需迁移数据库。
# ---------------------------------------------------------------------------

AGENT_RUN_STATUSES: tuple[str, ...] = (
    "created",
    "queued",
    "running",
    "waiting_input",  # 预留：等待用户审批 / 补充输入
    "completed",
    "failed",
    "cancelled",
)


class RuntimeConfigSchema(BaseModel):
    """AgentRun.runtime_config 的结构化 schema。

    作用：
      - 在入口 `runtime._sanitize_runtime_config` 统一做 Pydantic 校验，
        避免后续各处 `runtime_config.get(...)` 出现隐式约定。
      - 允许向前兼容的扩展字段（`extra = "allow"`），preset 自带的自定义
        键（如 `source_type`）不会被丢弃。
    """

    model_config = ConfigDict(extra="allow")

    priority: int = Field(default=5, ge=1, le=10, description="Task queue priority 1-10")
    process: Literal["sequential", "hierarchical"] = Field(
        default="sequential",
        description="CrewAI process mode",
    )
    max_tools_per_run: Optional[int] = Field(
        default=None,
        ge=1,
        description="Max tools exposed to LLM per run; None uses global default",
    )
    tool_allowlist: Optional[List[str]] = Field(
        default=None,
        description="Hard filter: only these tools are allowed; None means no restriction",
    )

    # --- Human-in-the-loop 预留字段 ---------------------------------------
    # 用于在将来支持"某些敏感工具需要用户二次确认才能执行"。
    # 目前 Worker 尚未消费该字段，仅做结构化占位，避免后续再次 schema 迁移。
    require_approval_for: List[str] = Field(
        default_factory=list,
        description=(
            "Tool names that require user approval before execution. Placeholder; worker does not implement approval yet."
        ),
    )

    @field_validator("process", mode="before")
    @classmethod
    def _coerce_process(cls, value: Any) -> Any:
        """兼容大小写 / 空值：历史入参可能是 'SEQUENTIAL' 或 None。"""
        if value is None:
            return "sequential"
        text = str(value).strip().lower()
        return text or "sequential"

    @field_validator("priority", mode="before")
    @classmethod
    def _coerce_priority(cls, value: Any) -> Any:
        """兼容字符串数字以及 None，保持与历史 _sanitize 行为一致。"""
        if value is None or value == "":
            return 5
        try:
            return int(value)
        except (TypeError, ValueError):
            return 5

    @field_validator("tool_allowlist", mode="before")
    @classmethod
    def _coerce_tool_allowlist(cls, value: Any) -> Any:
        """允许字符串 / 单值传入，统一规整为 list[str]。"""
        if value is None:
            return None
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return cleaned
        text = str(value).strip()
        return [text] if text else []

    @field_validator("require_approval_for", mode="before")
    @classmethod
    def _coerce_require_approval(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


@dataclass
class AgentCommand:
    """统一的 Agent 入站命令对象。"""

    user_id: str
    agent_id: str
    message: str
    source_type: str = "web"
    source_ref: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    crew_tool_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "message": self.message,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "attachments": list(self.attachments),
            "context": dict(self.context),
            "runtime_config": dict(self.runtime_config),
            "conversation_id": self.conversation_id,
            "parent_run_id": self.parent_run_id,
            "crew_tool_name": self.crew_tool_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCommand":
        payload = dict(data or {})
        return cls(
            user_id=str(payload.get("user_id") or "").strip(),
            agent_id=str(payload.get("agent_id") or "").strip(),
            message=str(payload.get("message") or "").strip(),
            source_type=str(payload.get("source_type") or "web").strip() or "web",
            source_ref=payload.get("source_ref"),
            attachments=list(payload.get("attachments") or []),
            context=dict(payload.get("context") or {}),
            runtime_config=dict(payload.get("runtime_config") or {}),
            conversation_id=payload.get("conversation_id") or None,
            parent_run_id=payload.get("parent_run_id") or None,
            crew_tool_name=payload.get("crew_tool_name") or None,
        )


@dataclass
class AgentResult:
    """统一的 Agent 出站结果对象。"""

    run_id: str
    task_id: str
    status: str
    summary: str
    output_text: str
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
            "output_text": self.output_text,
            "artifacts": list(self.artifacts),
            "error": self.error,
            "metrics": dict(self.metrics),
        }
