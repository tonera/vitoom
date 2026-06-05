"""Archive a URL file or conversation summary into the local knowledge base."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from backend.services.agent.tools.builtin.knowledge_base_core.archive import archive_conversation, archive_url
from backend.services.agent.tools.registry import register_tool

KNOWLEDGE_BASE_ARCHIVE_TOOL_NAME = "knowledge_base_archive"

KNOWLEDGE_BASE_ARCHIVE_DESCRIPTION = (
    "将用户指定的文件 URL 或当前对话总结归档到本地知识库，并立即更新 Elasticsearch 索引。"
    "适合用户说“把这个文件存档/加入知识库/保存到知识库”或“总结当前对话并存档”。"
    "如果是文件 URL，本工具会下载文件、分类、解析、入库并刷新索引；音频、视频、图片类文件会被拒绝入库。"
    "如果是对话归档，应把需要归档的对话内容传入 content，工具可先总结成 Markdown 再入库。"
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


@register_tool(
    name=KNOWLEDGE_BASE_ARCHIVE_TOOL_NAME,
    description=KNOWLEDGE_BASE_ARCHIVE_DESCRIPTION,
    tags=["knowledge", "archive", "kb", "知识库", "存档", "归档", "保存资料", "加入知识库"],
    provider="local",
    enabled=True,
)
def build_knowledge_base_archive_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()

    try:
        from crewai.tools import BaseTool  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("crewai is required to register native agent tools") from exc

    try:
        from pydantic import BaseModel, Field  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("pydantic is required to build knowledge_base_archive tool") from exc

    class KnowledgeBaseArchiveArgs(BaseModel):
        mode: str = Field(default="url", description="Archive mode: url or conversation.")
        url: str = Field(default="", description="File URL to download and archive when mode=url.")
        content: str = Field(
            default="",
            description=(
                "Conversation content or organized Markdown to archive when mode=conversation. "
                "When user asks to archive the current conversation, organize full content from context first."
            ),
        )
        title: str = Field(default="", description="Archive document title or filename stem.")
        summarize: bool = Field(default=True, description="When mode=conversation, whether to summarize via LLM to Markdown first.")
        classify: bool = Field(default=True, description="Whether to classify via LLM.")
        knowledge_base_id: str = Field(default="default", description="Knowledge base ID.")

    class KnowledgeBaseArchiveTool(BaseTool):
        name: str = KNOWLEDGE_BASE_ARCHIVE_TOOL_NAME
        description: str = KNOWLEDGE_BASE_ARCHIVE_DESCRIPTION
        args_schema: type = KnowledgeBaseArchiveArgs

        def _run(
            self,
            mode: str = "url",
            url: str = "",
            content: str = "",
            title: str = "",
            summarize: bool = True,
            classify: bool = True,
            knowledge_base_id: str = "default",
            **_ignored: Any,
        ) -> str:
            user_id = bound_user_id
            if not user_id:
                return "```json\n" + _json_dumps({"status": "failed", "error": "缺少用户上下文，无法执行知识库归档。"}) + "\n```"
            try:
                normalized_mode = str(mode or "url").strip().lower()
                if normalized_mode == "url":
                    if not str(url or "").strip():
                        raise ValueError("mode=url 需要提供 url")
                    result = archive_url(
                        str(url).strip(),
                        user_id=user_id,
                        title=title,
                        classify=classify,
                        knowledge_base_id=knowledge_base_id or "default",
                    )
                elif normalized_mode in {"conversation", "chat", "markdown"}:
                    if not str(content or "").strip():
                        raise ValueError("mode=conversation 需要提供 content")
                    result = archive_conversation(
                        str(content),
                        user_id=user_id,
                        title=title or "对话归档",
                        summarize=summarize,
                        classify=classify,
                        knowledge_base_id=knowledge_base_id or "default",
                    )
                else:
                    raise ValueError("mode 只支持 url 或 conversation")
                document = result.get("document") or {}
                payload = {
                    "status": "success",
                    "document_id": document.get("document_id"),
                    "file_name": document.get("file_name"),
                    "domain": document.get("domain"),
                    "topic": document.get("topic"),
                    "subtopic": document.get("subtopic"),
                    "canonical_path": document.get("canonical_path"),
                    "ingest": result.get("ingest"),
                    "message": "已归档到知识库并刷新索引，可立即查询。",
                }
                return "```json\n" + _json_dumps(payload) + "\n```"
            except Exception as exc:
                return "```json\n" + _json_dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}) + "\n```"

    return KnowledgeBaseArchiveTool()
