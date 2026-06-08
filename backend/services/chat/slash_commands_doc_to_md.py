"""/doc-to-md 斜杠命令：直接调用 document_to_markdown 工具，跳过 LLM 选择。"""

from __future__ import annotations

import argparse
import asyncio
from typing import List

from .slash_commands import (
    ChatSlashCommandResult,
    build_artifacts_from_tool_result,
    register_slash_command,
)


class _ArgparseError(Exception):
    pass


class _NonExitingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _ArgparseError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _NonExitingArgumentParser(
        prog="/doc-to-md",
        description="Convert one or more document URLs into markdown.",
        add_help=False,
    )
    parser.add_argument("instruction", nargs="*", help="附加说明，可为空")
    parser.add_argument("--url", dest="urls", action="append", default=[],
                        help="待转换文档 URL 或路径，可重复")
    parser.add_argument("--timeout", dest="timeout", type=float, default=None)
    return parser


_USAGE_TEXT = """用法：/doc-to-md [说明文本] --url <文档URL> [--url <更多URL>...]

示例：
  /doc-to-md 把这个文档转为 md --url https://example.com/a.docx
  /doc-to-md 把这几个文件都转成 markdown --url https://example.com/a.pdf --url https://example.com/b.txt

说明：
  - PDF 会优先走 OCR 文档链路
  - 其它常见办公/文本文档走本地转换器
  - 聊天窗口只显示首页预览，完整结果通过下载链接提供
"""


def _is_help_request(argv: List[str]) -> bool:
    if not argv:
        return True
    return any(tok in {"--help", "-h", "/help", "/?"} for tok in argv)


def _render_result_markdown(payload: dict) -> str:
    items = payload.get("items") or []
    if not items:
        return "文档转换未产出任何结果。"

    lines: List[str] = []
    total = int(payload.get("total") or 0)
    completed = int(payload.get("completed") or 0)
    lines.append(f"已处理 {completed}/{total} 个文档。")

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or f"document_{index}")
        status = str(item.get("status") or "unknown")
        provider = str(item.get("provider") or "")
        task_id = str(item.get("task_id") or "")
        files = item.get("files") or []
        download_url = None
        file_name = None
        if files and isinstance(files[0], dict):
            download_url = files[0].get("url") or files[0].get("http_url")
            file_name = files[0].get("file_name")

        lines.append("")
        lines.append(f"### {index}. `{source}`")
        lines.append(f"- 状态：`{status}`")
        if provider:
            lines.append(f"- 路径：`{provider}`")
        if task_id:
            lines.append(f"- Task：`{task_id}`")
        if download_url:
            label = file_name or "下载结果文件"
            lines.append(f"- 下载：[{label}]({download_url})")
        if item.get("error"):
            lines.append(f"- 错误：{item['error']}")
        if item.get("message"):
            lines.append(f"- 提示：{item['message']}")

        preview = str(item.get("preview_md") or "").strip()
        if preview:
            lines.append("")
            lines.append("首页预览：")
            lines.append("```md")
            lines.append(preview)
            lines.append("```")

    return "\n".join(lines).strip()


class DocumentToMarkdownSlashCommand:
    name = "doc-to-md"

    async def handle(
        self,
        *,
        user_id: str,
        argv: List[str],
        raw_message: str,
    ) -> ChatSlashCommandResult:
        del raw_message
        if _is_help_request(argv):
            return ChatSlashCommandResult(
                handled=True,
                status="usage",
                assistant_text=_USAGE_TEXT,
            )

        parser = _build_parser()
        try:
            ns = parser.parse_args(argv)
        except _ArgparseError as exc:
            return ChatSlashCommandResult(
                handled=True,
                status="failed",
                assistant_text=f"参数错误：{exc}\n\n{_USAGE_TEXT}",
                error=str(exc),
            )

        if not ns.urls:
            return ChatSlashCommandResult(
                handled=True,
                status="failed",
                assistant_text=f"缺少 `--url` 参数。\n\n{_USAGE_TEXT}",
                error="missing --url",
            )

        from backend.services.agent.tools.builtin.document_to_markdown import (
            convert_documents_sync,
        )

        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: convert_documents_sync(
                user_id=user_id,
                urls=list(ns.urls or []),
                timeout=ns.timeout,
            ),
        )

        task_ids = [
            str(item.get("task_id") or "").strip()
            for item in (payload.get("items") or [])
            if isinstance(item, dict) and str(item.get("task_id") or "").strip()
        ]
        artifacts = []
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            item_payload = {
                "task_id": item.get("task_id"),
                "files": item.get("files") or [],
            }
            artifacts.extend(build_artifacts_from_tool_result(item_payload, default_category="text"))

        return ChatSlashCommandResult(
            handled=True,
            status=str(payload.get("status") or "failed"),
            assistant_text=_render_result_markdown(payload),
            task_ids=task_ids,
            error=str(payload.get("error") or "") or None,
            artifacts=artifacts,
        )


register_slash_command(DocumentToMarkdownSlashCommand())

