import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tool_catalog import ToolCatalogEntry
from backend.services.agent.tool_selection import ToolSelectionService, warm_tool_selection_embedding_model
from backend.services.agent.types import AgentCommand
import backend.services.agent.embeddings as embeddings_module
import backend.services.agent.tool_selection as tool_selection_module


class FakeCatalog:
    def __init__(self, entries):
        self._entries = entries

    def get(self, name: str):
        return self._entries.get(name)

    def all(self):
        return dict(self._entries)


def test_default_tool_selection_services_share_catalog_instance():
    assert ToolSelectionService()._catalog is ToolSelectionService()._catalog


def test_warm_tool_selection_embedding_model_returns_backend_readiness(monkeypatch):
    class FakeBackend:
        cache_key = "onnx:resources/models/multilingual-e5-small-onnx"

        def is_ready(self):
            return True

    class FakeManager:
        def get_backend(self):
            return FakeBackend()

    monkeypatch.setattr(tool_selection_module, "_EMBEDDING_BACKENDS", FakeManager())

    assert warm_tool_selection_embedding_model() is True


def test_warm_knowledge_base_embedding_model_returns_service_readiness(monkeypatch):
    class FakeService:
        def is_ready(self):
            return False

    class FakeManager:
        def get_service(self):
            return FakeService()

    monkeypatch.setattr(embeddings_module, "_manager", FakeManager())

    assert embeddings_module.warm_knowledge_base_embedding_model() is False


def test_tool_selection_prefers_relevant_tools_and_respects_runtime_allowlist():
    catalog = FakeCatalog(
        {
            "openclaw_browser_search": ToolCatalogEntry(
                name="openclaw_browser_search",
                description="在浏览器页面中搜索指定关键词。",
                tags=["浏览器", "页面", "搜索"],
                provider="openclaw",
                enabled=True,
                requires_openclaw=True,
                target_tool_name="browser_search",
            ),
            "openclaw_sessions_list": ToolCatalogEntry(
                name="openclaw_sessions_list",
                description="列出当前会话信息。",
                tags=["会话", "浏览器"],
                provider="openclaw",
                enabled=True,
                requires_openclaw=True,
                target_tool_name="sessions_list",
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="请帮我在当前页面里搜索订单号关键词",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=1),
    ):
        selected = selector.select_tool_names(
            ["openclaw_sessions_list", "openclaw_browser_search"],
            command=command,
            runtime_allowlist=["browser_search"],
        )

    assert selected == ["openclaw_browser_search"]


def test_tool_selection_skips_openclaw_tools_when_integration_disabled():
    catalog = FakeCatalog(
        {
            "openclaw_sessions_list": ToolCatalogEntry(
                name="openclaw_sessions_list",
                description="列出当前会话信息。",
                tags=["会话", "浏览器"],
                provider="openclaw",
                enabled=True,
                requires_openclaw=True,
                target_tool_name="sessions_list",
            ),
            "local_generate_image": ToolCatalogEntry(
                name="local_generate_image",
                description="根据文本描述生成图片。",
                tags=["图片", "生成"],
                provider="local",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="请帮我生成一张海边日落图片",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=2),
    ):
        selected = selector.select_tool_names(
            ["openclaw_sessions_list", "local_generate_image"],
            command=command,
        )

    assert selected == ["local_generate_image"]


def test_tool_selection_returns_empty_when_all_scores_are_zero():
    catalog = FakeCatalog(
        {
            "analyze_media": ToolCatalogEntry(
                name="analyze_media",
                description="分析图片或视频内容。",
                tags=["图片", "视频", "媒体"],
                provider="local",
                enabled=True,
            ),
            "travel_planner": ToolCatalogEntry(
                name="travel_planner",
                description="规划多日旅行行程。",
                tags=["旅行", "行程", "规划"],
                provider="crew",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="解释一下 Python 装饰器和闭包的区别",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=2),
    ):
        selected = selector.select_tool_names(
            ["analyze_media", "travel_planner"],
            command=command,
        )

    assert selected == []


def test_tool_selection_uses_history_for_contextual_followup():
    catalog = FakeCatalog(
        {
            "hr_business_query": ToolCatalogEntry(
                name="hr_business_query",
                description="HR 员工查询工具，查询员工人数、员工名单、办公室人员、组织关系和简历资料。",
                tags=["HR", "员工", "名单", "办公室"],
                provider="local",
                enabled=True,
            ),
            "travel_planner": ToolCatalogEntry(
                name="travel_planner",
                description="规划多日旅行行程。",
                tags=["旅行", "行程", "规划"],
                provider="crew",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="[过去对话]\nuser: 旧金山有多少员工\nassistant: 旧金山有 40 名员工。\n\n[本轮输入]\nuser: 名单发我",
        context={"original_user_message": "名单发我"},
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=1),
    ):
        selected = selector.select_tool_names(
            ["hr_business_query", "travel_planner"],
            command=command,
        )

    assert selected == ["hr_business_query"]


def test_tool_selection_keeps_analyze_media_when_media_url_is_present():
    catalog = FakeCatalog(
        {
            "analyze_media": ToolCatalogEntry(
                name="analyze_media",
                description="分析图片或视频内容。",
                tags=["图片", "视频", "媒体"],
                provider="local",
                enabled=True,
            ),
            "travel_planner": ToolCatalogEntry(
                name="travel_planner",
                description="规划多日旅行行程。",
                tags=["旅行", "行程", "规划"],
                provider="crew",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="帮我描述这个图片 https://example.com/a.jpg",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=2),
    ):
        selected = selector.select_tool_names(
            ["analyze_media", "travel_planner"],
            command=command,
        )

    assert selected == ["analyze_media"]


def test_tool_selection_exports_chat_content_to_markdown():
    catalog = FakeCatalog(
        {
            "document_to_markdown": ToolCatalogEntry(
                name="document_to_markdown",
                description="把文档或聊天内容转换/导出为 Markdown，支持总结以上信息并导出为 md 文档。",
                tags=["文档", "markdown", "md", "导出md", "转markdown"],
                aliases=["export content to markdown", "save answer as markdown"],
                examples=["请总结以上信息，导出为 md 文档"],
                negative_examples=["导出为pdf", "保存成PDF"],
                intent_anchors=["把聊天内容导出为 MD", "保存回答为 Markdown 文件"],
                input_modalities=["document", "text"],
                provider="local",
                enabled=True,
            ),
            "document_to_pdf": ToolCatalogEntry(
                name="document_to_pdf",
                description="把文档或聊天内容转换/导出为 PDF。",
                tags=["文档", "pdf", "导出pdf", "转pdf"],
                aliases=["export content to pdf"],
                examples=["把刚才的回答导出为 PDF"],
                negative_examples=["导出为md", "转markdown"],
                intent_anchors=["把聊天内容导出为 PDF"],
                input_modalities=["document", "text"],
                provider="local",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="请总结以上信息，导出为 md 文档",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=1),
    ):
        selected = selector.select_tool_names(
            ["document_to_markdown", "document_to_pdf"],
            command=command,
        )

    assert selected == ["document_to_markdown"]


def test_tool_selection_exports_chat_content_to_pdf():
    catalog = FakeCatalog(
        {
            "document_to_markdown": ToolCatalogEntry(
                name="document_to_markdown",
                description="把文档或聊天内容转换/导出为 Markdown。",
                tags=["文档", "markdown", "md", "导出md", "转markdown"],
                aliases=["export content to markdown"],
                examples=["请总结以上信息，导出为 md 文档"],
                negative_examples=["导出为pdf", "保存成PDF"],
                intent_anchors=["把聊天内容导出为 MD"],
                input_modalities=["document", "text"],
                provider="local",
                enabled=True,
            ),
            "document_to_pdf": ToolCatalogEntry(
                name="document_to_pdf",
                description="把文档或聊天内容转换/导出为 PDF，支持把刚才回答导出为 PDF。",
                tags=["文档", "pdf", "导出pdf", "保存pdf", "转pdf"],
                aliases=["export content to pdf", "save answer as pdf"],
                examples=["把刚才的回答导出为 PDF"],
                negative_examples=["导出为md", "转markdown"],
                intent_anchors=["把聊天内容导出为 PDF", "保存回答为 PDF 文件"],
                input_modalities=["document", "text"],
                provider="local",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="把刚才的回答导出为 PDF",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=1),
    ):
        selected = selector.select_tool_names(
            ["document_to_markdown", "document_to_pdf"],
            command=command,
        )

    assert selected == ["document_to_pdf"]


def test_tool_selection_exports_chat_table_to_excel():
    catalog = FakeCatalog(
        {
            "table_to_excel": ToolCatalogEntry(
                name="table_to_excel",
                description="把聊天中 LLM 生成或整理出的表格内容导出为 Excel .xlsx 文件。",
                tags=["excel", "xlsx", "table", "表格", "导出excel", "保存表格"],
                aliases=["export table to excel", "save table as xlsx"],
                examples=["把上面的表格导出为 Excel"],
                negative_examples=["导出为pdf", "导出为md"],
                intent_anchors=["把聊天表格导出为 Excel", "保存表格为 xlsx 文件"],
                input_modalities=["text"],
                provider="local",
                enabled=True,
            ),
            "document_to_markdown": ToolCatalogEntry(
                name="document_to_markdown",
                description="把文档或聊天内容转换/导出为 Markdown。",
                tags=["文档", "markdown", "md", "导出md"],
                provider="local",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="把上面的表格导出为 Excel",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=1),
    ):
        selected = selector.select_tool_names(
            ["table_to_excel", "document_to_markdown"],
            command=command,
        )

    assert selected == ["table_to_excel"]


def test_tool_selection_reads_single_web_page_over_search_or_media():
    catalog = FakeCatalog(
        {
            "web_page_reader": ToolCatalogEntry(
                name="web_page_reader",
                description="读取用户给出的单个普通网页 URL 正文，适合总结这个网页、这篇文章讲什么、这个页面主要内容是什么。",
                tags=["web", "webpage", "url", "网页", "网页读取", "网页总结", "文章总结"],
                aliases=["read web page", "summarize webpage", "读取网页", "总结网页", "链接内容"],
                examples=["总结一下这个网页 https://example.com/article"],
                negative_examples=["搜索最新新闻", "分析这张图片", "这视频讲了什么"],
                intent_anchors=["读取单个网页 URL 内容", "总结用户提供的网页正文"],
                input_modalities=["text"],
                provider="local",
                enabled=True,
            ),
            "tavily_search": ToolCatalogEntry(
                name="tavily_search",
                description="使用 Tavily 联网搜索公开网页信息，适合搜索最新新闻、旅行规划、事实补充。",
                tags=["web", "search", "联网", "搜索"],
                aliases=["internet search", "web search"],
                examples=["搜索最新新闻"],
                provider="local",
                enabled=True,
            ),
            "analyze_media": ToolCatalogEntry(
                name="analyze_media",
                description="分析图片或视频内容。",
                tags=["图片", "视频", "媒体"],
                provider="local",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="这个链接主要讲什么？https://example.com/article",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=True),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=1),
    ):
        selected = selector.select_tool_names(
            ["web_page_reader", "tavily_search", "analyze_media"],
            command=command,
        )

    assert selected == ["web_page_reader"]


def test_tool_selection_collapses_url_content_members_when_facade_selected():
    selected = ToolSelectionService._collapse_url_content_members(
        [
            "process_url_content",
            "document_to_markdown",
            "web_page_reader",
            "knowledge_base_archive",
            "table_to_excel",
        ]
    )

    assert selected == ["process_url_content", "knowledge_base_archive"]


def test_tool_selection_does_not_collapse_members_without_facade():
    selected = ToolSelectionService._collapse_url_content_members(
        ["document_to_markdown", "web_page_reader"]
    )

    assert selected == ["document_to_markdown", "web_page_reader"]


def test_tool_selection_prefers_process_url_content_for_markdown_url():
    catalog = FakeCatalog(
        {
            "process_url_content": ToolCatalogEntry(
                name="process_url_content",
                description="处理用户给定 URL 的内容，读取和总结网页、文档、Markdown 文件。",
                tags=["url", "document", "markdown"],
                provider="local",
                enabled=True,
            ),
            "document_to_markdown": ToolCatalogEntry(
                name="document_to_markdown",
                description="把文档转换为 Markdown。",
                tags=["document", "markdown"],
                provider="local",
                enabled=True,
            ),
            "document_to_pdf": ToolCatalogEntry(
                name="document_to_pdf",
                description="把文档转换为 PDF。",
                tags=["document", "pdf"],
                provider="local",
                enabled=True,
            ),
            "knowledge_base_query": ToolCatalogEntry(
                name="knowledge_base_query",
                description="查询本地知识库文档。",
                tags=["knowledge", "kb"],
                provider="local",
                enabled=True,
            ),
        }
    )
    selector = ToolSelectionService(catalog=catalog)
    command = AgentCommand(
        user_id="user-1",
        agent_id="agent-1",
        message="请基于这份文档 http://example.com/report.md 给出回复",
    )

    with (
        patch("backend.services.agent.tool_selection.is_openclaw_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.is_tool_selection_enabled", return_value=False),
        patch("backend.services.agent.tool_selection.get_tool_selection_max_tools", return_value=2),
    ):
        selected = selector.select_tool_names(
            ["knowledge_base_query", "document_to_pdf", "document_to_markdown", "process_url_content"],
            command=command,
        )

    assert selected == ["process_url_content"]


def test_tool_selection_prefers_process_url_content_over_kb_for_contextual_markdown_document():
    selected = ToolSelectionService._prefer_url_content_facade(
        ["knowledge_base_query"],
        filtered_entries={
            "process_url_content": ToolCatalogEntry(name="process_url_content"),
            "knowledge_base_query": ToolCatalogEntry(name="knowledge_base_query"),
        },
        query_text="请基于这个 md 文档信息回答",
        command=AgentCommand(
            user_id="user-1",
            agent_id="agent-1",
            message=(
                "[过去对话]\n"
                "assistant: 完整 Markdown 文件已生成，文件名为 report.md。\n\n"
                "[本轮输入]\n"
                "user: 请基于这个 md 文档信息回答"
            ),
            context={"original_user_message": "请基于这个 md 文档信息回答"},
        ),
        limit=2,
    )

    assert selected == ["process_url_content"]


def test_tool_selection_keeps_kb_when_query_asks_knowledge_base_about_markdown_url():
    selected = ToolSelectionService._prefer_url_content_facade(
        ["knowledge_base_query"],
        filtered_entries={
            "process_url_content": ToolCatalogEntry(name="process_url_content"),
            "knowledge_base_query": ToolCatalogEntry(name="knowledge_base_query"),
        },
        query_text="请问知识库里有 http://example.com/report.md 相关的内容吗？",
        command=AgentCommand(
            user_id="user-1",
            agent_id="agent-1",
            message="请问知识库里有 http://example.com/report.md 相关的内容吗？",
        ),
        limit=2,
    )

    assert selected == ["knowledge_base_query"]
