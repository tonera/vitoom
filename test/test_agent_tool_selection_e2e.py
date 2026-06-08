"""端到端工具选择回归套件。

跟 ``test_agent_tool_selection.py``（用 ``FakeCatalog`` 做小型语义单元测试）不同，
这个文件用**真实** ``config/agent_tools.yaml`` + 真实 ONNX embedding，串起
完整召回/打分/筛选链路，作为：

- 多语言（中/英/日）TTS / ASR 消歧的金标
- 模态路由（带 URL 的图/视/音/文档）的金标
- 闲聊不应召回任何工具的金标
- 短 query / 跨语意图剥离 / 用户原始报告 case 的金标

新增或修改工具时，请在 ``CASES`` 里追加/修订条目；这里失败就意味着工具选择
策略层面发生了行为漂移。

环境要求：
- ``config/agent_tools.yaml`` 可加载
- ``resources/models/multilingual-e5-small-onnx`` 模型可加载
若 ONNX 模型不可用（CI 等场景），整组用例自动 skip 而非失败。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent import tool_selection as tool_selection_module  # noqa: E402
from backend.services.agent.tool_selection import ToolSelectionService  # noqa: E402


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
# 每条：``(query, expected_top_tool_name_or_None, scenario_tag)``。
# 期望为 ``None`` 表示『闲聊 / 工具不相关』，即不应召回任何工具。
#
# 命名约定（scenario）：
#   basic-*       — 各工具的标准正例（每工具至少 1 条）
#   modality-*    — query 含媒体 URL 的模态路由
#   tts-asr-*     — TTS / ASR 跨语消歧（最敏感）
#   chitchat-*    — 闲聊场景，必须返回空
#   short-cn-*    — 短中文 query 的 vector 方向性兜底
#   user-*        — 用户实际报告过的 case
CASES: list[tuple[str, Optional[str], str]] = [
    # --- basic 正例（中/英/日） ---
    ("画一只赛博朋克少女", "image_generator", "basic-image-gen-cn"),
    ("generate an image of a cat", "image_generator", "basic-image-gen-en"),
    ("猫の画像を作って", "image_generator", "basic-image-gen-ja"),
    ("把这段录音转成文字", "audio_asr", "basic-asr-cn"),
    ("transcribe this audio", "audio_asr", "basic-asr-en"),
    ("把这段文字念出来", "audio_tts", "basic-tts-cn"),
    ("read this text aloud", "audio_tts", "basic-tts-en"),
    ("把这个pdf转成markdown", "document_to_markdown", "basic-doc-cn"),
    ("请总结以上信息，导出为 md 文档", "document_to_markdown", "basic-export-md-cn"),
    ("把刚才的回答导出为 PDF", "document_to_pdf", "basic-export-pdf-cn"),
    ("把上面的表格导出为 Excel", "table_to_excel", "basic-export-excel-cn"),
    ("帮我搜索 2026 年北京 GDP 数据", "tavily_search", "basic-search-cn"),
    ("帮我规划一个东京3日游", "travel_planner", "basic-travel-cn"),

    # --- modality 路由：带 URL 时按媒体类型路由 ---
    ("分析一下这张图片", "analyze_media", "basic-analyze-cn"),
    ("这视频讲了什么 http://example.com/a.mp4", "analyze_media", "modality-video"),
    ("把这张图换成水彩风 http://example.com/foo.png", "image_editor", "modality-image-edit"),
    ("把这段录音转成文字 http://example.com/a.mp3", "audio_asr", "modality-audio-asr"),
    (
        "参考这张图的风格画一只小狗 http://example.com/ref.jpg",
        "image_generator",
        "modality-image-gen-with-ref",
    ),
    (
        "把这个pdf转成markdown http://example.com/a.pdf",
        "document_to_markdown",
        "modality-doc",
    ),

    # --- TTS / ASR 跨语消歧：用户主驱动的核心场景 ---
    # 命令式 TTS（动词为『说 / 朗读 / 念』），后面跟着被朗读的内容；
    # 即便内容本身字面像 ASR，也必须命中 audio_tts。
    ("用英语说：把这段音频转为文字！", "audio_tts", "tts-asr-cn-instruct-en"),
    ("用沧桑的男声生气地说：把这段音频转为文字！", "audio_tts", "tts-asr-cn-style"),
    ("用日语朗读：今天天气真好", "audio_tts", "tts-asr-cn-instruct-ja"),
    ("用温柔的女声念这段话", "audio_tts", "tts-asr-cn-voice-style"),
    # 命令式 ASR（动词为『转文字 / 转录』）
    ("把这段音频转为文字", "audio_asr", "tts-asr-bare-asr"),

    # --- 短中文 query：BM25 字面命中弱，靠 vector 方向性兜底 ---
    ("画个猫", "image_generator", "short-cn-image-gen-2char"),
    ("画个小猫", "image_generator", "short-cn-image-gen-3char"),
    ("画个可爱的小猫", "image_generator", "short-cn-image-gen-6char"),
    ("画个可爱的小狗", "image_generator", "short-cn-image-gen-with-dog"),

    # --- chitchat / 工具不相关：必须返回空，绝不能凭语义模糊召回 ---
    ("你好啊", None, "chitchat-greeting"),
    ("今晚月色真美", None, "chitchat-poetry"),
    ("解释一下 Python 闭包是什么", None, "chitchat-tech-qa"),
    ("解释一下 Python 闭包和装饰器", None, "chitchat-tech-qa-2"),
    ("你支持哪些命令？", "list_slash_commands", "slash-command-list-cn"),
    ("你有哪些可用命令", "list_slash_commands", "slash-command-list-cn-available"),
    ("你现在能用工具吗", "list_available_tools", "capability-list-cn-tools"),
    (
        "[过去对话]\nuser: 旧金山有多少员工\nassistant: 旧金山有 40 名员工。\n\n[本轮输入]\nuser: 名单发我",
        "hr_business_query",
        "hr-followup-list-from-history",
    ),

    # --- 用户实际报告过的 case ---
    (
        "这张ai生成的图片里有小狗吗"
        "http://local.fluxsd.cn/2026/04/12/301716522266017792_0_s.jpeg",
        "analyze_media",
        "user-original-ai-image-with-url",
    ),
    ("这张图里有小狗吗", "analyze_media", "user-image-question"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tool_selection_service() -> ToolSelectionService:
    """构造一次共享 service，避免每条 case 重新加载 ONNX 模型。"""
    service = ToolSelectionService()
    # 触发一次空跑，强制 catalog + ONNX 加载。
    service.debug_select_tool_names(
        [],
        command=SimpleNamespace(
            message="bootstrap",
            context={"original_user_message": "bootstrap"},
            runtime_config={},
        ),
        runtime_allowlist=[],
        max_tools=1,
        pool="global",
        preferred_tool_names=[],
    )
    backend = tool_selection_module._EMBEDDING_BACKENDS.get_backend()
    if not backend.is_ready():
        pytest.skip(
            "ONNX embedding backend not ready; "
            "set TOOL_SELECTION_EMBEDDING_BACKEND=onnx + model path to run e2e suite."
        )
    return service


def _select(service: ToolSelectionService, query: str) -> list[str]:
    if "[本轮输入]" in query:
        original = query.rsplit("[本轮输入]", 1)[-1].strip()
        if original.lower().startswith("user:"):
            original = original[len("user:"):].strip()
    else:
        original = query
    command = SimpleNamespace(
        message=query,
        context={"original_user_message": original},
        runtime_config={},
    )
    result = service.debug_select_tool_names(
        [],
        command=command,
        runtime_allowlist=[],
        max_tools=5,
        pool="global",
        preferred_tool_names=[],
    )
    return list(result.selected_names)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query, expected_top, scenario",
    [pytest.param(q, expected, sc, id=sc) for q, expected, sc in CASES],
)
def test_tool_selection_e2e(
    tool_selection_service: ToolSelectionService,
    query: str,
    expected_top: Optional[str],
    scenario: str,
) -> None:
    selected = _select(tool_selection_service, query)
    actual_top = selected[0] if selected else None
    if expected_top is None:
        assert actual_top is None, (
            f"[{scenario}] expected no candidates for {query!r}, got top={actual_top!r}"
        )
    else:
        assert actual_top == expected_top, (
            f"[{scenario}] expected top={expected_top!r} for {query!r}, "
            f"got top={actual_top!r}, full selection={selected}"
        )
