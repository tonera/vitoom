from __future__ import annotations

import math
import re
import threading
import time
import importlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from backend.core.logger import get_app_logger

from .settings import (
    get_tool_selection_always_include,
    get_tool_selection_bm25_top_k,
    get_tool_selection_embedding_backend,
    get_tool_selection_embedding_model_path,
    get_tool_selection_embedding_timeout_ms,
    get_tool_selection_max_tools,
    get_tool_selection_min_ratio_of_top,
    get_tool_selection_min_score,
    get_tool_selection_query_cache_size,
    get_tool_selection_rebuild_check_interval_seconds,
    get_tool_selection_strategy,
    get_tool_selection_vector_top_k,
    is_tool_selection_embedding_enabled,
    is_openclaw_enabled,
    is_tool_selection_enabled,
    get_tool_catalog_path,
)
from .tool_catalog import ToolCatalog, ToolCatalogEntry

logger = get_app_logger(__name__)

try:
    np = importlib.import_module("numpy")
except Exception:  # pragma: no cover - numpy is optional for BM25-only mode.
    np = None  # type: ignore[assignment]

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+")
_WHITESPACE_RE = re.compile(r"\s+")

# 中文 / 英文虚词停用词。char-1gram 切分会让"的""了""和""一"这种高频虚字
# 在 query 与几乎任意工具描述之间产生 BM25 假命中，通过停用词过滤可以稳定
# 地把"解释一下 Python 闭包"这种闲聊 query 推回零信号。
# 这里只覆盖**单字 / 极短**的虚词；更长的词组靠 char-2/3gram 自然区分，无需在此列举。
# BM25 停用词。命中规则：token（不限长度）若在此集合内，跳过收录。
# 中文 char-1gram 整体已在 _tokenize 里跳过；这里主要兜底覆盖**语气助词性
# 短语**（"一下""什么""怎么""如何"...）和英文虚词，这些短语在工具
# `examples` 字段里非常常见，但对工具区分没有信号。
# 多语言高频虚词/助词的字符串黑名单（**不是**句式或语义枚举）。
# 中、日、英在工具描述/query 里都不可避免会出现这些函数词，char-ngram 又会
# 把 "是什么" 切成 "是什/什么" 这种残段，残段在 BM25 里跟工具描述假命中
# （例如 analyze_media examples 含『是什么』），导致闲聊 query 也拿到分。
# 解决办法是把这些残段一并屏蔽。新增一个 token 是 O(1) 改动，与工具列表/语种解耦。
_STOPWORDS: frozenset[str] = frozenset(
    {
        # 中文虚词/语气助词短语（含 char-2gram 残段）
        "一下", "什么", "怎么", "如何", "为何", "为什么", "多少",
        "请问", "可以", "能否", "能够", "可否", "需要", "想要",
        "帮我", "帮忙", "麻烦", "请帮", "辛苦", "谢谢",
        "一个", "一只", "一张", "一段", "一份", "一篇", "一些",
        "这个", "这张", "这段", "这只", "这份", "这种",
        "那个", "那张", "那段", "那只",
        # "X 是什么 / 什么的 / 是什" 这类 char-2/3gram 残段
        "是什", "是什么", "什么是", "什么的", "了吗", "好吗", "对吗", "啥的",
        # 日文动词词尾/助动词的 char-2gram 残段
        # （"猫の画像を作って" 切出 "って" 跟 query "怒った声で言って" 假命中）
        "って", "です", "ます", "した", "して", "ない", "ある", "いる",
        "から", "けど", "ます", "なる", "あり", "され", "せて", "せる",
        "下さ", "ださ", "くだ", "ださ", "そう", "よう", "ので", "てい",
        # 英文虚词
        "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or",
        "is", "are", "was", "were", "be", "been", "being",
        "i", "me", "my", "we", "us", "our", "you", "your",
        "this", "that", "these", "those",
        "with", "from", "by", "as", "but", "if", "so", "not", "no",
        "please", "help",
    }
)

_HYBRID_STRATEGIES = {"bm25", "hybrid"}
_MUTEX_TOOL_PAIRS = {
    frozenset(("image_generator", "image_editor")),
    frozenset(("image_generator", "analyze_media")),
    frozenset(("image_editor", "analyze_media")),
    frozenset(("audio_asr", "audio_tts")),
}

PROCESS_URL_CONTENT_TOOL_NAME = "process_url_content"
URL_CONTENT_MEMBER_TOOLS: frozenset[str] = frozenset(
    {
        "web_page_reader",
        "analyze_media",
        "document_to_markdown",
        "document_to_pdf",
        "table_to_excel",
    }
)

# 通过 URL 后缀粗粒度推断 query 携带了什么模态资源。
# 故意只匹配 https? URL，避免把『.md 文件』『a.txt 这一段』这种纯文本误判。
# 注意：不能在 ``https?`` 前用 ``\b``，因为中文紧邻 URL（如 "图片吗http://..."）
# 时 "吗"/"h" 都是 ``\w`` 字符，``\b`` 不会成立 → 整段 URL 检测失效。
# 末尾的 ``(?=[?#/]|$|\s)`` lookahead 让 query string / anchor / 句末标点不破坏后缀匹配。
_URL_MODALITY_PATTERNS: List[tuple[str, "re.Pattern[str]"]] = [
    (
        "image",
        re.compile(
            r"https?://\S+?\.(?:jpe?g|png|gif|webp|bmp|tiff?|svg)(?=[?#/]|$|\s)",
            re.IGNORECASE,
        ),
    ),
    (
        "video",
        re.compile(
            r"https?://\S+?\.(?:mp4|mov|webm|mkv|avi|flv|m4v|3gp)(?=[?#/]|$|\s)",
            re.IGNORECASE,
        ),
    ),
    (
        "audio",
        re.compile(
            r"https?://\S+?\.(?:mp3|wav|m4a|ogg|flac|aac|opus|amr)(?=[?#/]|$|\s)",
            re.IGNORECASE,
        ),
    ),
    (
        "document",
        re.compile(
            r"https?://\S+?\.(?:pdf|docx?|pptx?|xlsx?|csv|md|txt|rtf|epub)(?=[?#/]|$|\s)",
            re.IGNORECASE,
        ),
    ),
]

_URL_CONTENT_DOCUMENT_RE = re.compile(
    r"https?://\S+?\.(?:pdf|docx?|pptx?|xlsx?|csv|md|markdown|txt|rtf|epub|zip)(?=[?#/]|$|\s)",
    re.IGNORECASE,
)
_TRANSIENT_DOCUMENT_MENTION_RE = re.compile(
    r"(?:md|markdown|pdf|文档|文件|报告|附件)",
    re.IGNORECASE,
)
_KNOWLEDGE_BASE_INTENT_RE = re.compile(
    r"(?:知识库|kb|knowledge\s*base|已入库|入库|归档|存档)",
    re.IGNORECASE,
)


# 跨语言剥离 query 中的"被引内容"：所有语言里"指令 + 引出内容"句式都靠
# **冒号**（中英）或**成对引号**做分隔，比如：
#   中: "用沧桑的男声生气地说：把这段音频转为文字！"
#   英: "In a deep voice, he angrily said: 'Transcribe this audio.'"
#   日: "怒った声で言って：「この音声を書き起こして！」"
#   法: "D'une voix grave, il a dit: \"Transcrire cet audio.\""
# 把"被引内容"剥掉后剩下的部分才是真正的工具意图，这条剥离规则**完全不依赖
# 具体语言**——既不用枚举每种语言的"说/朗读/voice/...", 也不用扩 yaml。
_COLON_CHARS = (":", "：", "︰")
_QUOTE_PAIRS: List[tuple[str, str]] = [
    ('"', '"'),
    # 注意：故意不放入 ASCII 单引号 (' ')。它和英文撇号同字符
    # （Don't / it's / 'cute'），靠『最早 / 最晚』成对会误切分。
    # 真要切分时用户多半会用全角弯引号 (\u2018 \u2019) 或具语言专属对。
    ("\u201c", "\u201d"),  # “ ”
    ("\u2018", "\u2019"),  # ‘ ’
    ("「", "」"),
    ("『", "』"),
    ("«", "»"),
    ("‹", "›"),
    ("《", "》"),
]
# URL scheme 后的 ``:`` 不能当成 intent/content 分隔符。
_URL_SCHEMES_BEFORE_COLON = ("http", "https", "ftp", "file", "ws", "wss", "data")
_META_INTENT_LABEL_RE = re.compile(
    r"^(?:用户\s*)?(?:需求|请求|问题|任务|输入|指令|query|request)(?:是|为)?$",
    re.IGNORECASE,
)

# Vector 方向性闸门阈值。
# 当 BM25 完全无字面命中（max_bm25 == 0）、且 query 也没有 intent/content 剥离信号时，
# 仅在 vector top1 跟 top2 的 raw cosine 差距 ≥ 此值时才纳入 vector 召回，
# 否则视为"语义模糊"丢弃。
#
# 经验值 0.02 的来源（multilingual-e5-small 实测分布）：
#   - 真正方向性命中：『画个猫』/『画个小猫』/『画个可爱的小猫』top1-top2 ≥ 0.030
#   - 真正语义模糊：『你好啊』/『今晚月色真美』/『解释 Python 闭包』top1-top2 ≤ 0.003
# 这两个簇之间留有近 10 倍差距。阈值取 0.02 在中间留出安全边际。
_VECTOR_DIRECTIONAL_GAP = 0.02


def _split_query_intent_content(query: str) -> tuple[str, str]:
    """跨语言把 query 拆成 ``(intent, content)``，没有可识别分隔符时
    ``content`` 为空字符串、``intent`` 为整段 query。

    优先级：先尝试**最早一个非 URL 冒号**做切分；若无冒号，再尝试**最外层成对引号**。
    """
    text = str(query or "").strip()
    if not text:
        return "", ""

    # 1) 冒号优先：扫描所有冒号字符，挑最早一个非 URL scheme 后缀的位置
    earliest_idx = -1
    for sep in _COLON_CHARS:
        cursor = 0
        while cursor < len(text):
            idx = text.find(sep, cursor)
            if idx < 0:
                break
            preceding = text[max(0, idx - 8):idx].lower()
            if any(preceding.endswith(scheme) for scheme in _URL_SCHEMES_BEFORE_COLON):
                cursor = idx + 1
                continue
            if earliest_idx < 0 or idx < earliest_idx:
                earliest_idx = idx
            break

    if earliest_idx >= 0:
        intent = text[:earliest_idx].strip()
        content = text[earliest_idx + 1:].strip()
        # 去掉 content 外层引号（不影响 intent）
        for opener, closer in _QUOTE_PAIRS:
            if content.startswith(opener) and content.endswith(closer) and len(content) >= len(opener) + len(closer):
                content = content[len(opener):-len(closer)].strip()
                break
        if intent and content:
            if _META_INTENT_LABEL_RE.match(intent):
                return content, ""
            return intent, content

    # 2) 引号回退：找最外层成对引号
    for opener, closer in _QUOTE_PAIRS:
        oi = text.find(opener)
        if oi < 0:
            continue
        ci = text.rfind(closer)
        if ci > oi + len(opener):
            content = text[oi + len(opener):ci].strip()
            intent = (text[:oi] + " " + text[ci + len(closer):]).strip()
            if intent and content:
                return intent, content

    return text, ""


def _detect_query_modalities(query_text: str) -> Set[str]:
    """根据 query 中出现的 URL 后缀返回检测到的非文本模态集合。

    没匹配到任何媒体 URL 时返回 ``{"text"}``，调用方据此跳过模态打分。
    """
    detected: Set[str] = set()
    text = str(query_text or "")
    if not text:
        return {"text"}
    for modality, pattern in _URL_MODALITY_PATTERNS:
        if pattern.search(text):
            detected.add(modality)
    return detected or {"text"}


def _has_url_content_document(query_text: str) -> bool:
    return bool(_URL_CONTENT_DOCUMENT_RE.search(str(query_text or "")))


def _mentions_transient_document(query_text: str) -> bool:
    return bool(_TRANSIENT_DOCUMENT_MENTION_RE.search(str(query_text or "")))


def _has_knowledge_base_intent(query_text: str) -> bool:
    return bool(_KNOWLEDGE_BASE_INTENT_RE.search(str(query_text or "")))


def _modality_scores(
    query_modalities: Set[str],
    tool_modalities: Iterable[str],
) -> tuple[float, float]:
    """根据 query 模态和工具声明的 ``input_modalities`` 返回 ``(match, mismatch)``。

    - query 未携带非文本模态 → 模态打分整体跳过，返回 ``(0.0, 0.0)``。
    - 工具未声明 ``input_modalities`` → 保守跳过打分，返回 ``(0.0, 0.0)``。
      这样 catalog 里没标注的工具不会因为新机制被错误降权，给逐步迁移留口子。
    - 工具声明了模态：与 query 的非文本模态有交集 → ``(1.0, 0.0)``；
      完全无交集 → ``(0.0, 1.0)``，工具被显著降权。
    """
    tool_set = {str(item).strip().lower() for item in tool_modalities if str(item).strip()}
    if not tool_set:
        return 0.0, 0.0
    query_non_text = query_modalities - {"text"}
    if not query_non_text:
        return 0.0, 0.0
    if query_non_text & tool_set:
        return 1.0, 0.0
    return 0.0, 1.0


def _normalize_list(raw_value: Any) -> List[str]:
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return [str(raw_value).strip()] if str(raw_value).strip() else []


def _ngrams(text: str, *, min_n: int = 1, max_n: int = 3) -> List[str]:
    length = len(text)
    tokens: List[str] = []
    for size in range(min_n, min(max_n, length) + 1):
        for idx in range(length - size + 1):
            tokens.append(text[idx : idx + size])
    return tokens


def _split_ascii_token(token: str) -> List[str]:
    pieces = [token]
    for delimiter in ("_", "-"):
        if delimiter in token:
            pieces.extend(part for part in token.split(delimiter) if part)
    for part in list(pieces):
        pieces.extend(piece.lower() for piece in _CAMEL_BOUNDARY_RE.split(part) if piece)
    return [piece.lower() for piece in pieces if piece]


def _tokenize(text: str) -> List[str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    tokens: List[str] = []
    for match in _ASCII_TOKEN_RE.finditer(raw_text):
        token = match.group(0)
        for piece in _split_ascii_token(token):
            if piece.lower() in _STOPWORDS:
                continue
            tokens.append(piece)

    normalized = raw_text.lower()
    for match in _CJK_RE.finditer(normalized):
        chunk = _WHITESPACE_RE.sub("", match.group(0))
        if not chunk:
            continue
        if chunk not in _STOPWORDS:
            tokens.append(chunk)
        # 中文 BM25 召回**只用 char-2/3gram + chunk 整体**：
        # char-1gram 的命中信号在工具描述这种短文本上 95% 是噪声（"的""下""器""一"
        # 这类高频字与几乎任意 query 都能配对），会让闲聊 query 也拿到虚假分数；
        # 真正的语义短词都至少 2 字（"分析""图片""视频""规划"），靠 2/3gram 自然召回。
        # 多 char 语气助词短语（"一下""什么""帮我"...）在 _STOPWORDS 里兜底过滤。
        for ngram in _ngrams(chunk, min_n=2, max_n=3):
            if ngram in _STOPWORDS:
                continue
            tokens.append(ngram)
    return tokens


def _vectorize(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _extract_primary_user_text(command: Any) -> str:
    context = getattr(command, "context", None) or {}
    if isinstance(context, dict):
        original = str(context.get("original_user_message") or "").strip()
        if original:
            return original
    return str(getattr(command, "message", "") or "").strip()


def _conversation_prompt_to_selection_text(prompt: str, original: str) -> str:
    lines: List[str] = []
    for raw_line in str(prompt or "").splitlines():
        line = raw_line.strip()
        if not line or line in {"[历史摘要]", "[过去对话]", "[本轮输入]"}:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        for prefix in ("user:", "assistant:", "unknown:"):
            if line.lower().startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            lines.append(line)
    if original and (not lines or lines[-1] != original):
        lines.append(original)
    return " ".join(lines).strip()


def _build_contextual_selection_text(command: Any, original: str) -> str:
    full_message = str(getattr(command, "message", "") or "").strip()
    if not full_message or full_message == original:
        return ""
    if original and original not in full_message:
        return ""
    contextual = _conversation_prompt_to_selection_text(full_message, original)
    if contextual and contextual != original:
        return contextual
    return ""


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left.keys()) & set(right.keys())
    numerator = sum(left[token] * right[token] for token in shared)
    if numerator <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _safe_float_ratio(value: float, divisor: float) -> float:
    if divisor <= 0:
        return 0.0
    return value / divisor


def _contains_exact_match(query_text: str, entry: ToolCatalogEntry) -> float:
    query = str(query_text or "").strip().lower()
    if not query:
        return 0.0
    exact_values = [entry.name, entry.runtime_tool_name, *entry.aliases]
    for raw_value in exact_values:
        value = str(raw_value or "").strip().lower()
        if value and (value in query or query in value):
            return 1.0
    return 0.0


def _is_mutex_pair(left: str, right: str) -> bool:
    return frozenset((left, right)) in _MUTEX_TOOL_PAIRS


def _normalize_dense_vector(vector: Any) -> Optional[Any]:
    if np is None:
        return None
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 0:
        return None
    if array.ndim > 1:
        array = array.reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        return None
    return array / norm


def _normalize_dense_matrix(matrix: Any) -> Optional[Any]:
    if np is None:
        return None
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        return None
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return array / norms


@dataclass(frozen=True)
class ToolSearchDocument:
    name: str
    entry: ToolCatalogEntry
    positive_text: str
    negative_text: str
    index: int


@dataclass(frozen=True)
class ScoredCandidate:
    name: str
    score: float
    index: int
    bm25_score: float = 0.0
    vector_score: float = 0.0
    exact_match_score: float = 0.0
    preferred_boost: float = 0.0
    negative_match_score: float = 0.0
    modality_match_score: float = 0.0
    modality_mismatch_score: float = 0.0
    intent_anchor_score: float = 0.0


@dataclass(frozen=True)
class ToolSelectionDebugResult:
    selected_names: List[str]
    candidates: List[ScoredCandidate]
    strategy: str
    index_version: str
    embedding_key: str
    latency_ms: int


class BM25Index:
    def __init__(self, documents: List[ToolSearchDocument], *, k1: float = 1.5, b: float = 0.75):
        self._documents = documents
        self._k1 = k1
        self._b = b
        self._doc_tokens: Dict[str, Counter[str]] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._postings: Dict[str, Dict[str, int]] = {}
        self._idf: Dict[str, float] = {}
        self._avg_doc_len = 0.0
        self._build()

    def _build(self) -> None:
        total_len = 0
        for document in self._documents:
            token_counts = Counter(_tokenize(document.positive_text))
            self._doc_tokens[document.name] = token_counts
            doc_len = sum(token_counts.values())
            self._doc_lengths[document.name] = doc_len
            total_len += doc_len
            for token, count in token_counts.items():
                self._postings.setdefault(token, {})[document.name] = count

        doc_count = max(1, len(self._documents))
        self._avg_doc_len = total_len / doc_count if total_len > 0 else 0.0
        for token, posting in self._postings.items():
            df = len(posting)
            self._idf[token] = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))

    def search(self, query_text: str, *, allowed_names: Set[str], top_k: int) -> Dict[str, float]:
        query_tokens = Counter(_tokenize(query_text))
        if not query_tokens or not allowed_names:
            return {}

        scores: Dict[str, float] = {}
        avg_doc_len = self._avg_doc_len or 1.0
        for token, query_count in query_tokens.items():
            posting = self._postings.get(token)
            if not posting:
                continue
            idf = self._idf.get(token, 0.0)
            for name, term_freq in posting.items():
                if name not in allowed_names:
                    continue
                doc_len = self._doc_lengths.get(name, 0)
                denominator = term_freq + self._k1 * (1.0 - self._b + self._b * doc_len / avg_doc_len)
                if denominator <= 0:
                    continue
                contribution = idf * (term_freq * (self._k1 + 1.0) / denominator) * max(1, query_count)
                scores[name] = scores.get(name, 0.0) + contribution

        if not scores:
            return {}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(1, top_k)]
        return dict(ranked)


class NegativeExampleIndex:
    """工具负例 BM25 子索引。

    每个工具的 ``negative_examples`` 拼成一段独立 doc 进 BM25。
    打分时返回该 query 对所有工具的归一化负例分（max-norm 到 ``[0, 1]``）。

    相比旧的 1-3gram cosine：
    - BM25 用 IDF，能突出"这张图片中是什么"这种关键短语，
      避免被『的、了、吗』这种高频虚词稀释；
    - 同一份 BM25 实现，正负打分行为更可预测、便于调权。
    """

    def __init__(self, documents: List[ToolSearchDocument]):
        negative_documents = [
            ToolSearchDocument(
                name=document.name,
                entry=document.entry,
                positive_text=document.negative_text,
                negative_text="",
                index=document.index,
            )
            for document in documents
            if document.negative_text
        ]
        self._has_negatives = bool(negative_documents)
        self._bm25: Optional[BM25Index] = (
            BM25Index(negative_documents) if negative_documents else None
        )
        self._allowed_names: Set[str] = {document.name for document in negative_documents}

    def scores(self, query_text: str) -> Dict[str, float]:
        """返回所有有负例的工具，对该 query 的归一化负例分。

        ``[0, 1]`` 区间，越大表示 query 越像该工具的反例 → 该工具应被降权。
        没有命中任何负例时返回空字典；调用方据此把 score 视为 0。
        """
        if not self._has_negatives or self._bm25 is None or not self._allowed_names:
            return {}
        raw_scores = self._bm25.search(
            query_text,
            allowed_names=self._allowed_names,
            top_k=len(self._allowed_names),
        )
        if not raw_scores:
            return {}
        max_score = max(raw_scores.values(), default=0.0)
        if max_score <= 0:
            return {}
        return {name: value / max_score for name, value in raw_scores.items()}


class EmbeddingBackend:
    cache_key = "none"

    def is_ready(self) -> bool:
        return False

    def embed_query(self, text: str, timeout_ms: int) -> Optional[Any]:
        del text, timeout_ms
        return None

    def embed_documents(self, texts: List[str]) -> Optional[Any]:
        del texts
        return None


class NoopEmbeddingBackend(EmbeddingBackend):
    pass


class OnnxEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_path: str):
        self.model_path = str(model_path or "").strip()
        self.cache_key = f"onnx:{self.model_path}"
        self._ready = False
        self._tokenizer: Any = None
        self._session: Any = None
        self._input_names: List[str] = []
        self._pad_token_id = 0
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.model_path:
            return
        model_dir = Path(self.model_path)
        onnx_path = model_dir
        if model_dir.is_dir():
            candidates = [
                model_dir / "model.onnx",
                model_dir / "onnx" / "model.onnx",
                model_dir / "model_quantized.onnx",
            ]
            onnx_path = next((path for path in candidates if path.exists()), candidates[0])
        if not onnx_path.exists():
            logger.warning("Tool embedding ONNX model not found: %s", onnx_path)
            return
        try:
            import onnxruntime as ort  # type: ignore
            from tokenizers import Tokenizer  # type: ignore

            tokenizer_path = str(model_dir if model_dir.is_dir() else onnx_path.parent)
            tokenizer_json = Path(tokenizer_path) / "tokenizer.json"
            if not tokenizer_json.exists():
                logger.warning("Tool embedding tokenizer.json not found: %s", tokenizer_json)
                return
            self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
            self._pad_token_id = (
                self._tokenizer.token_to_id("[PAD]")
                or self._tokenizer.token_to_id("<pad>")
                or self._tokenizer.token_to_id("<|endoftext|>")
                or 0
            )
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(onnx_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [item.name for item in self._session.get_inputs()]
            self._ready = True
            logger.info("Loaded tool embedding ONNX backend: %s", onnx_path)
        except Exception as exc:
            logger.warning("Failed to load tool embedding ONNX backend: %s", exc)
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def embed_query(self, text: str, timeout_ms: int) -> Optional[Any]:
        started = time.perf_counter()
        vector = self._embed_texts([text])
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if timeout_ms > 0 and elapsed_ms > timeout_ms:
            logger.warning("Tool query embedding exceeded timeout: elapsed_ms=%d timeout_ms=%d", elapsed_ms, timeout_ms)
            return None
        if vector is None or len(vector) == 0:
            return None
        return _normalize_dense_vector(vector[0])

    def embed_documents(self, texts: List[str]) -> Optional[Any]:
        matrix = self._embed_texts(texts)
        return _normalize_dense_matrix(matrix) if matrix is not None else None

    def _embed_texts(self, texts: List[str]) -> Optional[Any]:
        if np is None or not self._ready or not texts:
            return None
        try:
            encoded = self._encode_texts(texts)
            inputs = {name: encoded[name] for name in self._input_names if name in encoded}
            if not inputs:
                return None
            with self._lock:
                outputs = self._session.run(None, inputs)
            return self._pool_outputs(outputs, encoded.get("attention_mask"))
        except Exception as exc:
            logger.warning("Tool embedding failed: %s", exc)
            return None

    def _encode_texts(self, texts: List[str]) -> Dict[str, Any]:
        encoded_items = self._tokenizer.encode_batch(texts)
        max_len = min(256, max((len(item.ids) for item in encoded_items), default=0))
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        token_type_ids: List[List[int]] = []
        for item in encoded_items:
            ids = list(item.ids[:max_len])
            type_ids = list(item.type_ids[:max_len]) if item.type_ids else [0] * len(ids)
            mask = [1] * len(ids)
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids.extend([self._pad_token_id] * pad_len)
                type_ids.extend([0] * pad_len)
                mask.extend([0] * pad_len)
            input_ids.append(ids)
            token_type_ids.append(type_ids)
            attention_mask.append(mask)
        return {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            "token_type_ids": np.asarray(token_type_ids, dtype=np.int64),
        }

    @staticmethod
    def _pool_outputs(outputs: List[Any], attention_mask: Any) -> Optional[Any]:
        if np is None or not outputs:
            return None
        first = np.asarray(outputs[0])
        if first.ndim == 2:
            return first.astype(np.float32)
        if first.ndim != 3:
            return None
        if attention_mask is None:
            return first[:, 0, :].astype(np.float32)
        mask = np.asarray(attention_mask).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        summed = np.sum(first * mask, axis=1)
        counts = np.clip(np.sum(mask, axis=1), 1e-9, None)
        return (summed / counts).astype(np.float32)


class _EmbeddingBackendManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._backend: EmbeddingBackend = NoopEmbeddingBackend()
        self._config_key = ""

    def get_backend(self) -> EmbeddingBackend:
        backend_name = get_tool_selection_embedding_backend()
        model_path = get_tool_selection_embedding_model_path()
        enabled = is_tool_selection_embedding_enabled()
        config_key = f"{enabled}:{backend_name}:{model_path}"
        with self._lock:
            if config_key == self._config_key:
                return self._backend
            self._config_key = config_key
            if not enabled:
                self._backend = NoopEmbeddingBackend()
            elif backend_name == "onnx":
                self._backend = OnnxEmbeddingBackend(model_path)
            else:
                logger.warning("Unknown tool embedding backend=%s; embedding disabled", backend_name)
                self._backend = NoopEmbeddingBackend()
            return self._backend


_EMBEDDING_BACKENDS = _EmbeddingBackendManager()


class QueryEmbeddingCache:
    def __init__(self, max_size: int):
        self._max_size = max(0, int(max_size))
        self._items: Dict[str, Any] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        if self._max_size <= 0:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            return value

    def set(self, key: str, value: Any) -> None:
        if self._max_size <= 0 or value is None:
            return
        with self._lock:
            if key not in self._items and len(self._items) >= self._max_size and self._order:
                evicted = self._order.pop(0)
                self._items.pop(evicted, None)
            self._items[key] = value
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)


_QUERY_EMBEDDINGS = QueryEmbeddingCache(get_tool_selection_query_cache_size())


@dataclass(frozen=True)
class ToolSearchIndex:
    version: str
    built_at: float
    catalog_mtime: float
    catalog_key: int
    catalog_obj: Any
    documents: List[ToolSearchDocument]
    by_name: Dict[str, ToolSearchDocument]
    bm25: BM25Index
    negative_examples: NegativeExampleIndex
    embedding_key: str
    vector_matrix: Optional[Any]
    vector_names: List[str]
    # 工具的"意图锚点"嵌入矩阵：仅声明了 ``intent_anchors`` 的工具才进入此矩阵。
    # 一个工具可声明多条 anchor，每条 anchor 占一行；``intent_anchor_owners`` 把
    # 每行映射回工具名。召回阶段一次性把 ``query × matrix`` 算出，再按 owner
    # group 取每个工具的 max cosine 作为该工具的 anchor_score。
    # 这条信号**不进入工具 BM25 文档，也不进入 LLM 看到的 system prompt**，纯
    # 粹是召回侧的内部对齐锚点；用来在 query 极短或跨语言场景下做高精度意图对齐。
    intent_anchor_matrix: Optional[Any]
    intent_anchor_owners: List[str]


class _ToolSearchIndexManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._index: Optional[ToolSearchIndex] = None
        self._last_checked = 0.0

    def get_index(self, catalog: ToolCatalog) -> ToolSearchIndex:
        now = time.monotonic()
        interval = get_tool_selection_rebuild_check_interval_seconds()
        with self._lock:
            # 加 ``catalog_key`` 失配判定：生产里 catalog 是进程级单例，
            # ``id(catalog)`` 不变，不会引入额外重建；测试或运行时热替换
            # catalog 实例时立即触发重建，避免拿到来自旧 catalog 的索引。
            catalog_changed = self._index is not None and self._index.catalog_obj is not catalog
            if self._index is None or catalog_changed or now - self._last_checked >= interval:
                self._last_checked = now
                catalog_mtime = self._catalog_mtime()
                catalog_key = id(catalog)
                embedding_backend = _EMBEDDING_BACKENDS.get_backend()
                embedding_key = embedding_backend.cache_key if embedding_backend.is_ready() else "none"
                if (
                    self._index is None
                    or catalog_key != self._index.catalog_key
                    or catalog_mtime != self._index.catalog_mtime
                    or embedding_key != self._index.embedding_key
                ):
                    if self._index is not None and catalog_mtime != self._index.catalog_mtime:
                        catalog.reload()
                    self._index = self._build_index(
                        catalog,
                        catalog_mtime=catalog_mtime,
                        catalog_key=catalog_key,
                        embedding_backend=embedding_backend,
                        embedding_key=embedding_key,
                    )
            return self._index

    @staticmethod
    def _catalog_mtime() -> float:
        path = get_tool_catalog_path()
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def _build_index(
        catalog: ToolCatalog,
        *,
        catalog_mtime: float,
        catalog_key: int,
        embedding_backend: EmbeddingBackend,
        embedding_key: str,
    ) -> ToolSearchIndex:
        documents: List[ToolSearchDocument] = []
        for index, entry in enumerate(catalog.all().values()):
            documents.append(
                ToolSearchDocument(
                    name=entry.name,
                    entry=entry,
                    positive_text=entry.searchable_text,
                    negative_text=entry.negative_searchable_text,
                    index=index,
                )
            )
        built_at = time.time()
        version = f"{int(built_at * 1000)}:{len(documents)}"
        vector_matrix = None
        vector_names: List[str] = []
        intent_anchor_matrix = None
        intent_anchor_owners: List[str] = []
        if embedding_backend.is_ready():
            vector_names = [document.name for document in documents]
            vector_matrix = embedding_backend.embed_documents([document.positive_text for document in documents])
            if vector_matrix is None:
                vector_names = []
                embedding_key = "none"
            else:
                # 多 anchor 展开：一个工具的 N 条 anchor 各占一行。每行 owner 标
                # 记归属工具，scoring 阶段对同一 owner 的 cosine 取 max——这能
                # 让"工具的多个意图角度"彼此独立地参与召回，超短跨语 query 上
                # 比单 anchor 鲁棒（任一角度命中即可）。
                anchor_owners: List[str] = []
                anchor_texts: List[str] = []
                for document in documents:
                    for anchor in document.entry.intent_anchors:
                        anchor_owners.append(document.name)
                        anchor_texts.append(anchor)
                if anchor_texts:
                    anchor_matrix = embedding_backend.embed_documents(anchor_texts)
                    if anchor_matrix is not None:
                        intent_anchor_matrix = anchor_matrix
                        intent_anchor_owners = anchor_owners
        logger.info(
            "Built tool search index: version=%s tools=%d embedding=%s anchors=%d/%d",
            version,
            len(documents),
            embedding_key,
            len(set(intent_anchor_owners)),
            len(intent_anchor_owners),
        )
        return ToolSearchIndex(
            version=version,
            built_at=built_at,
            catalog_mtime=catalog_mtime,
            catalog_key=catalog_key,
                catalog_obj=catalog,
            documents=documents,
            by_name={document.name: document for document in documents},
            bm25=BM25Index(documents),
            negative_examples=NegativeExampleIndex(documents),
            embedding_key=embedding_key,
            vector_matrix=vector_matrix,
            vector_names=vector_names,
            intent_anchor_matrix=intent_anchor_matrix,
            intent_anchor_owners=intent_anchor_owners,
        )


_INDEX_MANAGER = _ToolSearchIndexManager()
_DEFAULT_TOOL_CATALOG = ToolCatalog()


def warm_tool_selection_index() -> None:
    """Build the process-wide tool search index before the first user turn."""

    started = time.perf_counter()
    index = _INDEX_MANAGER.get_index(_DEFAULT_TOOL_CATALOG)
    logger.info(
        "Tool selection index warmed: version=%s tools=%d embedding=%s elapsed_ms=%d",
        index.version,
        len(index.documents),
        index.embedding_key,
        int((time.perf_counter() - started) * 1000),
    )


def warm_tool_selection_embedding_model() -> bool:
    """Preload the configured tool-selection embedding model if it is available."""

    backend = _EMBEDDING_BACKENDS.get_backend()
    ready = backend.is_ready()
    if ready:
        logger.info("Tool selection embedding model warmed: %s", backend.cache_key)
    else:
        logger.info("Tool selection embedding warmup skipped: %s", backend.cache_key)
    return ready


class ToolSelectionService:
    """在工具实例化之前，根据请求语义筛选最相关的候选工具。"""

    def __init__(self, *, catalog: Optional[ToolCatalog] = None):
        self._catalog = catalog or _DEFAULT_TOOL_CATALOG

    def select_tool_names(
        self,
        tool_names: Iterable[str],
        *,
        command: Any,
        task_specs: Optional[List[Any]] = None,
        runtime_allowlist: Optional[List[str]] = None,
        max_tools: Optional[int] = None,
        pool: str = "declared",
        preferred_tool_names: Optional[Iterable[str]] = None,
    ) -> List[str]:
        unique_tool_names = self._build_candidate_pool(
            tool_names,
            pool=pool,
            preferred_tool_names=preferred_tool_names,
        )

        if not unique_tool_names:
            return []

        filtered_entries = self._apply_hard_filters(
            unique_tool_names,
            runtime_allowlist=runtime_allowlist,
        )
        if not filtered_entries:
            return []

        limit = max(1, int(max_tools or get_tool_selection_max_tools()))
        always_include = self._collect_always_include(filtered_entries)
        if len(always_include) >= limit:
            return self._collapse_url_content_members(always_include[:limit])

        if not is_tool_selection_enabled():
            remaining = [name for name in unique_tool_names if name in filtered_entries and name not in always_include]
            selected = (always_include + remaining)[:limit]
            selected = self._prefer_url_content_facade(
                selected,
                filtered_entries=filtered_entries,
                query_text=self._build_query_text(command, task_specs=task_specs),
                command=command,
                limit=limit,
            )
            return self._collapse_url_content_members(selected)

        strategy = get_tool_selection_strategy()
        if strategy not in _HYBRID_STRATEGIES:
            logger.warning("Unknown tool selection strategy=%s; using bm25", strategy)

        return self._select_with_bm25(
            unique_tool_names,
            filtered_entries=filtered_entries,
            always_include=always_include,
            limit=limit,
            command=command,
            task_specs=task_specs,
            preferred_tool_names=preferred_tool_names,
            use_vector=strategy == "hybrid",
        )

    def debug_select_tool_names(
        self,
        tool_names: Iterable[str],
        *,
        command: Any,
        task_specs: Optional[List[Any]] = None,
        runtime_allowlist: Optional[List[str]] = None,
        max_tools: Optional[int] = None,
        pool: str = "declared",
        preferred_tool_names: Optional[Iterable[str]] = None,
    ) -> ToolSelectionDebugResult:
        started = time.perf_counter()
        unique_tool_names = self._build_candidate_pool(
            tool_names,
            pool=pool,
            preferred_tool_names=preferred_tool_names,
        )
        if not unique_tool_names:
            return ToolSelectionDebugResult([], [], "bm25", "", "none", int((time.perf_counter() - started) * 1000))

        filtered_entries = self._apply_hard_filters(
            unique_tool_names,
            runtime_allowlist=runtime_allowlist,
        )
        if not filtered_entries:
            return ToolSelectionDebugResult([], [], "bm25", "", "none", int((time.perf_counter() - started) * 1000))

        limit = max(1, int(max_tools or get_tool_selection_max_tools()))
        always_include = self._collect_always_include(filtered_entries)
        strategy = get_tool_selection_strategy()
        use_vector = strategy == "hybrid"
        query_text = self._build_query_text(command, task_specs=task_specs)
        candidates, search_index, vector_used = self._score_candidates(
            unique_tool_names,
            filtered_entries=filtered_entries,
            always_include=always_include,
            command=command,
            task_specs=task_specs,
            preferred_tool_names=preferred_tool_names,
            use_vector=use_vector,
            query_text_override=query_text,
        )

        selected_names = list(always_include)
        for candidate in self._apply_score_gates(candidates):
            if len(selected_names) >= limit:
                break
            selected_names.append(candidate.name)

        if len(selected_names) == len(always_include):
            contextual_query = _build_contextual_selection_text(command, query_text)
            if contextual_query:
                fallback_candidates, search_index, fallback_vector_used = self._score_candidates(
                    unique_tool_names,
                    filtered_entries=filtered_entries,
                    always_include=always_include,
                    command=command,
                    task_specs=task_specs,
                    preferred_tool_names=preferred_tool_names,
                    use_vector=use_vector,
                    query_text_override=contextual_query,
                )
                for candidate in self._apply_score_gates(fallback_candidates):
                    if len(selected_names) >= limit:
                        break
                    selected_names.append(candidate.name)
                if len(selected_names) > len(always_include):
                    candidates = fallback_candidates
                    vector_used = fallback_vector_used

        selected_names = self._prefer_url_content_facade(
            selected_names,
            filtered_entries=filtered_entries,
            query_text=query_text,
            command=command,
            limit=limit,
        )
        selected_names = self._collapse_url_content_members(selected_names)

        return ToolSelectionDebugResult(
            selected_names=selected_names[:limit],
            candidates=sorted(candidates, key=lambda item: (-item.score, item.index, item.name)),
            strategy="hybrid" if use_vector and vector_used else "bm25",
            index_version=search_index.version,
            embedding_key=search_index.embedding_key,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _build_candidate_pool(
        self,
        tool_names: Iterable[str],
        *,
        pool: str,
        preferred_tool_names: Optional[Iterable[str]] = None,
    ) -> List[str]:
        pool_mode = str(pool or "declared").strip().lower() or "declared"
        if pool_mode != "global":
            return self._unique_names(tool_names)

        catalog_names = list(self._catalog.all().keys())
        declared_set = {str(n).strip() for n in tool_names if str(n).strip()}
        preferred_set = {str(n).strip() for n in (preferred_tool_names or []) if str(n).strip()}
        ordered: List[str] = []
        seen = set()
        for name in list(preferred_set) + list(declared_set) + catalog_names:
            normalized = str(name or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def _select_with_bm25(
        self,
        unique_tool_names: List[str],
        *,
        filtered_entries: Dict[str, ToolCatalogEntry],
        always_include: List[str],
        limit: int,
        command: Any,
        task_specs: Optional[List[Any]] = None,
        preferred_tool_names: Optional[Iterable[str]] = None,
        use_vector: bool = False,
    ) -> List[str]:
        started = time.perf_counter()
        query_text = self._build_query_text(command, task_specs=task_specs)
        candidates, search_index, vector_used = self._score_candidates(
            unique_tool_names,
            filtered_entries=filtered_entries,
            always_include=always_include,
            command=command,
            task_specs=task_specs,
            preferred_tool_names=preferred_tool_names,
            use_vector=use_vector,
            query_text_override=query_text,
        )

        selected_names = list(always_include)
        for candidate in self._apply_score_gates(candidates):
            if len(selected_names) >= limit:
                break
            selected_names.append(candidate.name)

        selection_query = query_text
        if len(selected_names) == len(always_include):
            contextual_query = _build_contextual_selection_text(command, query_text)
            if contextual_query:
                fallback_candidates, fallback_index, fallback_vector_used = self._score_candidates(
                    unique_tool_names,
                    filtered_entries=filtered_entries,
                    always_include=always_include,
                    command=command,
                    task_specs=task_specs,
                    preferred_tool_names=preferred_tool_names,
                    use_vector=use_vector,
                    query_text_override=contextual_query,
                )
                for candidate in self._apply_score_gates(fallback_candidates):
                    if len(selected_names) >= limit:
                        break
                    selected_names.append(candidate.name)
                if len(selected_names) > len(always_include):
                    selection_query = contextual_query
                    search_index = fallback_index
                    vector_used = fallback_vector_used

        selected_names = self._prefer_url_content_facade(
            selected_names,
            filtered_entries=filtered_entries,
            query_text=query_text,
            command=command,
            limit=limit,
        )
        selected_names = self._collapse_url_content_members(selected_names)

        message_preview = str(getattr(command, "message", "") or "")[:80]
        logger.info(
            "Selected agent tools: strategy=%s index=%s embedding=%s latency_ms=%d query=%s message_preview=%s selected=%s",
            "hybrid" if use_vector and vector_used else "bm25",
            search_index.version,
            search_index.embedding_key,
            int((time.perf_counter() - started) * 1000),
            selection_query[:120],
            message_preview,
            selected_names,
        )
        return selected_names[:limit]

    @staticmethod
    def _collapse_url_content_members(selected_names: List[str]) -> List[str]:
        """若 URL 内容 facade 已入选，则不再把同组底层工具暴露给 LLM。"""

        if PROCESS_URL_CONTENT_TOOL_NAME not in selected_names:
            return selected_names
        collapsed: List[str] = []
        seen = set()
        for name in selected_names:
            if name in URL_CONTENT_MEMBER_TOOLS:
                logger.debug(
                    "Dropped URL content member because %s is selected: %s",
                    PROCESS_URL_CONTENT_TOOL_NAME,
                    name,
                )
                continue
            if name in seen:
                continue
            seen.add(name)
            collapsed.append(name)
        return collapsed

    @staticmethod
    def _prefer_url_content_facade(
        selected_names: List[str],
        *,
        filtered_entries: Dict[str, ToolCatalogEntry],
        query_text: str,
        command: Any,
        limit: int,
    ) -> List[str]:
        """文档 URL / 临时文档跟进优先交给 URL 内容 facade，而不是 KB 或底层转换工具。"""

        if PROCESS_URL_CONTENT_TOOL_NAME not in filtered_entries:
            return selected_names
        contextual_query = ""
        if _has_knowledge_base_intent(query_text):
            return selected_names
        if PROCESS_URL_CONTENT_TOOL_NAME in selected_names:
            return selected_names

        should_prefer = _has_url_content_document(query_text)
        if not should_prefer and "knowledge_base_query" in selected_names and _mentions_transient_document(query_text):
            contextual_query = _build_contextual_selection_text(command, query_text)
            if _has_knowledge_base_intent(contextual_query):
                return selected_names
            should_prefer = _has_url_content_document(contextual_query) or _mentions_transient_document(contextual_query)

        if not should_prefer:
            return selected_names

        logger.debug(
            "Preferring %s for URL/transient document query; previous selected=%s",
            PROCESS_URL_CONTENT_TOOL_NAME,
            selected_names,
        )
        merged: List[str] = [PROCESS_URL_CONTENT_TOOL_NAME]
        for name in selected_names:
            if name == PROCESS_URL_CONTENT_TOOL_NAME:
                continue
            if name == "knowledge_base_query":
                continue
            if name not in merged:
                merged.append(name)
            if len(merged) >= limit:
                break
        return merged

    def _score_candidates(
        self,
        unique_tool_names: List[str],
        *,
        filtered_entries: Dict[str, ToolCatalogEntry],
        always_include: List[str],
        command: Any,
        task_specs: Optional[List[Any]] = None,
        preferred_tool_names: Optional[Iterable[str]] = None,
        use_vector: bool = False,
        query_text_override: str = "",
    ) -> tuple[List[ScoredCandidate], ToolSearchIndex, bool]:
        query_text = str(query_text_override or "").strip() or self._build_query_text(command, task_specs=task_specs)
        query_vector = _vectorize(query_text)

        search_index = _INDEX_MANAGER.get_index(self._catalog)
        allowed_names = {name for name in unique_tool_names if name in filtered_entries and name not in always_include}
        if not query_vector or not allowed_names:
            return [], search_index, False

        # 跨语言剥离：『指令 + 引出内容』句式（任何语言：用冒号或成对引号）下，
        # 真正决定工具的是『指令』部分，被引内容只是 payload。
        # 例如 "In a deep voice, he angrily said: 'Transcribe this audio'" 里
        # ASR 关键词（transcribe/audio）出现在 content 而非 intent，绝不能让它们
        # 反向把工具拉成 audio_asr。这是标点级别的硬信号，跨语言通用，不需要
        # 在 yaml/regex 里枚举各种语言的『说/say/言う/...』。
        intent_text, content_text = _split_query_intent_content(query_text)
        has_intent_split = bool(intent_text and content_text)
        search_text = intent_text if has_intent_split else query_text

        bm25_scores = search_index.bm25.search(
            search_text,
            allowed_names=allowed_names,
            top_k=get_tool_selection_bm25_top_k(),
        )
        max_bm25 = max(bm25_scores.values(), default=0.0)
        # 先无条件跑 vector，再用方向性判定决定是否纳入打分。
        #
        # 字面+语义双证据原则的本质：单凭向量给出高 cosine，可能只是 e5 的
        # "无意义语义模糊"（任何中文 query 跟『分析图片』cosine 也常 ≥ 0.6）。
        # 但**也可能是真正的方向性命中**（比如 query『画个猫』跟 image_generator
        # 的 cosine 显著高于其他工具——这是有效信号，硬丢就误伤）。
        # 区别在于"方向性"：闲聊 query 在 vector 上所有工具 cosine 都接近，
        # 真正的工具命中 query 会让 top1 跟 top2 raw cosine 拉开 ≥ 阈值的差距。
        vector_scores = self._search_vectors(
            search_index,
            query_text=search_text,
            allowed_names=allowed_names,
            top_k=get_tool_selection_vector_top_k(),
        ) if use_vector else {}
        # 方向性闸门：BM25 全 0、且非剥离场景下，要求 vector top1 跟 top2 raw
        # cosine 差距 ≥ ``_VECTOR_DIRECTIONAL_GAP``。否则视为"无方向性"，丢弃 vector
        # 防止闲聊 query（"你好啊"/"今晚月色真美"）凭空被语义模糊召回到工具上。
        # has_intent_split 场景天然有强意图信号（用户已经写了指令句式），跳过此闸门。
        if vector_scores and max_bm25 == 0 and not has_intent_split:
            sorted_raw = sorted(vector_scores.values(), reverse=True)
            if len(sorted_raw) < 2 or (sorted_raw[0] - sorted_raw[1]) < _VECTOR_DIRECTIONAL_GAP:
                vector_scores = {}
        vector_min = min(vector_scores.values(), default=0.0)
        vector_max = max(vector_scores.values(), default=0.0)
        vector_span = vector_max - vector_min
        preferred_set = {str(name).strip() for name in (preferred_tool_names or []) if str(name).strip()}
        # 负例匹配也走 intent（剥离后），避免被 content 带偏：
        # "用 X 说：把这段音频转为文字" 里 audio_tts 的负例 BM25 不应被
        # content 中的『转为文字』触发——它们只是被朗读的素材。
        negative_scores = search_index.negative_examples.scores(search_text)
        # query 模态识别仍用全 query：URL 通常出现在 content 里，但模态归属
        # 是『整段 query 携带了什么资源』，与 intent/content 切分无关。
        query_modalities = _detect_query_modalities(query_text)
        has_non_text_modality = bool(query_modalities - {"text"})

        # intent_anchor 召回：只在 query 经过指令/内容剥离时启用。
        # 这条信号专门用来在 short intent + 跨语言场景下做"工具本质意图"对齐
        # （比如『用英语说』『怒った声で言って』这种 4-12 字符的纯指令短语）。
        # 不剥离的 query 走原有 BM25/vector 路径，已稳定，不引入这条信号以免漂移。
        intent_anchor_scores: Dict[str, float] = {}
        if use_vector and has_intent_split:
            intent_anchor_scores = self._search_intent_anchors(
                search_index,
                query_text=search_text,
                allowed_names=allowed_names,
            )
        intent_anchor_max = max(intent_anchor_scores.values(), default=0.0)
        intent_anchor_min = min(intent_anchor_scores.values(), default=0.0)
        intent_anchor_span = intent_anchor_max - intent_anchor_min

        # 候选池：BM25 / vector top-k 的并集，再补上 exact-match。
        # 当 query 含非文本模态时，把所有"声明了 input_modalities 且与 query 不匹配"的工具
        # 也加入候选池，确保它们能被显式打上 modality mismatch 惩罚后参与排序，
        # 而不是因为 BM25/vector 都没召回就静默逃过减分。
        candidate_names = set(bm25_scores.keys()) | set(vector_scores.keys())
        # intent_anchor 命中的工具也进入候选池：保证 short intent 场景下 BM25=0
        # 但 anchor cosine 拉满的工具能浮上来。
        candidate_names.update(intent_anchor_scores.keys())
        for name in allowed_names:
            document = search_index.by_name.get(name)
            if not document:
                continue
            if _contains_exact_match(search_text, document.entry) > 0:
                candidate_names.add(name)
            if has_non_text_modality and document.entry.input_modalities:
                candidate_names.add(name)
        if not candidate_names:
            return [], search_index, bool(vector_scores)

        scored_candidates: List[ScoredCandidate] = []
        for name in candidate_names:
            document = search_index.by_name.get(name)
            entry = filtered_entries.get(name)
            if not document or not entry:
                continue
            bm25_norm = _safe_float_ratio(bm25_scores.get(name, 0.0), max_bm25)
            if name in vector_scores and vector_span >= 0.02:
                vector_norm = (vector_scores[name] - vector_min) / vector_span
            else:
                vector_norm = 0.0
            # intent_anchor 归一化：min-max 跨候选；span 不足时归 0，避免噪声放大。
            if name in intent_anchor_scores and intent_anchor_span >= 0.02:
                intent_anchor_norm = (intent_anchor_scores[name] - intent_anchor_min) / intent_anchor_span
            else:
                intent_anchor_norm = 0.0
            exact_match = _contains_exact_match(search_text, entry)
            preferred_boost = 1.0 if name in preferred_set else 0.0
            negative_score = negative_scores.get(name, 0.0)
            modality_match, modality_mismatch = _modality_scores(
                query_modalities, entry.input_modalities
            )
            if use_vector and vector_scores:
                # hybrid 打分。bm25/vector 是工具描述层的字面+语义召回；
                # intent_anchor 是工具的本质意图与 query.intent 的跨语 cosine。
                #
                # 剥离场景下（has_intent_split=True）vector 走的是 short intent ↔
                # 工具描述的跨语 cosine，e5-small 在 ≤ 6 字 query 上方向性较弱
                # （所有候选 cosine 跨度只有 0.03-0.05）；此时 anchor 路径相对更纯，
                # 把权重往 anchor 倾斜。非剥离 query 仍以 bm25/vector 为主。
                if has_intent_split:
                    score = (
                        0.30 * bm25_norm
                        + 0.15 * vector_norm
                        + 0.30 * intent_anchor_norm
                        + 0.10 * exact_match
                        + 0.05 * preferred_boost
                        + 0.15 * modality_match
                        - 0.20 * modality_mismatch
                        - 0.20 * negative_score
                    )
                else:
                    # hybrid：稀释 bm25/vector 各 0.05 给模态特征，让"输入资源类型"
                    # 也直接进入排序，否则向量在『ai生成的图片』这种 query 上会强行
                    # 把 image_generator 拉到第一。
                    score = (
                        0.40 * bm25_norm
                        + 0.30 * vector_norm
                        + 0.10 * exact_match
                        + 0.05 * preferred_boost
                        + 0.15 * modality_match
                        - 0.20 * modality_mismatch
                        - 0.20 * negative_score
                    )
            else:
                score = (
                    0.65 * bm25_norm
                    + 0.10 * exact_match
                    + 0.10 * preferred_boost
                    + 0.15 * modality_match
                    - 0.20 * modality_mismatch
                    - 0.20 * negative_score
                )
            scored_candidates.append(
                ScoredCandidate(
                    name=name,
                    score=max(0.0, score),
                    index=document.index,
                    bm25_score=bm25_norm,
                    vector_score=vector_norm,
                    exact_match_score=exact_match,
                    preferred_boost=preferred_boost,
                    negative_match_score=negative_score,
                    modality_match_score=modality_match,
                    modality_mismatch_score=modality_mismatch,
                    intent_anchor_score=intent_anchor_norm,
                )
            )

        return scored_candidates, search_index, bool(vector_scores)

    @staticmethod
    def _search_vectors(
        search_index: ToolSearchIndex,
        *,
        query_text: str,
        allowed_names: Set[str],
        top_k: int,
    ) -> Dict[str, float]:
        if np is None or search_index.vector_matrix is None or not search_index.vector_names:
            return {}
        backend = _EMBEDDING_BACKENDS.get_backend()
        if not backend.is_ready() or backend.cache_key != search_index.embedding_key:
            return {}
        cache_key = f"{backend.cache_key}:{query_text}"
        query_embedding = _QUERY_EMBEDDINGS.get(cache_key)
        if query_embedding is None:
            query_embedding = backend.embed_query(
                query_text,
                timeout_ms=get_tool_selection_embedding_timeout_ms(),
            )
            if query_embedding is None:
                return {}
            _QUERY_EMBEDDINGS.set(cache_key, query_embedding)

        scores = np.asarray(search_index.vector_matrix @ query_embedding, dtype=np.float32)
        ranked_indices = np.argsort(-scores)
        results: Dict[str, float] = {}
        for raw_index in ranked_indices:
            name = search_index.vector_names[int(raw_index)]
            if name not in allowed_names:
                continue
            results[name] = float(scores[int(raw_index)])
            if len(results) >= max(1, top_k):
                break
        return results

    @staticmethod
    def _search_intent_anchors(
        search_index: ToolSearchIndex,
        *,
        query_text: str,
        allowed_names: Set[str],
    ) -> Dict[str, float]:
        """用 query 的 embedding × intent_anchor_matrix 得到每个工具的意图 cosine。

        ``intent_anchor_owners[i]`` 标记第 i 行 anchor 归属的工具，scoring
        对同一工具的多条 anchor 取 max cosine——意图的多个表述角度只要任一
        命中即可拉高该工具的 intent_score。

        返回：``{tool_name: max_anchor_cosine}``，由调用方决定是否归一化。
        """
        if np is None or search_index.intent_anchor_matrix is None or not search_index.intent_anchor_owners:
            return {}
        backend = _EMBEDDING_BACKENDS.get_backend()
        if not backend.is_ready() or backend.cache_key != search_index.embedding_key:
            return {}
        cache_key = f"{backend.cache_key}:{query_text}"
        query_embedding = _QUERY_EMBEDDINGS.get(cache_key)
        if query_embedding is None:
            query_embedding = backend.embed_query(
                query_text,
                timeout_ms=get_tool_selection_embedding_timeout_ms(),
            )
            if query_embedding is None:
                return {}
            _QUERY_EMBEDDINGS.set(cache_key, query_embedding)

        scores = np.asarray(search_index.intent_anchor_matrix @ query_embedding, dtype=np.float32)
        results: Dict[str, float] = {}
        for raw_index, owner in enumerate(search_index.intent_anchor_owners):
            if owner not in allowed_names:
                continue
            cosine = float(scores[int(raw_index)])
            if cosine > results.get(owner, float("-inf")):
                results[owner] = cosine
        return results

    @staticmethod
    def _apply_score_gates(candidates: List[ScoredCandidate]) -> List[ScoredCandidate]:
        if not candidates:
            return []
        ranked = sorted(candidates, key=lambda item: (-item.score, item.index, item.name))
        min_score = get_tool_selection_min_score()
        min_ratio = get_tool_selection_min_ratio_of_top()
        top_score = ranked[0].score
        threshold = max(min_score, top_score * min_ratio)

        filtered = [candidate for candidate in ranked if candidate.score >= threshold]
        if len(filtered) < 2:
            return filtered

        first = filtered[0]
        # 高置信精确命中：直接踢掉所有与 top1 互斥的工具，避免把 image_generator
        # 这种正例命中后又把 image_editor 一并塞给 LLM。
        if first.score >= 0.75 and first.exact_match_score > 0:
            return [first] + [candidate for candidate in filtered[1:] if not _is_mutex_pair(first.name, candidate.name)]

        # 主动剔除"工具自己用 negative_examples 强烈拒绝该 query"的互斥候选。
        # 例如 query="把这段录音转成文字" 时 audio_tts 的 negative ≈ 1.0，
        # 即便它的 BM25/向量分还在阈值内，也不应作为伴随工具暴露给 LLM。
        # 这条规则只动『与 top1 互斥』的候选，避免误伤其他正常工具。
        survivors: List[ScoredCandidate] = [first]
        for candidate in filtered[1:]:
            if (
                _is_mutex_pair(first.name, candidate.name)
                and candidate.negative_match_score >= 0.5
            ):
                logger.debug(
                    "Dropped mutex tool candidate with strong self-declared negative: "
                    "kept=%s dropped=%s scores=(%.4f, %.4f) negative=%.4f",
                    first.name,
                    candidate.name,
                    first.score,
                    candidate.score,
                    candidate.negative_match_score,
                )
                continue
            survivors.append(candidate)

        if len(survivors) < 2:
            return survivors

        # 兼容老规则：top1/top2 分差极小且互斥，仅保留 top1。
        if (
            _is_mutex_pair(survivors[0].name, survivors[1].name)
            and abs(survivors[0].score - survivors[1].score) <= 0.05
        ):
            logger.debug(
                "Dropped mutex tool candidate due to narrow score gap: kept=%s dropped=%s scores=(%.4f, %.4f)",
                survivors[0].name,
                survivors[1].name,
                survivors[0].score,
                survivors[1].score,
            )
            return [survivors[0]] + survivors[2:]
        return survivors

    @staticmethod
    def _unique_names(tool_names: Iterable[str]) -> List[str]:
        results: List[str] = []
        seen = set()
        for raw_name in tool_names:
            normalized = str(raw_name or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
        return results

    def _apply_hard_filters(
        self,
        tool_names: List[str],
        *,
        runtime_allowlist: Optional[List[str]] = None,
    ) -> Dict[str, ToolCatalogEntry]:
        results: Dict[str, ToolCatalogEntry] = {}
        runtime_list = {item for item in _normalize_list(runtime_allowlist)}
        openclaw_enabled = is_openclaw_enabled()

        for name in tool_names:
            entry = self._catalog.get(name) or ToolCatalogEntry(name=name)
            if not entry.enabled:
                continue
            if (entry.provider == "openclaw" or entry.requires_openclaw) and not openclaw_enabled:
                continue
            if runtime_list and "*" not in runtime_list and (entry.provider == "openclaw" or entry.requires_openclaw):
                allowed_names = {name, entry.runtime_tool_name}
                if not allowed_names & runtime_list:
                    continue
            results[name] = entry
        return results

    def _collect_always_include(self, entries: Dict[str, ToolCatalogEntry]) -> List[str]:
        configured = set(get_tool_selection_always_include())
        selected: List[str] = []
        for name, entry in entries.items():
            if entry.always_include or name in configured:
                selected.append(name)
        return selected

    @staticmethod
    def _build_query_text(command: Any, *, task_specs: Optional[List[Any]] = None) -> str:
        del task_specs
        return _extract_primary_user_text(command)
