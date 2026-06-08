from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, get_args

import yaml

from backend.core.logger import get_app_logger

from .crew_tools.registry import get_crew_tool_registry
from .settings import get_tool_catalog_path
from .tools.registry import get_tool_plugin_registry

logger = get_app_logger(__name__)


# 所有合法的 tool provider。新增 provider 时在这里加一项，其余校验自动生效。
# 当前四类：
#   - local    ：backend/services/agent/tools/builtin 下 @register_tool 的原子工具
#   - crew     ：Crew-as-Tool，把一个子 Crew 打包成可调工具（详见 crew_tools/）
#   - openclaw ：通过 OpenClaw HTTP 桥接调用的外部浏览器/会话工具
#   - mcp      ：占位，未来接入 Model Context Protocol 外部工具服务
ToolProviderName = Literal["local", "crew", "openclaw", "mcp"]
TOOL_PROVIDERS: tuple = get_args(ToolProviderName)


def _normalize_provider(raw_value: Any, *, source: str) -> ToolProviderName:
    """把任意输入归一为合法 ``ToolProviderName``；非法值直接抛错，不做静默降级。

    ``source`` 用于报错信息定位（例如 ``tool=tavily_search (yaml)``）。
    历史上此处会 silently 回退成 ``"local"``，这会让 YAML 拼错的 ``providr:``
    退化成本地工具并在运行期才炸——这里改为启动期即失败。
    """
    normalized = str(raw_value or "").strip().lower()
    if not normalized:
        return "local"
    if normalized in TOOL_PROVIDERS:
        return normalized  # type: ignore[return-value]
    raise ValueError(
        f"Unknown tool provider '{raw_value}' at {source}. "
        f"Allowed values: {list(TOOL_PROVIDERS)}"
    )


# 工具声明的合法输入模态。未在此集合内的取值会让 YAML 加载期失败，
# 避免拼写错误的模态被静默忽略后，运行期才出现莫名其妙的排名漂移。
ALLOWED_INPUT_MODALITIES: tuple = ("text", "image", "video", "audio", "document")


@dataclass(frozen=True)
class ToolCatalogEntry:
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    # 工具的"意图锚点"：≤ 几条紧凑的本质意图描述，**仅用于内部召回的语义对齐**
    # （把 query.intent 跟所有 anchor 算 cosine），**不进入** LLM 看到的 system prompt
    # 也不参与 BM25 文本召回（避免污染字面命中）。
    # 用途：当 query 极短（如『用英语说』『怒った声で言って』），靠工具描述做
    # 跨语 cosine 信号太弱，多角度紧凑 anchor 跟 e5 的多语对齐能力配合更稳。
    # 设计准则：每条 ≤ 30 字符；只描"做什么"，不描"什么时候用"；尽量含核心动词
    # （生成/编辑/朗读/转写/搜索…），让多种语言的 query 都能对齐。
    # YAML 里既支持单字符串也支持 list[str]——前者会被归一为单元素列表。
    intent_anchors: List[str] = field(default_factory=list)
    # 工具实际接受的输入模态。空列表 = 未声明，工具检索时不参与模态加/减分。
    # 已声明的工具，用户 query 携带的非文本模态（image/video/audio/document URL）
    # 会进入 fusion：命中 → +modality_match；完全无交集 → -modality_mismatch。
    input_modalities: List[str] = field(default_factory=list)
    provider: ToolProviderName = "local"
    enabled: bool = True
    always_include: bool = False
    requires_openclaw: bool = False
    target_tool_name: Optional[str] = None

    @property
    def runtime_tool_name(self) -> str:
        return str(self.target_tool_name or self.name).strip()

    @property
    def searchable_text(self) -> str:
        parts = [
            self.name,
            self.description,
            self.provider,
            " ".join(self.tags),
            " ".join(self.aliases),
            " ".join(self.examples),
        ]
        if self.target_tool_name:
            parts.append(self.target_tool_name)
        return " ".join(part for part in parts if part).strip()

    @property
    def negative_searchable_text(self) -> str:
        return " ".join(self.negative_examples).strip()


def _normalize_string_list(raw_value: Any) -> List[str]:
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return [str(raw_value).strip()] if str(raw_value).strip() else []


def _normalize_tags(raw_value: Any) -> List[str]:
    return _normalize_string_list(raw_value)


def _normalize_input_modalities(raw_value: Any, *, source: str) -> List[str]:
    """归一化 input_modalities 字段。

    - 缺省/空 → 返回空列表，表示『未声明，模态打分不生效』。
    - 单字符串/列表 → 拆分、去重、转小写、过滤空值。
    - 任一取值不在 ``ALLOWED_INPUT_MODALITIES`` → 启动期硬失败，避免拼写漂移。

    ``source`` 用于把错误信息定位到具体工具，方便排错。
    """
    raw_items = _normalize_string_list(raw_value)
    if not raw_items:
        return []
    seen: List[str] = []
    for item in raw_items:
        normalized = item.lower()
        if normalized not in ALLOWED_INPUT_MODALITIES:
            raise ValueError(
                f"Unknown input modality '{item}' at {source}. "
                f"Allowed values: {list(ALLOWED_INPUT_MODALITIES)}"
            )
        if normalized not in seen:
            seen.append(normalized)
    return seen


def _normalize_bool(raw_value: Any, default: bool) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(raw_value)


def _entry_from_mapping(item: Dict[str, Any]) -> Optional[ToolCatalogEntry]:
    name = str(item.get("name") or "").strip()
    if not name:
        return None
    return ToolCatalogEntry(
        name=name,
        description=str(item.get("description") or "").strip(),
        tags=_normalize_tags(item.get("tags")),
        aliases=_normalize_string_list(item.get("aliases")),
        examples=_normalize_string_list(item.get("examples")),
        negative_examples=_normalize_string_list(item.get("negative_examples")),
        # ``intent_anchor`` (单字符串) / ``intent_anchors`` (列表) 都支持；
        # 前者作为后者的便捷写法存在，运行时统一为 ``intent_anchors`` 列表。
        intent_anchors=_normalize_string_list(
            item.get("intent_anchors") if item.get("intent_anchors") is not None else item.get("intent_anchor")
        ),
        input_modalities=_normalize_input_modalities(
            item.get("input_modalities"),
            source=f"tool={name} (yaml)",
        ),
        provider=_normalize_provider(item.get("provider"), source=f"tool={name} (yaml)"),
        enabled=_normalize_bool(item.get("enabled"), True),
        always_include=_normalize_bool(item.get("always_include"), False),
        requires_openclaw=_normalize_bool(item.get("requires_openclaw"), False),
        target_tool_name=str(item.get("target_tool_name") or item.get("target") or "").strip() or None,
    )


def _default_entries() -> Dict[str, ToolCatalogEntry]:
    """返回由装饰器注册合并出的默认目录。

    数据源优先级（后者覆盖前者）：
      1. `@register_tool` 装饰器注册的本地/openclaw 工具（`tools/registry.py`）
      2. `@register_crew_tool` 注册的 Crew-as-Tool（`crew_tools/registry.py`）
      3. `config/agent_tools.yaml`（由 `ToolCatalog._load_entries` 加载，覆盖默认值）

    不再维护第二份硬编码清单（原 `EXPLICIT_OPENCLAW_TOOLS` 已移除），
    openclaw 工具完全通过 YAML 声明；新增工具时只改 YAML / 装饰器一处。
    """
    entries: Dict[str, ToolCatalogEntry] = {}

    for registration in get_tool_plugin_registry().all_registrations().values():
        meta = registration.metadata
        entries[meta.name] = ToolCatalogEntry(
            name=meta.name,
            description=meta.description,
            tags=list(meta.tags),
            aliases=[],
            examples=[],
            negative_examples=[],
            provider=_normalize_provider(meta.provider, source=f"tool={meta.name} (@register_tool)"),
            enabled=meta.enabled,
            always_include=meta.always_include,
            requires_openclaw=meta.requires_openclaw,
            target_tool_name=meta.target_tool_name,
        )

    for registration in get_crew_tool_registry().all_registrations().values():
        meta = registration.metadata
        if not meta.enabled:
            continue
        if meta.name in entries:
            continue
        entries[meta.name] = ToolCatalogEntry(
            name=meta.name,
            description=meta.description,
            tags=list(meta.tags) + ["crew", "multi-agent"],
            aliases=[],
            examples=[],
            negative_examples=[],
            provider="crew",
            enabled=meta.enabled,
            always_include=meta.always_include,
            requires_openclaw=False,
            target_tool_name=meta.preset_id,
        )
    return entries


class ToolCatalog:
    """从 YAML 读取工具目录元数据，并提供按名称查询能力。"""

    def __init__(self):
        self._entries: Optional[Dict[str, ToolCatalogEntry]] = None

    def _load_entries(self) -> Dict[str, ToolCatalogEntry]:
        entries = _default_entries()
        catalog_path = get_tool_catalog_path()
        if not catalog_path.exists():
            logger.info(f"Agent tool catalog not found, using built-in defaults: {catalog_path}")
            return entries

        try:
            with open(catalog_path, "r", encoding="utf-8") as handle:
                raw_data = yaml.safe_load(handle) or {}
        except Exception as exc:
            logger.warning(f"Failed to load agent tool catalog {catalog_path}: {exc}")
            return entries

        raw_tools = raw_data.get("tools")
        if not isinstance(raw_tools, list):
            return entries

        for item in raw_tools:
            if not isinstance(item, dict):
                continue
            entry = _entry_from_mapping(item)
            if entry:
                entries[entry.name] = entry
        return entries

    def all(self) -> Dict[str, ToolCatalogEntry]:
        if self._entries is None:
            self._entries = self._load_entries()
        return self._entries

    def reload(self) -> Dict[str, ToolCatalogEntry]:
        self._entries = self._load_entries()
        return self._entries

    def get(self, name: str) -> Optional[ToolCatalogEntry]:
        normalized = str(name or "").strip()
        if not normalized:
            return None
        return self.all().get(normalized)

    def list_by_names(self, names: Iterable[str]) -> List[ToolCatalogEntry]:
        results: List[ToolCatalogEntry] = []
        seen = set()
        for raw_name in names:
            normalized = str(raw_name or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            entry = self.get(normalized)
            if entry:
                results.append(entry)
        return results
