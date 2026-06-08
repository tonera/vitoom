from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from backend.core.config import get_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _coerce_list(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def get_agent_secret(name: str, default: str = "") -> str:
    normalized_name = str(name or "").strip()
    if not normalized_name:
        return str(default or "").strip()
    env_value = str(os.getenv(normalized_name) or "").strip()
    if env_value:
        return env_value
    return str(default or "").strip()


def _get_openclaw_config(key: str, legacy_key: str, default: Any = None) -> Any:
    nested_value = get_config(f"agents.openclaw.{key}", None)
    if nested_value not in (None, ""):
        return nested_value
    return get_config(f"agents.{legacy_key}", default)


def is_agents_enabled() -> bool:
    return _coerce_bool(get_config("agents.enabled", True), True)


def is_agent_recovery_enabled() -> bool:
    return _coerce_bool(get_config("agents.recover_pending_tasks", False), False)


def is_openclaw_enabled() -> bool:
    explicit = get_config("agents.openclaw.enabled", None)
    if explicit is None:
        explicit = get_config("agents.openclaw_enabled", None)
    return _coerce_bool(explicit, False)


def get_openclaw_base_url() -> str:
    return str(_get_openclaw_config("base_url", "openclaw_base_url", "http://127.0.0.1:18789") or "").strip()


def get_openclaw_token() -> str:
    return str(_get_openclaw_config("token", "openclaw_token", "") or "").strip()


def get_tavily_api_key() -> str:
    return get_agent_secret("TAVILY_API_KEY", "")


def get_openclaw_timeout_seconds() -> float:
    raw_value = _get_openclaw_config("timeout_seconds", "openclaw_timeout_seconds", 30)
    try:
        return float(raw_value or 30)
    except (TypeError, ValueError):
        return 30.0


def get_openclaw_allowed_tools() -> List[str]:
    return _coerce_list(_get_openclaw_config("allowed_tools", "openclaw_allowed_tools", []))


def get_default_preset_agent_id() -> str:
    configured = str(get_config("agents.default_preset_agent_id", "preset-local-agent") or "").strip()
    return configured or "preset-local-agent"


def get_openclaw_preset_agent_id() -> str:
    configured = str(get_config("agents.openclaw.default_preset_agent_id", "preset-openclaw-agent") or "").strip()
    return configured or "preset-openclaw-agent"


def get_master_preset_agent_id() -> str:
    """统一聊天入口使用的 Master Agent ID。"""
    configured = str(get_config("agents.master_preset_agent_id", "preset-master-agent") or "").strip()
    return configured or "preset-master-agent"


def is_tool_selection_enabled() -> bool:
    return _coerce_bool(get_config("agents.tool_selection.enabled", True), True)


def get_tool_selection_max_tools() -> int:
    raw_value = get_config("agents.tool_selection.max_tools_per_run", 8)
    try:
        return max(1, int(raw_value or 8))
    except (TypeError, ValueError):
        return 8


def get_tool_selection_always_include() -> List[str]:
    return _coerce_list(get_config("agents.tool_selection.always_include", []))


def get_tool_selection_min_score() -> float:
    """绝对地板：工具相似度低于此值一律丢弃（默认 0.05）。"""
    raw = get_config("agents.tool_selection.min_score", 0.05)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.05


def get_tool_selection_min_ratio_of_top() -> float:
    """相对阈值：工具相似度必须不低于头部最高分的此比例，默认 0.35。

    这是为了解决 bag-of-ngrams cosine 在短 query 下头尾分数差一个数量级，
    但尾部依然 > 0 的情形，避免把毫无相关性的工具一并交给 LLM。
    """
    raw = get_config("agents.tool_selection.min_ratio_of_top", 0.35)
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.35


def get_tool_selection_strategy() -> str:
    configured = str(get_config("agents.tool_selection.strategy", "bm25") or "").strip().lower()
    return configured or "bm25"


def get_tool_selection_bm25_top_k() -> int:
    raw = get_config("agents.tool_selection.bm25_top_k", 100)
    try:
        return max(1, int(raw or 100))
    except (TypeError, ValueError):
        return 100


def get_tool_selection_vector_top_k() -> int:
    raw = get_config("agents.tool_selection.vector_top_k", 50)
    try:
        return max(1, int(raw or 50))
    except (TypeError, ValueError):
        return 50


def is_tool_selection_embedding_enabled() -> bool:
    return _coerce_bool(get_config("agents.tool_selection.embedding_enabled", False), False)


def get_tool_selection_embedding_backend() -> str:
    configured = str(get_config("agents.tool_selection.embedding_backend", "none") or "").strip().lower()
    return configured or "none"


def get_tool_selection_embedding_model_path() -> str:
    return str(get_config("agents.tool_selection.embedding_model_path", "") or "").strip()


def get_tool_selection_embedding_timeout_ms() -> int:
    raw = get_config("agents.tool_selection.embedding_timeout_ms", 60)
    try:
        return max(1, int(raw or 60))
    except (TypeError, ValueError):
        return 60


def get_tool_selection_query_cache_size() -> int:
    raw = get_config("agents.tool_selection.query_cache_size", 512)
    try:
        return max(0, int(raw or 512))
    except (TypeError, ValueError):
        return 512


def is_tool_selection_fallback_to_bm25_enabled() -> bool:
    return _coerce_bool(get_config("agents.tool_selection.fallback_to_bm25", True), True)


def get_tool_selection_rebuild_check_interval_seconds() -> float:
    raw = get_config("agents.tool_selection.rebuild_check_interval_seconds", 5)
    try:
        return max(0.0, float(raw or 5))
    except (TypeError, ValueError):
        return 5.0


def get_tool_catalog_path() -> Path:
    configured = str(get_config("agents.tool_selection.catalog_path", "config/agent_tools.yaml") or "").strip()
    path = Path(configured or "config/agent_tools.yaml")
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def is_knowledge_base_enabled() -> bool:
    return _coerce_bool(get_config("knowledge_base.enabled", True), True)


def get_knowledge_base_es_url() -> str:
    return str(get_config("knowledge_base.es.url", "http://127.0.0.1:9200") or "").strip()


def get_knowledge_base_es_username() -> str:
    return str(get_config("knowledge_base.es.username", "") or "").strip()


def get_knowledge_base_es_password() -> str:
    configured = str(get_config("knowledge_base.es.password", "") or "").strip()
    if configured:
        return configured
    return get_agent_secret("KNOWLEDGE_BASE_ES_PASSWORD", "")


def get_knowledge_base_document_index() -> str:
    return str(get_config("knowledge_base.es.document_index", "kb_document_v1") or "").strip() or "kb_document_v1"


def get_knowledge_base_chunk_index() -> str:
    return str(get_config("knowledge_base.es.chunk_index", "kb_chunk_v1") or "").strip() or "kb_chunk_v1"


def get_knowledge_base_request_timeout_seconds() -> float:
    raw = get_config("knowledge_base.es.request_timeout_seconds", 30)
    try:
        value = float(raw or 30)
    except (TypeError, ValueError):
        return 30.0
    return value if value > 0 else 30.0


def get_knowledge_base_source_scan_roots() -> List[str]:
    return _coerce_list(get_config("knowledge_base.source_organizer.scan_roots", []))


def get_knowledge_base_canonical_root() -> Path:
    configured = str(get_config("knowledge_base.source_organizer.canonical_root", "resources/knowledge_sources") or "").strip()
    path = Path(configured or "resources/knowledge_sources")
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def get_knowledge_base_manifest_path() -> Path:
    configured = str(get_config("knowledge_base.source_organizer.manifest_path", "resources/knowledge_sources/manifest.jsonl") or "").strip()
    path = Path(configured or "resources/knowledge_sources/manifest.jsonl")
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def get_knowledge_base_scan_state_path() -> Path:
    configured = str(get_config("knowledge_base.source_organizer.scan_state_path", "resources/knowledge_sources/scan_state.json") or "").strip()
    path = Path(configured or "resources/knowledge_sources/scan_state.json")
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def get_knowledge_base_markdown_root() -> Path:
    configured = str(get_config("knowledge_base.derived.markdown_root", "resources/knowledge_sources_derived/markdown") or "").strip()
    path = Path(configured or "resources/knowledge_sources_derived/markdown")
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def get_knowledge_base_parse_meta_root() -> Path:
    configured = str(get_config("knowledge_base.derived.parse_meta_root", "resources/knowledge_sources_derived/parse_meta") or "").strip()
    path = Path(configured or "resources/knowledge_sources_derived/parse_meta")
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def is_knowledge_base_embedding_enabled() -> bool:
    return _coerce_bool(get_config("knowledge_base.embedding.enabled", True), True)


def get_knowledge_base_embedding_backend() -> str:
    configured = str(get_config("knowledge_base.embedding.backend", "onnx") or "").strip().lower()
    return configured or "onnx"


def get_knowledge_base_embedding_model_path() -> str:
    return str(get_config("knowledge_base.embedding.model_path", "resources/models/multilingual-e5-small-onnx") or "").strip()


def get_knowledge_base_embedding_dims() -> int:
    raw = get_config("knowledge_base.embedding.dims", 384)
    try:
        return max(1, int(raw or 384))
    except (TypeError, ValueError):
        return 384


def get_knowledge_base_embedding_timeout_ms() -> int:
    raw = get_config("knowledge_base.embedding.timeout_ms", 300)
    try:
        return max(1, int(raw or 300))
    except (TypeError, ValueError):
        return 300


def get_knowledge_base_embedding_batch_size() -> int:
    raw = get_config("knowledge_base.embedding.batch_size", 32)
    try:
        return max(1, int(raw or 32))
    except (TypeError, ValueError):
        return 32


def get_knowledge_base_query_cache_size() -> int:
    raw = get_config("knowledge_base.embedding.query_cache_size", 1024)
    try:
        return max(0, int(raw or 1024))
    except (TypeError, ValueError):
        return 1024


def get_knowledge_base_retrieval_bm25_top_k() -> int:
    raw = get_config("knowledge_base.retrieval.bm25_top_k", 80)
    try:
        return max(1, int(raw or 80))
    except (TypeError, ValueError):
        return 80


def get_knowledge_base_retrieval_vector_top_k() -> int:
    raw = get_config("knowledge_base.retrieval.vector_top_k", 50)
    try:
        return max(0, int(raw or 50))
    except (TypeError, ValueError):
        return 50


def get_knowledge_base_retrieval_vector_num_candidates() -> int:
    raw = get_config("knowledge_base.retrieval.vector_num_candidates", 200)
    try:
        return max(1, int(raw or 200))
    except (TypeError, ValueError):
        return 200


def get_knowledge_base_retrieval_rrf_k() -> int:
    raw = get_config("knowledge_base.retrieval.rrf_k", 60)
    try:
        return max(1, int(raw or 60))
    except (TypeError, ValueError):
        return 60


def get_knowledge_base_retrieval_merged_top_k() -> int:
    raw = get_config("knowledge_base.retrieval.merged_top_k", 30)
    try:
        return max(1, int(raw or 30))
    except (TypeError, ValueError):
        return 30


def is_knowledge_base_rerank_enabled() -> bool:
    return _coerce_bool(get_config("knowledge_base.rerank.enabled", True), True)


def get_knowledge_base_rerank_backend() -> str:
    configured = str(get_config("knowledge_base.rerank.backend", "rule") or "").strip().lower()
    return configured or "rule"


def get_knowledge_base_rerank_top_n() -> int:
    raw = get_config("knowledge_base.rerank.top_n", 30)
    try:
        return max(1, int(raw or 30))
    except (TypeError, ValueError):
        return 30


def get_knowledge_base_rerank_final_top_k() -> int:
    raw = get_config("knowledge_base.rerank.final_top_k", 8)
    try:
        return max(1, int(raw or 8))
    except (TypeError, ValueError):
        return 8


def get_knowledge_base_answer_max_context_chars() -> int:
    raw = get_config("knowledge_base.answer.max_context_chars", 12000)
    try:
        return max(1000, int(raw or 12000))
    except (TypeError, ValueError):
        return 12000


def is_knowledge_base_answer_sources_enabled() -> bool:
    return _coerce_bool(get_config("knowledge_base.answer.include_sources", True), True)


def is_knowledge_base_answer_debug_enabled() -> bool:
    return _coerce_bool(get_config("knowledge_base.answer.include_debug", False), False)


def get_hr_business_query_es_url() -> str:
    return str(get_config("agents.business_queries.hr.es_url", "http://127.0.0.1:9200") or "").strip()


def get_hr_business_query_es_username() -> str:
    return str(get_config("agents.business_queries.hr.es_username", "") or "").strip()


def get_hr_business_query_es_password() -> str:
    configured = str(get_config("agents.business_queries.hr.es_password", "") or "").strip()
    if configured:
        return configured
    return get_agent_secret("HR_BUSINESS_QUERY_ES_PASSWORD", "")


def get_hr_business_query_employee_index() -> str:
    return str(get_config("agents.business_queries.hr.employee_index", "hr_employee_v1") or "").strip()


def get_hr_business_query_resume_chunk_index() -> str:
    return str(get_config("agents.business_queries.hr.resume_chunk_index", "hr_resume_chunk_v1") or "").strip()


def get_hr_business_query_resume_asset_index() -> str:
    return str(get_config("agents.business_queries.hr.resume_asset_index", "hr_resume_asset_v1") or "").strip()


def is_hr_business_query_vector_enabled() -> bool:
    return _coerce_bool(get_config("agents.business_queries.hr.vector_enabled", False), False)


def get_hr_business_query_request_timeout_seconds() -> float:
    raw = get_config("agents.business_queries.hr.request_timeout_seconds", 30)
    try:
        value = float(raw or 30)
    except (TypeError, ValueError):
        return 30.0
    return value if value > 0 else 30.0


def get_hr_business_query_planner_timeout_seconds() -> float:
    raw = get_config("agents.business_queries.hr.planner_timeout_seconds", get_agent_llm_timeout_seconds())
    try:
        value = float(raw or get_agent_llm_timeout_seconds())
    except (TypeError, ValueError):
        return get_agent_llm_timeout_seconds()
    return value if value > 0 else get_agent_llm_timeout_seconds()


def get_hr_business_query_max_limit() -> int:
    raw = get_config("agents.business_queries.hr.max_limit", 100)
    try:
        value = int(raw or 100)
    except (TypeError, ValueError):
        return 100
    return max(1, min(value, 500))


def get_hr_business_query_max_aggregation_buckets() -> int:
    raw = get_config("agents.business_queries.hr.max_aggregation_buckets", 100)
    try:
        value = int(raw or 100)
    except (TypeError, ValueError):
        return 100
    return max(1, min(value, 500))


def get_hr_business_query_dsl_max_limit() -> int:
    raw = get_config("agents.business_queries.hr.dsl_max_limit", get_hr_business_query_max_limit())
    try:
        value = int(raw or get_hr_business_query_max_limit())
    except (TypeError, ValueError):
        return get_hr_business_query_max_limit()
    return max(1, min(value, 500))


def get_hr_business_query_dsl_max_aggregation_buckets() -> int:
    raw = get_config("agents.business_queries.hr.dsl_max_aggregation_buckets", get_hr_business_query_max_aggregation_buckets())
    try:
        value = int(raw or get_hr_business_query_max_aggregation_buckets())
    except (TypeError, ValueError):
        return get_hr_business_query_max_aggregation_buckets()
    return max(1, min(value, 500))


def get_agents_presets_dir() -> Path:
    configured = str(get_config("agents.presets.directory", "config/agents/presets") or "").strip()
    path = Path(configured or "config/agents/presets")
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _normalize_loopback_host(host: str) -> str:
    normalized = str(host or "").strip()
    if normalized in {"", "0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return normalized


def get_agent_llm_model_name() -> str:
    configured = str(get_config("agents.llm.model", "") or "").strip()
    if configured:
        return configured
    return str(get_config("agents.default_model", "") or "").strip()


def get_agent_llm_base_url() -> str:
    configured = str(get_config("agents.llm.base_url", "") or "").strip().rstrip("/")
    if configured:
        return configured
    host = _normalize_loopback_host(str(get_config("server.host", "127.0.0.1") or "127.0.0.1"))
    port = int(get_config("server.port", 8888) or 8888)
    return f"http://{host}:{port}/v1"


def get_agent_llm_timeout_seconds() -> float:
    raw_value = get_config("agents.llm.timeout_seconds", 120)
    try:
        return float(raw_value or 120)
    except (TypeError, ValueError):
        return 120.0


def get_agent_internal_auth_token() -> str:
    configured = str(get_config("agents.llm.internal_auth_token", "vitoom-agent-internal") or "").strip()
    return configured or "vitoom-agent-internal"


def get_agent_internal_user_id() -> str:
    configured = str(get_config("agents.llm.internal_user_id", "agent-system") or "").strip()
    return configured or "agent-system"


def get_agent_effective_user_header_name() -> str:
    configured = str(get_config("agents.llm.effective_user_header", "X-Vitoom-Effective-User-Id") or "").strip()
    return configured or "X-Vitoom-Effective-User-Id"


def get_image_generator_default_timeout() -> float:
    """image_generator 工具等待推理器完成的默认超时秒数。"""
    raw_value = get_config("agents.tools.image_generator.default_timeout_seconds", 200)
    try:
        value = float(raw_value or 200)
    except (TypeError, ValueError):
        value = 200.0
    return value if value > 0 else 200.0


def get_video_generator_default_timeout() -> float:
    """video_generator 工具等待推理器完成的默认超时秒数。"""
    raw_value = get_config("agents.tools.video_generator.default_timeout_seconds", 600)
    try:
        value = float(raw_value or 600)
    except (TypeError, ValueError):
        value = 600.0
    return value if value > 0 else 600.0


def get_image_generator_default_model_name() -> str:
    """image_generator 工具未显式指定 model_name 时使用的默认模型名。"""
    return str(get_config(
        "agents.tools.image_generator.default_model_name", ""
    ) or "").strip()


def get_image_generator_default_num_inference_steps() -> int:
    """image_generator 工具未显式指定采样步数时使用的默认 num_inference_steps。"""
    raw_value = get_config("agents.tools.image_generator.default_num_inference_steps", 30)
    try:
        value = int(raw_value or 30)
    except (TypeError, ValueError):
        value = 30
    return max(1, min(100, value))


def get_image_generator_default_guidance_scale() -> float:
    """image_generator 工具未显式指定 CFG 时使用的默认 guidance_scale。"""
    raw_value = get_config("agents.tools.image_generator.default_guidance_scale", 7.5)
    try:
        value = float(raw_value if raw_value is not None else 7.5)
    except (TypeError, ValueError):
        value = 7.5
    return max(0.0, min(20.0, value))


def get_image_editor_default_timeout() -> float:
    """image_editor 工具等待推理器完成的默认超时秒数。"""
    raw_value = get_config("agents.tools.image_editor.default_timeout_seconds", 240)
    try:
        value = float(raw_value or 240)
    except (TypeError, ValueError):
        value = 240.0
    return value if value > 0 else 240.0


def get_image_editor_default_model_name() -> str:
    """image_editor 工具未显式指定 model_name 时使用的默认编辑模型名。"""
    return str(get_config(
        "agents.tools.image_editor.default_model_name", ""
    ) or "").strip()


def get_video_generator_default_model_name() -> str:
    """video_generator 工具未显式指定 model_name 时使用的默认模型名。"""
    return str(get_config(
        "agents.tools.video_generator.default_model_name", ""
    ) or "").strip()


def get_translate_default_model_name() -> str:
    """translate 任务未显式指定 load_name 时使用的默认模型名。"""
    return str(get_config(
        "agents.tools.translate.default_model_name", ""
    ) or "").strip()


def get_translate_default_family() -> str:
    """translate 任务 family 回退值（models 表未回填时使用）。"""
    return str(get_config(
        "agents.tools.translate.default_family", "TranslateGemma"
    ) or "TranslateGemma").strip()


def get_audio_asr_default_timeout() -> float:
    """audio_asr 工具等待推理器完成的默认超时秒数。"""
    raw_value = get_config("agents.tools.audio_asr.default_timeout_seconds", 180)
    try:
        value = float(raw_value or 180)
    except (TypeError, ValueError):
        value = 180.0
    return value if value > 0 else 180.0


def get_audio_tts_default_timeout() -> float:
    """audio_tts 工具等待推理器完成的默认超时秒数。"""
    raw_value = get_config("agents.tools.audio_tts.default_timeout_seconds", 180)
    try:
        value = float(raw_value or 180)
    except (TypeError, ValueError):
        value = 180.0
    return value if value > 0 else 180.0


def get_document_to_markdown_default_timeout() -> float:
    """document_to_markdown 工具的默认超时秒数。"""
    raw_value = get_config("agents.tools.document_to_markdown.default_timeout_seconds", 300)
    try:
        value = float(raw_value or 300)
    except (TypeError, ValueError):
        value = 300.0
    return value if value > 0 else 300.0


def get_document_to_markdown_pdf_model_name() -> str:
    """document_to_markdown 在 PDF 路径下使用的默认 mini OCR 模型名。"""
    return str(get_config(
        "agents.tools.document_to_markdown.pdf_model_name", "GLM-OCR"
    ) or "GLM-OCR").strip()


def get_document_to_pdf_default_timeout() -> float:
    """document_to_pdf 工具的默认超时秒数。"""
    raw_value = get_config("agents.tools.document_to_pdf.default_timeout_seconds", 300)
    try:
        value = float(raw_value or 300)
    except (TypeError, ValueError):
        value = 300.0
    return value if value > 0 else 300.0


def get_document_to_pdf_default_font() -> str:
    """document_to_pdf 缺字时默认替换字体。"""
    return str(get_config(
        "agents.tools.document_to_pdf.default_font", "Noto Sans CJK SC"
    ) or "Noto Sans CJK SC").strip()


