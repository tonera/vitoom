"""Shared ES DSL guard constants for business query tools."""

from __future__ import annotations


FORBIDDEN_ES_DSL_KEYS = {
    "script",
    "runtime_mappings",
    "script_fields",
    "query_string",
    "simple_query_string",
    "wildcard",
    "regexp",
    "prefix",
    "fuzzy",
    "knn",
    "collapse",
    "highlight",
    "rescore",
    "suggest",
    "pit",
    "search_after",
}
