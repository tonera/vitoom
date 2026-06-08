#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.agent.tool_selection import ToolSelectionService  # noqa: E402
from backend.services.agent import tool_selection as tool_selection_module  # noqa: E402
from backend.services.agent.settings import (  # noqa: E402
    get_tool_selection_embedding_backend,
    get_tool_selection_embedding_model_path,
    is_tool_selection_embedding_enabled,
)


def _split_csv(raw_value: str) -> List[str]:
    return [item.strip() for item in str(raw_value or "").split(",") if item.strip()]


def _format_names(names: Iterable[str]) -> str:
    values = [str(name).strip() for name in names if str(name).strip()]
    return ", ".join(values) if values else "-"


def _embedding_status() -> str:
    manager = tool_selection_module._EMBEDDING_BACKENDS
    backend = manager._backend
    if backend.is_ready():
        return f"ONNX embedding loaded: yes ({backend.cache_key})"
    enabled = is_tool_selection_embedding_enabled()
    backend_name = get_tool_selection_embedding_backend()
    model_path = get_tool_selection_embedding_model_path()
    if not enabled:
        return "ONNX embedding loaded: no (embedding disabled; using BM25)"
    if backend_name != "onnx":
        return f"ONNX embedding loaded: no (backend={backend_name}; using BM25)"
    if not model_path:
        return "ONNX embedding loaded: no (embedding_model_path is empty; using BM25)"
    return f"ONNX embedding loaded: no (configured path: {model_path}; loads on first query)"


def _print_result(query: str, result, *, top_candidates: int) -> None:
    print()
    print(f"Query: {query}")
    mode = "recommended/hybrid" if result.strategy == "hybrid" else "base/BM25"
    print(f"Mode: {mode}")
    print(f"Strategy: {result.strategy} | embedding: {result.embedding_key} | index: {result.index_version}")
    print(f"Latency: {result.latency_ms} ms")
    print(f"Selected: {_format_names(result.selected_names)}")
    print()

    candidates = list(result.candidates[: max(1, top_candidates)])
    if not candidates:
        print("No candidates.")
        print()
        return

    headers = ["rank", "tool", "final", "bm25", "vector", "anchor", "exact", "preferred", "mod+", "mod-", "negative"]
    rows = []
    for index, item in enumerate(candidates, start=1):
        rows.append(
            [
                str(index),
                item.name,
                f"{item.score:.4f}",
                f"{item.bm25_score:.4f}",
                f"{item.vector_score:.4f}",
                f"{item.intent_anchor_score:.4f}",
                f"{item.exact_match_score:.4f}",
                f"{item.preferred_boost:.4f}",
                f"{item.modality_match_score:.4f}",
                f"{item.modality_mismatch_score:.4f}",
                f"{item.negative_match_score:.4f}",
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    header_line = "  ".join(headers[col].ljust(widths[col]) for col in range(len(headers)))
    separator = "  ".join("-" * widths[col] for col in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rows:
        print("  ".join(row[col].ljust(widths[col]) for col in range(len(row))))
    print()


def _run_once(service: ToolSelectionService, query: str, args: argparse.Namespace) -> None:
    command = SimpleNamespace(
        message=query,
        context={"original_user_message": query},
        runtime_config={},
    )
    result = service.debug_select_tool_names(
        _split_csv(args.tools),
        command=command,
        runtime_allowlist=_split_csv(args.allowlist),
        max_tools=args.max_tools,
        pool=args.pool,
        preferred_tool_names=_split_csv(args.preferred),
    )
    _print_result(query, result, top_candidates=args.top_candidates)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactively inspect agent tool selection results and scores.",
    )
    parser.add_argument("--query", default="", help="Run one query and exit.")
    parser.add_argument("--pool", choices=["global", "declared"], default="global")
    parser.add_argument("--tools", default="", help="Comma-separated declared tool names.")
    parser.add_argument("--preferred", default="", help="Comma-separated preferred tool names.")
    parser.add_argument("--allowlist", default="", help="Comma-separated runtime allowlist.")
    parser.add_argument("--max-tools", type=int, default=5)
    parser.add_argument("--top-candidates", type=int, default=10)
    args = parser.parse_args()

    service = ToolSelectionService()
    if args.query.strip():
        _run_once(service, args.query.strip(), args)
        return 0

    print("Tool selection REPL. Type 'q', 'quit', or 'exit' to stop. Empty input is ignored.")
    print(_embedding_status())
    print(f"pool={args.pool} max_tools={args.max_tools} top_candidates={args.top_candidates}")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            return 0
        _run_once(service, query, args)


if __name__ == "__main__":
    raise SystemExit(main())
