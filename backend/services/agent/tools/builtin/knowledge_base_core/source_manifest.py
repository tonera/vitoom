from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


def read_manifest(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            yield parsed


def write_manifest(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def upsert_manifest_rows(path: Path, rows: Iterable[Dict[str, Any]], *, key_field: str = "document_id") -> int:
    existing = list(read_manifest(path))
    by_key: Dict[str, Dict[str, Any]] = {
        str(row.get(key_field) or ""): row for row in existing if str(row.get(key_field) or "")
    }
    order = [str(row.get(key_field) or "") for row in existing if str(row.get(key_field) or "")]
    count = 0
    for row in rows:
        key = str(row.get(key_field) or "").strip()
        if not key:
            continue
        if key not in by_key:
            order.append(key)
        by_key[key] = row
        count += 1
    write_manifest(path, [by_key[key] for key in order if key in by_key])
    return count
