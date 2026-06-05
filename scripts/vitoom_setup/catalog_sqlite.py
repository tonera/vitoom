"""Direct SQLite access for initial model catalog upserts (setup script only)."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

from vitoom_setup import REPO_ROOT

DOCKER_HOST_RESOURCES_DIR = REPO_ROOT / "data" / "resources"
LOCAL_RESOURCES_DIR = REPO_ROOT / "resources"
DOCKER_DEFAULT_DB_PATH = DOCKER_HOST_RESOURCES_DIR / "data" / "vitoom.db"
LOCAL_DEFAULT_DB_PATH = LOCAL_RESOURCES_DIR / "data" / "vitoom.db"
CONTAINER_DEFAULT_DB_PATH = Path("/app/resources/data/vitoom.db")


@dataclass(frozen=True)
class CatalogDbCandidate:
    label: str
    path: Path


class ModelCatalogDbNotFoundError(RuntimeError):
    """Raised when no existing SQLite DB with model_catalog can be found."""


def _map_container_path_to_host(path: Path) -> Path:
    """Map backend container paths to docker-compose host volume paths."""
    if not path.is_absolute():
        return path
    try:
        rel = path.relative_to("/app/resources")
    except ValueError:
        return path
    return DOCKER_HOST_RESOURCES_DIR / rel


def _sqlite_path_from_url(url: str) -> Path | None:
    base = url.split("?", 1)[0].strip()
    if not base.startswith("sqlite:"):
        return None
    if base.startswith("sqlite:////"):
        return _map_container_path_to_host(
            Path("/" + unquote(base[len("sqlite:////") :]))
        ).resolve()
    if base.startswith("sqlite:///"):
        raw = unquote(base[len("sqlite:///") :])
        if not raw or raw == ":memory:":
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return _map_container_path_to_host(path).resolve()
    return None


def has_model_catalog_table(db_path: Path) -> bool:
    if not db_path.is_file():
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'model_catalog'
                LIMIT 1
                """
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def _add_candidate(
    candidates: list[CatalogDbCandidate],
    seen: dict[Path, int],
    label: str,
    path: Path | None,
) -> None:
    if path is None:
        return
    resolved = path.resolve()
    if resolved in seen:
        index = seen[resolved]
        existing = candidates[index]
        candidates[index] = CatalogDbCandidate(f"{existing.label}, {label}", existing.path)
        return
    seen[resolved] = len(candidates)
    candidates.append(CatalogDbCandidate(label, resolved))


def collect_model_catalog_db_candidates(env: dict[str, str] | None = None) -> list[CatalogDbCandidate]:
    candidates: list[CatalogDbCandidate] = []
    seen: dict[Path, int] = {}

    _add_candidate(candidates, seen, "Docker Compose", DOCKER_DEFAULT_DB_PATH)
    _add_candidate(candidates, seen, "Local source", LOCAL_DEFAULT_DB_PATH)
    if CONTAINER_DEFAULT_DB_PATH.is_file():
        _add_candidate(candidates, seen, "Container runtime", CONTAINER_DEFAULT_DB_PATH)

    url = (env or {}).get("DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    url = url.strip()
    if url:
        resolved = _sqlite_path_from_url(url)
        _add_candidate(candidates, seen, "DATABASE_URL", resolved)

    return [candidate for candidate in candidates if has_model_catalog_table(candidate.path)]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def select_model_catalog_db_candidate(
    candidates: list[CatalogDbCandidate],
    *,
    input_func: Callable[[str], str] = input,
) -> CatalogDbCandidate:
    print("Detected multiple Vitoom SQLite databases with model_catalog:")
    for index, candidate in enumerate(candidates, start=1):
        print(f"  [{index}] {candidate.label}: {_display_path(candidate.path)}")
    while True:
        raw = input_func(f"Select database to update [1-{len(candidates)}]: ").strip()
        try:
            selected = int(raw)
        except ValueError:
            print("Invalid choice, try again.")
            continue
        if 1 <= selected <= len(candidates):
            return candidates[selected - 1]
        print("Invalid choice, try again.")


def resolve_model_catalog_db_path(
    env: dict[str, str] | None = None,
    *,
    interactive: bool = True,
    input_func: Callable[[str], str] = input,
) -> Path:
    candidates = collect_model_catalog_db_candidates(env)
    if not candidates:
        raise ModelCatalogDbNotFoundError(
            "No existing SQLite database with model_catalog was found. "
            "Start or initialize the backend first, then rerun this script."
        )
    if len(candidates) == 1:
        return candidates[0].path
    if not interactive:
        raise ModelCatalogDbNotFoundError(
            "Multiple SQLite databases with model_catalog were found; choose one interactively."
        )
    return select_model_catalog_db_candidate(candidates, input_func=input_func).path


def ensure_model_catalog_writable(db_path: Path) -> None:
    if not has_model_catalog_table(db_path):
        raise sqlite3.OperationalError(f"model_catalog table not found in {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
    finally:
        conn.close()


def _json_text(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def upsert_model_catalog(
    *,
    db_path: Path,
    load_name: str,
    seed: dict[str, Any],
    source: dict[str, Any],
) -> None:
    """Insert or update one model_catalog row. Raises sqlite3.Error on failure."""
    if not has_model_catalog_table(db_path):
        raise sqlite3.OperationalError(f"model_catalog table not found in {db_path}")
    model_key = str(seed["model_key"])
    now = _utc_now_iso()
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT 1 FROM model_catalog WHERE model_key = ? AND deleted_at IS NULL LIMIT 1",
            (model_key,),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE model_catalog
                SET download_status = ?, storage_mode = ?, service_status = ?,
                    source = ?, updated_at = ?
                WHERE model_key = ? AND deleted_at IS NULL
                """,
                (
                    "completed",
                    "local",
                    str(seed.get("service_status") or "inactive"),
                    _json_text(source),
                    now,
                    model_key,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO model_catalog (
                    model_key, name, modality, asset_type, family, capabilities,
                    runtime_engine, runtime_config, load_name, service_status,
                    storage_mode, download_status, source, thumb, tags,
                    trigger_words, description, created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    model_key,
                    str(seed["name"]),
                    str(seed["modality"]),
                    str(seed["asset_type"]),
                    str(seed.get("family") or ""),
                    _json_text(seed.get("capabilities") or {}),
                    str(seed.get("runtime_engine") or ""),
                    _json_text(seed.get("runtime_config") or {}),
                    load_name,
                    str(seed.get("service_status") or "inactive"),
                    "local",
                    "completed",
                    _json_text(source),
                    seed.get("thumb"),
                    _json_text(seed.get("tags") or []),
                    _json_text(seed.get("trigger_words") or []),
                    seed.get("description"),
                    now,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()
