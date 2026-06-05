import sys
import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vitoom_setup import catalog_sqlite  # noqa: E402
from vitoom_setup.catalog_sqlite import (  # noqa: E402
    ModelCatalogDbNotFoundError,
    ensure_model_catalog_writable,
    resolve_model_catalog_db_path,
)
from vitoom_setup import model_hub  # noqa: E402


def _create_sqlite_db(path: Path, *, with_model_catalog: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        if with_model_catalog:
            conn.execute("CREATE TABLE model_catalog (model_key TEXT PRIMARY KEY)")
        else:
            conn.execute("CREATE TABLE other_table (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()


def _patch_default_db_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    docker_db = tmp_path / "data" / "resources" / "data" / "vitoom.db"
    local_db = tmp_path / "resources" / "data" / "vitoom.db"
    container_db = tmp_path / "container" / "resources" / "data" / "vitoom.db"

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(catalog_sqlite, "DOCKER_DEFAULT_DB_PATH", docker_db)
    monkeypatch.setattr(catalog_sqlite, "LOCAL_DEFAULT_DB_PATH", local_db)
    monkeypatch.setattr(catalog_sqlite, "CONTAINER_DEFAULT_DB_PATH", container_db)

    return docker_db, local_db


def test_resolve_model_catalog_db_path_auto_selects_docker_db(monkeypatch, tmp_path):
    docker_db, local_db = _patch_default_db_paths(monkeypatch, tmp_path)
    _create_sqlite_db(docker_db)

    resolved = resolve_model_catalog_db_path({})

    assert resolved == docker_db.resolve()
    assert not local_db.exists()


def test_resolve_model_catalog_db_path_auto_selects_local_source_db(monkeypatch, tmp_path):
    docker_db, local_db = _patch_default_db_paths(monkeypatch, tmp_path)
    _create_sqlite_db(local_db)

    resolved = resolve_model_catalog_db_path({})

    assert resolved == local_db.resolve()
    assert not docker_db.exists()


def test_resolve_model_catalog_db_path_prompts_when_multiple_valid_dbs(monkeypatch, tmp_path):
    docker_db, local_db = _patch_default_db_paths(monkeypatch, tmp_path)
    _create_sqlite_db(docker_db)
    _create_sqlite_db(local_db)

    resolved = resolve_model_catalog_db_path({}, input_func=lambda prompt: "2")

    assert resolved == local_db.resolve()


def test_database_url_nonexistent_path_is_ignored_and_not_created(monkeypatch, tmp_path):
    docker_db, _ = _patch_default_db_paths(monkeypatch, tmp_path)
    missing_db = tmp_path / "missing" / "vitoom.db"
    _create_sqlite_db(docker_db)

    resolved = resolve_model_catalog_db_path({"DATABASE_URL": f"sqlite:///{missing_db}"})

    assert resolved == docker_db.resolve()
    assert not missing_db.exists()


def test_container_database_url_maps_to_valid_docker_host_db(monkeypatch, tmp_path):
    docker_host_resources = tmp_path / "data" / "resources"
    mapped_db = docker_host_resources / "data" / "vitoom.db"
    _patch_default_db_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(catalog_sqlite, "DOCKER_HOST_RESOURCES_DIR", docker_host_resources)
    _create_sqlite_db(mapped_db)

    resolved = resolve_model_catalog_db_path(
        {"DATABASE_URL": "sqlite:////app/resources/data/vitoom.db?check_same_thread=False"}
    )

    assert resolved == mapped_db.resolve()


def test_db_without_model_catalog_is_not_valid_candidate(monkeypatch, tmp_path):
    docker_db, _ = _patch_default_db_paths(monkeypatch, tmp_path)
    _create_sqlite_db(docker_db, with_model_catalog=False)

    with pytest.raises(ModelCatalogDbNotFoundError):
        resolve_model_catalog_db_path({})


def test_ensure_model_catalog_writable_raises_for_missing_catalog(tmp_path):
    db_path = tmp_path / "vitoom.db"
    _create_sqlite_db(db_path, with_model_catalog=False)

    with pytest.raises(sqlite3.OperationalError):
        ensure_model_catalog_writable(db_path)


def test_ensure_huggingface_tooling_installs_socksio_for_socks_proxy(monkeypatch):
    installed: list[tuple[str, ...]] = []

    monkeypatch.setenv("HTTPS_PROXY", "socks5://127.0.0.1:1080")
    monkeypatch.setattr(model_hub, "cli_available", lambda command: command == "hf")
    monkeypatch.setattr(model_hub, "module_available", lambda module: module != "socksio")
    monkeypatch.setattr(
        model_hub,
        "pip_install_packages",
        lambda locale, *packages: installed.append(packages),
    )

    model_hub.ensure_huggingface_tooling("en-US")

    assert installed == [("socksio",)]
