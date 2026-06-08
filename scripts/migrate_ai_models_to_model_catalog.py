#!/usr/bin/env python3
"""
Create and migrate fluxsd_com.model_catalog from legacy ai_models.

The migration keeps the old ai_models table untouched. New code should read the
new model_catalog table directly instead of adding compatibility fallbacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

try:
    import pymysql
except ImportError:
    print("ERROR: pymysql is required. Install it with: pip install pymysql")
    sys.exit(1)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `model_catalog` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `model_key` char(32) NOT NULL COMMENT 'md5(normalize(load_name)|normalize(runtime_engine)|normalize(asset_type))',
  `name` varchar(256) NOT NULL,
  `modality` varchar(16) NOT NULL DEFAULT 'image',
  `asset_type` varchar(32) NOT NULL DEFAULT 'checkpoint',
  `family` varchar(64) NOT NULL DEFAULT '',
  `capabilities` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`capabilities`)),
  `runtime_engine` varchar(64) NOT NULL DEFAULT '',
  `runtime_config` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`runtime_config`)),
  `load_name` varchar(256) NOT NULL,
  `service_status` varchar(16) NOT NULL DEFAULT 'inactive',
  `storage_mode` varchar(16) NOT NULL DEFAULT 'cloud',
  `download_status` varchar(16) NOT NULL DEFAULT 'pending',
  `source` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`source`)),
  `thumb` varchar(1024) DEFAULT NULL,
  `tags` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`tags`)),
  `trigger_words` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`trigger_words`)),
  `description` text DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  `deleted_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_model_catalog_model_key` (`model_key`),
  KEY `idx_model_catalog_modality` (`modality`),
  KEY `idx_model_catalog_asset_type` (`asset_type`),
  KEY `idx_model_catalog_family` (`family`),
  KEY `idx_model_catalog_runtime_engine` (`runtime_engine`),
  KEY `idx_model_catalog_service_status` (`service_status`),
  KEY `idx_model_catalog_storage_mode` (`storage_mode`),
  KEY `idx_model_catalog_deleted_at` (`deleted_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


UPSERT_SQL = """
INSERT INTO `model_catalog` (
  `model_key`, `name`, `modality`, `asset_type`, `family`, `capabilities`,
  `runtime_engine`, `runtime_config`, `load_name`, `service_status`,
  `storage_mode`, `download_status`, `source`, `thumb`, `tags`,
  `trigger_words`, `description`, `created_at`, `updated_at`, `deleted_at`
) VALUES (
  %(model_key)s, %(name)s, %(modality)s, %(asset_type)s, %(family)s, %(capabilities)s,
  %(runtime_engine)s, %(runtime_config)s, %(load_name)s, %(service_status)s,
  %(storage_mode)s, %(download_status)s, %(source)s, %(thumb)s, %(tags)s,
  %(trigger_words)s, %(description)s, %(created_at)s, %(updated_at)s, %(deleted_at)s
)
ON DUPLICATE KEY UPDATE
  `name` = VALUES(`name`),
  `modality` = VALUES(`modality`),
  `asset_type` = VALUES(`asset_type`),
  `family` = VALUES(`family`),
  `capabilities` = VALUES(`capabilities`),
  `runtime_engine` = VALUES(`runtime_engine`),
  `runtime_config` = VALUES(`runtime_config`),
  `load_name` = VALUES(`load_name`),
  `service_status` = VALUES(`service_status`),
  `storage_mode` = VALUES(`storage_mode`),
  `download_status` = VALUES(`download_status`),
  `source` = VALUES(`source`),
  `thumb` = VALUES(`thumb`),
  `tags` = VALUES(`tags`),
  `trigger_words` = VALUES(`trigger_words`),
  `description` = VALUES(`description`),
  `created_at` = VALUES(`created_at`),
  `updated_at` = VALUES(`updated_at`),
  `deleted_at` = VALUES(`deleted_at`)
"""


def normalize_required(value: Any, fallback: str) -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback


def normalize_key_part(value: Any, *, lower: bool = True) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower() if lower else text


def build_model_key(load_name: str, runtime_engine: str, asset_type: str) -> str:
    key_source = "|".join(
        [
            normalize_key_part(load_name, lower=False),
            normalize_key_part(runtime_engine),
            normalize_key_part(asset_type),
        ]
    )
    return hashlib.md5(key_source.encode("utf-8")).hexdigest()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def parse_json_object(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def parse_json_array(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def normalize_modality(value: Any) -> str:
    text = normalize_required(value, "image").lower()
    aliases = {
        "img": "image",
        "txt": "text",
        "tts": "voice",
    }
    text = aliases.get(text, text)
    allowed = {"image", "video", "text", "voice", "audio", "mini"}
    return text if text in allowed else "image"


def normalize_asset_type(value: Any) -> str:
    text = normalize_required(value, "checkpoint").lower()
    aliases = {
        "ckpt": "checkpoint",
        "checkpoints": "checkpoint",
        "loras": "lora",
    }
    return aliases.get(text, text)


def normalize_thumb(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        text = parsed.path.lstrip("/")
    return text.lstrip("/")


def maybe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def civitai_model_id_from_url(url: Optional[str]) -> Optional[int]:
    if not url:
        return None
    match = re.search(r"/models/(\d+)", url)
    if not match:
        return None
    return maybe_int(match.group(1))


def build_source(row: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
    model_url = model_info.get("civitai_link") or model_info.get("third_source")
    if model_url is not None:
        model_url = str(model_url).strip() or None

    civitai_version_id = maybe_int(row.get("civitai_version_id"))
    return {
        "model_url": model_url,
        "repo_id": str(civitai_version_id) if civitai_version_id is not None else None,
        "download_url": str(row.get("download_url")).strip() if row.get("download_url") else None,
        "civitai_model_id": civitai_model_id_from_url(model_url),
    }


def build_runtime_config(row: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(model_config)
    base_size = maybe_int(row.get("base_size"))
    base_steps = maybe_int(row.get("base_steps"))
    sampler = str(row.get("sampler")).strip() if row.get("sampler") else None
    prompt = str(row.get("prompt")).strip() if row.get("prompt") else None

    if base_size:
        config["base_size"] = base_size
    if base_steps:
        config["base_steps"] = base_steps
    if sampler:
        config["sampler"] = sampler
    if prompt:
        config["default_prompt"] = prompt
    return config


def build_capabilities(row: Dict[str, Any], modality: str) -> Dict[str, Any]:
    is_editable = bool(row.get("is_editable") or 0)
    capabilities: Dict[str, Any] = {
        "editable": is_editable,
        "nsfw": bool(row.get("nsfw") or 0),
    }

    combined = " ".join(
        str(row.get(key) or "").lower()
        for key in ("name", "engine", "model_type", "type", "base_model", "sd_name")
    )
    if modality == "image":
        capabilities["t2i"] = True
        if is_editable:
            capabilities["i2i"] = True
    elif modality == "video":
        capabilities["t2v"] = True
        if is_editable:
            capabilities["i2v"] = True
    elif modality in {"voice", "audio"}:
        if "asr" in combined or "speech-recognition" in combined:
            capabilities["asr"] = True
        else:
            capabilities["tts"] = True
    elif modality == "text":
        capabilities["chat"] = True
    elif modality == "mini":
        capabilities["mini"] = True
    return capabilities


def map_row(row: Dict[str, Any]) -> Dict[str, Any]:
    modality = normalize_modality(row.get("model_type"))
    asset_type = normalize_asset_type(row.get("type"))
    runtime_engine = normalize_required(row.get("engine"), "")
    load_name = normalize_required(row.get("sd_name"), normalize_required(row.get("name"), "unnamed"))
    model_info = parse_json_object(row.get("model_info"))
    model_config = parse_json_object(row.get("model_config"))

    storage_mode = "local" if runtime_engine == "FluxSD" else "cloud"
    if row.get("deleted_at"):
        service_status = "disabled"
    else:
        service_status = "active" if int(row.get("is_service") or 0) == 1 else "inactive"

    return {
        "model_key": build_model_key(load_name, runtime_engine, asset_type),
        "name": normalize_required(row.get("name"), load_name),
        "modality": modality,
        "asset_type": asset_type,
        "family": normalize_required(row.get("base_model"), ""),
        "capabilities": json_dumps(build_capabilities(row, modality)),
        "runtime_engine": runtime_engine,
        "runtime_config": json_dumps(build_runtime_config(row, model_config)),
        "load_name": load_name,
        "service_status": service_status,
        "storage_mode": storage_mode,
        "download_status": "completed" if storage_mode == "local" else "pending",
        "source": json_dumps(build_source(row, model_info)),
        "thumb": normalize_thumb(row.get("thumb")),
        "tags": json_dumps(parse_json_array(row.get("tags"))),
        "trigger_words": json_dumps([]),
        "description": str(row.get("note")).strip() if row.get("note") else None,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "deleted_at": row.get("deleted_at"),
    }


def connect(args: argparse.Namespace):
    return pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def fetch_ai_models(conn) -> Iterable[Dict[str, Any]]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM `ai_models` ORDER BY `id` ASC")
        return cursor.fetchall()


def migrate(args: argparse.Namespace) -> None:
    conn = connect(args)
    try:
        with conn.cursor() as cursor:
            if args.recreate:
                cursor.execute("DROP TABLE IF EXISTS `model_catalog`")
            cursor.execute(CREATE_TABLE_SQL)

        rows = fetch_ai_models(conn)
        mapped_rows = [map_row(row) for row in rows]

        duplicate_keys = len(mapped_rows) - len({row["model_key"] for row in mapped_rows})
        if duplicate_keys:
            print(f"WARNING: {duplicate_keys} source rows share an existing model_key and will be upserted.")

        if args.dry_run:
            print(f"DRY RUN: would migrate {len(mapped_rows)} rows into model_catalog.")
            if mapped_rows:
                print(json.dumps(mapped_rows[0], ensure_ascii=False, indent=2, default=json_default))
            conn.rollback()
            return

        with conn.cursor() as cursor:
            cursor.executemany(UPSERT_SQL, mapped_rows)

        conn.commit()
        print(f"Migrated {len(mapped_rows)} ai_models rows into model_catalog.")
        if duplicate_keys:
            print("Duplicate model_key rows were collapsed by ON DUPLICATE KEY UPDATE.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy ai_models to model_catalog.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3307)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default="fluxsd_com")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate model_catalog before migration.")
    parser.add_argument("--dry-run", action="store_true", help="Preview one mapped row without writing data.")
    return parser.parse_args()


if __name__ == "__main__":
    migrate(parse_args())
