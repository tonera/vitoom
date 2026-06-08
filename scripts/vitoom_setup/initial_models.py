"""Interactive initial model downloader for post-install setup."""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from backend.i18n.translator import t
from vitoom_setup import REPO_ROOT
from vitoom_setup.catalog_sqlite import (
    ModelCatalogDbNotFoundError,
    ensure_model_catalog_writable,
    resolve_model_catalog_db_path,
    upsert_model_catalog,
)
from vitoom_setup.env_file import parse_env_file
from vitoom_setup.model_hub import (
    ModelSource,
    RepoDownload,
    download_repo_with_fallback,
    region_default_model_source,
    repo_download_full,
)

ExistingAction = Literal["skip", "redownload", "abort"]

def resolve_inference_yaml_path() -> Path | None:
    """Local dev uses inference/config/; Docker entrypoint writes data/inference/config/."""
    candidates = (
        REPO_ROOT / "inference" / "config" / "inference.yaml",
        REPO_ROOT / "data" / "inference" / "config" / "inference.yaml",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


CACHED_MODEL_CATALOG_DB_PATH: Path | None = None

CATEGORY_IDS = ("llm", "audio", "video", "image", "basic")


@dataclass(frozen=True)
class InitialModelSpec:
    load_name: str
    hf_repo_id: str
    ms_repo_id: str
    category: str
    install_root: Literal["models", "weights"] = "models"


INITIAL_MODELS: tuple[InitialModelSpec, ...] = (
    InitialModelSpec(
        "Qwen3.5-4B-AWQ-4bit",
        "cyankiwi/Qwen3.5-4B-AWQ-4bit",
        "cyankiwi/Qwen3.5-4B-AWQ-4bit",
        "llm",
    ),
    InitialModelSpec(
        "translategemma-4b-it",
        "google/translategemma-4b-it",
        "google/translategemma-4b-it",
        "llm",
    ),
    InitialModelSpec("Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-0.6B", "audio"),
    InitialModelSpec(
        "Qwen3-ForcedAligner-0.6B",
        "Qwen/Qwen3-ForcedAligner-0.6B",
        "Qwen/Qwen3-ForcedAligner-0.6B",
        "audio",
    ),
    InitialModelSpec(
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "audio",
    ),
    InitialModelSpec(
        "Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "audio",
    ),
    InitialModelSpec(
        "Qwen3-TTS-Tokenizer-12Hz",
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "audio",
    ),
    InitialModelSpec(
        "VoxCPM2",
        "openbmb/VoxCPM2",
        "OpenBMB/VoxCPM2",
        "audio",
    ),
    InitialModelSpec(
        "TurboWan2.1-T2V-1.3B-480P",
        "TurboDiffusion/TurboWan2.1-T2V-1.3B-480P",
        "TurboDiffusion/TurboWan2.1-T2V-1.3B-480P",
        "video",
    ),
    InitialModelSpec("WanVideo", "tonera/WanVideo", "tonera/WanVideo", "video"),
    InitialModelSpec(
        "FLUX.2-klein-9B-Nunchaku",
        "tonera/FLUX.2-klein-9B-Nunchaku",
        "tonera/FLUX.2-klein-9B-Nunchaku",
        "image",
    ),
    InitialModelSpec(
        "Qwen3-text-Nunchaku",
        "tonera/Qwen3-text-Nunchaku",
        "tonera/Qwen3-text-Nunchaku",
        "image",
        install_root="weights",
    ),
    InitialModelSpec("RMBG-2.0", "1038lab/RMBG-2.0", "briaai/RMBG-2.0", "basic"),
    InitialModelSpec("GLM-OCR", "zai-org/GLM-OCR", "ZhipuAI/GLM-OCR", "basic"),
    InitialModelSpec("roop", "tonera/roop", "tonera/roop", "basic"),
)

CATEGORY_SIZE_HINTS = {
    "llm": "~12G",
    "audio": "~14.1G",
    "video": "~18.4G",
    "image": "~48G",
    "basic": "~9.8G",
}

# model_catalog 种子数据（自开发库 vitoom.db 导出；初始安装时表为空，运行时不再读库取字段）。
# 其中 WanVideo / RMBG-2.0 / roop / Qwen3-TTS-Tokenizer-12Hz 在导出时尚无行，字段按同类模型补全。
# service_status：Qwen3-ForcedAligner-0.6B / Qwen3-TTS-Tokenizer-12Hz / WanVideo / RMBG-2.0 / roop 默认 inactive，其余 active。
MODEL_CATALOG_SEED: dict[str, dict[str, Any]] = {
    "Qwen3.5-4B-AWQ-4bit": {
        "model_key": "27c33f9470a035c93b840c22058db7df",
        "name": "Qwen3.5-4B-AWQ-4bit",
        "modality": "text",
        "asset_type": "checkpoint",
        "family": "Qwen-text",
        "capabilities": {"editable": False},
        "runtime_engine": "",
        "runtime_config": {},
        "service_status": "active",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {},
        "thumb": None,
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "translategemma-4b-it": {
        "model_key": "3566e6363b31ef00c00b13078d1ccce8",
        "name": "translategemma-4b-it",
        "modality": "translate",
        "asset_type": "checkpoint",
        "family": "TranslateGemma",
        "capabilities": {"editable": False},
        "runtime_engine": "",
        "runtime_config": {},
        "service_status": "active",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {},
        "thumb": None,
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "Qwen3-ASR-0.6B": {
        "model_key": "5f051ee53fabad4be1654f841106b69a",
        "name": "Qwen/Qwen3-ASR-0.6B",
        "modality": "audio",
        "asset_type": "checkpoint",
        "family": "Qwen-asr",
        "capabilities": {"asr": True, "editable": False, "nsfw": False},
        "runtime_engine": "FluxSD",
        "runtime_config": {},
        "service_status": "active",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {"legacy_models_id": "644b3bf86a984badb97ab11a345e2d3d"},
        "thumb": "models/202604/3be986d7056a4848ba74a43f15da8d89.webp",
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "Qwen3-ForcedAligner-0.6B": {
        "model_key": "dbef8edfe2751221c72954fe19bbd27e",
        "name": "Qwen/Qwen3-ForcedAligner-0.6B",
        "modality": "audio",
        "asset_type": "checkpoint",
        "family": "Qwen-asr",
        "capabilities": {"asr": True, "editable": False, "nsfw": False},
        "runtime_engine": "FluxSD",
        "runtime_config": {},
        "service_status": "inactive",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {"legacy_models_id": "ade0844f92e0439e9d316726b5d56e98"},
        "thumb": "models/202604/4ec143862744449d9c688b5d921958f7.webp",
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "model_key": "5278dc0c2492d0f4ffdd53b7741e5cd6",
        "name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "modality": "audio",
        "asset_type": "checkpoint",
        "family": "Qwen-tts",
        "capabilities": {"editable": False, "nsfw": False, "tts": True},
        "runtime_engine": "FluxSD",
        "runtime_config": {},
        "service_status": "active",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {"legacy_models_id": "8cb9969d2f01467c975a1a7527f79ed8"},
        "thumb": "models/202604/0657ac527c364f24ad7e882e9f419cec.webp",
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "Qwen3-TTS-12Hz-0.6B-Base": {
        "model_key": "606aa034bd8993e1dd2abec82cc3ef09",
        "name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "modality": "audio",
        "asset_type": "checkpoint",
        "family": "Qwen-tts",
        "capabilities": {"editable": False, "nsfw": False, "tts": True},
        "runtime_engine": "FluxSD",
        "runtime_config": {},
        "service_status": "active",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {"legacy_models_id": "f38697f9b5ad448aa2ce653cc7eb5729"},
        "thumb": "models/202604/8721d04eed9743789c48aa9b4ab1b641.webp",
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "Qwen3-TTS-Tokenizer-12Hz": {
        "model_key": "f0958718fa2395a8a343eeb859e6e31a",
        "name": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "modality": "audio",
        "asset_type": "checkpoint",
        "family": "Qwen-tts",
        "capabilities": {"editable": False, "nsfw": False, "tts": True},
        "runtime_engine": "FluxSD",
        "runtime_config": {},
        "service_status": "inactive",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {},
        "thumb": None,
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "VoxCPM2": {
        "model_key": "92592b2d862de400f4a08e5fcf9d6082",
        "name": "openbmb/VoxCPM2",
        "modality": "audio",
        "asset_type": "checkpoint",
        "family": "Voxcpm",
        "capabilities": {"editable": False, "nsfw": False, "tts": True},
        "runtime_engine": "FluxSD",
        "runtime_config": {},
        "service_status": "active",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {},
        "thumb": None,
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "TurboWan2.1-T2V-1.3B-480P": {
        "model_key": "46b5797f12277efcb6c78b278c78da38",
        "name": "TurboWan2.1-T2V-1.3B-480P",
        "modality": "video",
        "asset_type": "checkpoint",
        "family": "T2V-1.3B",
        "capabilities": {"editable": False, "nsfw": False, "t2v": True},
        "runtime_engine": "FluxSD",
        "runtime_config": {
            "base_size": 1024,
            "base_steps": 30,
            "video": {
                "aspect_profile": "want2v",
                "defaults": {"aspect": "16:9", "duration": 5, "resolution": "480p"},
                "modes": {"TextToVideo": {"supported_resolutions": ["480p"]}},
                "supported_resolutions": ["480p"],
            },
        },
        "service_status": "active",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {
            "civitai_model_id": None,
            "download_url": None,
            "model_url": None,
            "repo_id": None,
        },
        "thumb": "202601/ba3694a307e54c0488edb76afd14edee.jpeg",
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "WanVideo": {
        "model_key": "79dedce12cc319f2b2570576c9fb32bc",
        "name": "WanVideo",
        "modality": "video",
        "asset_type": "checkpoint",
        "family": "",
        "capabilities": {"editable": False, "nsfw": False},
        "runtime_engine": "",
        "runtime_config": {},
        "service_status": "inactive",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {},
        "thumb": None,
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "FLUX.2-klein-9B-Nunchaku": {
        "model_key": "0a3ad584b5af9e87d1840c2cc396e02c",
        "name": "FLUX.2-klein-9B",
        "modality": "image",
        "asset_type": "checkpoint",
        "family": "Flux.2 Klein",
        "capabilities": {"editable": True, "i2i": True, "nsfw": False, "t2i": True},
        "runtime_engine": "FluxSD",
        "runtime_config": {"base_size": 1024, "base_steps": 5},
        "service_status": "active",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {
            "civitai_model_id": None,
            "download_url": None,
            "model_url": None,
            "repo_id": None,
        },
        "thumb": "atzoss/models/b1c49da5e785dd54e3ad94a83ffaf48b.jpeg",
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "Qwen3-text-Nunchaku": {
        "model_key": "99ea96f3d506ab824197bdfa91e770a0",
        "name": "Qwen3-text-Nunchaku",
        "modality": "image",
        "asset_type": "checkpoint",
        "family": "",
        "capabilities": {"editable": False, "nsfw": False},
        "runtime_engine": "",
        "runtime_config": {},
        "service_status": "inactive",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {},
        "thumb": None,
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "RMBG-2.0": {
        "model_key": "4b7f02b2b0a99a0fafbaa748ee3cbc32",
        "name": "RMBG-2.0",
        "modality": "image",
        "asset_type": "checkpoint",
        "family": "rmbg",
        "capabilities": {"editable": False, "nsfw": False, "rmbg": True},
        "runtime_engine": "",
        "runtime_config": {},
        "service_status": "inactive",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {},
        "thumb": None,
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "GLM-OCR": {
        "model_key": "d62ec2d9ae10ab4a60afab8c68269e32",
        "name": "ZhipuAI/GLM-OCR",
        "modality": "mini",
        "asset_type": "checkpoint",
        "family": "GLM-OCR",
        "capabilities": {"editable": False, "mini": True, "nsfw": False},
        "runtime_engine": "FluxSD",
        "runtime_config": {},
        "service_status": "active",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {"legacy_models_id": "7a970e89af474649a0a92d1d60374cd5"},
        "thumb": "models/202604/597572a653a7458fb2fbe922455ae210.webp",
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
    "roop": {
        "model_key": "c182a2c3f2fa3625d27337be5010b81f",
        "name": "roop",
        "modality": "image",
        "asset_type": "checkpoint",
        "family": "",
        "capabilities": {"editable": False, "nsfw": False},
        "runtime_engine": "",
        "runtime_config": {},
        "service_status": "inactive",
        "storage_mode": "local",
        "download_status": "pending",
        "source": {},
        "thumb": None,
        "tags": [],
        "trigger_words": [],
        "description": None,
    },
}


ENV_PATH = REPO_ROOT / ".env"


def _resolve_repo_relative_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _resolve_inference_dir(key: str, default: str) -> Path:
    inference_yaml = resolve_inference_yaml_path()
    if inference_yaml is None:
        return _resolve_repo_relative_dir(default)
    with inference_yaml.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    raw = str(cfg.get(key) or default).strip() or default
    return _resolve_repo_relative_dir(raw)


def resolve_models_dir() -> Path:
    return _resolve_inference_dir("models_dir", "resources/models")


def resolve_weights_dir() -> Path:
    return _resolve_inference_dir("weights_dir", "resources/weights")


def models_for_categories(categories: set[str]) -> list[InitialModelSpec]:
    return [spec for spec in INITIAL_MODELS if spec.category in categories]


def dir_has_content(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return False


def prompt_existing_action(locale: str, path: Path) -> ExistingAction:
    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    print(t("initial_models.prompt.existing_dir", locale, path=rel))
    print(t("initial_models.prompt.existing_skip", locale))
    print(t("initial_models.prompt.existing_redownload", locale))
    print(t("initial_models.prompt.existing_abort", locale))
    while True:
        raw = input("> ").strip().lower()
        if raw in ("", "1", "s", "skip", "y", "yes"):
            return "skip"
        if raw in ("2", "r", "redownload", "re"):
            return "redownload"
        if raw in ("3", "a", "abort", "q", "quit"):
            return "abort"
        print(t("initial_models.error.invalid_choice", locale))


def prompt_categories(locale: str) -> set[str]:
    print(t("initial_models.prompt.select_category", locale))
    labels = {
        "llm": t("initial_models.category.llm", locale, size=CATEGORY_SIZE_HINTS["llm"]),
        "audio": t("initial_models.category.audio", locale, size=CATEGORY_SIZE_HINTS["audio"]),
        "video": t("initial_models.category.video", locale, size=CATEGORY_SIZE_HINTS["video"]),
        "image": t("initial_models.category.image", locale, size=CATEGORY_SIZE_HINTS["image"]),
        "basic": t("initial_models.category.basic", locale, size=CATEGORY_SIZE_HINTS["basic"]),
    }
    for idx, category_id in enumerate(CATEGORY_IDS, start=1):
        print(f"  [{idx}] {labels[category_id]}")
    print(t("initial_models.prompt.select_all", locale))
    raw = input("> ").strip().lower()
    if not raw or raw in {"a", "all", "*"}:
        return set(CATEGORY_IDS)
    selected: set[str] = set()
    name_map = {str(i): cid for i, cid in enumerate(CATEGORY_IDS, start=1)}
    tokens = [token.strip() for token in raw.replace(" ", ",").split(",") if token.strip()]
    for token in tokens:
        if token in name_map:
            selected.add(name_map[token])
        elif token in CATEGORY_IDS:
            selected.add(token)
        elif token == "all":
            return set(CATEGORY_IDS)
        else:
            raise SystemExit(t("initial_models.error.unknown_category", locale, category=token))
    if not selected:
        raise SystemExit(t("initial_models.error.no_category_selected", locale))
    return selected


def _catalog_seed(spec: InitialModelSpec) -> dict[str, Any]:
    seed = MODEL_CATALOG_SEED.get(spec.load_name)
    if seed is None:
        raise KeyError(f"missing MODEL_CATALOG_SEED for {spec.load_name}")
    return seed


def _source_for_download(
    spec: InitialModelSpec,
    model_source: ModelSource,
    seed: dict[str, Any],
) -> dict[str, Any]:
    provider = "modelscope" if model_source == "modelscope" else "huggingface"
    repo_id = spec.ms_repo_id if model_source == "modelscope" else spec.hf_repo_id
    merged = dict(seed.get("source") or {})
    merged["provider"] = provider
    merged["repo_id"] = repo_id
    return merged


def register_downloaded_model(
    spec: InitialModelSpec,
    model_source: ModelSource,
    locale: str,
) -> bool:
    global CACHED_MODEL_CATALOG_DB_PATH
    seed = _catalog_seed(spec)
    source = _source_for_download(spec, model_source, seed)

    try:
        if CACHED_MODEL_CATALOG_DB_PATH is None:
            env = parse_env_file(ENV_PATH)
            CACHED_MODEL_CATALOG_DB_PATH = resolve_model_catalog_db_path(env)
        db_path = CACHED_MODEL_CATALOG_DB_PATH
        upsert_model_catalog(
            db_path=db_path,
            load_name=spec.load_name,
            seed=seed,
            source=source,
        )
        return True
    except (ModelCatalogDbNotFoundError, sqlite3.Error) as exc:
        print(
            t("initial_models.error.catalog_read_write_failed", locale, error=exc),
            file=sys.stderr,
        )
        return False


def resolve_writable_model_catalog_db(locale: str) -> Path | None:
    global CACHED_MODEL_CATALOG_DB_PATH
    try:
        if CACHED_MODEL_CATALOG_DB_PATH is None:
            env = parse_env_file(ENV_PATH)
            CACHED_MODEL_CATALOG_DB_PATH = resolve_model_catalog_db_path(env)
        ensure_model_catalog_writable(CACHED_MODEL_CATALOG_DB_PATH)
        return CACHED_MODEL_CATALOG_DB_PATH
    except (ModelCatalogDbNotFoundError, sqlite3.Error) as exc:
        print(
            t("initial_models.error.catalog_read_write_failed", locale, error=exc),
            file=sys.stderr,
        )
        print(t("initial_models.hint.catalog_permission", locale), file=sys.stderr)
        return None


def download_one_model(
    locale: str,
    region: str,
    models_dir: Path,
    weights_dir: Path,
    spec: InitialModelSpec,
) -> bool:
    base_dir = weights_dir if spec.install_root == "weights" else models_dir
    local_dir = base_dir / spec.load_name
    model_source = region_default_model_source(region, locale)

    if dir_has_content(local_dir):
        action = prompt_existing_action(locale, local_dir)
        if action == "abort":
            raise SystemExit(t("initial_models.status.aborted", locale))
        if action == "skip":
            print(t("initial_models.status.skip_existing", locale, path=local_dir.name))
            if register_downloaded_model(spec, model_source, locale):
                print(t("initial_models.status.catalog_updated", locale, name=spec.load_name))
            else:
                print(
                    t("initial_models.warn.catalog_not_updated", locale, name=spec.load_name),
                    file=sys.stderr,
                )
            return True
        print(t("initial_models.status.resume_existing", locale, path=local_dir.name))

    print(t("initial_models.status.download_start", locale, name=spec.load_name))
    repo_spec: RepoDownload = repo_download_full(
        hf_repo_id=spec.hf_repo_id,
        ms_repo_id=spec.ms_repo_id,
        local_dir=local_dir,
    )
    download_repo_with_fallback(
        locale,
        region,
        repo_spec,
        primary_source=model_source,
        allow_fallback=True,
    )

    if not dir_has_content(local_dir):
        raise RuntimeError(f"download finished but directory is empty: {local_dir}")

    if not register_downloaded_model(spec, model_source, locale):
        print(
            t("initial_models.warn.catalog_not_updated", locale, name=spec.load_name),
            file=sys.stderr,
        )
    else:
        print(t("initial_models.status.catalog_updated", locale, name=spec.load_name))
    print(t("initial_models.status.download_done", locale, name=spec.load_name))
    return True


def run_initial_model_download(locale: str, region: str, categories: set[str]) -> int:
    models_dir = resolve_models_dir()
    weights_dir = resolve_weights_dir()
    print(t("initial_models.status.models_dir", locale, path=str(models_dir)))
    print(t("initial_models.status.weights_dir", locale, path=str(weights_dir)))
    specs = models_for_categories(categories)
    if not specs:
        print(t("initial_models.error.no_models_for_selection", locale))
        return 1

    print(t("initial_models.status.plan", locale, count=len(specs)))
    for spec in specs:
        print(f"  - {spec.load_name}")

    raw = input(t("initial_models.prompt.confirm_download", locale)).strip().lower()
    if raw in ("n", "no"):
        print(t("initial_models.status.download_skipped", locale))
        return 0

    if resolve_writable_model_catalog_db(locale) is None:
        return 1

    failures = 0
    for spec in specs:
        try:
            download_one_model(locale, region, models_dir, weights_dir, spec)
        except SystemExit:
            raise
        except Exception as exc:
            failures += 1
            print(
                t("initial_models.error.download_failed", locale, name=spec.load_name, error=exc),
                file=sys.stderr,
            )

    if failures:
        print(t("initial_models.status.done_with_errors", locale, failures=failures))
        return 1
    print(t("initial_models.done", locale))
    return 0
