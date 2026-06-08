"""Download local artifacts required for Vitoom Docker image builds."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

from backend.i18n.translator import t
from vitoom_setup import REPO_ROOT
from vitoom_setup.constants import BUILD_COMPONENT_IDS
from vitoom_setup.model_hub import (
    RepoDownload,
    choose_model_source,
    download_http,
    download_repo_with_fallback,
    ensure_model_source_tooling,
    file_exists,
    load_dotenv_value,
    prefer_source_for_probe,
    repo_download_for_file,
)

MANIFEST_PATH = REPO_ROOT / "docker" / "build-artifacts.manifest.json"


def display_path(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


def detect_arch(locale: str) -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    raise SystemExit(t("build_artifacts.error.unrecognized_arch", locale, arch=platform.machine()))


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def format_template(template: str, context: dict[str, str]) -> str:
    return template.format(**context)


def build_context(manifest: dict[str, Any], arch: str, wheel_base_url: str) -> dict[str, str]:
    defaults = manifest["defaults"]
    arch_info = manifest["architectures"][arch]
    return {
        "arch": arch,
        "wheel_base_url": wheel_base_url.rstrip("/"),
        "pandoc_version": defaults["pandoc_version"],
        "elasticsearch_version": defaults["elasticsearch_version"],
        "es_arch": arch_info["es_arch"],
        "pandoc_deb_arch": arch_info["pandoc_deb_arch"],
        "e5_repo_id": defaults["e5_repo_id"],
    }


def resolve_dest(artifact: dict[str, Any], arch: str, context: dict[str, str]) -> str:
    if arch == "aarch64" and "dest_aarch64" in artifact:
        template = artifact["dest_aarch64"]
    else:
        template = artifact["dest"]
    return format_template(template, context)


def resolve_url(artifact: dict[str, Any], arch: str, context: dict[str, str]) -> str | None:
    key = "url_aarch64" if arch == "aarch64" else "url_x86_64"
    if key in artifact:
        return format_template(artifact[key], context)
    if "url" in artifact:
        return format_template(artifact["url"], context)
    return None


def collect_artifacts(
    manifest: dict[str, Any],
    components: set[str],
    arch: str,
    context: dict[str, str],
) -> list[tuple[str, dict[str, Any], Path, str | None]]:
    items: list[tuple[str, dict[str, Any], Path, str | None]] = []
    seen_dests: set[Path] = set()

    def add_artifacts(component_id: str, artifacts: list[dict[str, Any]]) -> None:
        for artifact in artifacts:
            dest = REPO_ROOT / resolve_dest(artifact, arch, context)
            if dest in seen_dests:
                continue
            seen_dests.add(dest)
            url = resolve_url(artifact, arch, context)
            items.append((component_id, artifact, dest, url))

    for component_id in sorted(components):
        if component_id == "mini":
            continue
        component = manifest["components"][component_id]
        add_artifacts(component_id, component.get("artifacts", []))

    if "mini" in components and "visual" not in components:
        visual_artifacts = manifest["components"]["visual"]["artifacts"]
        flash_attn = next(item for item in visual_artifacts if item["id"] == "flash_attn")
        add_artifacts("mini", [flash_attn])

    return items


def describe_components(components: set[str], locale: str) -> str:
    parts: list[str] = []
    for component_id in sorted(components):
        if component_id == "mini" and "visual" not in components:
            parts.append(t("build_artifacts.component.mini_flash_attn", locale))
        else:
            parts.append(t(f"build_artifacts.component.{component_id}", locale))
    return ", ".join(parts)


def download_artifact(
    locale: str,
    artifact: dict[str, Any],
    dest: Path,
    url: str | None,
    context: dict[str, str],
    model_source: str | None,
    region: str,
) -> None:
    if file_exists(dest):
        print(
            t(
                "build_artifacts.status.skip_existing",
                locale,
                path=dest.relative_to(REPO_ROOT),
            )
        )
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    artifact_type = artifact.get("type")

    if artifact_type in {"e5_model", "e5_file"}:
        repo_id = context["e5_repo_id"]
        if model_source is None:
            raise RuntimeError("model_source is required for E5 artifacts")
        remote_hf = artifact["huggingface_path"]
        remote_ms = artifact["modelscope_path"]
        local_dir = dest.parent
        if "/" in remote_hf:
            local_dir = dest.parent.parent
        spec = repo_download_for_file(
            hf_repo_id=repo_id,
            ms_repo_id=repo_id,
            local_dir=local_dir,
            hf_remote_path=remote_hf,
            ms_remote_path=remote_ms,
            dest=dest,
        )
        download_repo_with_fallback(
            locale,
            region,
            spec,
            primary_source=model_source,  # type: ignore[arg-type]
            allow_fallback=True,
        )
        return

    if not url:
        raise RuntimeError(
            t(
                "build_artifacts.error.missing_artifact_url",
                locale,
                artifact_id=artifact.get("id", "artifact"),
            )
        )
    print(t("build_artifacts.status.download_http", locale, url=url))
    download_http(url, dest)


def prompt_build_components(locale: str) -> set[str]:
    print(t("build_artifacts.prompt.select_components", locale))
    for idx, component_id in enumerate(BUILD_COMPONENT_IDS, start=1):
        label = t(f"build_artifacts.component.{component_id}", locale)
        print(f"  [{idx}] {component_id} ({label})")
    print(t("build_artifacts.prompt.select_all", locale))
    raw = input("> ").strip().lower()
    if not raw or raw in {"a", "all", "*"}:
        return set(BUILD_COMPONENT_IDS)
    selected: set[str] = set()
    tokens = [token.strip() for token in raw.replace(" ", ",").split(",") if token.strip()]
    name_map = {str(i): cid for i, cid in enumerate(BUILD_COMPONENT_IDS, start=1)}
    for token in tokens:
        if token in name_map:
            selected.add(name_map[token])
        elif token in BUILD_COMPONENT_IDS:
            selected.add(token)
        else:
            raise SystemExit(t("build_artifacts.error.unknown_component", locale, component=token))
    if not selected:
        raise SystemExit(t("build_artifacts.error.no_component_selected", locale))
    return selected


def run_build_download(
    locale: str,
    *,
    arch: str,
    components: set[str],
    region: str,
    wheel_base_url: str | None = None,
    env_values: dict[str, str] | None = None,
    skip_confirm: bool = False,
) -> int:
    manifest = load_manifest()
    if arch not in manifest["architectures"]:
        raise SystemExit(t("build_artifacts.error.unsupported_manifest_arch", locale, arch=arch))

    env_values = env_values or {}
    base = (
        wheel_base_url
        or env_values.get("VITOOM_WHEEL_BASE_URL")
        or load_dotenv_value("VITOOM_WHEEL_BASE_URL")
        or manifest["defaults"]["wheel_base_url"]
    )
    context = build_context(manifest, arch, base)
    items = collect_artifacts(manifest, components, arch, context)

    missing_items = [item for item in items if not file_exists(item[2])]
    existing_count = len(items) - len(missing_items)
    for _, _, dest, _ in items:
        if file_exists(dest) and dest not in {m[2] for m in missing_items}:
            print(
                t(
                    "build_artifacts.status.skip_existing",
                    locale,
                    path=dest.relative_to(REPO_ROOT),
                )
            )

    if not missing_items:
        print(t("build_artifacts.status.all_ready", locale))
        return 0

    print(
        t(
            "build_artifacts.status.download_plan",
            locale,
            existing=existing_count,
            missing=len(missing_items),
        )
    )

    model_source: str | None = None
    needs_e5 = any(item[1].get("type", "").startswith("e5_") for item in missing_items)
    if needs_e5:
        model_source = choose_model_source(context["e5_repo_id"], region, locale)
        print(t("build_artifacts.status.e5_model_source", locale, source=model_source))
        ensure_model_source_tooling(locale, model_source)  # type: ignore[arg-type]

    print(t("build_artifacts.status.target_arch", locale, arch=arch))
    print(t("build_artifacts.status.components", locale, components=describe_components(components, locale)))
    print(t("build_artifacts.status.wheel_base_url", locale, url=base))
    if not skip_confirm:
        confirm = input(t("build_artifacts.prompt.confirm_download", locale)).strip().lower()
        if confirm not in ("", "y", "yes"):
            print(t("build_artifacts.status.cancelled", locale))
            return 0

    for component_id, artifact, dest, url in missing_items:
        rel_dest = dest.relative_to(REPO_ROOT)
        print(
            t(
                "build_artifacts.status.artifact_line",
                locale,
                component=component_id,
                artifact_id=artifact.get("id", "artifact"),
                path=rel_dest,
            )
        )
        try:
            download_artifact(locale, artifact, dest, url, context, model_source, region)
        except Exception as exc:
            print(t("build_artifacts.error.download_failed", locale, error=exc), file=sys.stderr)
            return 1

    print(t("build_artifacts.status.done", locale))
    return 0


# ---------------------------------------------------------------------------
# Mini runtime model (DocLayout-YOLO; lives under resources/models, not docker/)
# ---------------------------------------------------------------------------

DOCLAYOUT_DIR_NAME = "DocLayout_YOLO_DocStructBench_imgsz1280_2501"
DOCLAYOUT_WEIGHT_FILE = "doclayout_yolo_docstructbench_imgsz1280_2501.pt"
DOCLAYOUT_HF_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
DOCLAYOUT_MS_REPO_ID = "JulioZhao97/DocLayout_YOLO_DocStructBench_imgsz1280_2501"


def resolve_models_dir(env: dict[str, str]) -> Path:
    raw = env.get("VITOOM_MODELS_HOST_DIR", "./resources/models").strip() or "./resources/models"
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def doclayout_weight_path(env: dict[str, str]) -> Path:
    return resolve_models_dir(env) / DOCLAYOUT_DIR_NAME / DOCLAYOUT_WEIGHT_FILE


def download_doclayout_weight(locale: str, region: str, env: dict[str, str]) -> int:
    dest = doclayout_weight_path(env)
    local_dir = dest.parent
    if file_exists(dest):
        print(
            t(
                "setup.mini_model.skip_existing",
                locale,
                path=display_path(dest),
            )
        )
        return 0

    prefer = prefer_source_for_probe(
        region,
        locale,
        hf_repo_id=DOCLAYOUT_HF_REPO_ID,
        ms_repo_id=DOCLAYOUT_MS_REPO_ID,
        probe_path=DOCLAYOUT_WEIGHT_FILE,
    )

    print(t("setup.mini_model.download_start", locale, path=display_path(dest)))

    spec = RepoDownload(
        hf_repo_id=DOCLAYOUT_HF_REPO_ID,
        ms_repo_id=DOCLAYOUT_MS_REPO_ID,
        local_dir=local_dir,
        include=(DOCLAYOUT_WEIGHT_FILE,),
        verify=(dest,),
        dest=dest,
    )

    try:
        download_repo_with_fallback(
            locale,
            region,
            spec,
            primary_source=prefer,  # type: ignore[arg-type]
            allow_fallback=True,
        )
        print(t("setup.mini_model.download_done", locale))
        return 0
    except Exception as exc:
        print(
            t("setup.mini_model.download_failed", locale, details=str(exc)),
            file=sys.stderr,
        )
        return 1


def maybe_download_mini_model(
    locale: str,
    region: str,
    env: dict[str, str],
    selected: set[str],
) -> None:
    if "mini" not in selected:
        return

    dest = doclayout_weight_path(env)
    if file_exists(dest):
        print(
            t(
                "setup.mini_model.skip_existing",
                locale,
                path=display_path(dest),
            )
        )
        return

    print(t("setup.prompt.download_mini_model", locale))
    raw = input("> ").strip().lower()
    if raw in ("n", "no"):
        print(t("setup.status.mini_model_skipped", locale))
        return

    exit_code = download_doclayout_weight(locale, region, env)
    if exit_code != 0:
        raise SystemExit(exit_code)
