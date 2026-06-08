"""Synchronous HuggingFace / ModelScope download helpers for setup scripts."""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from backend.i18n.translator import t
from vitoom_setup import REPO_ROOT

ModelSource = Literal["huggingface", "modelscope"]
PROBE_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class RepoDownload:
    """Download a model repo into ``local_dir``.

    * ``include`` / ``hf_include`` / ``ms_include`` — file patterns passed to CLI
      ``--include`` (e.g. a single ``.safetensors`` or ``onnx/model.onnx``).
    * When all include fields are unset or empty for a source, the full repo is
      downloaded (no ``--include``), matching::

          modelscope download --model ID --local_dir PATH

    * ``verify`` — absolute paths that must exist after a successful download.
    * ``dest`` — if set, copy the first downloaded include path to this file
      (used when HF/MS use different filenames but the same ``local_dir``).
    """

    hf_repo_id: str
    ms_repo_id: str
    local_dir: Path
    include: tuple[str, ...] | None = None
    hf_include: tuple[str, ...] | None = None
    ms_include: tuple[str, ...] | None = None
    verify: tuple[Path, ...] = ()
    dest: Path | None = None


def load_dotenv_value(key: str) -> str | None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        env_key, value = line.split("=", 1)
        if env_key.strip() == key:
            return value.strip().strip("'\"")
    return None


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def cli_available(command: str) -> bool:
    return shutil.which(command) is not None


def _uses_socks_proxy(value: str | None) -> bool:
    return bool(value and value.strip().lower().startswith("socks"))


def socks_proxy_configured() -> bool:
    proxy_keys = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")
    for key in proxy_keys:
        if _uses_socks_proxy(os.environ.get(key)) or _uses_socks_proxy(load_dotenv_value(key)):
            return True
    return False


def pip_install_packages(locale: str, *packages: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    pip_index_url = os.environ.get("PIP_INDEX_URL") or load_dotenv_value("PIP_INDEX_URL")
    if pip_index_url:
        cmd.extend(["--index-url", pip_index_url])
    pip_trusted_host = os.environ.get("PIP_TRUSTED_HOST") or load_dotenv_value("PIP_TRUSTED_HOST")
    if pip_trusted_host:
        cmd.extend(["--trusted-host", pip_trusted_host])
    print(t("build_artifacts.status.installing_packages", locale, packages=" ".join(packages)))
    subprocess.run(cmd, check=True)


def ensure_huggingface_tooling(locale: str) -> None:
    packages: list[str] = []
    if not cli_available("hf") and not module_available("huggingface_hub"):
        packages.append("huggingface_hub")
    if socks_proxy_configured() and not module_available("socksio"):
        packages.append("socksio")
    if packages:
        pip_install_packages(locale, *packages)


def ensure_modelscope_tooling(locale: str) -> None:
    if cli_available("modelscope") or module_available("modelscope"):
        return
    pip_install_packages(locale, "modelscope")


def ensure_initial_download_dependencies(locale: str) -> None:
    """Ensure hub packages exist before initial model download."""
    ensure_huggingface_tooling(locale)
    ensure_modelscope_tooling(locale)


def ensure_model_source_tooling(locale: str, model_source: ModelSource) -> None:
    if model_source == "huggingface":
        ensure_huggingface_tooling(locale)
    else:
        ensure_modelscope_tooling(locale)


def probe_url(url: str, timeout: float = PROBE_TIMEOUT_SECONDS) -> bool:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": "vitoom-setup/1.0",
            "Range": "bytes=0-0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status in {200, 206} or 200 <= response.status < 400
    except urllib.error.HTTPError as exc:
        return exc.code in {200, 206, 301, 302, 307, 308, 403, 416}
    except OSError:
        return False


def region_default_model_source(region: str, _locale: str) -> ModelSource:
    if region == "cn":
        return "modelscope"
    return "huggingface"


def alternate_model_source(source: ModelSource) -> ModelSource:
    return "modelscope" if source == "huggingface" else "huggingface"


def probe_huggingface(repo_id: str, path: str) -> bool:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{path}"
    return probe_url(url)


def probe_modelscope(repo_id: str, path: str) -> bool:
    url = (
        "https://www.modelscope.cn/api/v1/models/"
        f"{repo_id}/repo?Revision=master&FilePath={path}"
    )
    return probe_url(url)


def choose_model_source(repo_id: str, region: str, locale: str) -> ModelSource:
    prefer = region_default_model_source(region, locale)
    if prefer == "huggingface":
        hf_ok = probe_huggingface(repo_id, "tokenizer.json")
        ms_ok = probe_modelscope(repo_id, "tokenizer.json")
        if hf_ok and not ms_ok:
            return "huggingface"
        if ms_ok and not hf_ok:
            return "modelscope"
        if hf_ok and ms_ok:
            hf_start = time.perf_counter()
            hf_model_ok = probe_huggingface(repo_id, "onnx/model_quantized.onnx")
            hf_elapsed = time.perf_counter() - hf_start
            ms_start = time.perf_counter()
            ms_model_ok = probe_modelscope(repo_id, "onnx/model_int8.onnx")
            ms_elapsed = time.perf_counter() - ms_start
            if hf_model_ok and ms_model_ok:
                return "modelscope" if ms_elapsed < hf_elapsed else "huggingface"
            if ms_model_ok:
                return "modelscope"
            if hf_model_ok:
                return "huggingface"
        return "huggingface"

    hf_ok = probe_huggingface(repo_id, "tokenizer.json")
    ms_ok = probe_modelscope(repo_id, "tokenizer.json")
    if hf_ok and not ms_ok:
        return "huggingface"
    if ms_ok and not hf_ok:
        return "modelscope"
    if hf_ok and ms_ok:
        hf_start = time.perf_counter()
        hf_model_ok = probe_huggingface(repo_id, "onnx/model_quantized.onnx")
        hf_elapsed = time.perf_counter() - hf_start
        ms_start = time.perf_counter()
        ms_model_ok = probe_modelscope(repo_id, "onnx/model_int8.onnx")
        ms_elapsed = time.perf_counter() - ms_start
        if hf_model_ok and ms_model_ok:
            return "modelscope" if ms_elapsed < hf_elapsed else "huggingface"
        if ms_model_ok:
            return "modelscope"
        if hf_model_ok:
            return "huggingface"
        return "modelscope"
    fallback = region_default_model_source(region, locale)
    print(
        t(
            "build_artifacts.status.probe_inconclusive",
            locale,
            repo_id=repo_id,
            source=fallback,
        )
    )
    return fallback


def download_http(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vitoom-setup/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp_path.replace(dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def file_exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def includes_for_source(spec: RepoDownload, source: ModelSource) -> tuple[str, ...]:
    if source == "modelscope" and spec.ms_include is not None:
        return spec.ms_include
    if source == "huggingface" and spec.hf_include is not None:
        return spec.hf_include
    return spec.include or ()


def _run_cli(cmd: list[str], locale: str, label: str) -> None:
    shell_cmd = " ".join(shlex.quote(part) for part in cmd)
    print(t("model_hub.status.cli_exec", locale, label=label, cmd=shell_cmd))
    subprocess.run(cmd, check=True)


def _build_hf_cli_cmd(repo_id: str, local_dir: Path, includes: tuple[str, ...]) -> list[str]:
    cmd = ["hf", "download", repo_id, "--local-dir", str(local_dir)]
    for pattern in includes:
        cmd.extend(["--include", pattern])
    return cmd


def _build_ms_cli_cmd(repo_id: str, local_dir: Path, includes: tuple[str, ...]) -> list[str]:
    cmd = ["modelscope", "download", "--model", repo_id, "--local_dir", str(local_dir)]
    for pattern in includes:
        cmd.extend(["--include", pattern])
    return cmd


def _download_hf_sdk(locale: str, repo_id: str, local_dir: Path, includes: tuple[str, ...]) -> None:
    from huggingface_hub import snapshot_download

    print(t("model_hub.status.sdk_hf", locale, repo_id=repo_id))
    kwargs: dict[str, object] = {"repo_id": repo_id, "local_dir": str(local_dir)}
    if includes:
        kwargs["allow_patterns"] = list(includes)
    snapshot_download(**kwargs)


def _download_ms_sdk(locale: str, repo_id: str, local_dir: Path, includes: tuple[str, ...]) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    print(t("model_hub.status.sdk_ms", locale, repo_id=repo_id))
    kwargs: dict[str, object] = {"model_id": repo_id, "cache_dir": str(local_dir)}
    if includes:
        kwargs["allow_patterns"] = list(includes)
    snapshot_download(**kwargs)


def _finalize_dest(spec: RepoDownload, source: ModelSource) -> None:
    if spec.dest is None:
        return
    includes = includes_for_source(spec, source)
    if not includes:
        return
    src = (spec.local_dir / includes[0]).resolve()
    dest = spec.dest.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"expected download at {src}")
    if src == dest:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _check_verify(spec: RepoDownload) -> None:
    missing = [path for path in spec.verify if not file_exists(path)]
    if missing:
        raise FileNotFoundError(
            "missing after download: " + ", ".join(str(path) for path in missing)
        )


def download_repo(locale: str, source: ModelSource, spec: RepoDownload) -> None:
    """Download via ``hf download`` / ``modelscope download`` (with optional ``--include``)."""
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    includes = includes_for_source(spec, source)
    repo_id = spec.ms_repo_id if source == "modelscope" else spec.hf_repo_id

    if source == "modelscope":
        ensure_modelscope_tooling(locale)
        if includes:
            print(
                t(
                    "model_hub.status.download_partial",
                    locale,
                    source="modelscope",
                    patterns=", ".join(includes),
                )
            )
        else:
            print(t("model_hub.status.download_full", locale, source="modelscope", repo_id=repo_id))
        if cli_available("modelscope"):
            _run_cli(_build_ms_cli_cmd(repo_id, spec.local_dir, includes), locale, "modelscope")
        else:
            _download_ms_sdk(locale, repo_id, spec.local_dir, includes)
    else:
        ensure_huggingface_tooling(locale)
        if includes:
            print(
                t(
                    "model_hub.status.download_partial",
                    locale,
                    source="huggingface",
                    patterns=", ".join(includes),
                )
            )
        else:
            print(
                t("model_hub.status.download_full", locale, source="huggingface", repo_id=repo_id)
            )
        if cli_available("hf"):
            _run_cli(_build_hf_cli_cmd(repo_id, spec.local_dir, includes), locale, "huggingface")
        else:
            _download_hf_sdk(locale, repo_id, spec.local_dir, includes)

    _finalize_dest(spec, source)
    _check_verify(spec)


def download_repo_with_fallback(
    locale: str,
    region: str,
    spec: RepoDownload,
    *,
    primary_source: ModelSource | None = None,
    allow_fallback: bool = True,
) -> None:
    prefer = primary_source or region_default_model_source(region, locale)
    sources: list[ModelSource] = [prefer]
    if allow_fallback:
        alternate = alternate_model_source(prefer)
        if alternate not in sources:
            sources.append(alternate)

    errors: list[str] = []
    for index, source in enumerate(sources):
        if index > 0:
            print(
                t(
                    "build_artifacts.status.trying_fallback_source",
                    locale,
                    source=source,
                    error=errors[-1],
                )
            )
        print(t("build_artifacts.status.model_source", locale, source=source))
        try:
            download_repo(locale, source, spec)
            return
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    repo_label = spec.ms_repo_id or spec.hf_repo_id
    raise RuntimeError(
        t(
            "build_artifacts.error.model_download_failed",
            locale,
            repo_id=repo_label,
            details="; ".join(errors),
        )
    )


def prefer_source_for_probe(
    region: str,
    locale: str,
    *,
    hf_repo_id: str,
    ms_repo_id: str,
    probe_path: str,
) -> ModelSource:
    prefer = region_default_model_source(region, locale)
    if prefer == "modelscope" and not probe_modelscope(ms_repo_id, probe_path):
        if probe_huggingface(hf_repo_id, probe_path):
            return "huggingface"
    elif prefer == "huggingface" and not probe_huggingface(hf_repo_id, probe_path):
        if probe_modelscope(ms_repo_id, probe_path):
            return "modelscope"
    return prefer


def repo_download_for_file(
    *,
    hf_repo_id: str,
    ms_repo_id: str,
    local_dir: Path,
    hf_remote_path: str,
    ms_remote_path: str,
    dest: Path,
) -> RepoDownload:
    """Build a partial-download spec; copies the hub file into ``dest`` when names differ."""
    return RepoDownload(
        hf_repo_id=hf_repo_id,
        ms_repo_id=ms_repo_id,
        local_dir=local_dir,
        hf_include=(hf_remote_path,),
        ms_include=(ms_remote_path,),
        verify=(dest,),
        dest=dest,
    )


def repo_download_full(
    *,
    hf_repo_id: str,
    ms_repo_id: str,
    local_dir: Path,
    verify: tuple[Path, ...] = (),
) -> RepoDownload:
    """Build a full-repo download spec (no ``--include``)."""
    return RepoDownload(
        hf_repo_id=hf_repo_id,
        ms_repo_id=ms_repo_id,
        local_dir=local_dir,
        verify=verify,
    )
