from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import urlparse

from common.config_loader import InferenceConfig
from common.pipeline_cache import PipelineCache
from common.result_handler import ResultHandler
from schemas import InferenceRequestParams

from translate.runtime.runtime_resolver import merge_translate_runtime_cfg, resolve_translate_backend
from translate.runtime.translategemma_bridge import (
    TranslateGemmaPolicy,
    build_translate_messages,
    load_translategemma_bundle,
    normalize_model_ref,
)


def _pick_lang(extract: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(extract.get(key) or "").strip()
        if value:
            return value
    return ""


class TranslateGemmaHandler:
    def __init__(
        self,
        *,
        inference_config: InferenceConfig,
        bundle_cache: PipelineCache,
        result_handler: ResultHandler,
        service_id: str,
        logger,
        run_blocking: Callable[..., Awaitable[Any]],
        check_cancelled: Callable[[str], bool],
        service_model_cfg: Dict[str, Any],
    ) -> None:
        self.inference_config = inference_config
        self.bundle_cache = bundle_cache
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.run_blocking = run_blocking
        self.check_cancelled = check_cancelled
        self.service_cfg = service_model_cfg or {}

    async def handle(self, params: InferenceRequestParams) -> None:
        task_id = params.task_id
        load_name = str(getattr(params, "load_name", "") or "").strip()
        if not load_name:
            raise ValueError("translate task requires load_name")

        extract_raw = getattr(params, "extract", None) or {}
        extract: Dict[str, Any] = dict(extract_raw) if isinstance(extract_raw, dict) else {}
        source_lang = _pick_lang(extract, "source_lang", "source_lang_code")
        target_lang = _pick_lang(extract, "target_lang", "target_lang_code")
        if not source_lang or not target_lang:
            raise ValueError(
                "translate task requires extract.source_lang and extract.target_lang "
                "(aliases: source_lang_code / target_lang_code)"
            )

        prompt_text = str(getattr(params, "prompt", "") or "").strip()
        tpl_list = getattr(params, "tpl_list", None) or []
        image_sources = [str(item).strip() for item in tpl_list if str(item or "").strip()]

        if not prompt_text and not image_sources:
            raise ValueError("translate task requires prompt (text) or tpl_list (image)")

        runtime_name = resolve_translate_backend(self.service_cfg)
        svc_runtime = merge_translate_runtime_cfg(self._service_runtime(), backend=runtime_name)
        policy = TranslateGemmaPolicy(
            dtype=str(svc_runtime.get("dtype", "auto")),
            device_map=str(svc_runtime.get("device_map", "auto")),
            trust_remote_code=bool(svc_runtime.get("trust_remote_code", True)),
            temperature=float(svc_runtime.get("temperature", 0.0)),
            top_p=float(svc_runtime.get("top_p", 1.0)),
            max_new_tokens=int(svc_runtime.get("max_new_tokens", 768)),
            model_kwargs=dict(svc_runtime.get("model_kwargs") or {}),
        )
        models_dir = getattr(self.inference_config, "models_dir", None)
        model_ref = normalize_model_ref(load_name, models_dir=models_dir)
        cache_key = f"translate|runtime={runtime_name}|{model_ref}|{policy.cache_key}"

        async def _load() -> Any:
            return await self.run_blocking(load_translategemma_bundle, model_ref, policy)

        bundle, cache_hit = await self.bundle_cache.acquire(key=cache_key, create_fn=_load)
        self.logger.info(
            "[translate] bundle %s runtime=%s key=%s",
            "hit" if cache_hit else "load",
            runtime_name,
            cache_key,
        )

        temp_paths: list[Path] = []
        try:
            jobs: list[tuple[str, Optional[str], Optional[str]]] = []
            if prompt_text:
                jobs.append(("text", prompt_text, None))
            for src in image_sources:
                jobs.append(("image", None, src))

            total = len(jobs)
            for idx, (mode, text, image_src) in enumerate(jobs):
                if self.check_cancelled(task_id):
                    self.logger.info("[translate] task=%s cancelled before idx=%s", task_id, idx)
                    return None

                image_ref = image_src
                if mode == "image" and image_src:
                    parsed = urlparse(image_src)
                    if parsed.scheme in ("http", "https"):
                        local_path = await self.run_blocking(_download_to_tempfile, image_src)
                        temp_paths.append(local_path)
                        image_ref = local_path.as_uri()
                    else:
                        local_path = Path(image_src).expanduser()
                        if not local_path.is_absolute():
                            candidate = Path("resources") / image_src.lstrip("/")
                            if candidate.exists():
                                local_path = candidate.resolve()
                        if not local_path.exists():
                            raise FileNotFoundError(f"translate image input not found: {image_src}")
                        image_ref = local_path.resolve().as_uri()

                messages = build_translate_messages(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    text=text,
                    image_ref=image_ref if mode == "image" else None,
                )

                t0 = time.time()
                translated = await self.run_blocking(bundle.translate, messages)
                gen_time = time.time() - t0
                file_data = translated.encode("utf-8")
                params_for_send = params.model_copy(update={"file_type": "txt"})
                await self.result_handler.process_single_result(
                    file_data=file_data,
                    request_params=params_for_send,
                    generate_time=gen_time,
                    service_id=self.service_id,
                    index=idx,
                    total=total,
                    extra_message_fields={"content": translated},
                )
        finally:
            await self.bundle_cache.release_use(key=cache_key)
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

        return None

    def _service_runtime(self) -> Dict[str, Any]:
        runtime = self.service_cfg.get("runtime")
        return dict(runtime) if isinstance(runtime, dict) else {}


def _download_to_tempfile(url: str) -> Path:
    import requests  # type: ignore

    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    ext = _guess_ext(url, resp.headers.get("content-type"))
    fd, tmp_path = tempfile.mkstemp(prefix="translate_", suffix=ext)
    try:
        with os.fdopen(fd, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    handle.write(chunk)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
    return Path(tmp_path)


def _guess_ext(url: str, content_type: Optional[str]) -> str:
    path = urlparse(url).path or ""
    match = re.search(r"\.([A-Za-z0-9]{1,6})$", path)
    if match:
        ext = "." + match.group(1).lower()
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}:
            return ext
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct == "image/jpeg":
            return ".jpg"
        if ct == "image/png":
            return ".png"
        if ct == "image/webp":
            return ".webp"
    return ".bin"
