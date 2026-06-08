"""
GLM-OCR handler

输入协议（来自 InferenceRequestParams）：
    tpl_list    : List[str]       # 图片或 PDF 的 URL / 本地路径；支持多文件
    extract     : Optional[dict]
        task   : "text" | "table" | "formula" | "extract"  (默认 "text")
        schema : Optional[dict]    # task="extract" 必填
    load_name  : 必填（由 backend 查 models 表后回填；handler 自身不持有默认值）
    family : 必填（由 backend 查库时从 models.family 自动回填到 params.family）

输出协议：
- 每个输入文件 → 一条 result WS 消息（index/total 标记进度），同时：
    * 走 ResultHandler 落盘为 .md（text/table/formula）或 .json（extract）
    * 在 result 消息里内联 content 字段，携带完整文本

设计要点：
- PDF 优先交给 GLM-OCR 官方 SDK 的 PageLoader；如果环境缺失 SDK，再回退到 PyMuPDF
  渲染为 image 后多页调用模型。这样保证"模型天然支持 PDF"在工程上落地。
- 每次请求进入时通过 MiniInferrer 共享的 bundle_cache.acquire 拿到 bundle；
  不同 policy（模型/TP/memory）天然走不同 cache key，切换时旧 bundle 自动 shutdown。
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from common.pipeline_cache import PipelineCache
from common.config_loader import InferenceConfig
from common.result_handler import ResultHandler
from schemas import InferenceRequestParams

from mini.runtime.ocr_runtime import OcrBundleLike
from mini.runtime.runtime_resolver import merge_mini_ocr_runtime_cfg
from mini.runtime.hf_ocr_bridge import (
    HfOcrPolicy,
    load_hf_ocr_bundle,
    _normalize_model_ref as _hf_normalize_model_ref,
)
from mini.runtime.vllm_ocr_bridge import (
    VllmOcrPolicy,
    _normalize_model_ref as _vllm_normalize_model_ref,
    build_ocr_messages,
    load_vllm_ocr_bundle,
)
from mini.pipeline.doc_pipeline import DocPipelineConfig, build_doc_zip
from mini.pipeline.layout_bridge import DocLayoutDetector, build_default_layout_detector


# OCR runtime 后端枚举
_RUNTIME_HF = "transformers"
_RUNTIME_VLLM = "vllm"
_DEFAULT_RUNTIME = _RUNTIME_HF


# ---------------------------------------------------------------------------
# 常量：不同 OCR 任务的 prompt 模板
# 参考 GLM-OCR 官方 README 推荐 prompt：
# ---------------------------------------------------------------------------

_OCR_TEXT_PROMPT = "Text Recognition:"
_OCR_TABLE_PROMPT = "Table Recognition:"
_OCR_FORMULA_PROMPT = "Formula Recognition:"
_OCR_EXTRACT_PROMPT_TEMPLATE = (
    "Information Extraction:\n"
    "Extract structured fields from the image and return a JSON object that strictly follows the schema below.\n"
    "If a field is not present, use an empty string.\n"
    "\n"
    "Schema:\n{schema_json}\n"
    "\n"
    "Return only the JSON object, no explanation."
)


_SUPPORTED_TASKS = {"text", "table", "formula", "extract"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _OcrRunRequest:
    task: str                       # text / table / formula / extract
    prompt: str
    file_path: Path                 # 本地真实文件路径
    is_pdf: bool
    input_display: str              # 原始 URL 或路径，仅用于日志
    # text 模式下是否走"图文混排 zip"新路径（由 handler 根据 layout 配置 + 请求 file_type 决定）
    use_doc_zip: bool = False


class OcrHandler:
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
        ws_client: Any = None,
    ) -> None:
        self.inference_config = inference_config
        self.bundle_cache = bundle_cache
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.run_blocking = run_blocking
        self.check_cancelled = check_cancelled
        self.service_cfg = service_model_cfg or {}
        # 用于发 processing+progress 心跳给客户端，避免长 PDF/图册 OCR 期间
        # 客户端看不到进度直到 300s 超时（见 handler._run_doc_zip 里的 progress_cb）
        self.ws_client = ws_client

        # text 模式的版面分析器（doc_pipeline 用）。按需懒加载；失败会降级到旧 md 路径。
        self._layout_detector: Optional[DocLayoutDetector] = None
        self._layout_load_failed: bool = False

    # -----------------------------------------------------------------------
    # Entry
    # -----------------------------------------------------------------------

    async def handle(self, params: InferenceRequestParams) -> None:
        task_id = params.task_id

        extract_raw = getattr(params, "extract", None) or {}
        extract: Dict[str, Any] = dict(extract_raw) if isinstance(extract_raw, dict) else {}
        ocr_task = str(extract.get("task") or "text").strip().lower()
        if ocr_task not in _SUPPORTED_TASKS:
            raise ValueError(
                f"extract.task must be one of {sorted(_SUPPORTED_TASKS)}; got {ocr_task!r}"
            )

        schema: Optional[Dict[str, Any]] = None
        if ocr_task == "extract":
            schema_raw = extract.get("schema")
            if not isinstance(schema_raw, dict) or not schema_raw:
                raise ValueError("extract.schema is required (non-empty dict) when extract.task='extract'")
            schema = dict(schema_raw)

        tpl_list = list(getattr(params, "tpl_list", None) or [])
        if not tpl_list:
            # 兼容：允许用单字段 url
            single = getattr(params, "url", None)
            if single:
                tpl_list = [single]
        if not tpl_list:
            raise ValueError("tpl_list (or url) is required for OCR task")

        family = str(getattr(params, "family", "") or "").strip()
        load_name = str(getattr(params, "load_name", "") or "").strip()
        self.logger.info(
            "[mini-ocr] task=%s family=%s load_name=%s ocr_task=%s files=%d",
            task_id, family, load_name, ocr_task, len(tpl_list),
        )

        # text 模式的新路径（图文混排 zip）触发条件：
        #   1) extract.task == "text"
        #   2) 请求没有显式指定 file_type="md"（显式旁路，回退到旧纯文字输出）
        #   3) 本进程未出现过不可恢复的 layout 加载失败
        req_file_type = str(getattr(params, "file_type", "") or "").strip().lower()
        use_doc_zip_for_text = (
            ocr_task == "text"
            and req_file_type != "md"
            and not self._layout_load_failed
        )

        # 解析输入为本地文件，并根据扩展名决定是 PDF 还是 image；
        # PDF 统一先展开为逐页 image，一个 "请求 -> 一条结果消息" 的协议不变
        ocr_requests: List[_OcrRunRequest] = []
        base_prompt = self._build_prompt(ocr_task, schema)

        temp_dirs: List[Path] = []
        try:
            for src in tpl_list:
                local_path = await self._fetch_to_local(src)
                is_pdf = local_path.suffix.lower() == ".pdf"
                ocr_requests.append(_OcrRunRequest(
                    task=ocr_task,
                    prompt=base_prompt,
                    file_path=local_path,
                    is_pdf=is_pdf,
                    input_display=str(src),
                    use_doc_zip=use_doc_zip_for_text,
                ))

            # 取/建 bundle（LRU=1 + TTL）
            runtime_name, policy, model_ref = self._resolve_policy_and_model(params)
            cache_key = f"ocr|runtime={runtime_name}|{model_ref}|{policy.cache_key}"

            async def _load() -> OcrBundleLike:
                if runtime_name == _RUNTIME_VLLM:
                    return await self.run_blocking(load_vllm_ocr_bundle, model_ref, policy)
                return await self.run_blocking(load_hf_ocr_bundle, model_ref, policy)

            bundle, cache_hit = await self.bundle_cache.acquire(key=cache_key, create_fn=_load)
            self.logger.info(
                "[mini-ocr] bundle %s runtime=%s key=%s",
                "hit" if cache_hit else "load",
                runtime_name,
                cache_key,
            )

            try:
                total = len(ocr_requests)
                t_all_start = time.time()
                for idx, req in enumerate(ocr_requests):
                    if self.check_cancelled(task_id):
                        self.logger.info(f"[mini-ocr] task={task_id} cancelled before idx={idx}")
                        return None

                    t0 = time.time()
                    file_data, file_ext, content_inline = await self._run_single(
                        bundle, req, load_name=load_name, task_id=task_id,
                    )
                    gen_time = time.time() - t0

                    # 写入并通过 ResultHandler 广播（按每个输入文件一条 result 消息）
                    params_for_send = self._clone_request_params(params, file_type=file_ext)
                    extra = {"content": content_inline}
                    await self.result_handler.process_single_result(
                        file_data=file_data,
                        request_params=params_for_send,
                        generate_time=gen_time,
                        service_id=self.service_id,
                        index=idx,
                        total=total,
                        extra_message_fields=extra,
                    )

                self.logger.info(
                    f"[mini-ocr] task={task_id} done; total_items={total} elapsed={time.time()-t_all_start:.2f}s"
                )
            finally:
                await self.bundle_cache.release_use(key=cache_key)
        finally:
            for d in temp_dirs:
                try:
                    for p in d.glob("*"):
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    d.rmdir()
                except Exception:
                    pass

        return None

    # -----------------------------------------------------------------------
    # Prompt & output shaping
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_prompt(ocr_task: str, schema: Optional[Dict[str, Any]]) -> str:
        if ocr_task == "text":
            return _OCR_TEXT_PROMPT
        if ocr_task == "table":
            return _OCR_TABLE_PROMPT
        if ocr_task == "formula":
            return _OCR_FORMULA_PROMPT
        if ocr_task == "extract":
            schema_json = json.dumps(schema or {}, ensure_ascii=False, indent=2)
            return _OCR_EXTRACT_PROMPT_TEMPLATE.format(schema_json=schema_json)
        raise ValueError(f"unsupported ocr_task: {ocr_task}")

    @staticmethod
    def _shape_text_output(text: str, ocr_task: str) -> Tuple[Any, str, str]:
        """旧路径：text/table/formula/extract 返回字符串，统一成 (data, ext, inline)。"""
        text = text or ""
        if ocr_task == "extract":
            parsed = _try_parse_json(text)
            if parsed is None:
                payload = {"_parse_error": True, "raw": text}
            else:
                payload = parsed
            content_str = json.dumps(payload, ensure_ascii=False, indent=2)
            return content_str, "json", content_str

        # text/table/formula 统一 md 落盘；content 内联整段文本
        return text, "md", text

    # -----------------------------------------------------------------------
    # Layout detector (lazy + fail-safe)
    # -----------------------------------------------------------------------

    def _ensure_layout_detector(self) -> Optional[DocLayoutDetector]:
        """懒加载版面检测器；失败一次后不再重试，触发降级到旧 md 路径。

        权重位置固定（与 audio/image 推理器一致）：
            {models_dir}/DocLayout_YOLO_DocStructBench_imgsz1280_2501/doclayout_yolo_docstructbench_imgsz1280_2501.pt
            {weights_dir}/DocLayout_YOLO_DocStructBench_imgsz1280_2501/doclayout_yolo_docstructbench_imgsz1280_2501.pt
        """
        if self._layout_detector is not None:
            return self._layout_detector
        if self._layout_load_failed:
            return None

        models_dir = getattr(self.inference_config, "models_dir", None)
        weights_dir = getattr(self.inference_config, "weights_dir", None)

        try:
            detector = build_default_layout_detector(
                models_dir=models_dir,
                weights_dir=weights_dir,
            )
        except Exception as e:
            self.logger.warning(
                "[mini-ocr] layout detector init failed, will fall back to plain-text md: %s", e,
            )
            self._layout_load_failed = True
            return None

        self._layout_detector = detector
        return detector

    # -----------------------------------------------------------------------
    # Single input inference (supports pdf/image + plain-md / doc-zip paths)
    # -----------------------------------------------------------------------

    async def _run_single(
        self,
        bundle: OcrBundleLike,
        req: _OcrRunRequest,
        *,
        load_name: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[Any, str, str]:
        """返回 (file_data, file_ext, content_inline)。"""
        if req.use_doc_zip and req.task == "text":
            detector = self._ensure_layout_detector()
            if detector is not None:
                try:
                    return await self._run_doc_zip(
                        bundle, detector, req,
                        load_name=load_name,
                        task_id=task_id,
                    )
                except Exception as e:
                    self.logger.warning(
                        "[mini-ocr] doc-zip path failed, falling back to plain md: %s", e,
                        exc_info=True,
                    )
                    # 不再重试本请求的 detector，但不永久拉黑（下次请求还可再试）
            # 降级到旧路径

        if req.is_pdf:
            text = await self._run_pdf(bundle, req, task_id=task_id)
        else:
            text = await self._run_image(bundle, req)
        return self._shape_text_output(text, req.task)

    async def _run_doc_zip(
        self,
        bundle: OcrBundleLike,
        detector: DocLayoutDetector,
        req: _OcrRunRequest,
        *,
        load_name: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[Any, str, str]:
        """text 模式新路径：版面分析 + GLM-OCR + 打 zip。"""
        cfg = DocPipelineConfig()  # 采用固定默认（pdf_dpi/max_pages/title_font_ratio 等写死）

        # 把每页完成事件转成一条 WS task_status processing 心跳；让前端/客户端
        # 在 PDF 长耗时（可能 >5 分钟）期间仍能稳定看到进度，而不是卡在 "processing"
        # 然后直接 300s 超时。
        progress_cb = self._make_doc_progress_cb(task_id)

        def _do() -> Tuple[bytes, str]:
            return build_doc_zip(
                bundle=bundle,
                detector=detector,
                source_path=req.file_path,
                cfg=cfg,
                load_name=load_name,
                layout_backend="doclayout_yolo",
                progress_cb=progress_cb,
            )

        zip_bytes, md_inline = await self.run_blocking(_do)
        # 每份文档收尾：主动回收一次显存碎片。
        # 不动 bundle（由 PipelineCache TTL 负责驱逐），只清掉 layout/中间 tensor 的残留。
        await self.run_blocking(_release_doc_run_memory)
        return zip_bytes, "zip", md_inline

    def _make_doc_progress_cb(
        self,
        task_id: Optional[str],
    ) -> Optional[Callable[[int, int, float], None]]:
        """构造一个"每页完成心跳"回调，线程安全。

        - 需要 ws_client + task_id 才会生效，否则返回 None（doc_pipeline 会跳过）
        - 回调在 run_blocking 线程里被触发；通过 run_coroutine_threadsafe
          fire-and-forget 地调度回事件循环，避免阻塞 OCR 主线程
        - 任何异常（包括 loop 已关闭、ws 已断开）都被吞掉，绝不影响 OCR 流水线
        """
        if self.ws_client is None or not task_id:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None

        ws = self.ws_client
        logger = self.logger

        def _cb(done: int, total: int, elapsed: float) -> None:
            # 进度按页数折算到 0~99，避免 "100" 被前端解读为任务完成
            pct = 0
            if total > 0:
                pct = int(round((done / total) * 100))
                if done < total:
                    pct = min(99, pct)
                else:
                    pct = 99  # 整份文档结束后还有 meta.json / zip 打包 / 回传等工作
            try:
                coro = ws.send_task_status(
                    task_id=task_id,
                    status="processing",
                    progress=pct,
                    stage="ocr",
                    page=done,
                    total_pages=total,
                    elapsed=round(elapsed, 2),
                )
                asyncio.run_coroutine_threadsafe(coro, loop)
            except Exception as e:
                logger.debug("[mini-ocr] progress heartbeat failed: %s", e)

        return _cb

    async def _run_image(self, bundle: OcrBundleLike, req: _OcrRunRequest) -> str:
        messages = build_ocr_messages(
            image_paths_or_urls=[str(req.file_path)],
            text_prompt=req.prompt,
        )

        def _do() -> str:
            return bundle.generate_from_messages(messages)

        return await self.run_blocking(_do)

    async def _run_pdf(
        self,
        bundle: OcrBundleLike,
        req: _OcrRunRequest,
        *,
        task_id: Optional[str] = None,
    ) -> str:
        """PDF 输入：逐页渲染为图片后调用模型，最后拼接为一段 markdown（或 JSON 数组）。"""
        image_paths = await self.run_blocking(_render_pdf_to_images, req.file_path)
        if not image_paths:
            raise RuntimeError(f"Failed to render PDF to images: {req.file_path}")

        total = len(image_paths)
        self.logger.info(
            "[mini-ocr] pdf rendered task=%s source=%s pages=%d",
            task_id, req.input_display, total,
        )

        page_outputs: List[str] = []
        t_doc = time.time()
        try:
            for i, img_path in enumerate(image_paths):
                t_page = time.time()
                messages = build_ocr_messages(
                    image_paths_or_urls=[str(img_path)],
                    text_prompt=req.prompt,
                )

                def _do(msgs=messages) -> str:
                    return bundle.generate_from_messages(msgs)

                page_text = await self.run_blocking(_do)
                page_outputs.append(page_text or "")

                done = i + 1
                self.logger.info(
                    "[mini-ocr] pdf page %d/%d done in %.2fs (total %.2fs) task=%s",
                    done, total, time.time() - t_page, time.time() - t_doc, task_id,
                )
                # 借用同一个心跳工具给客户端发 task_status=processing+progress
                if task_id and self.ws_client is not None:
                    try:
                        pct = int(round((done / total) * 100))
                        if done < total:
                            pct = min(99, pct)
                        else:
                            pct = 99
                        await self.ws_client.send_task_status(
                            task_id=task_id,
                            status="processing",
                            progress=pct,
                            stage="ocr",
                            page=done,
                            total_pages=total,
                            elapsed=round(time.time() - t_doc, 2),
                        )
                    except Exception as e:
                        self.logger.debug("[mini-ocr] pdf heartbeat failed: %s", e)
        finally:
            for p in image_paths:
                try:
                    Path(p).unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass

        if req.task == "extract":
            # 抽取模式：每页独立解析，返回一个 JSON 数组
            items: List[Any] = []
            for t in page_outputs:
                parsed = _try_parse_json(t)
                items.append(parsed if parsed is not None else {"_parse_error": True, "raw": t})
            return json.dumps(items, ensure_ascii=False, indent=2)

        joined: List[str] = []
        for i, t in enumerate(page_outputs, start=1):
            joined.append(f"<!-- page {i} -->\n\n{t.strip()}")
        return "\n\n".join(joined)

    # -----------------------------------------------------------------------
    # Policy / model resolution
    # -----------------------------------------------------------------------

    def _resolve_policy_and_model(
        self,
        params: InferenceRequestParams,
    ) -> Tuple[str, Any, str]:
        """根据 service_mini.yaml 的 runtime.backend 选择 runtime，并装配对应 policy。

        配置布局与 text/audio 服务一致：公共项与 ``backend`` 同级；``transformers`` /
        ``vllm`` 子 dict 为各后端专属（见 ``merge_mini_ocr_runtime_cfg``）。

        返回 (runtime_name, policy, model_ref)。
            runtime_name ∈ {"transformers", "vllm"}；默认 "transformers"
            policy       : HfOcrPolicy 或 VllmOcrPolicy
            model_ref    : 归一化后的本地路径 / HF repo id
        """
        svc_runtime_raw = self._service_section("runtime") or {}

        # 模型名必由请求携带（/v1/tasks 后端已按 models 表查库并回填）。
        # mini 服务不持有默认模型：所有小模型都必须注册到 models 表统一管理。
        load_name = (getattr(params, "load_name", None) or "").strip()
        if not load_name:
            raise ValueError(
                "OCR task requires load_name (must be registered in models table). "
                "The backend /v1/tasks endpoint normally guarantees this; this error indicates "
                "the request bypassed backend validation."
            )

        backend_raw = str(svc_runtime_raw.get("backend") or _DEFAULT_RUNTIME).strip().lower()
        # 别名兼容
        if backend_raw in ("hf", "huggingface", "transformers"):
            runtime_name = _RUNTIME_HF
        elif backend_raw in ("vllm",):
            runtime_name = _RUNTIME_VLLM
        else:
            self.logger.warning(
                "[mini-ocr] unknown runtime.backend=%r; fallback to %s",
                backend_raw, _DEFAULT_RUNTIME,
            )
            runtime_name = _DEFAULT_RUNTIME

        svc_runtime = merge_mini_ocr_runtime_cfg(svc_runtime_raw, backend=runtime_name)

        models_dir = getattr(self.inference_config, "models_dir", None)

        if runtime_name == _RUNTIME_VLLM:
            policy: Any = VllmOcrPolicy(
                tensor_parallel_size=int(svc_runtime.get("tensor_parallel_size", 1)),
                gpu_memory_utilization=float(svc_runtime.get("gpu_memory_utilization", 0.35)),
                max_model_len=_as_optional_int(svc_runtime.get("max_model_len", 8192)),
                trust_remote_code=bool(svc_runtime.get("trust_remote_code", True)),
                dtype=str(svc_runtime.get("dtype", "auto")),
                max_images_per_prompt=int(svc_runtime.get("max_images_per_prompt", 4)),
                temperature=float(svc_runtime.get("temperature", 0.0)),
                top_p=float(svc_runtime.get("top_p", 1.0)),
                max_new_tokens=int(svc_runtime.get("max_new_tokens", 8192)),
                engine_kwargs=dict(svc_runtime.get("engine_kwargs") or {}),
            )
            model_ref = _vllm_normalize_model_ref(load_name, models_dir=models_dir)
            return runtime_name, policy, model_ref

        # transformers（默认）
        policy = HfOcrPolicy(
            dtype=str(svc_runtime.get("dtype", "auto")),
            device_map=str(svc_runtime.get("device_map", "auto")),
            trust_remote_code=bool(svc_runtime.get("trust_remote_code", True)),
            temperature=float(svc_runtime.get("temperature", 0.0)),
            top_p=float(svc_runtime.get("top_p", 1.0)),
            max_new_tokens=int(svc_runtime.get("max_new_tokens", 8192)),
            model_kwargs=dict(svc_runtime.get("model_kwargs") or {}),
        )
        model_ref = _hf_normalize_model_ref(load_name, models_dir=models_dir)
        return runtime_name, policy, model_ref

    def _service_section(self, key: str) -> Dict[str, Any]:
        val = self.service_cfg.get(key)
        return val if isinstance(val, dict) else {}

    # -----------------------------------------------------------------------
    # Input fetching (URL / path → local file)
    # -----------------------------------------------------------------------

    async def _fetch_to_local(self, src: str) -> Path:
        """把 URL 或路径归一化为本地真实文件（保留扩展名）。"""
        if not isinstance(src, str) or not src.strip():
            raise ValueError("OCR tpl_list item must be a non-empty string")

        s = src.strip()
        parsed = urlparse(s)
        if parsed.scheme in ("http", "https"):
            return await self.run_blocking(_download_to_tempfile, s)

        # 本地路径（绝对 / 相对）
        p = Path(s).expanduser()
        if not p.is_absolute():
            # 相对路径首选 resources 目录（与 common.image_utils.load_image 约定一致）
            candidate = Path("resources") / s.lstrip("/")
            if candidate.exists():
                p = candidate.resolve()
        if not p.exists():
            raise FileNotFoundError(f"OCR input not found: {src}")
        return p.resolve()

    # -----------------------------------------------------------------------
    # Cloning params for each per-file result message
    # -----------------------------------------------------------------------

    @staticmethod
    def _clone_request_params(
        params: InferenceRequestParams,
        *,
        file_type: str,
    ) -> InferenceRequestParams:
        """返回一份浅拷贝参数副本（仅修改 file_type，避免跨文件串扰）。"""
        dumped = params.model_dump(by_alias=False)
        dumped["file_type"] = file_type
        return InferenceRequestParams(**dumped)


# ---------------------------------------------------------------------------
# Module-level helpers (blocking, safe to run_blocking)
# ---------------------------------------------------------------------------


def _download_to_tempfile(url: str) -> Path:
    import requests  # type: ignore

    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()

    # 尝试从 URL 或 content-type 猜扩展名
    ext = _guess_ext_from_url_and_ct(url, resp.headers.get("content-type"))
    fd, tmp_path = tempfile.mkstemp(prefix="mini_ocr_", suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
    return Path(tmp_path)


def _guess_ext_from_url_and_ct(url: str, content_type: Optional[str]) -> str:
    # URL 后缀优先
    path = urlparse(url).path or ""
    m = re.search(r"\.([A-Za-z0-9]{1,6})$", path)
    if m:
        ext = "." + m.group(1).lower()
        if ext in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}:
            return ext
    # content-type 兜底
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        guessed = mimetypes.guess_extension(ct)
        if guessed:
            return guessed
        if ct == "application/pdf":
            return ".pdf"
    return ".bin"


def _render_pdf_to_images(pdf_path: Path) -> List[Path]:
    """把 PDF 逐页渲染为 PNG，返回生成的临时文件路径列表。

    优先使用 PyMuPDF（fitz）；若未安装则抛明确错误，提示用户安装依赖。
    这里刻意不耦合 GLM-OCR 官方 SDK：通用 mini 服务保持对第三方 SDK 的低依赖，
    官方 SDK 若已经在 vllm 里内置了 PDF 支持，未来可作为一个独立的 handler 分支接入。
    """
    try:
        import fitz  # type: ignore  # PyMuPDF
    except Exception as e:
        raise RuntimeError(
            "PDF input requires PyMuPDF. Please `pip install pymupdf` or pass image URLs instead."
        ) from e

    doc = fitz.open(str(pdf_path))
    out_paths: List[Path] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            # 150 DPI 够 OCR 用；大页面自动缩放
            pix = page.get_pixmap(dpi=150, alpha=False)
            fd, img_path = tempfile.mkstemp(prefix=f"mini_ocr_pdf_{i:03d}_", suffix=".png")
            os.close(fd)
            pix.save(img_path)
            out_paths.append(Path(img_path))
    finally:
        doc.close()
    return out_paths


def _try_parse_json(text: str) -> Optional[Any]:
    """尝试从模型输出里抽出合法 JSON。兼容 ```json ... ``` 代码块。"""
    if not text:
        return None
    s = text.strip()
    # 代码块兼容
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    # 直接尝试
    try:
        return json.loads(s)
    except Exception:
        pass
    # 找首个 { 到最后一个 } 之间的子串
    lbrace = s.find("{")
    rbrace = s.rfind("}")
    if lbrace >= 0 and rbrace > lbrace:
        candidate = s[lbrace : rbrace + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def _release_doc_run_memory() -> None:
    """每份文档 OCR 结束后，尽力回收显存碎片。

    - 不碰 vLLM bundle（由 PipelineCache TTL 驱逐），保留 KV Cache。
    - 只回收 doclayout-yolo 的中间 tensor、零散 CUDA 块；避免"每跑一份文档显存都增长"。
    """
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _as_optional_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        i = int(val)
        return i if i > 0 else None
    except Exception:
        return None
