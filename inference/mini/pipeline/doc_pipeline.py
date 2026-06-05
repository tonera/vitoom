"""
图文混排文档 pipeline

输入：本地 PDF 或 image 文件 + 已就绪的 GLM-OCR bundle + layout detector
输出：一个 zip 的 bytes + 内联 markdown 文本（供 WS 消息 content 字段使用）

zip 内部结构：
    document.md
    images/fig_{page:03d}_{idx:02d}.png
    meta.json

流程：
    1. 把输入展开成 List[PIL.Image]（image 单张；pdf 每页 150 DPI 渲染）
    2. 逐页：layout detect → 按 kind 分派：
        - figure → crop 保存进 images/ 目录，md 里 ![](images/xxx.png)
        - title/text/list/caption → 裁剪区域单图调 GLM-OCR `Text Recognition:`
        - table → `Table Recognition:`
        - formula → `Formula Recognition:`
    3. 按阅读顺序（y-band + x）合并到 document.md
    4. 打包为 zip；返回 (bytes, md_text)

注意：
- 所有 GLM-OCR 调用仍走 runtime/vllm_ocr_bridge 的同步接口；本 pipeline 不动模型生命周期
- 函数主体是同步阻塞的（单 vLLM generate），由 handler 放到 run_blocking 线程调用
"""
from __future__ import annotations

import gc
import io
import json
import statistics
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# pyright: reportMissingImports=false

from common.logger import get_logger

from mini.runtime.ocr_runtime import OcrBundleLike
from mini.pipeline.layout_bridge import (
    BLOCK_KIND_CAPTION,
    BLOCK_KIND_FIGURE,
    BLOCK_KIND_FORMULA,
    BLOCK_KIND_LIST,
    BLOCK_KIND_TABLE,
    BLOCK_KIND_TEXT,
    BLOCK_KIND_TITLE,
    BlockDetection,
    DocLayoutDetector,
)

logger = get_logger(__name__)


# 与 handler 里保持一致的三条 prompt
_OCR_TEXT_PROMPT = "Text Recognition:"
_OCR_TABLE_PROMPT = "Table Recognition:"
_OCR_FORMULA_PROMPT = "Formula Recognition:"


def _cuda_empty_cache() -> None:
    """尽力把 allocator 里未占用的块还给驱动；没有 CUDA 时 no-op。

    注意：vLLM 的 KV Cache 依然常驻（预分配且引用有效），这里只会回收
    layout/前后处理/零碎的中间 tensor，因此不会打断 GLM-OCR 服务。
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 对一些驱动还额外需要 ipc collect 才能真把"used"降下来
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


@dataclass
class DocPipelineConfig:
    # 注：pdf_dpi 是 PDF 栅格化分辨率，直接决定写入 zip 的 figure PNG 的像素数
    # （figure 块会被原样 PNG 编码塞进产物）。150 是"保真档"，120 省 ~10% 耗时
    # 但 figure 明显变糊。OCR 路径对 dpi 不敏感，所以这里优先保真。
    pdf_dpi: int = 150
    max_pages: int = 200
    title_font_ratio: float = 1.3       # block 高度 >= 正文中位高度 × ratio 视为标题
    figure_min_area: int = 50 * 50      # 过滤掉极小的 figure 误检
    crop_padding: int = 4               # 裁剪时的小外扩（像素）

    # --- OCR 生成长度：按 kind 分档 ---
    # 旧实现只有一个 max_new_tokens=8192（由 runtime policy 控制），对只有几十~几百 token
    # 的 text/title/caption 块严重过长，造成每块 1~3s 的无谓耗时。这里按语义分档透传给
    # bundle.generate_from_messages(max_new_tokens=...)，显著降低单块耗时。
    max_new_tokens_text: int = 512      # text / list / caption / title
    max_new_tokens_table: int = 4096
    max_new_tokens_formula: int = 1024

    # --- batch 化 ---
    # 同一页里 text/title/caption/list 共用 "Text Recognition:" prompt，可以把它们拼成
    # 一个 batch 一次 generate。0 或 1 代表关闭 batch（逐块跑）。
    text_batch_size: int = 4

    # --- 每多少页做一次 gc + cuda empty_cache（= 1 表示每页都做；越大越省时间） ---
    # 旧默认是 4，实测在小显卡上每次 empty_cache 会让下一次分配多花几十~上百 ms。
    # 除非显存吃紧，否则 16 即可；整文档结束的 finally 里还会再做一次。
    gc_every_n_pages: int = 16


# ---------------------------------------------------------------------------
# 页图准备
# ---------------------------------------------------------------------------


def _load_image_pages(source_path: Path, pdf_dpi: int, max_pages: int) -> List[Any]:
    """返回每页的 PIL.Image 列表。"""
    from PIL import Image  # type: ignore

    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return _render_pdf_pages(source_path, dpi=pdf_dpi, max_pages=max_pages)

    img = Image.open(source_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return [img]


def _render_pdf_pages(pdf_path: Path, dpi: int, max_pages: int) -> List[Any]:
    try:
        import fitz  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PDF input requires PyMuPDF (pip install pymupdf)."
        ) from e
    from PIL import Image  # type: ignore

    doc = fitz.open(str(pdf_path))
    pages: List[Any] = []
    try:
        total = doc.page_count
        n = min(total, int(max_pages) if max_pages and max_pages > 0 else total)
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=int(dpi), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
        if total > n:
            logger.warning(
                "PDF has %d pages; truncated to %d (max_pages).", total, n,
            )
    finally:
        doc.close()
    return pages


# ---------------------------------------------------------------------------
# 阅读顺序 / 标题启发式
# ---------------------------------------------------------------------------


def _assign_reading_order(blocks: List[BlockDetection], page_height: int) -> List[BlockDetection]:
    """按 (y_band, x_center) 排序，容忍同一行水平并排。"""
    if not blocks:
        return blocks
    band = max(1, int(page_height / 40))
    blocks.sort(key=lambda b: (int(b.y_center // band), b.x_center, b.bbox[1]))
    for i, b in enumerate(blocks):
        b.reading_order = i
    return blocks


def _decide_title_level(
    block: BlockDetection,
    text_heights: List[int],
    ratio: float,
) -> int:
    """返回 Markdown 标题等级（1~3），非标题返回 0。

    基于 block 高度相对于正文块中位高度的比例判定：
      >= ratio*1.6 → h1
      >= ratio*1.2 → h2
      >= ratio     → h3
      其它         → 0（按普通文本处理）
    """
    if not text_heights:
        return 2  # 没有正文参照时，检测到的 title 默认 h2
    median = statistics.median(text_heights) or 1
    r = block.height / median
    if r >= ratio * 1.6:
        return 1
    if r >= ratio * 1.2:
        return 2
    if r >= ratio:
        return 3
    return 0


# ---------------------------------------------------------------------------
# 裁剪 + OCR
# ---------------------------------------------------------------------------


def _crop(image: Any, bbox: Tuple[int, int, int, int], pad: int = 0) -> Any:
    w, h = image.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def _ocr_block(
    bundle: OcrBundleLike,
    crop_image: Any,
    prompt: str,
    *,
    max_new_tokens: Optional[int] = None,
    timings: Optional[Dict[str, List[float]]] = None,
    kind: Optional[str] = None,
) -> str:
    """对一个裁剪后的 PIL 小图跑 GLM-OCR，返回文本。

    max_new_tokens：透传给 bundle.generate_from_messages 的输出长度上限。按 kind 分档传入
      能显著降低短文本块的生成耗时（默认 8192 对 text 块过长）。

    timings / kind：可选诊断通道。若传入，把本次 generate 的耗时（秒）按 kind 分桶累计，
    供调用方（_process_page）末尾汇总打印——用来定位"单页慢"究竟是 block 数多还是单块慢。
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": crop_image},
            {"type": "text", "text": prompt},
        ],
    }]
    gen_kwargs: Dict[str, Any] = {}
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = int(max_new_tokens)

    if timings is None or kind is None:
        return (bundle.generate_from_messages(messages, **gen_kwargs) or "").strip()

    t0 = time.time()
    try:
        return (bundle.generate_from_messages(messages, **gen_kwargs) or "").strip()
    finally:
        timings.setdefault(kind, []).append(time.time() - t0)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


@dataclass
class _PageRender:
    """每页处理产物。"""
    page_index: int
    md_parts: List[str] = field(default_factory=list)
    image_entries: List[Tuple[str, bytes]] = field(default_factory=list)  # (arcname, png_bytes)
    counts: Dict[str, int] = field(default_factory=dict)


def _run_text_jobs_in_batches(
    *,
    bundle: OcrBundleLike,
    jobs: List[Dict[str, Any]],
    md_parts: List[Optional[str]],
    cfg: DocPipelineConfig,
    timings: Dict[str, List[float]],
    page_index: int,
) -> None:
    """把同页所有 Text Recognition: 类 block 的 crop 分批跑 OCR，并回填 md_parts。

    jobs: [{slot, crop, block, wrap}]
      slot : 对应 md_parts 里的占位索引
      crop : PIL.Image 子图
      block: 原 BlockDetection（仅用来打耗时桶 / 错误日志）
      wrap : str → str，最终文本包装（title 加 #、caption 包 *…*、text/list 原样）
    """
    batch_size = max(1, int(cfg.text_batch_size or 1))
    batch_fn = getattr(bundle, "generate_batch_from_messages", None)

    def _build_msgs(crop_image: Any) -> List[Dict[str, Any]]:
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": crop_image},
                {"type": "text", "text": _OCR_TEXT_PROMPT},
            ],
        }]

    i = 0
    while i < len(jobs):
        batch = jobs[i : i + batch_size]
        i += batch_size

        # 单条直接走原路径
        if batch_size == 1 or batch_fn is None or len(batch) == 1:
            for job in batch:
                try:
                    text = _ocr_block(
                        bundle, job["crop"], _OCR_TEXT_PROMPT,
                        max_new_tokens=cfg.max_new_tokens_text,
                        timings=timings, kind=job["block"].kind,
                    )
                    if text:
                        md_parts[job["slot"]] = job["wrap"](text)
                except Exception as e:
                    logger.warning(
                        "[doc-pipeline] block OCR failed page=%d kind=%s bbox=%s err=%s",
                        page_index, job["block"].kind, job["block"].bbox, e,
                    )
                    md_parts[job["slot"]] = (
                        f"<!-- ocr failed: page={page_index} kind={job['block'].kind} -->"
                    )
            continue

        messages_list = [_build_msgs(j["crop"]) for j in batch]
        t0 = time.time()
        texts: Optional[List[str]] = None
        try:
            texts = batch_fn(
                messages_list,
                max_new_tokens=cfg.max_new_tokens_text,
            )
        except Exception as e:
            logger.warning(
                "[doc-pipeline] batch OCR failed page=%d size=%d err=%s; "
                "fallback to per-block generate",
                page_index, len(batch), e,
            )
            texts = None

        if texts is not None:
            elapsed = time.time() - t0
            # 把 batch 的总耗时按等分回填到每个 job 的 kind 桶里——这样页末诊断行仍然
            # 能反映"text 类一共花了多久"。严格来讲不如单条精确，但足以监控趋势。
            per_item = elapsed / max(1, len(batch))
            for j in batch:
                timings.setdefault(j["block"].kind, []).append(per_item)

            for job, text in zip(batch, texts):
                text = (text or "").strip()
                if text:
                    md_parts[job["slot"]] = job["wrap"](text)
            continue

        # batch 失败：整批逐条回退
        for job in batch:
            try:
                text = _ocr_block(
                    bundle, job["crop"], _OCR_TEXT_PROMPT,
                    max_new_tokens=cfg.max_new_tokens_text,
                    timings=timings, kind=job["block"].kind,
                )
                if text:
                    md_parts[job["slot"]] = job["wrap"](text)
            except Exception as e:
                logger.warning(
                    "[doc-pipeline] block OCR failed (fallback) page=%d kind=%s bbox=%s err=%s",
                    page_index, job["block"].kind, job["block"].bbox, e,
                )
                md_parts[job["slot"]] = (
                    f"<!-- ocr failed: page={page_index} kind={job['block'].kind} -->"
                )


def _process_page(
    bundle: OcrBundleLike,
    detector: DocLayoutDetector,
    page_image: Any,
    page_index: int,
    cfg: DocPipelineConfig,
) -> _PageRender:
    render = _PageRender(page_index=page_index)
    pw, ph = page_image.size

    # --- 诊断：layout 耗时 + 每类 block 的 generate 耗时桶 ---
    # 格式：{kind: [elapsed_sec, ...]}；页末会打印
    #   page=N blocks=M layout=Xms ocr=[title:a/b.bms text:c/d.bms ...] total_gen=Yms
    timings: Dict[str, List[float]] = {}
    t_layout = time.time()
    blocks = detector.detect(page_image, page_index=page_index) or []
    layout_ms = (time.time() - t_layout) * 1000.0

    for b in blocks:
        render.counts[b.kind] = render.counts.get(b.kind, 0) + 1

    blocks = _assign_reading_order(blocks, page_height=ph)

    # 正文高度参照：收集 text/list 类 block 的高度作为中位值基准
    text_heights = [b.height for b in blocks if b.kind in (BLOCK_KIND_TEXT, BLOCK_KIND_LIST)]

    figure_counter = 0

    # 两阶段处理：
    # 1) 第一遍扫 blocks：figure 直接写入；text/title/caption/list（共用 Text Recognition:
    #    prompt）攒到 text_jobs 里并在 md_parts 预埋 None 占位；table/formula 数量少，继续
    #    单条处理即可。
    # 2) 把 text_jobs 按 cfg.text_batch_size 分批，调 bundle.generate_batch_from_messages
    #    （若不支持则逐条回退），回填到对应占位。
    #
    # 这样既保持了阅读顺序，也让 OCR 的大多数调用走批量路径。
    text_jobs: List[Dict[str, Any]] = []  # {slot, crop, block, wrap}
    md_parts: List[Optional[str]] = []

    def _wrap_text(text: str) -> str:
        return text

    def _wrap_caption(text: str) -> str:
        return f"*{text}*"

    def _make_title_wrapper(block: BlockDetection) -> Callable[[str], str]:
        lvl = _decide_title_level(block, text_heights, cfg.title_font_ratio) or 2
        hashes = "#" * max(1, min(6, lvl))

        def _wrap(text: str) -> str:
            return f"{hashes} {text}"

        return _wrap

    for block in blocks:
        try:
            crop = _crop(page_image, block.bbox, pad=cfg.crop_padding)
            if crop is None:
                continue

            if block.kind == BLOCK_KIND_FIGURE:
                if block.width * block.height < cfg.figure_min_area:
                    continue
                arcname = f"images/fig_p{page_index:03d}_{figure_counter:02d}.png"
                figure_counter += 1
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                render.image_entries.append((arcname, buf.getvalue()))
                md_parts.append(f"![figure p{page_index+1}-{figure_counter}]({arcname})")
                continue

            if block.kind == BLOCK_KIND_TABLE:
                text = _ocr_block(
                    bundle, crop, _OCR_TABLE_PROMPT,
                    max_new_tokens=cfg.max_new_tokens_table,
                    timings=timings, kind=block.kind,
                )
                if text:
                    md_parts.append(text)
                continue

            if block.kind == BLOCK_KIND_FORMULA:
                text = _ocr_block(
                    bundle, crop, _OCR_FORMULA_PROMPT,
                    max_new_tokens=cfg.max_new_tokens_formula,
                    timings=timings, kind=block.kind,
                )
                if text:
                    # 若模型已经返回带 $$ 的 LaTeX 块，就直接用；否则包一层 $$
                    t = text.strip()
                    if t.startswith("$$") or t.startswith(r"\[") or "\n" in t:
                        md_parts.append(t if t.startswith("$$") else f"$$\n{t}\n$$")
                    else:
                        # 宽度窄（<0.5 页宽）视为行内，否则块级
                        inline = block.width < pw * 0.5
                        md_parts.append(f"${t}$" if inline else f"$$\n{t}\n$$")
                continue

            # 下面都是共用 Text Recognition: prompt 的短文本类 block
            if block.kind == BLOCK_KIND_TITLE:
                wrap: Callable[[str], str] = _make_title_wrapper(block)
            elif block.kind == BLOCK_KIND_CAPTION:
                wrap = _wrap_caption
            else:  # text / list / 未知默认
                wrap = _wrap_text

            slot = len(md_parts)
            md_parts.append(None)  # 占位
            text_jobs.append({
                "slot": slot,
                "crop": crop,
                "block": block,
                "wrap": wrap,
            })
        except Exception as e:
            logger.warning(
                "[doc-pipeline] block OCR prepare failed page=%d kind=%s bbox=%s err=%s",
                page_index, block.kind, block.bbox, e,
            )
            md_parts.append(f"<!-- ocr failed: page={page_index} kind={block.kind} -->")

    # ---------- 执行 text 类 batch ----------
    if text_jobs:
        _run_text_jobs_in_batches(
            bundle=bundle,
            jobs=text_jobs,
            md_parts=md_parts,
            cfg=cfg,
            timings=timings,
            page_index=page_index,
        )

    # 把占位里可能还是 None 的（极少见：batch 失败且单条也抛）替换为失败注释
    for i, p in enumerate(md_parts):
        if p is None:
            md_parts[i] = f"<!-- ocr empty: page={page_index} slot={i} -->"

    render.md_parts.extend([p for p in md_parts if p is not None])

    # --- 诊断汇总：每类 OCR 调用次数 / 总耗时 / 平均耗时 ---
    # 示例：
    #   [doc-pipeline][diag] page=5/? blocks=12 layout=78.4ms ocr=[
    #     title:2x/0.65s(avg 325ms) text:8x/7.20s(avg 900ms) formula:1x/1.10s ...
    #   ] total_gen=9.10s
    if timings:
        per_kind_parts: List[str] = []
        total_gen = 0.0
        for k in sorted(timings.keys()):
            arr = timings[k]
            s = sum(arr)
            total_gen += s
            avg_ms = (s / len(arr)) * 1000.0 if arr else 0.0
            per_kind_parts.append(f"{k}:{len(arr)}x/{s:.2f}s(avg {avg_ms:.0f}ms)")
        logger.info(
            "[doc-pipeline][diag] page=%d blocks=%d counts=%s layout=%.1fms "
            "ocr=[%s] total_gen=%.2fs",
            page_index + 1,
            len(blocks),
            dict(render.counts),
            layout_ms,
            " ".join(per_kind_parts),
            total_gen,
        )
    else:
        logger.info(
            "[doc-pipeline][diag] page=%d blocks=%d counts=%s layout=%.1fms "
            "ocr=[none] total_gen=0.00s",
            page_index + 1, len(blocks), dict(render.counts), layout_ms,
        )

    return render


def build_doc_zip(
    *,
    bundle: OcrBundleLike,
    detector: DocLayoutDetector,
    source_path: Path,
    cfg: Optional[DocPipelineConfig] = None,
    load_name: Optional[str] = None,
    layout_backend: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[bytes, str]:
    """对一个 image 或 pdf，生成图文混排 markdown 并打 zip。

    Args:
        progress_cb: 每完成一页就被调用一次。签名 (page_done, total_pages, elapsed_sec)；
                     page_done 为已完成页数（1-based，等于刚完成的那页序号+1）。
                     用于让调用方上报 WS progress / 打心跳日志。任何异常都被静默吞掉，
                     不影响 OCR 主流程。

    返回 (zip_bytes, md_text)。
    """
    cfg = cfg or DocPipelineConfig()
    t0 = time.time()

    pages = _load_image_pages(source_path, pdf_dpi=cfg.pdf_dpi, max_pages=cfg.max_pages)
    if not pages:
        raise RuntimeError(f"No pages could be loaded from {source_path}")

    md_sections: List[str] = []
    all_image_entries: List[Tuple[str, bytes]] = []
    total_counts: Dict[str, int] = {}
    per_page_meta: List[Dict[str, Any]] = []

    total_pages = len(pages)
    logger.info(
        "[doc-pipeline] start source=%s total_pages=%d pdf_dpi=%d",
        source_path.name, total_pages, cfg.pdf_dpi,
    )

    try:
        for page_index in range(total_pages):
            t_page = time.time()
            page_image = pages[page_index]
            pw, ph = page_image.size
            page_render = _process_page(bundle, detector, page_image, page_index, cfg)

            md_sections.append(
                f"<!-- page {page_index + 1} -->\n\n" + "\n\n".join(page_render.md_parts)
            )
            all_image_entries.extend(page_render.image_entries)
            for k, v in page_render.counts.items():
                total_counts[k] = total_counts.get(k, 0) + int(v)
            per_page_meta.append({
                "page": page_index + 1,
                "width": pw,
                "height": ph,
                "counts": dict(page_render.counts),
                "images": [name for name, _ in page_render.image_entries],
            })

            page_elapsed = time.time() - t_page
            total_elapsed = time.time() - t0
            done = page_index + 1
            logger.info(
                "[doc-pipeline] page %d/%d done in %.2fs (total %.2fs) counts=%s",
                done, total_pages, page_elapsed, total_elapsed, page_render.counts,
            )

            if progress_cb is not None:
                try:
                    progress_cb(done, total_pages, total_elapsed)
                except Exception as _cb_err:
                    logger.debug("[doc-pipeline] progress_cb raised: %s", _cb_err)

            try:
                page_image.close()
            except Exception:
                pass
            pages[page_index] = None  # type: ignore[assignment]

            gc_every = max(1, int(getattr(cfg, "gc_every_n_pages", 4) or 4))
            if done % gc_every == 0:
                gc.collect()
                _cuda_empty_cache()
    finally:
        # 整文档收尾：彻底释放页图列表，再 empty_cache 一次
        try:
            pages.clear()
        except Exception:
            pass
        gc.collect()
        _cuda_empty_cache()

    md_text = "\n\n".join(md_sections).strip() + "\n"

    meta = {
        "generated_at": int(time.time()),
        "elapsed_seconds": round(time.time() - t0, 3),
        "load_name": load_name,
        "layout_backend": layout_backend,
        "total_pages": total_pages,
        "total_counts": total_counts,
        "pages": per_page_meta,
    }

    # 打 zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("document.md", md_text)
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        for arcname, data in all_image_entries:
            zf.writestr(arcname, data)

    return buf.getvalue(), md_text


__all__ = ["DocPipelineConfig", "build_doc_zip"]
