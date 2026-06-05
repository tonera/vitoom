"""document_to_markdown 工具的单元测试：覆盖 pandoc 路径与 markitdown 路径的纯函数行为。

只测可独立验证的纯函数（zip 打包 / 抽图 / 路由探测 / pandoc 定位）；端到端 docx→zip 的转换
需要系统 pandoc 二进制 + python-docx，未安装时整项 skip，不影响 CI。
"""
from __future__ import annotations

import base64
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin import document_to_markdown as dtm


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_strip_html_img_alt_attributes_removes_alt_leaves_src_style():
    """pandoc docx → raw ``<img src="…" style="…" alt="长水印" />``：去 alt 保留图。"""
    s = (
        r'<img src="images/img_001.png" style="width:0.3in" alt="学科网长句水印" />'
        "  mid  "
        r'<img src="a" alt=unquoted style="width:1in" />'
    )
    out = dtm._strip_html_img_alt_attributes(s)
    assert re.search(r"<img[^>]+alt\s*=", out, re.IGNORECASE) is None
    assert 'src="images/img_001.png"' in out
    assert "style=" in out
    assert "学科网" not in out
    assert "unquoted" not in out


def test_strip_unrenderable_vector_image_refs_drops_wmf_not_png():
    """WMF/EMF 在 Web 中几乎不可见：删正文引用，保留 PNG 等可渲染图。"""
    s = '文字20<img src="images/img_013.wmf" style="h:1" /> 已知，![](images/x.wmf)end'
    s += r' 中间 <img src="a.png" /> 保留'
    out = dtm._strip_unrenderable_vector_image_refs(s)
    assert "img_013" not in out
    assert "x.wmf" not in out
    assert "20" in out and "已知" in out
    assert 'src="a.png"' in out
    assert ".wmf" not in out.lower()


def test_peek_source_suffix_handles_url_and_local_path():
    assert dtm._peek_source_suffix("https://x.com/a/b/foo.DOCX", timeout=5.0) == ".docx"
    assert dtm._peek_source_suffix("/tmp/some.pptx", timeout=5.0) == ".pptx"
    assert dtm._peek_source_suffix("https://x.com/no-suffix", timeout=5.0) == ""
    assert dtm._peek_source_suffix("plain-name", timeout=5.0) == ""


def test_extract_first_page_preview_splits_on_ocr_page_marker():
    """PDF/OCR 路径用 ``<!-- page N -->`` 注释分页（见 inference.mini OCR handler）。"""
    md = (
        "<!-- page 1 -->\n\n# 试卷第一页\n\n第一题题干……\n\n"
        "<!-- page 2 -->\n\n# 试卷第二页\n\n第二题题干……"
    )
    preview = dtm._extract_first_page_preview(md)
    assert "试卷第一页" in preview
    assert "第二题题干" not in preview, "must stop at the second OCR page marker"
    assert "<!-- page" not in preview


def test_extract_first_page_preview_falls_back_to_form_feed():
    md = "page-one body\fpage-two body"
    preview = dtm._extract_first_page_preview(md)
    assert preview == "page-one body"


def test_extract_first_page_preview_truncates_with_ellipsis_when_no_pagebreak():
    body = "x" * 5000
    preview = dtm._extract_first_page_preview(body)
    assert len(preview) <= 800
    assert preview.endswith("…")


def test_extract_first_page_preview_keeps_short_text_intact():
    md = "# Hello\n\nshort body."
    assert dtm._extract_first_page_preview(md) == md.strip()


def test_locate_pandoc_binary_finds_executable():
    """pandoc 已经装在 vitoom env 里时，定位逻辑应能拿到一个可执行路径。

    若整个开发机上确实没装 pandoc（罕见，因为本仓库 docx 高保真依赖它），
    返回 None 也是允许的；这里只做"非空就必须可执行"的强约束。
    """
    found = dtm._locate_pandoc_binary()
    if found is None:
        pytest.skip("pandoc binary not available on this machine; skipping locator check")
    p = Path(found)
    assert p.is_file()


def test_extract_data_uri_images_decodes_and_rewrites():
    raw = b"\x89PNG fake-bytes-1"
    b64 = base64.b64encode(raw).decode("ascii")
    md = (
        f"# title\n\nbefore ![hello](data:image/png;base64,{b64}) after\n"
        f"another line ![](data:image/jpeg;base64,{base64.b64encode(b'jpg-bytes').decode('ascii')})\n"
    )

    new_md, entries = dtm._extract_data_uri_images(md)

    assert len(entries) == 2
    assert entries[0][0].startswith("images/img_001.")
    assert entries[0][0].endswith(".png")
    assert entries[0][1] == raw
    assert entries[1][0].endswith(".jpg")
    assert "data:image/png;base64" not in new_md
    assert "data:image/jpeg;base64" not in new_md
    assert "![hello](images/img_001.png)" in new_md
    assert "(images/img_002.jpg)" in new_md


def test_extract_data_uri_images_deduplicates_by_content_hash():
    raw = b"shared-bytes"
    b64 = base64.b64encode(raw).decode("ascii")
    md = (
        f"![a](data:image/png;base64,{b64})\n\n"
        f"![b](data:image/png;base64,{b64})\n"
    )

    new_md, entries = dtm._extract_data_uri_images(md)

    assert len(entries) == 1, "same bytes must dedupe to a single image entry"
    arc = entries[0][0]
    assert new_md.count(arc) == 2


def test_extract_archive_media_reads_pptx_style_zip(tmp_path: Path):
    src = tmp_path / "fake.pptx"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("ppt/media/image1.png", b"png-1")
        zf.writestr("ppt/media/image2.jpg", b"jpg-2")
        zf.writestr("ppt/slides/slide1.xml", b"<unrelated/>")

    seen: set[str] = set()
    entries = dtm._extract_archive_media(src, skip_hashes=seen)

    assert {e[0] for e in entries} == {"images/extra_001.png", "images/extra_002.jpg"}
    assert {e[1] for e in entries} == {b"png-1", b"jpg-2"}
    # skip_hashes 应被填上，下次重复调用就不再返回这两张
    again = dtm._extract_archive_media(src, skip_hashes=seen)
    assert again == []


def test_extract_archive_media_returns_empty_for_unsupported_or_broken(tmp_path: Path):
    txt = tmp_path / "plain.txt"
    txt.write_text("not a zip", encoding="utf-8")
    assert dtm._extract_archive_media(txt, skip_hashes=set()) == []

    # 后缀像 docx 但内容不是 zip：应静默返回空
    fake = tmp_path / "broken.docx"
    fake.write_bytes(b"definitely not a zip header")
    assert dtm._extract_archive_media(fake, skip_hashes=set()) == []


def test_normalize_source_list_filters_empty_like_tokens():
    sources = dtm._normalize_source_list(
        url="None",
        urls=["https://example.com/a.docx", None, " null ", " /tmp/demo.txt "],
    )
    assert sources == ["https://example.com/a.docx", "/tmp/demo.txt"]


def test_effective_pdf_timeout_has_ocr_floor():
    assert dtm._effective_pdf_timeout(60.0) == 300.0
    assert dtm._effective_pdf_timeout("120") == 300.0
    assert dtm._effective_pdf_timeout(600.0) == 600.0


def test_convert_documents_uses_pdf_timeout_floor(monkeypatch):
    seen: dict[str, float] = {}

    monkeypatch.setattr(dtm, "_detect_pdf_source", lambda *_args, **_kwargs: True)

    def fake_pdf_convert(**kwargs):
        seen["timeout"] = kwargs["timeout"]
        return {
            "source": kwargs["source"],
            "task_id": "task-1",
            "status": "completed",
            "provider": "mini_ocr",
            "preview_md": "# ok",
            "files": [],
            "total": 0,
        }

    monkeypatch.setattr(dtm, "_convert_pdf_document_sync", fake_pdf_convert)

    payload = dtm.convert_documents_sync(
        user_id="user-1",
        url="https://example.com/a.pdf",
        timeout=60.0,
    )

    assert payload["status"] == "completed"
    assert seen["timeout"] == 300.0


def test_pdf_collect_timeout_returns_processing(monkeypatch):
    def fake_submit_and_collect(**kwargs):
        return {
            "task_id": "task-timeout",
            "status": "timeout",
            "files": [],
            "total": 0,
            "error": "document-pdf generation did not finish within 300s",
        }

    monkeypatch.setattr(dtm, "submit_and_collect", fake_submit_and_collect)

    item = dtm._convert_pdf_document_sync(
        user_id="user-1",
        source="https://example.com/a.pdf",
        timeout=300.0,
        storage="local",
    )

    assert item["status"] == "processing"
    assert item["message"] == "任务已提交，正在处理中"
    assert "error" not in item
    assert item["task_id"] == "task-timeout"


def test_convert_documents_processing_status_not_failed(monkeypatch):
    monkeypatch.setattr(dtm, "_detect_pdf_source", lambda *_args, **_kwargs: True)

    def fake_pdf_convert(**kwargs):
        return {
            "source": kwargs["source"],
            "task_id": "task-processing",
            "status": "processing",
            "provider": "mini_ocr",
            "preview_md": "",
            "files": [],
            "total": 0,
            "message": "任务已提交，正在处理中",
        }

    monkeypatch.setattr(dtm, "_convert_pdf_document_sync", fake_pdf_convert)

    payload = dtm.convert_documents_sync(
        user_id="user-1",
        url="https://example.com/a.pdf",
        timeout=60.0,
    )

    assert payload["status"] == "processing"
    assert payload["completed"] == 0
    assert payload["processing"] == 1


def test_convert_markdown_content_persists_directly(monkeypatch):
    seen: dict[str, object] = {}

    def fake_persist(**kwargs):
        seen.update(kwargs)
        return {
            "source": kwargs["source"],
            "task_id": "task-md",
            "status": "completed",
            "provider": kwargs["provider"],
            "preview_md": "# 总结",
            "files": [{"url": "http://example.com/summary.md"}],
            "total": 1,
        }

    monkeypatch.setattr(dtm, "_persist_local_markdown_result", fake_persist)

    payload = dtm.convert_documents_sync(
        user_id="user-1",
        content="# 总结\n\n- 要点",
        title="summary",
    )

    assert payload["status"] == "completed"
    assert payload["total"] == 1
    assert seen["source"] == "summary.md"
    assert seen["markdown_text"] == "# 总结\n\n- 要点"
    assert seen["provider"] == "content"
    assert seen["source_kind"] == "content"


def test_convert_markdown_requires_source_or_content():
    payload = dtm.convert_documents_sync(user_id="user-1", url=None, urls=None)
    assert payload["status"] == "failed"
    assert "markdown/content" in payload["error"]


def test_materialize_source_locally_rejects_empty_source(tmp_path: Path):
    with pytest.raises(ValueError, match="document source is required"):
        dtm._materialize_source_locally(None, work_dir=tmp_path, timeout=5.0)


def test_pack_doc_zip_writes_md_meta_and_images():
    md = "# hello\n\n![](images/img_001.png)"
    img_bytes = b"png-payload"
    entries = [("images/img_001.png", img_bytes)]

    blob = dtm._pack_doc_zip(
        md_text=md,
        image_entries=entries,
        source="https://example.com/foo.docx",
        provider="pandoc",
        source_kind="docx",
    )

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
        assert {"document.md", "meta.json", "images/img_001.png"} <= names
        assert zf.read("document.md").decode("utf-8").startswith("# hello")
        # md 末尾应被规范成换行结尾
        assert zf.read("document.md").endswith(b"\n")
        assert zf.read("images/img_001.png") == img_bytes
        meta = zf.read("meta.json").decode("utf-8")
        assert "\"source_kind\": \"docx\"" in meta
        assert "\"image_count\": 1" in meta


def test_libreoffice_sidecar_rewrite_respects_start_index(tmp_path):
    """sidecar 配图序号接在 data URI 抽取之后，避免 zip 内重复 arcname。"""
    from backend.services.agent.tools.builtin._libreoffice import (
        rewrite_libreoffice_markdown_sidecars,
    )

    export = tmp_path / "lo_out"
    export.mkdir()
    png = b"\x89PNG\r\n\x1a\n\x00"
    (export / "pic.png").write_bytes(png)
    md_file = export / "x.md"
    md_file.write_text("See ![](pic.png)\n", encoding="utf-8")

    md_out, entries = rewrite_libreoffice_markdown_sidecars(
        md_file.read_text(encoding="utf-8"),
        export,
        md_file,
        image_suffixes=frozenset({".png"}),
        start_index=3,
    )
    assert entries == [("images/img_003.png", png)]
    assert "images/img_003.png" in md_out


# ---------------------------------------------------------------------------
# Args schema：兜住 LLM 把可选参数序列化成 "None"/"null"/"" 的常见 quirk
# ---------------------------------------------------------------------------


def _build_args_schema():
    """复制 build_document_to_markdown_tool 内 args_schema 的最小构造逻辑，便于在测试里直接校验。

    这里**不调用** build_document_to_markdown_tool 本体，避免触发 crewai 工具实例化路径
    （单元测试不应该依赖 crewai 装载）。schema 部分是纯 pydantic，可以在测试里独立验证。
    """
    pytest.importorskip("pydantic", reason="pydantic missing")
    tool = dtm.build_document_to_markdown_tool(context={"user_id": "test-user"})
    return tool.args_schema


def test_args_schema_coerces_string_nones_to_none():
    schema = _build_args_schema()
    parsed = schema(url="None", urls="null", timeout="None")
    assert parsed.url is None
    assert parsed.urls is None
    assert parsed.timeout is None

    parsed2 = schema(url="", urls="  ", timeout="undefined")
    assert parsed2.url is None
    assert parsed2.urls is None
    assert parsed2.timeout is None


def test_args_schema_keeps_real_values():
    schema = _build_args_schema()
    parsed = schema(
        url="https://example.com/a.docx",
        urls=["https://example.com/b.pdf"],
        timeout=30.0,
    )
    assert parsed.url == "https://example.com/a.docx"
    assert parsed.urls == ["https://example.com/b.pdf"]
    assert parsed.timeout == 30.0


# ---------------------------------------------------------------------------
# E2E: real docx → pandoc → zip （需要 pandoc 二进制 + python-docx）
# ---------------------------------------------------------------------------


def _make_docx_with_image(tmp_path: Path) -> Path:
    """构造一个最小可用的 docx，含一段标题 + 一张内嵌 PNG。"""
    docx_lib = pytest.importorskip("docx", reason="python-docx not installed")
    from PIL import Image  # type: ignore  # markitdown 已带依赖

    img_path = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), color=(0, 128, 255)).save(img_path, format="PNG")

    doc = docx_lib.Document()
    doc.add_heading("Hello pandoc", level=1)
    doc.add_paragraph("一段中文正文，含数学公式占位。")
    doc.add_picture(str(img_path))
    out = tmp_path / "fixture.docx"
    doc.save(str(out))
    return out


def test_run_pandoc_docx_extracts_image_and_rewrites_refs(tmp_path: Path):
    if dtm._locate_pandoc_binary() is None:
        pytest.skip("pandoc binary not available")
    docx = _make_docx_with_image(tmp_path)

    work = tmp_path / "work"
    work.mkdir()
    md_text, image_entries = dtm._run_pandoc_docx(docx, work, timeout=60.0)

    assert "Hello pandoc" in md_text
    assert image_entries, "pandoc 应能抽出至少一张图片"
    arc = image_entries[0][0]
    assert arc.startswith("images/img_")
    # 引用必须已被重写为 images/imgN.ext，原始 ./media/... 不应残留
    assert "./media/" not in md_text
    assert arc in md_text


def test_pack_doc_zip_round_trip_with_real_docx(tmp_path: Path):
    if dtm._locate_pandoc_binary() is None:
        pytest.skip("pandoc binary not available")
    docx = _make_docx_with_image(tmp_path)

    work = tmp_path / "work2"
    work.mkdir()
    md_text, image_entries = dtm._run_pandoc_docx(docx, work, timeout=60.0)
    blob = dtm._pack_doc_zip(
        md_text=md_text,
        image_entries=image_entries,
        source=str(docx),
        provider="pandoc",
        source_kind="docx",
    )

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        assert "document.md" in names
        assert "meta.json" in names
        assert any(n.startswith("images/") for n in names)
        # 还原后图片可被读出且非空
        first_img = next(n for n in names if n.startswith("images/"))
        assert len(zf.read(first_img)) > 0
