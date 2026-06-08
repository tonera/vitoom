from __future__ import annotations

import io
import os
import shutil
import zipfile
from pathlib import Path

import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin import _libreoffice as lo_builtin
from backend.services.agent.tools.builtin import document_to_pdf as dtp
from backend.services.agent.tools.builtin import md_sanitize_for_pdf as mds


def test_normalize_source_list_filters_empty_like_tokens():
    sources = dtp._normalize_source_list(
        url="None",
        urls=["https://example.com/a.md", None, " null ", " /tmp/demo.zip "],
    )
    assert sources == ["https://example.com/a.md", "/tmp/demo.zip"]


def test_markdown_exts_support_lowercase_rmd():
    assert ".rmd" in dtp._MARKDOWN_EXTS


def test_inner_pdf_name_from_rel_distinguishes_subdirs():
    # 曾仅用 basename 时 sub/a 与 "sub_a" 会争用 a.pdf
    n1 = dtp._inner_pdf_name_from_rel("dir/readme.md")
    n2 = dtp._inner_pdf_name_from_rel("readme.md")
    assert n1 != n2
    assert n1 == "dir_readme.pdf"
    assert n2 == "readme.pdf"


def test_unique_zip_entry_when_inner_names_hash_collide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """模拟 a/b.md 与 a_b.md 均映射为 a_b.pdf 时追加 _2 后缀。"""
    src_zip = tmp_path / "c.zip"
    with zipfile.ZipFile(src_zip, "w") as zf:
        zf.writestr("a/b.md", "# a")
        zf.writestr("a_b.md", "# b")

    def _fast_md(path: Path, **kwargs: object) -> Path:
        out = path.with_suffix(".pdf")
        out.write_bytes(b"%PDF-1")
        return out

    monkeypatch.setattr(dtp, "_convert_markdown_with_pandoc", _fast_md)
    items, _report = dtp._convert_zip_mixed_to_pdfs(src_zip, timeout=10, default_font="")
    names = {n for n, _p in items}
    # 两路合并为同一逻辑名时，第二个应变成 _2
    assert len(names) == 2
    assert "a_b.pdf" in names
    assert "a_b_2.pdf" in names


def test_build_pdf_zip_contains_all_outputs(tmp_path: Path):
    p1 = tmp_path / "a.pdf"
    p2 = tmp_path / "b.pdf"
    p1.write_bytes(b"%PDF-a")
    p2.write_bytes(b"%PDF-b")
    blob = dtp._build_pdf_zip([("a.pdf", p1), ("b.pdf", p2)])
    with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
        names = set(zf.namelist())
        assert "a.pdf" in names
        assert "b.pdf" in names


def test_pandoc_engines_tectonic_first_then_latex(monkeypatch: pytest.MonkeyPatch):
    def _which(name: str) -> str | None:  # type: ignore[valid-type]
        return f"/bin/{name}" if name in ("tectonic", "xelatex", "pdflatex") else None

    monkeypatch.setattr(shutil, "which", _which)
    o = dtp._pandoc_markdown_pdf_engine_steps(default_font="")
    assert [e for e, _ in o] == ["tectonic", "xelatex", "pdflatex"]
    o2 = dtp._pandoc_markdown_pdf_engine_steps(default_font="Noto")
    assert [e for e, _ in o2] == ["tectonic", "tectonic", "xelatex", "xelatex", "pdflatex"]


def test_pandoc_engines_only_latex_no_tectonic(monkeypatch: pytest.MonkeyPatch):
    def _which(name: str) -> str | None:  # type: ignore[valid-type]
        if name in ("xelatex", "pdflatex"):
            return f"/bin/{name}"
        return None

    monkeypatch.setattr(shutil, "which", _which)
    o = dtp._pandoc_markdown_pdf_engine_steps(default_font="")
    assert [e for e, _ in o] == ["xelatex", "pdflatex"]


def test_is_zip_source_true_by_suffix():
    assert dtp._is_zip_source("/tmp/demo.zip", timeout=5.0) is True


def test_convert_zip_mixed_to_pdfs_with_monkeypatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src_zip = tmp_path / "mixed.zip"
    with zipfile.ZipFile(src_zip, "w") as zf:
        zf.writestr("a.md", "# hello")
        zf.writestr("b.docx", "fake-docx")
        zf.writestr("c.xls", "fake-xls")
        zf.writestr("d.txt", "skip-me")

    def _fake_pandoc(path: Path, *, timeout: float, default_font: str) -> Path:
        out = path.with_suffix(".pdf")
        out.write_bytes(b"%PDF-md")
        return out

    def _fake_lo(path: Path, *, timeout: float) -> Path:
        out = path.with_suffix(".pdf")
        out.write_bytes(b"%PDF-office")
        return out

    monkeypatch.setattr(dtp, "_convert_markdown_with_pandoc", _fake_pandoc)
    monkeypatch.setattr(dtp, "_convert_with_libreoffice", _fake_lo)

    zip_items, report = dtp._convert_zip_mixed_to_pdfs(src_zip, timeout=30, default_font="Noto Sans CJK SC")
    assert len(zip_items) == 3
    assert len(report["converted"]) == 3
    assert len(report["skipped"]) == 1
    assert report["skipped"][0]["path"] == "d.txt"


def test_zip_backslash_path_normalizes_to_nested_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Windows 风格 ``dir\\file.md`` 在 zip 内应解压为子目录，仍按 .md 转换。"""
    z = tmp_path / "bs.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(r"notes\chapter.md", "# ch")

    def _fake_pandoc(path: Path, **kwargs: object) -> Path:
        assert path.suffix.lower() == ".md"
        out = path.with_suffix(".pdf")
        out.write_bytes(b"%PDF")
        return out

    monkeypatch.setattr(dtp, "_convert_markdown_with_pandoc", _fake_pandoc)
    items, report = dtp._convert_zip_mixed_to_pdfs(z, timeout=10, default_font="")
    assert len(items) == 1
    assert report["converted"][0]["engine"] == "pandoc"


def test_zip_only_apple_metadata_skipped(tmp_path: Path):
    z = tmp_path / "mac.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("__MACOSX/._readme.md", b"garbage")
    with pytest.raises(RuntimeError, match="无支持的后缀"):
        dtp._convert_zip_mixed_to_pdfs(z, timeout=10, default_font="")


def test_zip_markdown_conversion_error_in_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    z = tmp_path / "one.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("a.md", "# x")

    def _boom(*a: object, **k: object) -> Path:
        raise RuntimeError("service_unavailable: pandoc not installed")

    monkeypatch.setattr(dtp, "_convert_markdown_with_pandoc", _boom)
    with pytest.raises(RuntimeError, match="未生成任何 PDF"):
        dtp._convert_zip_mixed_to_pdfs(z, timeout=10, default_font="")


def test_single_docx_not_routed_to_zip_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src_docx = tmp_path / "a.docx"
    src_docx.write_bytes(b"fake-docx-content")

    called = {"zip": False, "lo": False}

    def _fake_materialize(source: str, *, work_dir: Path, timeout: float) -> Path:
        return src_docx

    def _fake_is_pdf(source: str, *, timeout: float) -> bool:
        return False

    def _fake_zip(*args, **kwargs):
        called["zip"] = True
        raise AssertionError("zip branch should not be used for plain .docx input")

    def _fake_lo(path: Path, *, timeout: float) -> Path:
        called["lo"] = True
        out = tmp_path / "a.pdf"
        out.write_bytes(b"%PDF")
        return out

    def _fake_persist(**kwargs):
        return {"status": "completed", "provider": "libreoffice", "files": [{}], "total": 1}

    monkeypatch.setattr(dtp, "_materialize_source_locally", _fake_materialize)
    monkeypatch.setattr(dtp, "_detect_pdf_source", _fake_is_pdf)
    monkeypatch.setattr(dtp, "_convert_zip_mixed_to_pdfs", _fake_zip)
    monkeypatch.setattr(dtp, "_convert_with_libreoffice", _fake_lo)
    monkeypatch.setattr(dtp, "_persist_output_bytes", _fake_persist)

    out = dtp._convert_single_document_sync(
        user_id="u1",
        source=str(src_docx),
        timeout=30,
        storage="local",
        default_font="Noto Sans CJK SC",
    )
    assert out["status"] == "completed"
    assert called["zip"] is False
    assert called["lo"] is True


def test_docx_zip_container_not_treated_as_zip_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # 真实 docx 本质是 zip 容器；即便 zipfile.is_zipfile 为 True，也不应走 zip 批处理分支
    src_docx = tmp_path / "a.docx"
    with zipfile.ZipFile(src_docx, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")

    called = {"zip": False, "lo": False}

    def _fake_materialize(source: str, *, work_dir: Path, timeout: float) -> Path:
        return src_docx

    def _fake_is_pdf(source: str, *, timeout: float) -> bool:
        return False

    def _fake_is_zip_source(source: str, *, timeout: float) -> bool:
        # 模拟某些远端响应把 docx 标成 zip 的情况
        return True

    def _fake_zip(*args, **kwargs):
        called["zip"] = True
        raise AssertionError("zip branch should not be used for .docx input")

    def _fake_lo(path: Path, *, timeout: float) -> Path:
        called["lo"] = True
        out = tmp_path / "a.pdf"
        out.write_bytes(b"%PDF")
        return out

    def _fake_persist(**kwargs):
        return {"status": "completed", "provider": "libreoffice", "files": [{}], "total": 1}

    monkeypatch.setattr(dtp, "_materialize_source_locally", _fake_materialize)
    monkeypatch.setattr(dtp, "_detect_pdf_source", _fake_is_pdf)
    monkeypatch.setattr(dtp, "_is_zip_source", _fake_is_zip_source)
    monkeypatch.setattr(dtp, "_convert_zip_mixed_to_pdfs", _fake_zip)
    monkeypatch.setattr(dtp, "_convert_with_libreoffice", _fake_lo)
    monkeypatch.setattr(dtp, "_persist_output_bytes", _fake_persist)

    out = dtp._convert_single_document_sync(
        user_id="u1",
        source="https://example.com/download?id=1",
        timeout=30,
        storage="local",
        default_font="Noto Sans CJK SC",
    )
    assert out["status"] == "completed"
    assert called["zip"] is False
    assert called["lo"] is True


def test_zip_like_source_without_zip_suffix_routes_to_zip_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = tmp_path / "payload"
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("a.md", "# a")

    called = {"zip": False}

    def _fake_materialize(source: str, *, work_dir: Path, timeout: float) -> Path:
        return payload

    def _fake_is_pdf(source: str, *, timeout: float) -> bool:
        return False

    def _fake_is_zip_source(source: str, *, timeout: float) -> bool:
        return True

    def _fake_zip(path: Path, *, timeout: float, default_font: str):
        called["zip"] = True
        out = tmp_path / "a.pdf"
        out.write_bytes(b"%PDF")
        return [("a.pdf", out)], {"converted": [{"path": "a.md"}], "failed": [], "skipped": []}

    def _fake_persist(**kwargs):
        return {"status": "completed", "provider": "mixed_zip", "files": [{}], "total": 1}

    monkeypatch.setattr(dtp, "_materialize_source_locally", _fake_materialize)
    monkeypatch.setattr(dtp, "_detect_pdf_source", _fake_is_pdf)
    monkeypatch.setattr(dtp, "_is_zip_source", _fake_is_zip_source)
    monkeypatch.setattr(dtp, "_convert_zip_mixed_to_pdfs", _fake_zip)
    monkeypatch.setattr(dtp, "_persist_output_bytes", _fake_persist)

    out = dtp._convert_single_document_sync(
        user_id="u1",
        source="https://example.com/download?id=1",
        timeout=30,
        storage="local",
        default_font="Noto Sans CJK SC",
    )
    assert out["status"] == "completed"
    assert called["zip"] is True


def test_libreoffice_headless_env_svp_on_posix():
    if os.name == "nt":
        return
    env = lo_builtin.libreoffice_headless_env()
    assert env.get("SAL_USE_VCLPLUGIN") == "svp"
    assert env.get("SAL_NO_DISPLAY_FOR_OPENGL") == "1"


def test_libreoffice_headless_env_respects_vitoom_lo_no_svp(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VITOOM_LO_NO_SVP", "1")
    monkeypatch.delenv("SAL_USE_VCLPLUGIN", raising=False)
    env = lo_builtin.libreoffice_headless_env()
    assert "SAL_USE_VCLPLUGIN" not in env


def test_coerce_timeout_accepts_string_none_and_invalid():
    t_none = dtp._coerce_timeout_seconds("None")
    t_undef = dtp._coerce_timeout_seconds("undefined")
    t_bad = dtp._coerce_timeout_seconds("not-a-number")
    assert t_none == t_undef == t_bad == dtp._default_timeout()
    assert dtp._coerce_timeout_seconds("30") == 30.0
    assert dtp._coerce_timeout_seconds(0) == dtp._default_timeout()


def test_sanitize_markdown_replaces_img_with_md_image():
    t = r'<img src="images/x.png" style="width:1in"/> before'
    s = mds.sanitize_markdown_for_pdf_text(t)
    assert "![](images/x.png)" in s
    assert "<img" not in s


def test_sanitize_markdown_trims_inline_math_and_times():
    t = r"A.  $ 23×10 ^ { -8 }  $  B."
    s = mds.sanitize_markdown_for_pdf_text(t)
    assert r"\times" in s
    assert " $ " not in s


def test_sanitize_markdown_downgrades_broken_frac_to_text():
    t = r"设备折旧 $ \frac{\sum $设备实际报工时间 $ \times $工序单片实际时间"
    s = mds.sanitize_markdown_for_pdf_text(t)
    assert r"$\frac{\sum$" not in s
    assert "frac sum" in s
    assert r"$\times$" in s


def test_sanitize_markdown_keeps_valid_frac_and_cjk_text_font():
    t = r"$ \frac{\sum \mathrm{设备OEE}\times4320}{} $"
    s = mds.sanitize_markdown_for_pdf_text(t)
    assert r"\frac" in s
    assert r"\text{设备OEE}" in s


def test_sanitize_markdown_skips_fenced_blocks():
    t = "```\n$  x  $\n```\nline $  y  $ end"
    s = mds.sanitize_markdown_for_pdf_text(t)
    assert "$  x  $" in s
    assert "$y$" in s


def test_args_schema_coerces_string_nones_to_none():
    try:
        import crewai  # noqa: F401
    except Exception:
        return
    tool = dtp.build_document_to_pdf_tool(context={"user_id": "test-user"})
    schema = tool.args_schema
    parsed = schema(
        url="None",
        urls="null",
        timeout="None",
        default_font="undefined",
    )
    assert parsed.url is None
    assert parsed.urls is None
    assert parsed.timeout is None
    assert parsed.default_font is None


def test_convert_markdown_content_to_pdf_uses_pandoc_branch(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def _fake_convert(md_path: Path, *, timeout: float, default_font: str) -> Path:
        seen["md_name"] = md_path.name
        seen["md_text"] = md_path.read_text(encoding="utf-8")
        seen["timeout"] = timeout
        out = md_path.with_suffix(".pdf")
        out.write_bytes(b"%PDF-content")
        return out

    def _fake_persist(**kwargs: object) -> dict:
        seen["persist"] = kwargs
        return {
            "source": kwargs["source"],
            "status": "completed",
            "provider": kwargs["provider"],
            "files": [{"url": "http://example.com/report.pdf"}],
            "total": 1,
        }

    monkeypatch.setattr(dtp, "_convert_markdown_with_pandoc", _fake_convert)
    monkeypatch.setattr(dtp, "_persist_output_bytes", _fake_persist)

    payload = dtp.convert_documents_to_pdf_sync(
        user_id="u1",
        markdown="# 总结\n\n- 结论",
        filename="report.md",
        timeout=12,
        default_font="",
    )

    assert payload["status"] == "completed"
    assert payload["total"] == 1
    assert seen["md_name"] == "report.md"
    assert seen["md_text"] == "# 总结\n\n- 结论"
    assert seen["timeout"] == 12.0
    persist = seen["persist"]
    assert isinstance(persist, dict)
    assert persist["output_name"] == "report.pdf"
    assert persist["provider"] == "pandoc_content"


def test_convert_pdf_requires_source_or_content():
    payload = dtp.convert_documents_to_pdf_sync(user_id="u1", url=None, urls=None)
    assert payload["status"] == "failed"
    assert "markdown/content" in payload["error"]
