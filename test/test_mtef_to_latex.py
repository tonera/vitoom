"""Tests for the MathType-OLE → LaTeX recovery path used by ``document_to_markdown``.

Covers:

* :func:`mathtype_ole_to_latex` against a real MathType v5 OLE blob (fixture
  taken from a production ``.docx`` shipped to the team) — guards parser
  behaviour & the ``olefile``-based replacement of upstream's hand-rolled OLE
  reader, plus the stdout-silencing wrapper.
* :func:`_replace_image_refs_with_latex` — guards both Markdown ``![…](…)``
  and Pandoc 3.x raw-HTML ``<img src="…" />`` rewriting paths, including the
  LaTeX-as-``re.sub``-replacement gotcha (``\\frac`` looks like a backref).
* :func:`_extract_mathtype_latex_map` — guards the docx XML / rels traversal
  on a programmatically-built minimal ``.docx`` that embeds the same OLE
  fixture, so we don't ship a multi-MB binary in the repo.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from backend.services.agent.tools.builtin._vendor.mtef import mathtype_ole_to_latex
from backend.services.agent.tools.builtin.document_to_markdown import (
    _extract_mathtype_latex_map,
    _replace_image_refs_with_latex,
)


FIXTURE_OLE = Path(__file__).parent / "fixtures" / "mtef" / "sample_ole.bin"


def test_fixture_present_and_nonempty():
    """Sanity-check the fixture so a missing/zero-byte file fails loudly,
    rather than masquerading as a parser regression."""
    assert FIXTURE_OLE.exists(), f"fixture missing: {FIXTURE_OLE}"
    assert FIXTURE_OLE.stat().st_size > 0


def test_mathtype_ole_to_latex_recovers_known_equation():
    """The fixture is the first equation from the production exam paper:
    ``2.3 × 10^{-8}`` rendered via MathType. Output must contain a
    ``$ … $`` wrapper, the multiplication sign, and a ``^`` superscript so
    we know record-stream → AST → LaTeX is end-to-end functional."""
    latex = mathtype_ole_to_latex(FIXTURE_OLE.read_bytes())
    assert latex is not None
    assert latex.startswith("$") and latex.endswith("$")
    assert "10" in latex
    assert "^" in latex
    assert "-8" in latex or "- 8" in latex


@pytest.mark.parametrize(
    "blob",
    [b"", b"not an ole file", b"\x00" * 32],
    ids=["empty", "garbage", "zero-padded"],
)
def test_mathtype_ole_to_latex_returns_none_on_garbage(blob):
    """Robustness: anything that isn't a valid OLE compound with an
    ``Equation Native`` stream must return ``None`` rather than raise —
    callers fall back to keeping the source WMF on a ``None`` reply."""
    assert mathtype_ole_to_latex(blob) is None


def test_mathtype_ole_to_latex_silences_parser_stdout(capsys):
    """Upstream parser emits diagnostic ``print(...)`` for malformed
    streams. Our wrapper must not let any of that leak to stdout."""
    mathtype_ole_to_latex(b"\x00" * 4096)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_replace_image_refs_handles_markdown_and_html_forms():
    """Both Pandoc output flavours must be replaced; the result keeps the
    surrounding text untouched and inserts the LaTeX with breathing room
    so it doesn't fuse with adjacent characters."""
    md = (
        "before ![](/abs/img1.wmf) middle "
        "<img src=\"/abs/img1.wmf\" style=\"x\" /> "
        "<img src=\"/abs/img1.wmf\"/> after"
    )
    out = _replace_image_refs_with_latex(md, ["/abs/img1.wmf"], "$ x^2 $")
    # All three references gone, replaced by the LaTeX (with surrounding spaces).
    assert ".wmf" not in out
    assert out.count("$ x^2 $") == 3
    assert out.startswith("before ")
    assert out.endswith(" after")


def test_replace_image_refs_does_not_touch_unrelated_paths():
    md = "![](/abs/keep.png) and ![](/abs/img1.wmf)"
    out = _replace_image_refs_with_latex(md, ["/abs/img1.wmf"], "$ y $")
    assert "/abs/keep.png" in out
    assert "/abs/img1.wmf" not in out


def test_replace_image_refs_safe_against_backref_metacharacters():
    """``\\frac``, ``\\1`` etc. inside LaTeX must not be treated as
    ``re.sub`` back-references. Smoke-tested with the upstream wrapper —
    if we used a string ``repl`` directly this would raise
    ``error: bad escape \\f``."""
    md = "ref ![](/abs/x.wmf) end"
    latex = "$ \\frac{a}{b}_1 $"
    out = _replace_image_refs_with_latex(md, ["/abs/x.wmf"], latex)
    assert "\\frac{a}{b}_1" in out
    assert ".wmf" not in out


def test_replace_image_refs_with_no_candidates_is_noop():
    md = "no ![](/abs/img.wmf)"
    assert _replace_image_refs_with_latex(md, [], "$ x $") == md


def _build_minimal_docx(ole_bytes: bytes) -> bytes:
    """Construct a minimal valid ``.docx`` (a zip with the four files
    ``_extract_mathtype_latex_map`` touches), embedding the fixture OLE
    via the same ``<w:object>`` shape Word produces for MathType."""
    rels = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.wmf"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject" Target="embeddings/oleObject1.bin"/>
</Relationships>'''
    document = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:o="urn:schemas-microsoft-com:office:office"
  xmlns:v="urn:schemas-microsoft-com:vml">
  <w:body>
    <w:p>
      <w:r>
        <w:object>
          <v:shape><v:imagedata r:id="rId1"/></v:shape>
          <o:OLEObject ProgID="Equation.DSMT4" r:id="rId2"/>
        </w:object>
      </w:r>
    </w:p>
  </w:body>
</w:document>'''
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b"<types/>")  # placeholder; we never read it
        zf.writestr("word/_rels/document.xml.rels", rels)
        zf.writestr("word/document.xml", document)
        zf.writestr("word/embeddings/oleObject1.bin", ole_bytes)
    return buf.getvalue()


def test_extract_mathtype_latex_map_pairs_imagedata_with_ole(tmp_path):
    docx_path = tmp_path / "tiny.docx"
    docx_path.write_bytes(_build_minimal_docx(FIXTURE_OLE.read_bytes()))

    mapping = _extract_mathtype_latex_map(docx_path)

    assert mapping, "expected one (image1.wmf -> latex) entry"
    assert "image1.wmf" in mapping
    latex = mapping["image1.wmf"]
    assert latex.startswith("$") and latex.endswith("$")
    assert "10" in latex


def test_extract_mathtype_latex_map_returns_empty_for_non_docx(tmp_path):
    not_a_docx = tmp_path / "junk.docx"
    not_a_docx.write_bytes(b"definitely not a zip")
    assert _extract_mathtype_latex_map(not_a_docx) == {}


def test_extract_mathtype_latex_map_skips_non_docx_extensions(tmp_path):
    other_ext = tmp_path / "thing.pdf"
    other_ext.write_bytes(b"%PDF-1.4 ...")
    assert _extract_mathtype_latex_map(other_ext) == {}


def test_extract_mathtype_latex_map_ignores_non_equation_ole(tmp_path):
    """OLEs with a non-Equation ProgID (e.g. embedded Excel) must be
    ignored, leaving the wmf as a normal image — no false-positive
    LaTeX replacement that would corrupt unrelated embeds."""
    rels = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.wmf"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject" Target="embeddings/oleObject1.bin"/>
</Relationships>'''
    document = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:o="urn:schemas-microsoft-com:office:office"
  xmlns:v="urn:schemas-microsoft-com:vml">
  <w:body><w:p><w:r><w:object>
    <v:shape><v:imagedata r:id="rId1"/></v:shape>
    <o:OLEObject ProgID="Excel.Sheet.12" r:id="rId2"/>
  </w:object></w:r></w:p></w:body>
</w:document>'''
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b"<types/>")
        zf.writestr("word/_rels/document.xml.rels", rels)
        zf.writestr("word/document.xml", document)
        zf.writestr("word/embeddings/oleObject1.bin", FIXTURE_OLE.read_bytes())
    docx_path = tmp_path / "excel_embed.docx"
    docx_path.write_bytes(buf.getvalue())

    assert _extract_mathtype_latex_map(docx_path) == {}
