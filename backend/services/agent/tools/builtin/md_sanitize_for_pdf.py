"""Markitdown / docx 导出的脏 Markdown 的轻量归一，便于 Pandoc 识别行内数学并减轻 TeX 编译失败。

仅做窄场景、可预测替换：HTML ``<img>`` 转 ``![](...)``、围栏外对行内/块数学做去空白与常见 Unicode
算符替换。代码块（围栏）内不修改；不实现完整 Markdown/TeX 解析器。
"""

from __future__ import annotations

import re
from typing import List

# ``<img src="..." .../>``
_RE_HTML_IMG = re.compile(
    r'<img\s[^>]*?src\s*=\s*(["\'])([^"\'>]+)\1[^>]*>',
    re.IGNORECASE | re.DOTALL,
)
_RE_HTML_IMG_BARE = re.compile(
    r'<img\s[^>]*?src\s*=\s*([^\s"\'<>=]+)(?:\s|>)',
    re.IGNORECASE,
)

_MATH_UNICODE = (
    ("×", r"\times"),
    ("·", r"\cdot"),
    ("＝", "="),
    ("﹣", "-"),
    ("﹢", "+"),
    ("：", ":"),
)
_MATH_TEXT_COMMAND_RE = re.compile(r"\\(?:mathrm|text)\{([^{}]*)\}")


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")


def _tex_braces_balanced(text: str) -> bool:
    depth = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _math_body_to_plain_text(body: str) -> str:
    """Best-effort fallback for broken OCR math so TeX does not scan past EOF."""

    s = body.strip()
    s = _MATH_TEXT_COMMAND_RE.sub(lambda m: m.group(1), s)
    replacements = {
        r"\frac": "frac ",
        r"\sum": "sum",
        r"\times": "×",
        r"\cdot": "·",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _replace_img_tags(text: str) -> str:
    def rep(m: re.Match[str]) -> str:
        return f"![]({m.group(2)})"

    t = _RE_HTML_IMG.sub(rep, text)
    t = _RE_HTML_IMG_BARE.sub(
        lambda m: f"![]({m.group(1).rstrip('/>')})",
        t,
    )
    return t


def _normalize_math_body(body: str) -> str:
    s = body.strip()
    s = re.sub(r"[ \t]+", " ", s)
    for u, tex in _MATH_UNICODE:
        s = s.replace(u, tex)
    s = _MATH_TEXT_COMMAND_RE.sub(
        lambda m: rf"\text{{{m.group(1)}}}" if _has_cjk(m.group(1)) else m.group(0),
        s,
    )
    return s


def _render_math_span(delim: str, body: str) -> str:
    normalized = _normalize_math_body(body)
    if not _tex_braces_balanced(normalized):
        return _math_body_to_plain_text(normalized)
    return delim + normalized + delim


def _fix_math_on_line(line: str) -> str:
    """同一行内：``$$...$$`` 与单 ``$...$`` 成对处理，统一去掉数学体两端空白与多余空格。"""
    if "$" not in line:
        return line
    parts: List[str] = []
    i, n = 0, len(line)
    while i < n:
        if i + 1 < n and line[i : i + 2] == "$$":
            j = line.find("$$", i + 2)
            if j == -1:
                parts.append(line[i:])
                break
            parts.append(_render_math_span("$$", line[i + 2 : j]))
            i = j + 2
        elif line[i] == "$":
            j = line.find("$", i + 1)
            if j == -1:
                parts.append(line[i:])
                break
            parts.append(_render_math_span("$", line[i + 1 : j]))
            i = j + 1
        else:
            parts.append(line[i])
            i += 1
    return "".join(parts)


def _apply_outside_fences(text: str) -> str:
    out: List[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        s = line.lstrip("\ufeff")
        fstart = s.lstrip()
        if fstart.startswith("```") or fstart.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            out.append(_fix_math_on_line(line))
    return "".join(out)


def sanitize_markdown_for_pdf_text(text: str) -> str:
    """对 PDF 用 Pandoc+TeX 线：转 ``<img>``、行内/块数字符与空白规范化。"""
    if not text:
        return text
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _replace_img_tags(t)
    t = _apply_outside_fences(t)
    return t
