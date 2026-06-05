"""LibreOffice ``soffice`` 无头转换：供多个 builtin 工具复用。

涵盖：可执行文件探测、无显示器环境变量、``--convert-to`` 子进程封装。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

# 与 document_to_pdf 历史行为一致：批量/服务器场景下尽量稳妥的启动参数。
SOFFICE_FULL_HEADLESS_ARGS: Tuple[str, ...] = (
    "--headless",
    "--invisible",
    "--nologo",
    "--nolockcheck",
    "--nodefault",
    "--norestore",
    "--nofirststartwizard",
)

# 轻量调用（如仅需 Markdown 导出）：保留最常用的无头参数。
SOFFICE_LIGHT_HEADLESS_ARGS: Tuple[str, ...] = ("--headless", "--nologo")


def _env_truthy(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def locate_soffice_binary() -> Optional[str]:
    """定位 LibreOffice ``soffice`` 可执行文件。

    1. ``VITOOM_SOFFICE_BIN`` / ``LIBREOFFICE_BIN``：显式覆盖；
    2. ``PATH`` 中的 ``soffice`` / ``libreoffice``；
    3. 与当前 Python 同目录的 ``soffice``（conda / 打包场景）；
    4. macOS：标准 ``.app`` 安装路径（避免仅 GUI 安装未入 PATH）。
    """
    env_bin = (
        os.environ.get("VITOOM_SOFFICE_BIN", "").strip()
        or os.environ.get("LIBREOFFICE_BIN", "").strip()
    )
    if env_bin and Path(env_bin).is_file() and os.access(env_bin, os.X_OK):
        return env_bin
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    sibling = Path(sys.executable).resolve().parent / "soffice"
    if sibling.is_file() and os.access(str(sibling), os.X_OK):
        return str(sibling)
    if sys.platform == "darwin":
        mac_app = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
        if mac_app.is_file() and os.access(str(mac_app), os.X_OK):
            return str(mac_app)
    return None


def libreoffice_headless_env() -> Dict[str, str]:
    """为无显示器/无 X11 环境准备子进程环境。

    ``SAL_USE_VCLPLUGIN=svp`` 使用屏外后端，适合 Linux 无头 ``convert-to``。
    本机带显示器调试可设 ``VITOOM_LO_NO_SVP=1`` 跳过注入。

    **macOS**：即使用 ``--headless``，程序坞仍可能出现 LibreOffice 图标，属常见现象。
    """
    env: Dict[str, str] = dict(os.environ)
    if _env_truthy("VITOOM_LO_NO_SVP"):
        return env
    if os.name == "nt":
        return env
    env.setdefault("SAL_USE_VCLPLUGIN", "svp")
    env.setdefault("SAL_NO_DISPLAY_FOR_OPENGL", "1")
    return env


def build_soffice_convert_command(
    *,
    soffice_bin: str,
    input_path: Path,
    out_dir: Path,
    convert_to: str,
    profile_dir: Path,
    lo_args: Sequence[str],
) -> List[str]:
    """拼装 ``soffice … --convert-to …`` 命令（不含 cwd / env）。"""
    profile_uri = profile_dir.resolve().as_uri()
    return [
        soffice_bin,
        *lo_args,
        f"-env:UserInstallation={profile_uri}",
        "--convert-to",
        convert_to,
        "--outdir",
        str(out_dir.resolve()),
        str(input_path.resolve()),
    ]


def run_soffice_convert_sync(
    *,
    soffice_bin: str,
    input_path: Path,
    out_dir: Path,
    convert_to: str,
    profile_dir: Path,
    lo_args: Sequence[str],
    timeout: float,
    cwd: Optional[Path] = None,
    timeout_floor: float = 1.0,
) -> subprocess.CompletedProcess[str]:
    """执行 LibreOffice 批量转换；返回 ``CompletedProcess``（不 ``check``）。"""
    cmd = build_soffice_convert_command(
        soffice_bin=soffice_bin,
        input_path=input_path,
        out_dir=out_dir,
        convert_to=convert_to,
        profile_dir=profile_dir,
        lo_args=lo_args,
    )
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(float(timeout_floor), float(timeout)),
        cwd=str(cwd) if cwd is not None else None,
        env=libreoffice_headless_env(),
        check=False,
    )


def pick_convert_output_file(
    out_dir: Path,
    *,
    source_stem: str,
    output_suffix: str,
) -> Path:
    """在 ``out_dir`` 中选取转换产物：优先 ``{stem}{suffix}``，否则取最新匹配的 ``*{suffix}``。"""
    suf = output_suffix if output_suffix.startswith(".") else f".{output_suffix}"
    expected = out_dir / f"{source_stem}{suf}"
    if expected.is_file():
        return expected
    pattern = f"*{suf}"
    candidates = sorted(
        out_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"libreoffice conversion produced no {suf} file in {out_dir}")
    return candidates[0]


def normalize_lo_md_link_target(raw: str) -> str:
    """LibreOffice 导出 Markdown 内相对链接规范化（用于配图路径匹配）。"""
    from urllib.parse import unquote

    text = unquote((raw or "").strip().strip('"').strip("'"))
    text = text.split("#", 1)[0].split("?", 1)[0]
    text = text.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def rewrite_libreoffice_markdown_sidecars(
    md_text: str,
    export_dir: Path,
    md_file: Path,
    *,
    image_suffixes: AbstractSet[str],
    start_index: int = 1,
) -> Tuple[str, List[Tuple[str, bytes]]]:
    """收集 LibreOffice 导出目录中的配图，并把 Markdown/HTML 里的相对路径改为 ``images/…``。

    ``start_index``：首张 sidecar 使用的序号（生成 ``images/img_{序号:03d}.…``）。
    若同一段 Markdown 已由 ``_extract_data_uri_images`` 等写入 ``img_001`` … ``img_K``，
    应传入 ``start_index=K+1``（``K = len(此前条目)``），避免 zip 内同名 arc 冲突。
    """
    entries: List[Tuple[str, bytes]] = []
    export_resolved = export_dir.resolve()
    md_resolved = md_file.resolve()
    pairs: List[Tuple[str, Path]] = []
    for path in export_resolved.rglob("*"):
        if not path.is_file() or path.resolve() == md_resolved:
            continue
        suf = path.suffix.lower()
        if suf not in image_suffixes:
            continue
        try:
            rel_posix = path.relative_to(export_resolved).as_posix()
        except ValueError:
            continue
        pairs.append((rel_posix, path))
    if not pairs:
        return md_text, entries

    pairs.sort(key=lambda item: item[0])
    rel_to_arc: Dict[str, str] = {}
    basename_hits: Dict[str, List[str]] = {}
    for rel_posix, _path in pairs:
        basename_hits.setdefault(Path(rel_posix).name, []).append(rel_posix)

    try:
        base_slot = int(start_index)
    except (TypeError, ValueError):
        base_slot = 1
    base_slot = max(1, base_slot)
    slot = base_slot - 1
    for rel_posix, path in pairs:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        ext = path.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        slot += 1
        arc = f"images/img_{slot:03d}{ext}"
        entries.append((arc, data))
        rel_to_arc[rel_posix] = arc
        rel_to_arc[f"./{rel_posix}"] = arc

    def _lookup_arc(norm: str) -> Optional[str]:
        if norm in rel_to_arc:
            return rel_to_arc[norm]
        base = Path(norm).name
        cands = basename_hits.get(base)
        if cands and len(cands) == 1 and cands[0] in rel_to_arc:
            return rel_to_arc[cands[0]]
        return None

    def _repl_md_img(m: re.Match[str]) -> str:
        alt, raw_target = m.group(1), m.group(2)
        arc = _lookup_arc(normalize_lo_md_link_target(raw_target))
        if not arc:
            return m.group(0)
        return f"![{alt}]({arc})"

    out = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        _repl_md_img,
        md_text,
    )

    def _repl_html_img(m: re.Match[str]) -> str:
        full = m.group(0)
        sm = re.search(
            r"""\bsrc\s*=\s*([\"'])([^\"'>]+)\1""",
            full,
            re.IGNORECASE,
        )
        if not sm:
            return full
        arc = _lookup_arc(normalize_lo_md_link_target(sm.group(2)))
        if not arc:
            return full
        return re.sub(
            r"""(\bsrc\s*=\s*)([\"'])([^\"'>]+)(\2)""",
            rf"\1\2{arc}\4",
            full,
            count=1,
            flags=re.IGNORECASE,
        )

    out = re.sub(r"<img\b[^>]+/?>", _repl_html_img, out, flags=re.IGNORECASE)
    return out, entries
