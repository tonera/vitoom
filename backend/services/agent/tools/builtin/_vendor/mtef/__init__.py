"""MathType OLE → LaTeX (MTEF v5).

See ``./NOTICE.md`` and ``./LICENSE`` for upstream attribution.

Public surface is intentionally tiny: pass the raw bytes of an OLE compound
file (a ``word/embeddings/oleObjectN.bin`` extracted from a ``.docx``) and get
back a LaTeX string already wrapped with ``$ … $`` (suitable for inline
embedding inside Markdown), or ``None`` on any error / non-MathType OLE.

Errors are intentionally swallowed: the caller in ``document_to_markdown``
falls back to keeping the source WMF whenever LaTeX recovery fails, so a
single broken stream must never break the surrounding ``.docx → .md`` pipeline.
"""

from __future__ import annotations

import contextlib
import io
import logging
import struct
from typing import Optional

import olefile

from ._mtef import MTEF, oleCbHdr

logger = logging.getLogger(__name__)


__all__ = ["mathtype_ole_to_latex"]


def mathtype_ole_to_latex(ole_bytes: bytes) -> Optional[str]:
    """Translate one MathType OLE blob into LaTeX.

    Returns ``None`` if ``ole_bytes`` is empty, not a valid OLE compound
    file, lacks an ``Equation Native`` stream, or the embedded MTEF body
    fails to parse. The MTEF parser writes diagnostic ``print(...)`` calls
    onto stdout for malformed records — those are captured into a discarded
    ``StringIO`` buffer here so production logs stay clean.
    """
    if not ole_bytes:
        return None

    try:
        ole = olefile.OleFileIO(io.BytesIO(ole_bytes))
    except Exception as exc:  # not an OLE file at all
        logger.debug("mathtype_ole_to_latex: not an OLE compound file (%s)", exc)
        return None

    try:
        if not ole.exists("Equation Native"):
            return None
        try:
            eq_stream = ole.openstream("Equation Native").read()
        except Exception as exc:
            logger.debug(
                "mathtype_ole_to_latex: failed to read Equation Native stream (%s)",
                exc,
            )
            return None
    finally:
        try:
            ole.close()
        except Exception:
            pass

    if len(eq_stream) <= oleCbHdr:
        return None
    cb_hdr = struct.unpack_from("<H", eq_stream, 0)[0]
    if cb_hdr != oleCbHdr:
        # Header layout we don't recognise — bail rather than risk feeding the
        # parser garbage. ``EQNOLEFILEHDR`` always starts with cbHdr=0x001c.
        logger.debug(
            "mathtype_ole_to_latex: unexpected EQNOLEFILEHDR cbHdr=0x%04x", cb_hdr
        )
        return None
    cb_size = struct.unpack_from("<I", eq_stream, 6)[0]
    body = eq_stream[oleCbHdr : oleCbHdr + cb_size] if cb_size else eq_stream[oleCbHdr:]
    if not body:
        return None

    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            eqn = MTEF.parse_body(body)
            latex_obj = eqn.Translate()
    except Exception as exc:  # parser blew up on an exotic stream
        logger.debug("mathtype_ole_to_latex: parser raised (%s)", exc)
        return None

    if isinstance(latex_obj, tuple):
        latex = latex_obj[0]
    else:
        latex = latex_obj
    if not isinstance(latex, str):
        return None
    latex = latex.strip()
    return latex or None
