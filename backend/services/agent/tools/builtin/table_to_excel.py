"""table_to_excel 工具：把 LLM 生成的表格内容导出为 Excel xlsx。"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re
import zipfile
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

from backend.services.agent.tools.registry import register_tool
from backend.storage import get_storage_manager_by_mode
from backend.utils import generate_uuid, safe_filename

from ._arg_utils import clean_optional_str, coerce_optional

logger = logging.getLogger(__name__)

TABLE_TO_EXCEL_TOOL_NAME = "table_to_excel"
TABLE_TO_EXCEL_DESCRIPTION = (
    "把聊天中 LLM 生成或整理出的表格内容导出为 Excel .xlsx 文件。"
    "当用户明确要求『把上面的表格导出为 Excel』『保存成 xlsx』『导出表格』时使用。"
    "调用时只传 content（表格正文）以及可选 filename/sheet_name；content 可为 Markdown/HTML 表格、"
    "CSV/TSV 文本，或 JSON 行数组。"
    "工具返回 files[]：向用户写可点击链接时用 url，须原样复制。"
)
TABLE_TO_EXCEL_DOCSTRING = (
    "Export generated chat table content to an Excel .xlsx artifact. Accepts "
    "one main `content` argument containing a Markdown/HTML table, CSV/TSV text, or "
    "a JSON rows array. Optional: `filename`, `sheet_name`. Return JSON with files[].url."
)

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_MARKDOWN_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_HTML_TABLE_RE = re.compile(r"<\s*table\b", re.IGNORECASE)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _clean_str(value: Any) -> str:
    return clean_optional_str(value) or ""


def _coerce_table_text(*, markdown: Any = None, content: Any = None, table: Any = None) -> str:
    for value in (markdown, content, table):
        text = clean_optional_str(value)
        if text:
            return text
    return ""


def _safe_export_stem(*, filename: Any = None, title: Any = None, default: str = "table") -> str:
    raw = clean_optional_str(filename) or clean_optional_str(title) or default
    name = Path(str(raw or default)).name.strip() or default
    stem = Path(name).stem if Path(name).suffix else name
    return safe_filename(stem) or default


def _normalize_sheet_name(value: Any) -> str:
    name = clean_optional_str(value) or "Sheet1"
    name = re.sub(r"[\[\]:*?/\\]", " ", name).strip() or "Sheet1"
    return name[:31] or "Sheet1"


def _stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def _coerce_rows_argument(rows: Any) -> List[List[str]]:
    if rows in (None, "", [], {}):
        return []
    parsed = rows
    if isinstance(rows, str):
        text = rows.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
    if not isinstance(parsed, list):
        return []
    if not parsed:
        return []

    if all(isinstance(item, dict) for item in parsed):
        keys: List[str] = []
        seen = set()
        for item in parsed:
            for key in (item or {}).keys():
                key_text = str(key)
                if key_text not in seen:
                    seen.add(key_text)
                    keys.append(key_text)
        if not keys:
            return []
        table = [keys]
        for item in parsed:
            row = [_stringify_cell((item or {}).get(key)) for key in keys]
            table.append(row)
        return table

    table: List[List[str]] = []
    for item in parsed:
        if isinstance(item, (list, tuple)):
            table.append([_stringify_cell(cell) for cell in item])
        else:
            table.append([_stringify_cell(item)])
    return _pad_rows(table)


def _coerce_content_payload(content: Any) -> Dict[str, Any]:
    text = clean_optional_str(content)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {"content": text}
    if isinstance(parsed, dict):
        return dict(parsed)
    if isinstance(parsed, list):
        return {"rows": parsed}
    return {"content": text}


def _split_markdown_row(line: str) -> List[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    cells: List[str] = []
    buf: List[str] = []
    escaped = False
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "|":
            cells.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


def _parse_markdown_table(text: str) -> List[List[str]]:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    for idx in range(len(lines) - 1):
        if "|" not in lines[idx] or not _MARKDOWN_SEPARATOR_RE.match(lines[idx + 1]):
            continue
        table = [_split_markdown_row(lines[idx])]
        cursor = idx + 2
        while cursor < len(lines) and "|" in lines[cursor]:
            table.append(_split_markdown_row(lines[cursor]))
            cursor += 1
        return _pad_rows(table)
    return []


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[List[List[str]]] = []
        self._table_depth = 0
        self._current_table: List[List[str]] = []
        self._current_row: Optional[List[str]] = None
        self._current_cell: Optional[List[str]] = None
        self._cell_tag = ""

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        del attrs
        normalized = tag.lower()
        if normalized == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_table = []
            return
        if self._table_depth <= 0:
            return
        if normalized == "tr":
            self._current_row = []
            return
        if normalized in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
            self._cell_tag = normalized
            return
        if self._current_cell is not None and normalized in {"br", "p", "div"}:
            self._current_cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"td", "th"} and self._current_cell is not None:
            cell = re.sub(r"\s+", " ", "".join(self._current_cell)).strip()
            if self._current_row is not None:
                self._current_row.append(unescape(cell))
            self._current_cell = None
            self._cell_tag = ""
            return
        if normalized == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
            return
        if normalized == "table" and self._table_depth > 0:
            self._table_depth -= 1
            if self._table_depth == 0 and self._current_table:
                self.tables.append(self._current_table)
                self._current_table = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)


def _parse_html_table(text: str) -> List[List[str]]:
    body = str(text or "").strip()
    if not body or not _HTML_TABLE_RE.search(body):
        return []
    parser = _HtmlTableParser()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        return []
    if not parser.tables:
        return []
    # OCR table 模式通常只返回一个表；若有多个，取内容最多的那个。
    table = max(parser.tables, key=lambda rows: sum(len(row) for row in rows))
    return _pad_rows(table)


def _parse_delimited_table(text: str) -> List[List[str]]:
    body = str(text or "").strip()
    if not body:
        return []
    delimiter = "\t" if "\t" in body and "," not in body else ","
    try:
        rows = list(csv.reader(io.StringIO(body), delimiter=delimiter))
    except Exception:
        return []
    rows = [[cell.strip() for cell in row] for row in rows if any(str(cell).strip() for cell in row)]
    return _pad_rows(rows)


def _pad_rows(rows: Sequence[Sequence[Any]]) -> List[List[str]]:
    width = max((len(row) for row in rows), default=0)
    if width <= 0:
        return []
    return [[_stringify_cell(cell) for cell in row] + [""] * (width - len(row)) for row in rows]


def _coerce_table_rows(*, rows: Any = None, markdown: Any = None, content: Any = None, table: Any = None) -> List[List[str]]:
    payload = _coerce_content_payload(content)
    if payload:
        rows = rows if rows not in (None, "", [], {}) else payload.get("rows")
        markdown = markdown if markdown not in (None, "") else payload.get("markdown")
        table = table if table not in (None, "") else payload.get("table")
        content = payload.get("content") or payload.get("csv") or payload.get("tsv")
    from_rows = _coerce_rows_argument(rows)
    if from_rows:
        return from_rows
    text = _coerce_table_text(markdown=markdown, content=content, table=table)
    parsed = _parse_html_table(text)
    if parsed:
        return parsed
    parsed = _parse_markdown_table(text)
    if parsed:
        return parsed
    return _parse_delimited_table(text)


def _column_name(index: int) -> str:
    n = index
    name = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        name = chr(65 + rem) + name
    return name or "A"


def _worksheet_xml(rows: Sequence[Sequence[str]]) -> str:
    xml_rows: List[str] = []
    for r_idx, row in enumerate(rows, start=1):
        cells: List[str] = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{_column_name(c_idx)}{r_idx}"
            text = escape(_stringify_cell(value), {'"': "&quot;"})
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        "</worksheet>"
    )


def _build_xlsx(rows: Sequence[Sequence[str]], *, sheet_name: str = "Sheet1") -> bytes:
    safe_sheet = escape(_normalize_sheet_name(sheet_name), {'"': "&quot;"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="{safe_sheet}" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        zf.writestr("xl/worksheets/sheet1.xml", _worksheet_xml(rows))
    return buf.getvalue()


def _storage_file_to_tool_file(file_row: Dict[str, Any]) -> Dict[str, Any]:
    storage_path = str(file_row.get("storage_path") or "").strip()
    return {
        "file_id": str(file_row.get("id") or ""),
        "url": file_row.get("http_url") or file_row.get("url"),
        "thumb_url": file_row.get("thumb_url"),
        "storage_path": storage_path or None,
        "file_name": Path(storage_path).name if storage_path else file_row.get("file_name"),
        "file_size": file_row.get("file_size"),
        "mime_type": file_row.get("mime_type") or _XLSX_MIME,
        "index": 0,
    }


def _persist_xlsx_result(
    *,
    user_id: str,
    source: str,
    xlsx_bytes: bytes,
    output_name: str,
    row_count: int,
    column_count: int,
    storage: str,
    started_at: datetime,
) -> Dict[str, Any]:
    from backend.database import Task

    task_id = generate_uuid()
    completed_at = datetime.utcnow()
    Task.create(
        id=task_id,
        user_id=user_id,
        task_type="text",
        prompt=f"Export table to excel: {source}",
        params={
            "job_type": "TABLE_TO_EXCEL",
            "provider": "local",
            "row_count": row_count,
            "column_count": column_count,
        },
        status="completed",
        storage=storage,
    )
    storage_manager = get_storage_manager_by_mode(str(storage or "local"))
    file_row = asyncio.run(
        storage_manager.save_file(
            file_data=xlsx_bytes,
            user_id=user_id,
            category="document",
            filename=safe_filename(output_name),
            task_id=task_id,
            metadata={
                "job_type": "TABLE_TO_EXCEL",
                "provider": "local",
                "row_count": row_count,
                "column_count": column_count,
                "mime_type": _XLSX_MIME,
            },
            storage=storage,
        )
    )
    Task.update(task_id, status="completed", progress=100, started_at=started_at, completed_at=completed_at)
    return {
        "tool": TABLE_TO_EXCEL_TOOL_NAME,
        "source": source,
        "task_id": task_id,
        "status": "completed",
        "provider": "local",
        "files": [_storage_file_to_tool_file(file_row)],
        "total": 1,
        "row_count": row_count,
        "column_count": column_count,
        "mime_type": _XLSX_MIME,
    }


def _failure_payload(error: str) -> Dict[str, Any]:
    return {
        "tool": TABLE_TO_EXCEL_TOOL_NAME,
        "status": "failed",
        "files": [],
        "total": 0,
        "error": str(error),
    }


def export_table_to_excel_sync(
    *,
    user_id: str,
    markdown: Any = None,
    content: Any = None,
    table: Any = None,
    rows: Any = None,
    filename: Any = None,
    title: Any = None,
    sheet_name: Any = None,
    storage: str = "local",
) -> Dict[str, Any]:
    if not _clean_str(user_id):
        return _failure_payload("table_to_excel requires a user_id (bound via agent context)")
    table_rows = _coerce_table_rows(rows=rows, markdown=markdown, content=content, table=table)
    if not table_rows:
        return _failure_payload("table_to_excel requires table content: markdown/content/table or rows")
    stem = _safe_export_stem(filename=filename, title=title)
    output_name = f"{stem}.xlsx"
    started_at = datetime.utcnow()
    try:
        xlsx_bytes = _build_xlsx(table_rows, sheet_name=_normalize_sheet_name(sheet_name))
        return _persist_xlsx_result(
            user_id=user_id,
            source=f"content:{output_name}",
            xlsx_bytes=xlsx_bytes,
            output_name=output_name,
            row_count=len(table_rows),
            column_count=max((len(row) for row in table_rows), default=0),
            storage=storage,
            started_at=started_at,
        )
    except Exception as exc:
        logger.exception("table_to_excel export failed")
        return _failure_payload(str(exc))


@register_tool(
    name=TABLE_TO_EXCEL_TOOL_NAME,
    description=TABLE_TO_EXCEL_DESCRIPTION,
    tags=["excel", "xlsx", "table", "spreadsheet", "表格", "导出excel", "导出表格"],
    provider="local",
    enabled=True,
)
def build_table_to_excel_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()

    try:
        from crewai.tools import BaseTool
    except Exception as exc:
        raise RuntimeError("crewai is required to register native agent tools") from exc

    try:
        from pydantic import BaseModel, ConfigDict, Field, field_validator
    except Exception as exc:
        raise RuntimeError("pydantic is required to build table_to_excel tool") from exc

    class TableToExcelArgs(BaseModel):
        model_config = ConfigDict(extra="ignore")

        content: str = Field(
            default="",
            description=(
                "Table body. Pass Markdown table, TSV/CSV text, or JSON row array string; "
                "do not pass tool invocation as XML/HTML text."
            ),
        )
        filename: Optional[str] = Field(default=None, description="Output filename; .xlsx extension may be omitted.")
        sheet_name: Optional[str] = Field(default=None, description="Excel worksheet name.")

        @field_validator("content", "filename", "sheet_name", mode="before")
        @classmethod
        def _normalize_llm_string_nones(cls, value: Any) -> Any:
            return coerce_optional(value)

    class TableToExcelTool(BaseTool):
        name: str = TABLE_TO_EXCEL_TOOL_NAME
        description: str = TABLE_TO_EXCEL_DESCRIPTION
        args_schema: type = TableToExcelArgs

        def _run(self, **kwargs: Any) -> str:
            try:
                args = TableToExcelArgs.model_validate(kwargs)
            except Exception as exc:
                return (
                    "```json\n"
                    f"{_json_dumps({'tool': TABLE_TO_EXCEL_TOOL_NAME, 'status': 'failed', 'error': f'invalid tool arguments: {exc}'})}\n"
                    "```"
                )
            payload = export_table_to_excel_sync(
                user_id=bound_user_id,
                content=args.content,
                filename=args.filename,
                sheet_name=args.sheet_name,
            )
            return f"```json\n{_json_dumps(payload)}\n```"

    tool_instance = TableToExcelTool()
    tool_instance.__doc__ = TABLE_TO_EXCEL_DOCSTRING
    return tool_instance
