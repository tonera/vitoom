from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin import table_to_excel as tte


def test_parse_markdown_table():
    rows = tte._coerce_table_rows(
        markdown="""
| 姓名 | 分数 |
| --- | ---: |
| Alice | 95 |
| Bob | 88 |
"""
    )

    assert rows == [["姓名", "分数"], ["Alice", "95"], ["Bob", "88"]]


def test_parse_html_table_from_ocr_output():
    rows = tte._coerce_table_rows(
        content=(
            '<table border="1"><tr><td></td><td>VLM</td><td>Specialized VLM</td></tr>'
            "<tr><td>Recognition</td><td>94.0</td><td>75.1</td></tr>"
            "<tr><td>Extraction</td><td>93.7</td><td>-</td></tr></table>"
        )
    )

    assert rows == [
        ["", "VLM", "Specialized VLM"],
        ["Recognition", "94.0", "75.1"],
        ["Extraction", "93.7", "-"],
    ]


def test_coerce_rows_argument_from_dicts():
    rows = tte._coerce_table_rows(
        rows=[
            {"姓名": "Alice", "分数": 95},
            {"姓名": "Bob", "等级": "A"},
        ]
    )

    assert rows == [["姓名", "分数", "等级"], ["Alice", "95", ""], ["Bob", "", "A"]]


def test_content_accepts_json_rows_array():
    rows = tte._coerce_table_rows(
        content='[{"姓名":"Alice","分数":95},{"姓名":"Bob","分数":88}]'
    )

    assert rows == [["姓名", "分数"], ["Alice", "95"], ["Bob", "88"]]


def test_build_xlsx_contains_required_parts():
    blob = tte._build_xlsx([["姓名", "分数"], ["Alice", "95"]], sheet_name="成绩表")

    with zipfile.ZipFile(BytesIO(blob)) as zf:
        names = set(zf.namelist())
        assert "[Content_Types].xml" in names
        assert "_rels/.rels" in names
        assert "xl/workbook.xml" in names
        assert "xl/worksheets/sheet1.xml" in names
        sheet_xml = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
        assert "Alice" in sheet_xml
        assert "95" in sheet_xml
        workbook_xml = zf.read("xl/workbook.xml").decode("utf-8")
        assert "成绩表" in workbook_xml


def test_export_table_to_excel_sync_persists(monkeypatch):
    seen: dict[str, object] = {}

    def fake_persist(**kwargs):
        seen.update(kwargs)
        return {
            "tool": tte.TABLE_TO_EXCEL_TOOL_NAME,
            "source": kwargs["source"],
            "status": "completed",
            "files": [{"url": "http://example.com/table.xlsx"}],
            "total": 1,
            "row_count": kwargs["row_count"],
            "column_count": kwargs["column_count"],
        }

    monkeypatch.setattr(tte, "_persist_xlsx_result", fake_persist)

    payload = tte.export_table_to_excel_sync(
        user_id="u1",
        markdown="| A | B |\n|---|---|\n| 1 | 2 |",
        filename="demo.xlsx",
    )

    assert payload["status"] == "completed"
    assert payload["row_count"] == 2
    assert payload["column_count"] == 2
    assert seen["output_name"] == "demo.xlsx"
    assert isinstance(seen["xlsx_bytes"], bytes)


def test_export_table_to_excel_requires_content():
    payload = tte.export_table_to_excel_sync(user_id="u1")
    assert payload["status"] == "failed"
    assert "table content" in payload["error"]


def test_tool_args_schema_exposes_single_content_parameter():
    try:
        import crewai  # noqa: F401
    except Exception:
        return
    tool = tte.build_table_to_excel_tool(context={"user_id": "u1"})
    fields = set(tool.args_schema.model_fields.keys())
    assert fields == {"content", "filename", "sheet_name"}
