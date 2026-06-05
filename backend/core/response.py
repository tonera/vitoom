"""
统一 API 响应封装

约定：
- 成功：code = 1
- 失败：code != 1，使用 ErrorCode（1000+）或自定义 int

统一格式：
{ "code": int, "data": any, "msg": string }
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from .error_codes import ErrorCode


def ok(
    data: Any = None,
    msg: str = "ok",
    code: Union[int, ErrorCode] = ErrorCode.SUCCESS,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """成功响应封装。默认 code=ErrorCode.SUCCESS(=1)。"""
    c = int(code.value) if isinstance(code, ErrorCode) else int(code)
    resp: Dict[str, Any] = {"code": c, "data": data, "msg": msg}
    if meta is not None:
        resp["meta"] = meta
    return resp


def fail(
    code: Union[int, ErrorCode],
    msg: str,
    data: Any = None,
    message_code: Optional[str] = None,
    message_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """失败响应封装。"""
    c = int(code.value) if isinstance(code, ErrorCode) else int(code)
    resp: Dict[str, Any] = {"code": c, "data": data, "msg": msg}
    if message_code:
        resp["message_code"] = message_code
    if message_params:
        resp["message_params"] = message_params
    return resp


