from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx

from backend.core.logger import get_app_logger
from backend.services.agent.settings import (
    get_openclaw_allowed_tools,
    get_openclaw_base_url,
    get_openclaw_timeout_seconds,
    get_openclaw_token,
)

logger = get_app_logger(__name__)


class OpenClawBridgeError(RuntimeError):
    """OpenClaw 工具桥接异常。"""


def _normalize_allowlist(raw_value: Any) -> List[str]:
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return []


class OpenClawToolBridge:
    """受控的 OpenClaw tools 调用桥接层。"""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        allowed_tools: Optional[List[str]] = None,
    ):
        configured_base_url = base_url or get_openclaw_base_url()
        self.base_url = str(configured_base_url).rstrip("/")
        self.token = token or get_openclaw_token()
        self.timeout_seconds = float(timeout_seconds or get_openclaw_timeout_seconds())
        self.allowed_tools = _normalize_allowlist(
            allowed_tools if allowed_tools is not None else get_openclaw_allowed_tools()
        )

    def is_tool_allowed(self, tool_name: str, runtime_allowlist: Optional[List[str]] = None) -> bool:
        normalized_name = str(tool_name or "").strip()
        if not normalized_name:
            return False

        system_allowlist = self.allowed_tools
        if system_allowlist and "*" not in system_allowlist and normalized_name not in system_allowlist:
            return False

        runtime_list = _normalize_allowlist(runtime_allowlist)
        if runtime_list and "*" not in runtime_list and normalized_name not in runtime_list:
            return False

        if not system_allowlist and runtime_list:
            return normalized_name in runtime_list or "*" in runtime_list

        return bool(system_allowlist)

    def invoke_tool(
        self,
        *,
        agent_run_id: str,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
        dry_run: bool = False,
        session_key: Optional[str] = None,
        runtime_allowlist: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        normalized_tool_name = str(tool_name or "").strip()
        if not normalized_tool_name:
            raise OpenClawBridgeError("tool_name is required")
        if not self.is_tool_allowed(normalized_tool_name, runtime_allowlist=runtime_allowlist):
            raise OpenClawBridgeError(f"OpenClaw tool not allowed: {normalized_tool_name}")
        if args is not None and not isinstance(args, dict):
            raise OpenClawBridgeError("tool args must be a JSON object")

        request_timeout = float(timeout_seconds or self.timeout_seconds)
        payload: Dict[str, Any] = {
            "tool": normalized_tool_name,
            "args": dict(args or {}),
            "dryRun": bool(dry_run),
        }
        if session_key:
            payload["sessionKey"] = session_key

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        logger.info(
            f"Invoking OpenClaw tool: run={agent_run_id}, tool={normalized_tool_name}, dry_run={bool(dry_run)}"
        )

        started = time.perf_counter()
        try:
            with httpx.Client(timeout=request_timeout) as client:
                response = client.post(f"{self.base_url}/tools/invoke", headers=headers, json=payload)
        except Exception as e:
            raise OpenClawBridgeError(f"Failed to call OpenClaw /tools/invoke: {e}") from e
        duration_ms = int((time.perf_counter() - started) * 1000)

        raw_text = response.text
        try:
            raw_response: Any = response.json()
        except Exception:
            raw_response = {"text": raw_text}

        if response.status_code >= 400:
            raise OpenClawBridgeError(
                f"OpenClaw tool invoke failed: status={response.status_code}, body={raw_text[:500]}"
            )

        output: Any
        if isinstance(raw_response, dict):
            output = raw_response.get("data", raw_response)
        else:
            output = raw_response

        return {
            "ok": True,
            "tool_name": normalized_tool_name,
            "output": output,
            "error": None,
            "duration_ms": duration_ms,
            "raw_response": raw_response,
        }

    @staticmethod
    def stringify_output(output: Any) -> str:
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception:
            return str(output)
