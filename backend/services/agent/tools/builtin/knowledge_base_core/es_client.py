from __future__ import annotations

import base64
import json
import ssl
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class EsHttpError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Elasticsearch HTTP {status}: {message}")
        self.status = status
        self.message = message


class KnowledgeBaseEsClient:
    def __init__(self, *, url: str, username: str = "", password: str = "", timeout: float = 30.0) -> None:
        self.url = str(url or "http://127.0.0.1:9200").rstrip("/") + "/"
        self.username = str(username or "")
        self.password = str(password or "")
        self.timeout = timeout if timeout > 0 else 30.0

    def request(self, method: str, path: str, *, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        full_url = urljoin(self.url, str(path or "").lstrip("/"))
        request_headers = {"Accept": "application/json"}
        if headers:
            request_headers.update(headers)
        data: Optional[bytes] = None
        if body is not None:
            if isinstance(body, bytes):
                data = body
            elif isinstance(body, str):
                data = body.encode("utf-8")
            else:
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                request_headers.setdefault("Content-Type", "application/json")
        if self.username:
            token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
            request_headers["Authorization"] = f"Basic {token}"
        request = Request(full_url, data=data, headers=request_headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout, context=ssl.create_default_context()) as response:
                raw = response.read()
        except HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            raise EsHttpError(exc.code, raw_error) from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to connect Elasticsearch at {self.url}: {exc}") from exc
        if not raw:
            return {}
        text = raw.decode("utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def search(self, index_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        response = self.request("POST", f"/{index_name}/_search", body=body)
        return response if isinstance(response, dict) else {}
