"""
远端模型信息探测与缩略图候选提取（从 routes.py 抽离）。

支持：
- HuggingFace
- ModelScope
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from backend.utils.http_utils import HTTPClient


def extract_hf_repo_id(s: str) -> Optional[str]:
    """
    支持：
    - https://huggingface.co/{repo_id}
    - https://huggingface.co/{repo_id}/blob/main/README_CN.md
    - 以及其他 /tree/... /resolve/... 等
    """
    try:
        u = str(s or "").strip()
        if "huggingface.co" not in u.lower():
            return None
        p = urlparse(u)
        parts = [x for x in (p.path or "").split("/") if x]
        if len(parts) < 2:
            return None
        return f"{parts[0]}/{parts[1]}"
    except Exception:
        return None


def extract_ms_repo_id(s: str) -> Optional[str]:
    """
    支持：
    - https://modelscope.cn/models/{repo_id}
    - https://modelscope.cn/models/{repo_id}/something
    """
    try:
        u = str(s or "").strip()
        if "modelscope.cn" not in u.lower():
            return None
        p = urlparse(u)
        parts = [x for x in (p.path or "").split("/") if x]
        # modelscope: /models/{org}/{name}/...
        if len(parts) >= 3 and parts[0].lower() == "models":
            return f"{parts[1]}/{parts[2]}"
        # 兜底：/org/name/...
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None
    except Exception:
        return None


def looks_like_repo_id(s: str) -> bool:
    v = str(s or "").strip()
    return bool(v) and ("/" in v) and (" " not in v) and (not v.lower().startswith(("http://", "https://")))


async def probe_url_ok(url: str, timeout_seconds: float = 3.0) -> bool:
    try:
        async with HTTPClient(timeout=timeout_seconds) as client:
            resp = await client.get(url, follow_redirects=True)
            return 200 <= int(resp.status_code) < 400
    except Exception:
        return False


async def fetch_hf_model_info(repo_id: str) -> Dict[str, Any]:
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    params = {"full": "true"}
    async with HTTPClient(timeout=10.0) as client:
        resp = await client.get(api_url, params=params, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()


async def fetch_ms_model_info(repo_id: str) -> Dict[str, Any]:
    # modelscope 的公开接口（返回结构较大，前端可挑字段展示）
    api_url = f"https://modelscope.cn/api/v1/models/{repo_id}"
    async with HTTPClient(timeout=10.0) as client:
        resp = await client.get(api_url, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()


def _is_image_filename(p: str) -> bool:
    s = str(p or "").strip().lower()
    return bool(s) and any(s.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"))


def _rank_thumb_name(name: str) -> int:
    s = str(name or "").lower()
    keys = ["thumbnail", "thumb", "cover", "preview", "logo", "banner", "icon"]
    for i, k in enumerate(keys):
        if k in s:
            return i
    return 999


def extract_hf_thumb_candidates(info: Dict[str, Any], repo_id: str, limit: int = 10) -> List[str]:
    rid = str(repo_id or "").strip()
    if not rid:
        return []
    sibs = info.get("siblings") if isinstance(info, dict) else None
    if not isinstance(sibs, list):
        return []
    names: List[str] = []
    for s in sibs:
        fn = ""
        if isinstance(s, dict):
            fn = str(s.get("rfilename") or "").strip()
        if fn and _is_image_filename(fn):
            names.append(fn)
    names.sort(key=lambda x: (_rank_thumb_name(x), len(x)))
    urls = [f"https://huggingface.co/{rid}/resolve/main/{n}" for n in names[: max(0, int(limit))]]
    out: List[str] = []
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def extract_ms_thumb_candidates(info: Any, repo_id: str, limit: int = 10) -> List[str]:
    """
    ModelScope thumbs：
    - 直接 URL：Avatar / 其他字段里出现的 http(s) 图片 URL
    - README 里引用的 assets 图片：按
      https://modelscope.cn/models/{repo_id}/resolve/{revision}/assets/xxx.png
      拼接（revision 默认 master）
    """
    import re

    max_n = max(0, int(limit))
    out: List[str] = []
    seen = set()

    rid = str(repo_id or "").strip()
    rev = "master"
    if isinstance(info, dict):
        rv = str(info.get("Revision") or info.get("revision") or "").strip()
        if rv:
            rev = rv

    def push_url(u: str):
        url = str(u or "").strip()
        if not url or url in seen:
            return
        ul = url.lower()
        if not (ul.startswith("http://") or ul.startswith("https://")):
            return
        if not any(ul.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")):
            return
        seen.add(url)
        out.append(url)

    def push_assets_path(p: str):
        if not rid:
            return
        path = str(p or "").strip()
        # normalize: "/assets/.." or "./assets/.." -> "assets/.."
        path = path.lstrip("/").lstrip()
        if path.startswith("./"):
            path = path[2:]
        # allow case-insensitive "assets/"
        if not str(path).lower().startswith("assets/"):
            return
        lp = str(path).lower()
        if not any(lp.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")):
            return
        url = f"https://modelscope.cn/models/{rid}/resolve/{rev}/{path}"
        if url in seen:
            return
        seen.add(url)
        out.append(url)

    # 递归扫描整个 JSON：
    # - 绝对图片 URL（Avatar/cover 等）：直接收集
    # - 任意字符串里出现的 assets/xxx.png：提取并按 resolve/{rev} 拼成可访问 URL
    assets_pat = re.compile(r"(?:\./)?assets/[A-Za-z0-9._/\-]+\.(?:png|jpe?g|webp|gif)", re.IGNORECASE)

    def walk(x: Any):
        if len(out) >= max_n:
            return
        if isinstance(x, dict):
            for _, v in x.items():
                walk(v)
                if len(out) >= max_n:
                    return
        elif isinstance(x, list):
            for v in x:
                walk(v)
                if len(out) >= max_n:
                    return
        elif isinstance(x, str):
            s = x.strip()
            # absolute image url
            push_url(s)
            if len(out) >= max_n:
                return
            # assets paths (single or embedded in markdown/html)
            for m in assets_pat.findall(s):
                push_assets_path(m)
                if len(out) >= max_n:
                    return

    walk(info)
    return out[:max_n]

