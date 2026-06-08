# pyright: reportMissingImports=false

"""Ollama 文本运行时桥。

对齐 ``vllm_bridge`` / ``transformers_bridge`` 的公共接口：

    - ``load_ollama_text_bundle(model_ref, policy)``
    - ``stream_chat_text(bundle, *, messages, request_id, ...)``
    - ``generate_chat_text(bundle, *, messages, request_id, ...)``
    - ``abort_chat_request(bundle, request_id)``
    - ``shutdown_ollama_text_bundle(bundle)``

与 vllm 桥的主要差异：

1. **模型加载**：``model_ref`` 指向一个本地目录（``{models_dir}/{load_name}``），
   推理侧自动扫描里面的 ``.gguf`` 文件，识别主模型和可选的 ``mmproj`` 投影仪，
   如果目录下有 ``Modelfile`` 就优先用，否则自动生成一个，派生出一个稳定 tag
   并在首次加载时调用 ``client.create`` 注册到 Ollama 守护进程里。tag 的
   fingerprint 由主 gguf 的 mtime+size+Modelfile 文本派生，文件任何变动都会
   产生新 tag，天然支持热更新。
2. **abort 语义**：Ollama 没有 request_id 级别的 abort API。我们把 stream
   迭代挂在一个 ``asyncio.Task`` 上并登记到 ``active_requests[request_id]``，
   ``abort_chat_request`` 直接 ``task.cancel()``，底层 httpx 连接关闭后
   daemon 会自动停止本次生成。
3. **工具调用**：若 Ollama 把 ``<tool_call>`` 文本解析成结构化
   ``message.tool_calls`` 返回，桥层把它重新序列化成 Qwen 风格的
   ``<tool_call>{json}</tool_call>`` 文本拼到 delta 里，让既有的
   ``QwenToolCallParser`` 继续统一处理。模型原生吐出 ``<tool_call>`` 文本的
   场景也自然兼容。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlparse
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from common.io_utils import download_url_to_tempfile
from common.logger import get_logger

from text.runtime.common import count_multimodal_parts
from text.runtime.runtime_resolver import TextRuntimePolicy

logger = get_logger(__name__)


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_TAG_PREFIX = "vitoom"
DEFAULT_MULTIMODAL_MODE = "off"
DEFAULT_VIDEO_FRAME_COUNT = 8
DEFAULT_VIDEO_SAMPLE_FPS = 1.0
DEFAULT_VIDEO_MAX_FRAMES = 16
DEFAULT_MODEL_SOURCE = "local_gguf"


# Qwen3 家族的 tools-capable chat 模板：对齐官方 `ollama.com/library/qwen3` 发行版的
# go-template 语法。Ollama 判定一个 tag "是否 supports tools" 的方法是：扫 TEMPLATE
# 文本里有没有 ``.Tools`` / ``.ToolCalls`` 引用——换句话说，tag 要能接 tools 参数，
# 其 TEMPLATE 必须显式渲染这两个字段；否则 `client.chat(tools=[...])` 直接 400
# "does not support tools"。这里的 template 同时覆盖：
#   1. system prompt + tools schema 的 <tools>...</tools> 渲染
#   2. assistant tool_calls 的 <tool_call>...</tool_call> 序列化
#   3. role=tool 的历史回放（渲染成 <tool_response>）
# 外层 session_runtime 用的 QwenToolCallParser 按这套格式解析 delta，语义天然一致。
QWEN3_TOOLS_CHAT_TEMPLATE = '''{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}

{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <arguments-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{- if .Content }}
{{ .Content }}
{{- else if .ToolCalls }}
<tool_call>
{{- range .ToolCalls }}
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}
</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}'''


# 与上面 template 配套的 Qwen 特殊 token stop 集（Ollama 不会根据 template 自动
# 推断 stop，必须显式给）。放在这里集中声明，避免用户要手动维护。
QWEN_DEFAULT_STOP_TOKENS: Tuple[str, ...] = ("<|im_start|>", "<|im_end|>")


@dataclass
class OllamaTextBundle:
    model_source: str
    model_ref: str
    tag: str
    main_gguf: str
    mmproj_path: Optional[str]
    modelfile_source: str  # "user" | "generated"
    client: Any
    policy: TextRuntimePolicy
    multimodal_mode: str = DEFAULT_MULTIMODAL_MODE
    vision_enabled: bool = False
    host: str = DEFAULT_OLLAMA_HOST
    active_requests: Dict[str, "asyncio.Task[Any]"] = field(default_factory=dict, repr=False)
    request_lock: Any = field(default_factory=threading.Lock, repr=False)


# ---------------------------------------------------------------------------
# 目录扫描 / Modelfile / tag
# ---------------------------------------------------------------------------


_MMPROJ_RE = re.compile(r"mmproj", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_GGUF_MAGIC = b"GGUF"


def _ensure_valid_gguf_file(path: Path, *, role: str) -> None:
    try:
        with path.open("rb") as fh:
            magic = fh.read(4)
    except OSError as exc:
        raise RuntimeError(f"Failed to read {role} GGUF file {path}: {exc}") from exc
    if magic != _GGUF_MAGIC:
        raise RuntimeError(
            f"{role} file does not appear to be a valid GGUF: {path}. "
            f"Expected leading bytes {_GGUF_MAGIC!r}, got {magic!r}. "
            "This usually means the file is mislabeled, incomplete, or not a GGUF model component."
        )


def _scan_gguf_dir(model_ref: str) -> Tuple[str, Optional[str]]:
    """扫描模型目录，返回 (主 gguf 绝对路径, mmproj 绝对路径或 None)。"""
    path = Path(model_ref).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(
            f"Ollama runtime expects a directory containing .gguf files; "
            f"got non-directory path: {path}"
        )
    ggufs = sorted(p for p in path.glob("*.gguf") if p.is_file())
    if not ggufs:
        raise RuntimeError(
            f"No .gguf file found under {path}; "
            "please put the main gguf (and optional mmproj*.gguf) into this directory."
        )

    mmprojs = [p for p in ggufs if _MMPROJ_RE.search(p.name)]
    mains = [p for p in ggufs if p not in mmprojs]
    if not mains:
        raise RuntimeError(
            f"Only mmproj-*.gguf found under {path} but no main weight .gguf. "
            "Please also place the quantized main model (e.g. *-Q4_K_M.gguf) in this directory."
        )

    # 主模型取体积最大的那个；多个大小相近时也直接用最大的（Ollama Modelfile 的 FROM
    # 只认一个主 weight，额外的非 mmproj gguf 会被忽略，避免静默选错）。
    main = max(mains, key=lambda p: p.stat().st_size)
    mmproj = mmprojs[0] if mmprojs else None
    _ensure_valid_gguf_file(main, role="Main GGUF")
    if mmproj is not None:
        _ensure_valid_gguf_file(mmproj, role="mmproj GGUF")
    return str(main), (str(mmproj) if mmproj else None)


def _slugify(name: str) -> str:
    base = Path(name).name.lower()
    slug = _SLUG_RE.sub("-", base).strip("-")
    return slug or "model"


def _file_fingerprint(path: str) -> str:
    st = Path(path).stat()
    return f"{path}|{int(st.st_mtime)}|{st.st_size}"


def _compute_tag(
    *,
    model_ref: str,
    main_gguf: str,
    mmproj_path: Optional[str],
    modelfile_text: str,
    tag_prefix: str,
) -> str:
    slug = _slugify(model_ref)
    blob = "||".join(
        [
            _file_fingerprint(main_gguf),
            _file_fingerprint(mmproj_path) if mmproj_path else "",
            modelfile_text,
        ]
    )
    fingerprint = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    return f"{tag_prefix}/{slug}:{fingerprint}"


def _quote_modelfile_string(value: str) -> str:
    # 把字符串包成 Modelfile 里合法的多行字面量。Ollama Modelfile 的 TEMPLATE /
    # SYSTEM / PARAMETER 值支持三引号多行语义，与 Python 相似。只做最小转义：把
    # 内容里出现的三连双引号替换成 " \" " 形式，避免字面量提前闭合。
    TRIPLE = '"' * 3
    escaped = value.replace(TRIPLE, '"' + '\\"' + '"')
    return TRIPLE + escaped + TRIPLE


def _render_generated_modelfile(
    *,
    model_ref: str,
    main_gguf: str,
    mmproj_path: Optional[str],
    extra_lines: List[str],
    chat_template: Optional[str],
    stop_tokens: Tuple[str, ...],
    vision_enabled: bool,
) -> str:
    # 文本模式显式指向主 gguf，避免目录里碰巧存在 mmproj 时被误装成视觉模型。
    # 视觉模式优先按目录级 FROM 交给 Ollama 自己解析配套组件，比无脑双 FROM 更稳。
    base_from = str(Path(model_ref).expanduser().resolve()) if vision_enabled else main_gguf
    lines: List[str] = [f"FROM {base_from}"]

    # chat template 决定了一个 tag 是否被 ollama 判为 "supports tools"——必须显式声明，
    # 裸 gguf 没有 template 时 `client.chat(tools=[...])` 会直接 400。
    if chat_template:
        lines.append(f"TEMPLATE {_quote_modelfile_string(chat_template)}")
    for token in stop_tokens:
        token_str = str(token or "").strip()
        if token_str:
            lines.append(f'PARAMETER stop "{token_str}"')

    for entry in extra_lines:
        text = str(entry or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines) + "\n"


def _load_or_generate_modelfile(
    *,
    model_ref: str,
    main_gguf: str,
    mmproj_path: Optional[str],
    extra_lines: List[str],
    chat_template: Optional[str],
    stop_tokens: Tuple[str, ...],
    vision_enabled: bool,
) -> Tuple[str, str]:
    """返回 ``(modelfile_text, source)``。``source`` ∈ {``'user'``, ``'generated'``}。

    - 用户在模型目录里放了 ``Modelfile`` 就用他的（原文透传，只检查一条 FROM）
    - 否则按 gguf + 模板 + stop tokens + extra_lines 拼一个
    """
    modelfile_path = Path(model_ref) / "Modelfile"
    if modelfile_path.is_file():
        text = modelfile_path.read_text(encoding="utf-8")
        if "FROM" not in text:
            raise RuntimeError(
                f"User Modelfile at {modelfile_path} has no FROM directive; "
                "specify the main .gguf with `FROM <path>`."
            )
        return text, "user"

    if vision_enabled and mmproj_path is None:
        raise RuntimeError(
            f"Ollama multimodal mode is enabled, but no mmproj*.gguf was found under "
            f"{Path(model_ref).expanduser().resolve()}. "
            "For local GGUF vision models, place the compatible projector beside the main GGUF, "
            "or provide a custom Modelfile that points Ollama at a verified multimodal bundle."
        )

    return (
        _render_generated_modelfile(
            model_ref=model_ref,
            main_gguf=main_gguf,
            mmproj_path=mmproj_path,
            extra_lines=extra_lines,
            chat_template=chat_template,
            stop_tokens=stop_tokens,
            vision_enabled=vision_enabled,
        ),
        "generated",
    )


# ---------------------------------------------------------------------------
# Ollama 客户端封装
# ---------------------------------------------------------------------------


def _import_ollama() -> Any:
    try:
        import ollama  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import `ollama` python client. "
            "Please add `ollama>=0.4.0` to inference/text/requirements.txt and reinstall."
        ) from e
    return ollama


def _make_async_client(host: str) -> Any:
    ollama = _import_ollama()
    AsyncClient = getattr(ollama, "AsyncClient", None)
    if AsyncClient is None:
        raise RuntimeError("ollama.AsyncClient not found; pin `ollama>=0.4.0` in requirements.")
    # trust_env=False: 本机 daemon 不该被 HTTP(S)_PROXY 劫持；远端 daemon 也应直接在
    # base_url 里写地址，不依赖 env。
    return AsyncClient(host=host, trust_env=False)


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    """构造一个不走代理的 urllib opener。本机 daemon 不该经过 HTTP_PROXY。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_get_json_sync(host: str, path: str, *, timeout: float = 10.0) -> Any:
    url = f"{host.rstrip('/')}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with _no_proxy_opener().open(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body) if body.strip() else None


def _http_post_json_sync(
    host: str, path: str, payload: Dict[str, Any], *, timeout: float = 30.0
) -> None:
    url = f"{host.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    with _no_proxy_opener().open(req, timeout=timeout) as resp:
        resp.read()


def _list_tags_sync(host: str) -> List[str]:
    resp = _http_get_json_sync(host, "/api/tags")
    if not isinstance(resp, dict):
        return []
    items = resp.get("models") or []
    out: List[str] = []
    for item in items:
        if isinstance(item, dict):
            tag = item.get("model") or item.get("name")
            if tag:
                out.append(str(tag))
    return out


def _ensure_ollama_cli() -> str:
    cli = shutil.which("ollama")
    if not cli:
        raise RuntimeError(
            "`ollama` CLI not found in PATH. Install it (Linux: "
            "`curl -fsSL https://ollama.com/install.sh | sh`; macOS: `brew install ollama`)."
        )
    return cli


def _ensure_tag_registered(
    *,
    tag: str,
    modelfile_text: str,
    host: str,
) -> None:
    """如果 daemon 还没注册过这个 tag，就用 ``ollama create`` CLI 注册。

    全程 sync：
      - 查 tag 走 sync urllib（避免 ``ollama.AsyncClient`` 绑死到临时 event loop）
      - 注册走 subprocess CLI，daemon 自己读本地 gguf，不走任何 HTTP blob 上传
    """
    try:
        existing = set(_list_tags_sync(host))
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama daemon at base_url={host}: {e!r}. "
            f"Check: `pgrep -a ollama`, `curl -sS {host.rstrip('/')}/api/tags`, "
            "and ensure HTTP(S)_PROXY/ALL_PROXY doesn't intercept localhost."
        ) from e

    if tag in existing:
        logger.info("Ollama tag already registered: %s", tag)
        return

    cli = _ensure_ollama_cli()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(modelfile_text)
        modelfile_path = fh.name

    env = dict(os.environ)
    env["OLLAMA_HOST"] = host

    logger.info("Registering Ollama tag=%s via `ollama create -f %s`", tag, modelfile_path)
    try:
        proc = subprocess.run(
            [cli, "create", tag, "-f", modelfile_path],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    finally:
        try:
            os.unlink(modelfile_path)
        except OSError:
            pass

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
        raise RuntimeError(
            f"`ollama create {tag}` failed (exit={proc.returncode}). "
            f"stderr tail:\n" + "\n".join(tail)
        )
    logger.info("Ollama tag registered: %s", tag)


def _ensure_tag_available(*, tag: str, host: str, auto_pull: bool) -> None:
    try:
        existing = set(_list_tags_sync(host))
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama daemon at base_url={host}: {e!r}. "
            f"Check: `pgrep -a ollama`, `curl -sS {host.rstrip('/')}/api/tags`, "
            "and ensure HTTP(S)_PROXY/ALL_PROXY doesn't intercept localhost."
        ) from e

    if tag in existing:
        logger.info("Ollama tag already available: %s", tag)
        return
    if not auto_pull:
        raise RuntimeError(
            f"Ollama tag not found in daemon: {tag}. "
            "Pull it first with `ollama pull <tag>`, or set `config.runtime.ollama.auto_pull: true`."
        )

    cli = _ensure_ollama_cli()
    env = dict(os.environ)
    env["OLLAMA_HOST"] = host
    logger.info("Pulling missing Ollama tag=%s via `ollama pull`", tag)
    proc = subprocess.run(
        [cli, "pull", tag],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
        raise RuntimeError(
            f"`ollama pull {tag}` failed (exit={proc.returncode}). stderr tail:\n" + "\n".join(tail)
        )
    logger.info("Ollama tag pulled: %s", tag)


# ---------------------------------------------------------------------------
# 加载入口
# ---------------------------------------------------------------------------


def _coerce_str_list(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _coerce_optional_positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    return number if number > 0 else None


def _normalize_multimodal_mode(value: Any) -> str:
    text = str(value or DEFAULT_MULTIMODAL_MODE).strip().lower()
    aliases = {
        "": DEFAULT_MULTIMODAL_MODE,
        "false": DEFAULT_MULTIMODAL_MODE,
        "none": DEFAULT_MULTIMODAL_MODE,
        "text": DEFAULT_MULTIMODAL_MODE,
        "disabled": DEFAULT_MULTIMODAL_MODE,
        "image_only": "image",
        "images": "image",
        "vision": "image",
        "multimodal": "image",
        "image+video": "image_and_video_frames",
        "image_video": "image_and_video_frames",
        "video": "image_and_video_frames",
        "video_frames": "image_and_video_frames",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"off", "image", "image_and_video_frames"}:
        raise RuntimeError(
            "Unsupported ollama.multimodal_mode=%r; expected one of off/image/image_and_video_frames"
            % (value,)
        )
    return normalized


def _resolve_multimodal_mode(ollama_cfg: Dict[str, Any]) -> str:
    return _normalize_multimodal_mode(ollama_cfg.get("multimodal_mode"))


def _resolve_model_source(ollama_cfg: Dict[str, Any]) -> str:
    raw = str(ollama_cfg.get("model_source") or DEFAULT_MODEL_SOURCE).strip().lower()
    aliases = {
        "": DEFAULT_MODEL_SOURCE,
        "local": DEFAULT_MODEL_SOURCE,
        "gguf": DEFAULT_MODEL_SOURCE,
        "path": DEFAULT_MODEL_SOURCE,
        "directory": DEFAULT_MODEL_SOURCE,
        "ollama_tag": "tag",
        "registry": "tag",
        "name": "tag",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"local_gguf", "tag"}:
        raise RuntimeError(
            "Unsupported ollama.model_source=%r; expected one of local_gguf/tag" % (raw,)
        )
    return normalized


def load_ollama_text_bundle(model_ref: str, policy: TextRuntimePolicy) -> OllamaTextBundle:
    ollama_cfg: Dict[str, Any] = dict(policy.ollama_cfg or {})
    # 优先读 `base_url`（更明确的语义，表达"推理侧作为客户端连的 daemon 地址"），
    # 同时保留 `host` 兼容——它对齐 ollama.AsyncClient(host=...) 的 SDK 参数名。
    raw_base = ollama_cfg.get("base_url")
    host = str(raw_base or DEFAULT_OLLAMA_HOST).strip() or DEFAULT_OLLAMA_HOST
    tag_prefix = str(ollama_cfg.get("tag_prefix") or DEFAULT_TAG_PREFIX).strip() or DEFAULT_TAG_PREFIX
    extra_lines = _coerce_str_list(ollama_cfg.get("modelfile_extra"))

    # chat template：auto-generated Modelfile 里默认注入 Qwen3 tools-capable 模板。
    # 用户可用 ollama_cfg.template 完全覆盖（string），或设 ollama_cfg.template=null /
    # "" 明确关闭模板（某些非 Qwen 的 base gguf 自带 template，交给 gguf metadata 决定）。
    raw_template = ollama_cfg.get("template", "__default__")
    if raw_template == "__default__":
        chat_template: Optional[str] = QWEN3_TOOLS_CHAT_TEMPLATE
    elif raw_template in (None, "", False):
        chat_template = None
    else:
        chat_template = str(raw_template)

    # stop tokens：同样 auto 场景默认 Qwen 的 <|im_start|> / <|im_end|>。用户可用
    # ollama_cfg.stop_tokens 显式覆盖（list[str]）；传空 list 即不加默认 stop。
    if "stop_tokens" in ollama_cfg:
        stop_tokens: Tuple[str, ...] = tuple(_coerce_str_list(ollama_cfg.get("stop_tokens")))
    else:
        stop_tokens = QWEN_DEFAULT_STOP_TOKENS if chat_template is QWEN3_TOOLS_CHAT_TEMPLATE else tuple()

    model_source = _resolve_model_source(ollama_cfg)
    multimodal_mode = _resolve_multimodal_mode(ollama_cfg)
    vision_enabled = multimodal_mode != "off"
    main_gguf: str
    mmproj_path: Optional[str]
    modelfile_text: str
    modelfile_source: str
    tag: str

    if model_source == "tag":
        main_gguf = ""
        mmproj_path = None
        modelfile_text = ""
        modelfile_source = "tag"
        tag = str(model_ref or "").strip()
        if not tag:
            raise RuntimeError("Ollama tag source requires a non-empty load_name/tag.")
    else:
        main_gguf, mmproj_path = _scan_gguf_dir(model_ref)
        modelfile_text, modelfile_source = _load_or_generate_modelfile(
            model_ref=model_ref,
            main_gguf=main_gguf,
            mmproj_path=mmproj_path,
            extra_lines=extra_lines,
            chat_template=chat_template,
            stop_tokens=stop_tokens,
            vision_enabled=vision_enabled,
        )
        tag = _compute_tag(
            model_ref=model_ref,
            main_gguf=main_gguf,
            mmproj_path=mmproj_path,
            modelfile_text=modelfile_text,
            tag_prefix=tag_prefix,
        )

    logger.info(
        "Loading Ollama text bundle model_ref=%s main=%s mmproj=%s modelfile=%s tag=%s host=%s model_source=%s multimodal_mode=%s vision_enabled=%s",
        model_ref,
        Path(main_gguf).name if main_gguf else "-",
        Path(mmproj_path).name if mmproj_path else "-",
        modelfile_source,
        tag,
        host,
        model_source,
        multimodal_mode,
        vision_enabled,
    )

    # 注册走 sync 路径：sync urllib 查 tag + subprocess CLI 创建。
    # AsyncClient 在这里**只构造不使用**——第一次 await 它才是在用户的 event loop 里，
    # 避免被临时 loop 绑定导致之后 chat/close 时触发 "Event loop is closed"。
    if model_source == "tag":
        _ensure_tag_available(
            tag=tag,
            host=host,
            auto_pull=_coerce_bool(ollama_cfg.get("auto_pull"), False),
        )
    else:
        _ensure_tag_registered(tag=tag, modelfile_text=modelfile_text, host=host)
    client = _make_async_client(host)

    return OllamaTextBundle(
        model_source=model_source,
        model_ref=model_ref,
        tag=tag,
        main_gguf=main_gguf,
        mmproj_path=mmproj_path,
        modelfile_source=modelfile_source,
        client=client,
        policy=policy,
        multimodal_mode=multimodal_mode,
        vision_enabled=vision_enabled,
        host=host,
    )


# ---------------------------------------------------------------------------
# messages 归一化 / options 构造
# ---------------------------------------------------------------------------


def _flatten_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: List[str] = []
    for part in content:
        if isinstance(part, dict):
            ptype = str(part.get("type") or "").strip().lower()
            if ptype in {"", "text", "input_text"}:
                parts.append(str(part.get("text") or ""))
    return "".join(parts)


def _extract_image_url(part: Dict[str, Any]) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        return str(image_url.get("url") or "").strip()
    if isinstance(image_url, str):
        return image_url.strip()
    return str(part.get("image") or part.get("url") or "").strip()


def _extract_video_url(part: Dict[str, Any]) -> str:
    video_url = part.get("video_url")
    if isinstance(video_url, dict):
        return str(video_url.get("url") or "").strip()
    if isinstance(video_url, str):
        return video_url.strip()
    return str(part.get("video") or part.get("url") or "").strip()


def _is_data_url(value: str) -> bool:
    return str(value or "").startswith("data:")


def _parse_data_url(data_url: str) -> Tuple[str, bytes]:
    try:
        header, payload = str(data_url).split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid data URL payload") from exc
    meta = header[5:]
    mime = meta.split(";", 1)[0] if meta else ""
    if ";base64" in meta:
        try:
            blob = base64.b64decode(payload, validate=True)
        except binascii.Error as exc:
            raise ValueError("Invalid base64 data URL payload") from exc
    else:
        blob = unquote_to_bytes(payload)
    return mime, blob


def _guess_suffix_from_mime(mime: str, *, default_suffix: str) -> str:
    normalized = str(mime or "").strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/x-msvideo": ".avi",
    }
    return mapping.get(normalized, default_suffix)


def _file_url_to_path(url: str) -> Optional[Path]:
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "file":
        return None
    return Path(urllib.request.url2pathname(parsed.path))


def _cleanup_temp_path(path: Optional[Path], *, keep: bool) -> None:
    if keep or path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


async def _resolve_binary_input(
    url_or_path: str,
    *,
    default_suffix: str,
    max_bytes: Optional[int] = None,
) -> Tuple[bytes, Optional[Path]]:
    ref = str(url_or_path or "").strip()
    if not ref:
        raise ValueError("empty media reference")

    if _is_data_url(ref):
        _mime, blob = _parse_data_url(ref)
        return blob, None

    file_path = _file_url_to_path(ref)
    if file_path is not None:
        return await asyncio.to_thread(file_path.read_bytes), None

    if ref.startswith(("http://", "https://")):
        temp_path = await download_url_to_tempfile(
            ref,
            default_suffix=default_suffix,
            timeout_seconds=60.0,
            max_bytes=max_bytes,
        )
        try:
            blob = await asyncio.to_thread(temp_path.read_bytes)
        except Exception:
            _cleanup_temp_path(temp_path, keep=False)
            raise
        return blob, temp_path

    path = Path(ref).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return await asyncio.to_thread(path.read_bytes), None


async def _resolve_video_input_to_path(url_or_path: str) -> Tuple[Path, bool]:
    ref = str(url_or_path or "").strip()
    if not ref:
        raise ValueError("empty video reference")

    if _is_data_url(ref):
        mime, blob = _parse_data_url(ref)
        suffix = _guess_suffix_from_mime(mime, default_suffix=".mp4")
        fd, raw_path = tempfile.mkstemp(prefix="vitoom_ollama_video_", suffix=suffix)
        os.close(fd)
        video_path = Path(raw_path)
        await asyncio.to_thread(video_path.write_bytes, blob)
        return video_path, True

    file_path = _file_url_to_path(ref)
    if file_path is not None:
        return file_path, False

    if ref.startswith(("http://", "https://")):
        video_path = await download_url_to_tempfile(
            ref,
            default_suffix=".mp4",
            timeout_seconds=120.0,
            max_bytes=512 * 1024 * 1024,
        )
        return video_path, True

    return Path(ref).expanduser().resolve(), False


def _select_frame_indices(total_frames: int, desired_frames: int, sample_every: Optional[int]) -> List[int]:
    if total_frames <= 0:
        return []
    if sample_every is not None and sample_every > 1:
        return list(range(0, total_frames, sample_every))
    desired = min(total_frames, max(1, desired_frames))
    if desired == 1:
        return [0]
    if desired >= total_frames:
        return list(range(total_frames))
    points = {
        min(total_frames - 1, int(round((total_frames - 1) * idx / float(desired - 1))))
        for idx in range(desired)
    }
    return sorted(points)


def _frame_to_png_bytes(frame: Any) -> bytes:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Video frame extraction requires Pillow to encode sampled frames to PNG bytes."
        ) from exc
    image = Image.fromarray(frame)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _extract_video_frames_with_cv2(
    video_path: Path,
    *,
    desired_frames: int,
    sample_fps: Optional[float],
    max_frames: int,
) -> Optional[List[bytes]]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        sample_every = None
        if sample_fps and native_fps > 0:
            sample_every = max(1, int(round(native_fps / sample_fps)))
        indices = _select_frame_indices(total_frames, desired_frames, sample_every)
        if not indices:
            indices = list(range(min(desired_frames, max_frames)))

        wanted = set(indices[:max_frames])
        output: List[bytes] = []
        cursor = 0
        last_needed = max(wanted) if wanted else -1
        while cursor <= last_needed:
            ok, frame = capture.read()
            if not ok:
                break
            if cursor in wanted:
                ok_encoded, encoded = cv2.imencode(".png", frame)
                if ok_encoded:
                    output.append(encoded.tobytes())
                    if len(output) >= max_frames:
                        break
            cursor += 1
        return output
    finally:
        capture.release()


def _extract_video_frames_with_imageio(
    video_path: Path,
    *,
    desired_frames: int,
    sample_fps: Optional[float],
    max_frames: int,
) -> Optional[List[bytes]]:
    try:
        import imageio  # type: ignore
    except Exception:
        return None

    reader = None
    try:
        reader = imageio.get_reader(str(video_path))
        meta = reader.get_meta_data() if hasattr(reader, "get_meta_data") else {}
        total_frames = int(reader.count_frames()) if hasattr(reader, "count_frames") else 0
        native_fps = float((meta or {}).get("fps") or 0.0)
        sample_every = None
        if sample_fps and native_fps > 0:
            sample_every = max(1, int(round(native_fps / sample_fps)))
        indices = _select_frame_indices(total_frames, desired_frames, sample_every)
        if not indices:
            indices = list(range(min(desired_frames, max_frames)))
        output: List[bytes] = []
        for frame_index in indices[:max_frames]:
            frame = reader.get_data(frame_index)
            output.append(_frame_to_png_bytes(frame))
        return output
    except Exception:
        return None
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()


def _extract_video_frames_sync(
    video_path: Path,
    *,
    desired_frames: int,
    sample_fps: Optional[float],
    max_frames: int,
) -> List[bytes]:
    for extractor in (_extract_video_frames_with_cv2, _extract_video_frames_with_imageio):
        frames = extractor(
            video_path,
            desired_frames=desired_frames,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
        if frames:
            return frames[:max_frames]
    raise RuntimeError(
        "Unable to sample frames from video input. Install a usable decoder backend such as "
        "`opencv-python` or `imageio` in the text runtime environment."
    )


def _build_video_sampling_config(
    bundle: OllamaTextBundle,
    mm_processor_kwargs: Optional[Dict[str, Any]],
) -> Tuple[int, Optional[float], int]:
    cfg = bundle.policy.ollama_cfg
    desired_frames = _coerce_positive_int(
        (mm_processor_kwargs or {}).get("video_frame_count") or cfg.get("video_frame_count")
    ) or DEFAULT_VIDEO_FRAME_COUNT
    sample_fps = _coerce_optional_positive_float(
        (mm_processor_kwargs or {}).get("fps")
        or (mm_processor_kwargs or {}).get("video_sample_fps")
        or cfg.get("video_sample_fps")
    )
    max_frames = _coerce_positive_int(
        (mm_processor_kwargs or {}).get("video_max_frames") or cfg.get("video_max_frames")
    ) or DEFAULT_VIDEO_MAX_FRAMES
    desired_frames = min(desired_frames, max_frames)
    return desired_frames, sample_fps, max_frames


async def _sample_video_frames(
    bundle: OllamaTextBundle,
    video_ref: str,
    *,
    mm_processor_kwargs: Optional[Dict[str, Any]],
) -> List[bytes]:
    desired_frames, sample_fps, max_frames = _build_video_sampling_config(bundle, mm_processor_kwargs)
    video_path, should_cleanup = await _resolve_video_input_to_path(video_ref)
    try:
        return await asyncio.to_thread(
            _extract_video_frames_sync,
            video_path,
            desired_frames=desired_frames,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
    finally:
        _cleanup_temp_path(video_path, keep=not should_cleanup)


def _build_video_frame_note(video_index: int, frame_count: int) -> str:
    return (
        f"[Video {video_index} is attached as {frame_count} sampled frames in chronological order. "
        "Use the full frame sequence to answer the user's question.]"
    )


def _serialize_assistant_tool_calls_to_text(tool_calls: Any) -> str:
    """把 assistant.tool_calls 回写成 Qwen 风格 ``<tool_call>...</tool_call>`` 文本。

    Ollama 对 ``messages`` 里的历史 tool_calls 支持不够一致，一些 GGUF 模板直接对
    messages 渲染文本。保守起见把它 inline 进 content 末尾，等价于模型自己吐过的
    格式，对 Qwen/Hermes-Pro 模板都友好。
    """
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    blocks: List[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        args_raw = function.get("arguments")
        if isinstance(args_raw, str):
            try:
                args_value: Any = json.loads(args_raw) if args_raw.strip() else {}
            except Exception:
                args_value = {"_raw": args_raw}
        elif isinstance(args_raw, (dict, list)):
            args_value = args_raw
        else:
            args_value = {}
        payload = json.dumps({"name": name, "arguments": args_value}, ensure_ascii=False)
        blocks.append(f"<tool_call>\n{payload}\n</tool_call>")
    return ("\n".join(blocks) + "\n") if blocks else ""


async def _to_ollama_messages(
    bundle: OllamaTextBundle,
    messages: List[Dict[str, Any]],
    *,
    mm_processor_kwargs: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip() or "user"
        content = message.get("content")
        text = _flatten_text_content(content)
        images: List[bytes] = []
        if isinstance(content, list):
            text_parts: List[str] = []
            video_index = 0
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = str(part.get("type") or "").strip().lower()
                if ptype in {"", "text", "input_text"}:
                    chunk = str(part.get("text") or "")
                    if chunk:
                        text_parts.append(chunk)
                    continue
                if ptype in {"image", "image_url"}:
                    image_ref = _extract_image_url(part)
                    if not image_ref:
                        raise ValueError("image_url content item requires a non-empty url")
                    image_bytes, temp_path = await _resolve_binary_input(
                        image_ref,
                        default_suffix=".png",
                        max_bytes=50 * 1024 * 1024,
                    )
                    try:
                        images.append(image_bytes)
                    finally:
                        _cleanup_temp_path(temp_path, keep=False)
                    continue
                if ptype in {"video", "video_url"}:
                    video_ref = _extract_video_url(part)
                    if not video_ref:
                        raise ValueError("video_url content item requires a non-empty url")
                    video_index += 1
                    video_frames = await _sample_video_frames(
                        bundle,
                        video_ref,
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
                    if not video_frames:
                        raise RuntimeError(f"Video input produced no sampled frames: {video_ref}")
                    images.extend(video_frames)
                    text_parts.append(_build_video_frame_note(video_index, len(video_frames)))
            text = "\n".join(part for part in text_parts if part).strip()
        if role == "assistant":
            tool_text = _serialize_assistant_tool_calls_to_text(message.get("tool_calls"))
            if tool_text:
                text = (text + ("\n" if text else "") + tool_text).rstrip()
        entry: Dict[str, Any] = {"role": role, "content": text}
        if images:
            entry["images"] = images
        if role == "tool":
            name = str(message.get("name") or "").strip()
            if name:
                entry["name"] = name
        converted.append(entry)
    return converted


def _build_unsupported_multimodal_message(
    bundle: OllamaTextBundle,
    messages: List[Dict[str, Any]],
) -> Optional[str]:
    image_count, video_count = count_multimodal_parts(messages)
    if image_count <= 0 and video_count <= 0:
        return None

    if bundle.multimodal_mode == "off":
        return (
            "当前所选 Ollama 服务未开启多模态支持，暂时不能处理图片或视频输入。"
            "请切换到支持多模态的服务配置，或改用纯文本提问。"
        )
    if video_count > 0 and bundle.multimodal_mode != "image_and_video_frames":
        return (
            "当前所选 Ollama 服务未开启视频输入支持。"
            "请切换到支持视频抽帧输入的配置，或仅发送文本/图片。"
        )
    return None


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except Exception:
        return None
    return number if number > 0 else None


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


_OLLAMA_EXTRA_OPTION_KEYS = (
    "num_gpu",
    "num_thread",
    "num_batch",
    "repeat_penalty",
    "repeat_last_n",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "seed",
    "stop",
)


def _build_ollama_options(
    *,
    policy: TextRuntimePolicy,
    temperature: Any,
    max_tokens: Any,
    top_p: Any,
    top_k: Any,
    presence_penalty: Any,
    frequency_penalty: Any,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    num_ctx = policy.max_model_len
    if isinstance(num_ctx, int) and num_ctx > 0:
        options["num_ctx"] = num_ctx

    num_predict = _coerce_positive_int(max_tokens)
    if num_predict is not None:
        options["num_predict"] = num_predict

    temp = _coerce_optional_float(temperature)
    if temp is not None:
        options["temperature"] = temp
    tp = _coerce_optional_float(top_p)
    if tp is not None:
        options["top_p"] = tp
    tk = _coerce_positive_int(top_k)
    if tk is not None:
        options["top_k"] = tk
    pp = _coerce_optional_float(presence_penalty)
    if pp is not None:
        options["presence_penalty"] = pp
    fp = _coerce_optional_float(frequency_penalty)
    if fp is not None:
        options["frequency_penalty"] = fp

    for key in _OLLAMA_EXTRA_OPTION_KEYS:
        if key in policy.ollama_cfg and policy.ollama_cfg[key] not in (None, ""):
            options[key] = policy.ollama_cfg[key]

    return options


# ---------------------------------------------------------------------------
# stream / abort
# ---------------------------------------------------------------------------


def _register_task(bundle: OllamaTextBundle, request_id: str, task: "asyncio.Task[Any]") -> None:
    if not request_id:
        return
    with bundle.request_lock:
        bundle.active_requests[request_id] = task


def _unregister_task(bundle: OllamaTextBundle, request_id: str) -> None:
    if not request_id:
        return
    with bundle.request_lock:
        bundle.active_requests.pop(request_id, None)


def _ns_to_seconds(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if number <= 0:
        return None
    return number / 1e9


def _chunk_get(chunk: Any, key: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)


def _chunk_message_field(chunk: Any, key: str) -> Any:
    message = _chunk_get(chunk, "message")
    if message is None:
        return None
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _chunk_tool_calls_to_text(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    blocks: List[str] = []
    for call in tool_calls:
        fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
        name = None
        args_raw: Any = None
        if isinstance(fn, dict):
            name = fn.get("name")
            args_raw = fn.get("arguments")
        elif fn is not None:
            name = getattr(fn, "name", None)
            args_raw = getattr(fn, "arguments", None)
        name = str(name or "").strip()
        if not name:
            continue
        if isinstance(args_raw, str):
            try:
                args_value: Any = json.loads(args_raw) if args_raw.strip() else {}
            except Exception:
                args_value = {"_raw": args_raw}
        elif isinstance(args_raw, (dict, list)):
            args_value = args_raw
        elif args_raw is None:
            args_value = {}
        else:
            try:
                args_value = dict(args_raw)  # type: ignore[arg-type]
            except Exception:
                args_value = {"_raw": str(args_raw)}
        payload = json.dumps({"name": name, "arguments": args_value}, ensure_ascii=False)
        blocks.append(f"<tool_call>\n{payload}\n</tool_call>")
    return "\n".join(blocks)


def _build_final_stats(
    *,
    chunk: Any,
    started_at: float,
    first_delta_at: Optional[float],
    finished_at: float,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    prompt_tokens = _chunk_get(chunk, "prompt_eval_count")
    output_tokens = _chunk_get(chunk, "eval_count")
    if isinstance(prompt_tokens, int):
        stats["prompt_tokens"] = prompt_tokens
    if isinstance(output_tokens, int):
        stats["output_tokens"] = output_tokens

    total_seconds_wall = max(0.0, finished_at - started_at)
    # 优先用 wall clock 的 ttft（最贴近用户感知）；拿不到再退到 ollama 反推。
    ttft_seconds: Optional[float] = None
    if first_delta_at is not None:
        ttft_seconds = max(0.0, first_delta_at - started_at)
    else:
        load_s = _ns_to_seconds(_chunk_get(chunk, "load_duration"))
        prompt_s = _ns_to_seconds(_chunk_get(chunk, "prompt_eval_duration"))
        if load_s is not None or prompt_s is not None:
            ttft_seconds = (load_s or 0.0) + (prompt_s or 0.0)

    ollama_total_s = _ns_to_seconds(_chunk_get(chunk, "total_duration"))
    total_seconds = ollama_total_s if ollama_total_s is not None else total_seconds_wall

    decode_seconds = _ns_to_seconds(_chunk_get(chunk, "eval_duration"))
    if decode_seconds is None and ttft_seconds is not None:
        decode_seconds = max(0.0, total_seconds - ttft_seconds)

    stats["total_seconds"] = total_seconds
    if ttft_seconds is not None:
        stats["ttft_seconds"] = ttft_seconds
    if decode_seconds is not None:
        stats["decode_seconds"] = decode_seconds

    if isinstance(output_tokens, int) and total_seconds > 0:
        stats["tok_s_total"] = float(output_tokens) / total_seconds
        if decode_seconds is not None and decode_seconds > 0:
            stats["tok_s_decode"] = float(output_tokens) / decode_seconds
    return stats


async def _call_chat_stream(
    *,
    client: Any,
    tag: str,
    messages: List[Dict[str, Any]],
    options: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
    keep_alive: Any,
    think: Optional[bool],
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": tag,
        "messages": messages,
        "stream": True,
        "options": options or None,
        "keep_alive": keep_alive,
    }
    if tools:
        kwargs["tools"] = tools
    if think is not None:
        kwargs["think"] = think

    try:
        return await client.chat(**kwargs)
    except TypeError as exc:
        # ollama<0.4 没有 `think` / `tools` 参数；逐一剥离重试。
        removed: List[str] = []
        for optional_key in ("think", "tools"):
            if optional_key in kwargs and f"'{optional_key}'" in str(exc):
                kwargs.pop(optional_key, None)
                removed.append(optional_key)
        if not removed:
            raise
        logger.info(
            "ollama.AsyncClient.chat does not accept %s; retrying without it. Upgrade `ollama` to unlock.",
            ",".join(removed),
        )
        return await client.chat(**kwargs)


async def stream_chat_text(
    bundle: OllamaTextBundle,
    *,
    messages: List[Dict[str, Any]],
    request_id: str,
    max_tokens: Any = None,
    temperature: Any = None,
    enable_thinking: bool | None = None,
    top_p: Any = None,
    top_k: Any = None,
    presence_penalty: Any = None,
    frequency_penalty: Any = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    unsupported_message = _build_unsupported_multimodal_message(bundle, messages)
    if unsupported_message:
        yield {
            "delta": unsupported_message,
            "finished": True,
            "finish_reason": "unsupported_multimodal",
        }
        return

    ollama_messages = await _to_ollama_messages(
        bundle,
        messages,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    options = _build_ollama_options(
        policy=bundle.policy,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    keep_alive = bundle.policy.ollama_cfg.get("keep_alive", DEFAULT_KEEP_ALIVE)
    # 对支持 thinking 的官方模型（如 qwen3.6），False 必须显式传给 Ollama；否则 daemon
    # 可能按模型默认行为返回 `message.thinking`，而不是最终 `message.content`。
    think = bool(bundle.policy.enable_thinking) if enable_thinking is None else bool(enable_thinking)

    normalized_tools = (
        [dict(tool) for tool in tools if isinstance(tool, dict)] if tools else None
    )

    async def _run() -> AsyncIterator[Dict[str, Any]]:
        started_at = time.perf_counter()
        first_delta_at: Optional[float] = None
        stream = await _call_chat_stream(
            client=bundle.client,
            tag=bundle.tag,
            messages=ollama_messages,
            options=options,
            tools=normalized_tools,
            keep_alive=keep_alive,
            think=think,
        )

        try:
            async for chunk in stream:  # type: ignore[union-attr]
                content_delta = str(_chunk_message_field(chunk, "content") or "")
                thinking_delta = str(_chunk_message_field(chunk, "thinking") or "")
                tool_calls = _chunk_message_field(chunk, "tool_calls")
                tool_text = _chunk_tool_calls_to_text(tool_calls)
                combined_delta = content_delta
                if tool_text:
                    combined_delta = (combined_delta + "\n" + tool_text) if combined_delta else tool_text
                if think and not combined_delta and thinking_delta:
                    combined_delta = thinking_delta

                if combined_delta and first_delta_at is None:
                    first_delta_at = time.perf_counter()

                finished = bool(_chunk_get(chunk, "done", False))

                payload: Dict[str, Any] = {
                    "delta": combined_delta,
                    "finished": finished,
                }
                if thinking_delta:
                    payload["thinking_delta"] = thinking_delta
                done_reason = _chunk_get(chunk, "done_reason")
                if done_reason:
                    payload["finish_reason"] = str(done_reason)
                if finished:
                    finished_at = time.perf_counter()
                    payload.update(
                        _build_final_stats(
                            chunk=chunk,
                            started_at=started_at,
                            first_delta_at=first_delta_at,
                            finished_at=finished_at,
                        )
                    )
                    payload.setdefault("finish_reason", "stop")

                if combined_delta or finished:
                    yield payload
                await asyncio.sleep(0)
        finally:
            # 主动关流，防止 httpx 连接残留把 ollama 的 slot 占住。
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass

    # 把真正的消费挂到一个 Task 上，登记下来用于 abort。这里不能直接把 async generator
    # 本身塞进 active_requests——generator 没有 cancel 语义；真正能 cancel 的是包一层
    # 的 Task。外层 `async for` 消费下面这个 queue 即可。
    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
    sentinel: Dict[str, Any] = {"__sentinel__": True}

    async def _pump() -> None:
        try:
            async for item in _run():
                await queue.put(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put({"__error__": exc})
        finally:
            await queue.put(sentinel)

    task = asyncio.create_task(_pump(), name=f"ollama-chat-{request_id or 'anon'}")
    _register_task(bundle, request_id, task)

    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if "__error__" in item:
                raise item["__error__"]  # type: ignore[misc]
            yield item
    except asyncio.CancelledError:
        task.cancel()
        raise
    finally:
        _unregister_task(bundle, request_id)
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        else:
            # 把已完成任务里潜在的 exception 消费掉，避免 "Task exception was never retrieved"。
            exc = task.exception() if not task.cancelled() else None
            if exc is not None:
                logger.debug("ollama chat pump finished with exception (already surfaced): %s", exc)


async def generate_chat_text(
    bundle: OllamaTextBundle,
    *,
    messages: List[Dict[str, Any]],
    request_id: str,
    max_tokens: Any = None,
    temperature: Any = None,
    enable_thinking: bool | None = None,
    top_p: Any = None,
    top_k: Any = None,
    presence_penalty: Any = None,
    frequency_penalty: Any = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    parts: List[str] = []
    async for item in stream_chat_text(
        bundle,
        messages=messages,
        request_id=request_id,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        mm_processor_kwargs=mm_processor_kwargs,
        tools=tools,
    ):
        delta = str(item.get("delta") or "")
        if delta:
            parts.append(delta)
    return "".join(parts)


async def abort_chat_request(bundle: OllamaTextBundle, request_id: Optional[str]) -> None:
    if not request_id:
        return
    with bundle.request_lock:
        task = bundle.active_requests.get(request_id)
    if task is None or task.done():
        return
    task.cancel()


def shutdown_ollama_text_bundle(bundle: OllamaTextBundle) -> None:
    # 取消所有尚未结束的 chat 任务。
    with bundle.request_lock:
        tasks = list(bundle.active_requests.values())
        bundle.active_requests.clear()
    for task in tasks:
        try:
            if not task.done():
                task.cancel()
        except Exception:
            pass

    # sync 请求 daemon 立即卸载模型（keep_alive=0）。不走 AsyncClient，外层有没有
    # 在跑的 event loop 都不影响。失败不致命。
    try:
        _http_post_json_sync(
            bundle.host,
            "/api/generate",
            {"model": bundle.tag, "prompt": "", "keep_alive": 0},
            timeout=30.0,
        )
    except Exception as exc:
        logger.warning("ollama unload for tag=%s failed: %s", bundle.tag, exc)
