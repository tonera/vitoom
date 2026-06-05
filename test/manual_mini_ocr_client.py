"""
Mini（GLM-OCR）联调客户端：从 POST /v1/tasks 起跑完整链路。

覆盖四种 OCR 模式：
  - text     : 通用文本识别（Markdown）
  - table    : 表格识别（Markdown 表格 / 也可 HTML）
  - formula  : 公式识别（LaTeX）
  - extract  : 按 schema 做信息抽取（返回 JSON）

流程（对齐 test/text_ws_experiment.py 的做法）：
  1) 自动从 users 表挑一个 active 用户，读 config/default.yaml + config/app.yaml 的
     security.jwt 本地签 JWT（等价于 backend.auth.create_access_token）。
     完全自动，不需要你跑别的脚本、也不需要 export 任何环境变量。
  2) POST /v1/tasks 创建 mini 任务（task_type=mini, job_type=OCR, model_name=GLM-OCR）
  3) 连接 /ws/task/{task_id}?token=... 收状态 + 多条 result
  4) 每条 result 打印 content（若有）、url/file_size 等；status=completed 结束

运行前提：
  - 在能跑 backend 的那个 Python 环境里执行（需要能 import backend.database.User + python-jose）
  - backend API/WS 已启动（默认 http://127.0.0.1:8888）
  - inference/mini 已启动（service_type: "mini"）
  - models 表已登记 `GLM-OCR`（full_name=GLM-OCR, family=GLM-OCR, is_local_model=true）
  - 测 PDF 时 mini 服务机器需装 pymupdf（见 inference/mini/requirements.txt）

示例（直接跑，不需要任何前置步骤）：
  python test/manual_mini_ocr_client.py text    -i /path/to/invoice.jpg
  python test/manual_mini_ocr_client.py text    -i a.jpg -i b.png
  python test/manual_mini_ocr_client.py text    -i ./doc.pdf
  python test/manual_mini_ocr_client.py table   -i ./table.png
  python test/manual_mini_ocr_client.py formula -i ./formula.png
  python test/manual_mini_ocr_client.py extract -i ./invoice.jpg \\
      --schema '{"invoice_no":"", "total_amount":"", "date":""}'
  python test/manual_mini_ocr_client.py smoke   -i ./invoice.jpg

可选：
  - --token <jwt>         调试远端/别人账号时用；不传就按上面流程自动签
  - --api-url / --ws-url  指向非本机 backend；不传就走默认 127.0.0.1:8888

依赖：httpx、websockets、PyYAML、python-jose。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import websockets

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_BASE = os.environ.get("VITOOM_API_BASE", "http://127.0.0.1:8888")
WS_BASE = os.environ.get("VITOOM_WS_BASE", "ws://127.0.0.1:8888")
DEFAULT_MODEL_NAME = "GLM-OCR"

# JWT 默认项（与 backend/auth 的 fallback 保持一致，实际值由 security.jwt 配置覆盖）
DEFAULT_JWT_SECRET = "vitoom-default-secret-key-change-in-production"
DEFAULT_JWT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE = 86400

OUTPUT_DIR = PROJECT_ROOT / "test" / "outputs" / "mini_ocr"


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
@dataclass
class OcrOptions:
    mode: str  # text | table | formula | extract
    inputs: List[str]
    schema: Optional[Dict[str, Any]] = None
    model_name: str = DEFAULT_MODEL_NAME
    # 硬超时：整个订阅最多跑多久（秒）
    timeout: float = 1800.0
    # 空闲超时：连续没收到任何 WS 消息多久就放弃（秒）
    idle_timeout: float = 120.0
    save: bool = True
    storage: str = "local"
    # 显式 file_type 覆盖（典型用法：text 模式传 "md" 强制走纯文字旁路做对照测试）
    file_type_override: Optional[str] = None


@dataclass
class CollectedResult:
    index: Optional[int]
    total: Optional[int]
    content: Optional[str]
    url: Optional[str]
    file_size: Optional[int]
    file_type: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 账号 / token（参考 test/text_ws_experiment.py：本地查库 + 签 JWT）
# ---------------------------------------------------------------------------
def load_security_settings() -> Dict[str, Any]:
    """合并 config/default.yaml + config/app.yaml 的 security 段。"""
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("自动生成 token 需要安装 PyYAML，或直接使用 --token") from e

    def _read_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}

    config_dir = PROJECT_ROOT / "config"
    merged: Dict[str, Any] = {}
    for candidate in (config_dir / "default.yaml", config_dir / "app.yaml"):
        data = _read_yaml(candidate)
        security = data.get("security")
        if isinstance(security, dict):
            merged.update(security)
    return merged


def create_access_token_local(data: Dict[str, Any]) -> str:
    """用本机 config 的 security.jwt 直接签 access token（等价 backend.auth.create_access_token）。"""
    try:
        from jose import jwt  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "自动生成 token 需要 python-jose；当前 Python 环境可能装了错误的 py2 `jose`。"
            "建议在能运行 backend 的环境里执行本脚本，或直接传 --token / $VITOOM_TOKEN。"
        ) from e

    security = load_security_settings()
    jwt_cfg = security.get("jwt") if isinstance(security.get("jwt"), dict) else {}
    secret = str(jwt_cfg.get("secret_key") or DEFAULT_JWT_SECRET)
    algorithm = str(jwt_cfg.get("algorithm") or DEFAULT_JWT_ALGORITHM)
    access_token_expire = int(jwt_cfg.get("access_token_expire") or DEFAULT_ACCESS_TOKEN_EXPIRE)

    payload = dict(data)
    payload.update(
        {
            "exp": datetime.utcnow() + timedelta(seconds=access_token_expire),
            "iat": datetime.utcnow(),
            "type": "access",
        }
    )
    return jwt.encode(payload, secret, algorithm=algorithm)


def pick_active_user_and_token() -> Tuple[str, str]:
    """从 users 表挑一个 active 用户，签一个本地 JWT。返回 (user_id, token)。"""
    try:
        from backend.database import User  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "未提供 --token，且无法导入 backend.database.User 自动选择测试用户。"
            "请改用 --token / $VITOOM_TOKEN，或在能跑 backend 的环境里执行本脚本。"
        ) from e

    users = User.list_all(limit=20)
    if not users:
        raise RuntimeError("users 表中没有可用用户，请先创建用户，或直接用 --token")

    user = next((u for u in users if str(u.get("status") or "").lower() == "active"), users[0])
    user_id = str(user.get("id") or "")
    email = str(user.get("email") or "")
    if not user_id:
        raise RuntimeError(f"选中的用户缺少 id: {user}")

    token = create_access_token_local({"sub": user_id, "email": email})
    print(f"[auth] user_id={user_id} email={email}")
    return user_id, token


def resolve_token(explicit_token: Optional[str]) -> Tuple[str, str]:
    """按优先级拿 (user_id, token)：--token > $VITOOM_TOKEN > 本地签发。"""
    tk = explicit_token or os.environ.get("VITOOM_TOKEN")
    if tk:
        print("[auth] use external token")
        return "<external-token>", tk
    return pick_active_user_and_token()


# ---------------------------------------------------------------------------
# HTTP：创建任务
# ---------------------------------------------------------------------------
def _build_extract_payload(opts: OcrOptions) -> Dict[str, Any]:
    """把 CLI 侧的 mode/schema 打包成 mini 服务约定的 extract dict。"""
    extract: Dict[str, Any] = {"task": opts.mode}
    if opts.mode == "extract":
        if not opts.schema:
            raise ValueError("extract 模式必须提供 --schema（JSON 对象）")
        extract["schema"] = opts.schema
    return extract


async def create_task(opts: OcrOptions, token: str) -> str:
    if not opts.inputs:
        raise ValueError("至少需要一个 --input（本地路径或可访问 URL）")

    payload: Dict[str, Any] = {
        "task_type": "mini",
        "job_type": "OCR",
        "model_name": opts.model_name,
        "tpl_list": opts.inputs,
        "extract": _build_extract_payload(opts),
        "storage": opts.storage,
    }
    # 只有 extract 模式需要客户端显式声明 file_type=json；其他模式不传，
    # 由推理端按模式默认决定（text → zip 图文混排，table/formula → md）。
    # 注意：text 模式如果显式传 file_type="md" 会触发 handler 的"强制纯文字 md 旁路"。
    if opts.mode == "extract":
        payload["file_type"] = "json"
    elif opts.file_type_override:
        payload["file_type"] = opts.file_type_override

    print("→ POST /v1/tasks")
    print(f"   payload = {json.dumps(payload, ensure_ascii=False)}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{API_BASE}/v1/tasks",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code != 201:
        raise RuntimeError(f"创建任务失败: HTTP {resp.status_code} {resp.text}")

    body = resp.json()
    # 统一响应：{"code":1, "data":{"task_id": "...", "status": "..."}, "msg": "..."}
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        # 兼容直接返回扁平对象
        data = body if isinstance(body, dict) else {}
    task_id = data.get("task_id") or body.get("task_id")
    if not task_id:
        raise RuntimeError(f"响应缺少 task_id: {body}")

    print(f"✓ task_id={task_id} status={data.get('status') or body.get('status')}")
    return task_id


# ---------------------------------------------------------------------------
# WS：订阅任务
# ---------------------------------------------------------------------------
async def subscribe_task(
    task_id: str,
    token: str,
    timeout: float,
    *,
    idle_timeout: Optional[float] = None,
) -> Tuple[str, List[CollectedResult], Dict[str, Any]]:
    """
    返回 (final_status, results_by_order, last_status_message)。
    完成条件：收到 status ∈ {completed, failed, cancelled}。

    超时语义（双保险）：
      - idle_timeout: "空闲超时"；每收到任意一条 WS 消息都会重置空闲计时，
        只有持续 idle_timeout 秒完全没收到任何消息才视为超时。这样可以容忍
        PDF OCR 这类单请求跑数分钟的场景（推理端每页都会发 task_status
        processing+progress 心跳），不再被 300s 总超时一刀切。
        未显式传入时，默认取 timeout 值（保持向后兼容的"总超时"语义）。
      - timeout: 硬上限；无论是否有心跳，超过这个时长也会超时退出，作为兜底。
    """
    ws_url = f"{WS_BASE}/ws/task/{task_id}?token={token}"
    print(f"→ WS {ws_url}")

    if idle_timeout is None or idle_timeout <= 0:
        idle_timeout = timeout

    results: List[CollectedResult] = []
    final_status: str = "unknown"
    final_msg: Dict[str, Any] = {}

    loop = asyncio.get_event_loop()
    t_start = loop.time()

    async with websockets.connect(ws_url, ping_interval=None) as ws:
        while True:
            # 硬上限：整个订阅过程不允许超过 timeout 秒（作为兜底，避免极端情况死等）
            remaining_hard = timeout - (loop.time() - t_start)
            if remaining_hard <= 0:
                raise asyncio.TimeoutError(
                    f"subscribe_task hit hard timeout={timeout:.1f}s"
                )
            wait = min(idle_timeout, remaining_hard)

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=wait)
            except asyncio.TimeoutError:
                # 连 idle_timeout 秒都没有收到任何一条 WS 消息：认为推理端已经卡住/失联
                raise asyncio.TimeoutError(
                    f"subscribe_task idle for {idle_timeout:.1f}s "
                    f"(no ws message received); giving up"
                )
            except websockets.ConnectionClosed:
                break

            try:
                data = json.loads(raw)
            except Exception:
                print(f"[WS-raw] {raw!r}")
                continue

            msg_type = data.get("type")
            status = data.get("status")

            if msg_type == "result":
                collected = CollectedResult(
                    index=data.get("index"),
                    total=data.get("total"),
                    content=data.get("content"),
                    url=data.get("big") or data.get("url"),
                    file_size=data.get("file_size"),
                    file_type=data.get("file_type"),
                    raw=data,
                )
                results.append(collected)
                _print_result_snippet(collected)
            else:
                # task_status / progress / 心跳 / error
                extra = []
                if data.get("page") is not None and data.get("total_pages") is not None:
                    extra.append(f"page={data.get('page')}/{data.get('total_pages')}")
                if data.get("stage"):
                    extra.append(f"stage={data.get('stage')}")
                if data.get("elapsed") is not None:
                    extra.append(f"elapsed={data.get('elapsed')}s")
                extra_str = (" " + " ".join(extra)) if extra else ""
                print(
                    f"[WS] type={msg_type} status={status}"
                    f" progress={data.get('progress')}{extra_str}"
                    f" msg={data.get('message') or data.get('msg') or ''}"
                )

            if status in ("completed", "failed", "cancelled"):
                final_status = status
                final_msg = data
                break

    return final_status, results, final_msg


def _print_result_snippet(r: CollectedResult) -> None:
    head = f"[WS-result]"
    if r.index is not None or r.total is not None:
        head += f" [{r.index}/{r.total}]"
    if r.file_type:
        head += f" file_type={r.file_type}"
    if r.file_size is not None:
        head += f" file_size={r.file_size}"
    if r.url:
        head += f" url={r.url}"
    print(head)
    if r.content:
        snippet = r.content if len(r.content) <= 400 else r.content[:400] + " ...<truncated>"
        print("    content:")
        for line in snippet.splitlines() or [snippet]:
            print(f"      {line}")


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------
def persist_results(mode: str, task_id: str, results: List[CollectedResult]) -> Optional[Path]:
    if not results:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # text 模式服务端默认返回图文混排 zip（file_type=zip），需按条单独下载落盘
    zip_results = [r for r in results if (r.file_type or "").lower() == "zip"]
    if mode == "text" and zip_results:
        return _persist_zip_results(mode, task_id, zip_results)

    ext = "json" if mode == "extract" else "md"
    fname = f"{mode}__{task_id}.{ext}"
    fpath = OUTPUT_DIR / fname

    # 单条直接落；多条按顺序拼（md 之间插入分隔，json 合成数组）
    if mode == "extract":
        payloads: List[Any] = []
        for r in sorted(results, key=lambda x: (x.index if x.index is not None else 0)):
            if not r.content:
                continue
            try:
                payloads.append(json.loads(r.content))
            except Exception:
                payloads.append({"_raw": r.content})
        fpath.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        parts: List[str] = []
        for r in sorted(results, key=lambda x: (x.index if x.index is not None else 0)):
            if not r.content:
                continue
            header = f"\n\n<!-- result #{r.index} / {r.total} -->\n"
            parts.append(header + r.content)
        fpath.write_text("".join(parts).lstrip(), encoding="utf-8")

    print(f"✓ 结果已落盘：{fpath}")
    return fpath


def _persist_zip_results(mode: str, task_id: str, results: List[CollectedResult]) -> Optional[Path]:
    """text 模式图文混排：每条 result 对应一个 zip（document.md + images/），从 url 下载落盘。"""
    import urllib.request

    saved: List[Path] = []
    for r in sorted(results, key=lambda x: (x.index if x.index is not None else 0)):
        if not r.url:
            print(f"⚠ result index={r.index} 缺少 url，跳过下载")
            continue
        idx = r.index if r.index is not None else 0
        fname = f"{mode}__{task_id}__{idx}.zip"
        fpath = OUTPUT_DIR / fname
        try:
            with urllib.request.urlopen(r.url, timeout=120) as resp:
                data = resp.read()
            fpath.write_bytes(data)
            print(f"✓ zip 已下载落盘：{fpath} ({len(data)} bytes)")
            saved.append(fpath)
        except Exception as e:
            print(f"⚠ 下载 zip 失败 url={r.url}: {e}")

    # 额外把内联 md 文本（content）也落一份，方便快速查看
    md_parts: List[str] = []
    for r in sorted(results, key=lambda x: (x.index if x.index is not None else 0)):
        if not r.content:
            continue
        header = f"\n\n<!-- result #{r.index} / {r.total}; see also zip -->\n"
        md_parts.append(header + r.content)
    if md_parts:
        md_path = OUTPUT_DIR / f"{mode}__{task_id}.inline.md"
        md_path.write_text("".join(md_parts).lstrip(), encoding="utf-8")
        print(f"✓ 内联 md 预览已落盘：{md_path}")
        saved.append(md_path)

    return saved[0] if saved else None


# ---------------------------------------------------------------------------
# 单次执行
# ---------------------------------------------------------------------------
async def run_one(opts: OcrOptions, token: str) -> Tuple[str, List[CollectedResult]]:
    print(f"\n=========== 模式: {opts.mode}  inputs={len(opts.inputs)} ===========")
    task_id = await create_task(opts, token)
    status, results, last = await subscribe_task(
        task_id, token,
        timeout=opts.timeout,
        idle_timeout=opts.idle_timeout,
    )
    print(f"\n✓ 最终状态: {status}  result 条数={len(results)}")
    if status == "failed":
        err = last.get("error") or last.get("message") or last
        print(f"⚠ 失败原因: {err}")
    if opts.save and results:
        persist_results(opts.mode, task_id, results)
    return status, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--input", "-i", action="append", default=[],
        help="输入文件：本地绝对路径或可访问 URL（可多次指定）",
    )
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="模型名（对应 models.full_name）")
    p.add_argument(
        "--timeout", type=float, default=1800.0,
        help="WS 等待最终状态的硬超时（秒）。transformers backend 跑多页 PDF 很容易 >5min，默认给 30min。",
    )
    p.add_argument(
        "--idle-timeout", type=float, default=120.0,
        help=(
            "WS 空闲超时（秒）：连续多久没收到任何 WS 消息就放弃。推理端每页会发一次"
            " task_status processing+progress 心跳，只要服务还活着，中间就不会沉默超过这个值。"
            "默认 120s，对比 --timeout 是两级保险：任何一个先触发都会终止订阅。"
        ),
    )
    p.add_argument("--no-save", action="store_true", help="不落盘结果到 test/outputs/mini_ocr/")
    p.add_argument("--storage", default="local", help="存储后端：local / oss / ...")
    p.add_argument(
        "--token", default=None,
        help="显式 JWT token；不传则从 users 表挑 active 用户、本地按 security.jwt 签发",
    )
    p.add_argument("--api-url", default=None, help=f"后端 API 基地址（默认 {API_BASE}）")
    p.add_argument("--ws-url", default=None, help=f"后端 WS 基地址（默认 {WS_BASE}）")
    p.add_argument(
        "--plain-md", action="store_true",
        help="text 模式下强制走纯文字 md 旁路（file_type=md），不触发图文混排 zip 路径",
    )


def _parse_schema(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--schema 不是合法 JSON: {e}") from e
    if not isinstance(obj, dict):
        raise SystemExit("--schema 必须是 JSON 对象（dict）")
    return obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manual_mini_ocr_client",
        description="Mini（GLM-OCR）联调客户端",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    for m in ("text", "table", "formula"):
        sp = sub.add_parser(m, help=f"OCR - {m}")
        _add_common_args(sp)

    sp_ext = sub.add_parser("extract", help="OCR - 信息抽取（需 --schema）")
    _add_common_args(sp_ext)
    sp_ext.add_argument("--schema", required=True, help='抽取模板 JSON，例如 \'{"name":"", "age":""}\'')

    sp_smoke = sub.add_parser("smoke", help="一把梭：依次跑 text/table/formula（+ 可选 extract）")
    _add_common_args(sp_smoke)
    sp_smoke.add_argument(
        "--extract-schema", default=None,
        help="若提供则追加 extract 模式；不提供则跳过 extract",
    )

    return parser


async def cmd_single(args: argparse.Namespace) -> int:
    inputs = _normalize_inputs(args.input)
    opts = OcrOptions(
        mode=args.mode,
        inputs=inputs,
        schema=_parse_schema(getattr(args, "schema", None)),
        model_name=args.model_name,
        timeout=args.timeout,
        idle_timeout=args.idle_timeout,
        save=not args.no_save,
        storage=args.storage,
        file_type_override="md" if getattr(args, "plain_md", False) else None,
    )
    _, token = resolve_token(args.token)
    status, _ = await run_one(opts, token)
    return 0 if status == "completed" else 1


async def cmd_smoke(args: argparse.Namespace) -> int:
    inputs = _normalize_inputs(args.input)
    _, token = resolve_token(args.token)

    plans: List[OcrOptions] = [
        OcrOptions(mode="text", inputs=inputs, model_name=args.model_name,
                   timeout=args.timeout, idle_timeout=args.idle_timeout,
                   save=not args.no_save, storage=args.storage),
        OcrOptions(mode="table", inputs=inputs, model_name=args.model_name,
                   timeout=args.timeout, idle_timeout=args.idle_timeout,
                   save=not args.no_save, storage=args.storage),
        OcrOptions(mode="formula", inputs=inputs, model_name=args.model_name,
                   timeout=args.timeout, idle_timeout=args.idle_timeout,
                   save=not args.no_save, storage=args.storage),
    ]
    if args.extract_schema:
        plans.append(OcrOptions(
            mode="extract",
            inputs=inputs,
            schema=_parse_schema(args.extract_schema),
            model_name=args.model_name,
            timeout=args.timeout,
            idle_timeout=args.idle_timeout,
            save=not args.no_save,
            storage=args.storage,
        ))

    summary: List[Tuple[str, str, int]] = []
    for plan in plans:
        try:
            status, results = await run_one(plan, token)
        except Exception as e:
            print(f"⚠ 模式 {plan.mode} 抛异常: {e}")
            summary.append((plan.mode, f"error: {e.__class__.__name__}", 0))
            continue
        summary.append((plan.mode, status, len(results)))

    print("\n=========== Smoke 汇总 ===========")
    for mode, status, n in summary:
        print(f"  {mode:8s} -> status={status:10s} results={n}")

    return 0 if all(s == "completed" for _, s, _ in summary) else 1


def _normalize_inputs(raw: List[str]) -> List[str]:
    """把本地相对路径统一成绝对路径，URL 原样返回。"""
    out: List[str] = []
    for item in raw:
        if not item:
            continue
        if item.startswith(("http://", "https://", "file://")):
            out.append(item)
            continue
        p = Path(item).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            print(f"⚠ 警告：本地文件不存在 {p}（仍然按路径发出，服务端会再试一次）")
        out.append(str(p))
    return out


async def _amain() -> int:
    parser = build_parser()
    args = parser.parse_args()

    global API_BASE, WS_BASE
    if getattr(args, "api_url", None):
        API_BASE = args.api_url.rstrip("/")
    if getattr(args, "ws_url", None):
        WS_BASE = args.ws_url.rstrip("/")
    print(f"[env] API_BASE={API_BASE}  WS_BASE={WS_BASE}")

    if args.mode in ("text", "table", "formula", "extract"):
        return await cmd_single(args)
    if args.mode == "smoke":
        return await cmd_smoke(args)
    parser.error(f"unknown mode: {args.mode}")
    return 2


def main() -> None:
    try:
        rc = asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\n(中断)")
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
