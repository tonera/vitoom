# pyright: reportMissingImports=false
"""
直测 inference/text/runtime/ollama_bridge.py —— 不经过后端、不经过 CrewAI / LiteLLM / WS，
只在一个 Python 进程里把 bridge 跑起来。用来：
  1) 验证 ollama daemon 连得上
  2) 验证模型目录里的 gguf / mmproj 扫描正确
  3) 验证 `ollama create` 的 tag 注册 + blob 上传真的能跑完
  4) 验证 streaming chat 能出 token

必须在**推理侧**（即 ollama daemon 所在机器）的 Python 环境里跑：
    conda activate vitoom-text
    pip install ollama>=0.4.0,<0.7.0
    # 如果 ollama serve 在 127.0.0.1:11434 并且这台机器有 HTTP(S)_PROXY 环境变量，
    # 要么 `export NO_PROXY=127.0.0.1,localhost,::1`，要么不管（bridge 内部已经
    # 给 httpx 传了 trust_env=False，不走代理）。

基本用法：
    python test/test_ollama_bridge_direct.py \
        --service-yaml inference/config/service_text_qwen.yaml \
        --model-name "Qwen3.6-35B-A3B-GGUF" \
        --prompt "你好，介绍一下你自己"

只做 tag 注册、不做推理（先确认大 blob 上传能过）：
    python test/test_ollama_bridge_direct.py \
        --service-yaml inference/config/service_text_qwen.yaml \
        --model-name "Qwen3.6-35B-A3B-GGUF" \
        --load-only

实用辅助排障提示：
  - 跑前另开一个终端 `watch -n 1 'du -sh ~/.ollama/models/blobs/ 2>/dev/null; \
                              ss -tnp | grep 11434'`，
    可以看到 blob store 在实时增长、HTTP/2 连接数、流量。
  - 跑前 `ollama list` 如果能列出 tag 而不是报 timeout，说明 daemon 正常。
  - 想强制重建 tag：删 `~/.ollama/models/manifests/registry.ollama.ai/vitoom/*`。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

from text.runtime.ollama_bridge import (  # noqa: E402
    generate_chat_text,
    load_ollama_text_bundle,
    shutdown_ollama_text_bundle,
    stream_chat_text,
)
from text.runtime.runtime_resolver import (  # noqa: E402
    resolve_text_model_ref,
    resolve_text_runtime,
    resolve_text_runtime_policy,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _parse_yaml(service_yaml: Path) -> Dict[str, Any]:
    if not service_yaml.is_file():
        raise SystemExit(f"service yaml not found: {service_yaml}")
    with service_yaml.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"service yaml is not a mapping: {service_yaml}")
    return data


def _build_params(service_cfg: Dict[str, Any], model_name: str) -> SimpleNamespace:
    """把 YAML 里的 `config.runtime.*` 复刻成一个最小的 params 对象。

    与 ``TextInferrer._build_request_spec`` 对齐：解析器只读 ``service_runtime``（服务端
    ``config.runtime``），不读 ``model_cfg[\"runtime\"]``。
    """
    runtime_cfg = ((service_cfg.get("config") or {}).get("runtime")) or {}
    runtime_cfg = dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}
    return SimpleNamespace(
        model_name=model_name,
        service_runtime=runtime_cfg,
    )


def _resolve_models_dir(service_cfg: Dict[str, Any], override: Optional[str]) -> str:
    """与 InferenceConfig._resolve_dir 等价的最小实现：绝对路径原样；
    相对路径相对仓库根解析。

    CLI `--models-dir` > 服务 YAML 里的 `models_dir` > 仓库根的 `resources/models`。
    """
    if override:
        p = Path(override).expanduser()
        return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())
    raw = service_cfg.get("models_dir") or "resources/models"
    p = Path(raw).expanduser()
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


def _build_messages(
    prompt: str,
    system: Optional[str],
    *,
    image_url: Optional[str] = None,
    video_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    content: str | List[Dict[str, Any]]
    if image_url or video_url:
        parts: List[Dict[str, Any]] = []
        if image_url:
            parts.append({"type": "image_url", "image_url": {"url": image_url}})
        if video_url:
            parts.append({"type": "video_url", "video_url": {"url": video_url}})
        if prompt:
            parts.append({"type": "text", "text": prompt})
        content = parts
    else:
        content = prompt
    msgs.append({"role": "user", "content": content})
    return msgs


async def _run_streaming(bundle, messages, *, max_tokens: int, temperature: float) -> None:
    print("\n===== streaming =====")
    started = time.perf_counter()
    first_delta_at: Optional[float] = None
    total_chars = 0
    final_stats: Dict[str, Any] = {}

    async for chunk in stream_chat_text(
        bundle,
        messages=messages,
        request_id="direct-smoke",
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        delta = chunk.get("delta") or ""
        if delta:
            if first_delta_at is None:
                first_delta_at = time.perf_counter()
            sys.stdout.write(delta)
            sys.stdout.flush()
            total_chars += len(delta)
        if chunk.get("finished"):
            final_stats = {k: v for k, v in chunk.items() if k not in ("delta", "finished")}
            break

    ended = time.perf_counter()
    ttft = (first_delta_at - started) if first_delta_at else None
    print("\n\n===== stats =====")
    print(f"ttft={ttft:.3f}s" if ttft is not None else "ttft=N/A (no delta emitted)")
    print(f"wall={ended - started:.3f}s  delta_chars={total_chars}")
    for k in (
        "prompt_tokens",
        "output_tokens",
        "tok_s_total",
        "tok_s_decode",
        "finish_reason",
    ):
        if k in final_stats:
            print(f"{k}={final_stats[k]}")


async def _run_blocking(bundle, messages, *, max_tokens: int, temperature: float) -> None:
    print("\n===== blocking generate =====")
    started = time.perf_counter()
    text = await generate_chat_text(
        bundle,
        messages=messages,
        request_id="direct-smoke-blk",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    ended = time.perf_counter()
    print(text or "")
    print("\n===== stats =====")
    print(f"wall={ended - started:.3f}s  output_chars={len(text or '')}")


async def _amain(args: argparse.Namespace) -> int:
    service_yaml = Path(args.service_yaml).resolve()
    service_cfg = _parse_yaml(service_yaml)
    params = _build_params(service_cfg, args.model_name)

    runtime = resolve_text_runtime(params)
    if runtime != "ollama":
        raise SystemExit(
            f"This smoke test targets the ollama bridge, but {service_yaml.name} "
            f"resolves to runtime={runtime!r}. Set `config.runtime.backend: ollama` or "
            f"point --service-yaml at a file that does."
        )

    policy = resolve_text_runtime_policy(params)
    models_dir = _resolve_models_dir(service_cfg, args.models_dir)
    model_ref = resolve_text_model_ref(params, models_dir=models_dir)

    print("=" * 60)
    print(f"service_yaml    = {service_yaml}")
    print(f"model_name      = {args.model_name}")
    print(f"models_dir      = {models_dir}")
    print(f"resolved_ref    = {model_ref}")
    print(f"runtime         = {runtime}")
    print(f"ollama_cfg keys = {sorted(policy.ollama_cfg.keys())}")
    print(f"max_model_len   = {policy.max_model_len}")
    print(f"enable_thinking = {policy.enable_thinking}")
    print("=" * 60)
    # 代理环境变量提示：bridge 已经 trust_env=False，但如果 ollama daemon 本身
    # 是远程的、必须走代理，仍然可能出问题，所以这里把现状打出来方便排障。
    proxy_env = {
        k: v for k, v in os.environ.items()
        if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
    }
    if proxy_env:
        print(f"[note] proxy env detected (bridge passes trust_env=False, so localhost "
              f"traffic should bypass anyway): {proxy_env}")

    t0 = time.perf_counter()
    # load 可能很慢（首次注册要把大 gguf copy 进 blob store），放线程跑避免阻塞当前 loop。
    print("\n[load] loading ollama text bundle (may be slow on first run; "
          "big gguf → blob store is ~1-2 GB/s on local NVMe)...")
    bundle = await asyncio.to_thread(load_ollama_text_bundle, model_ref, policy)
    print(f"[load] done in {time.perf_counter() - t0:.2f}s  tag={bundle.tag}")

    try:
        if args.load_only:
            print("\n--load-only specified; skipping chat.")
            return 0

        messages = _build_messages(
            args.prompt,
            args.system,
            image_url=args.image_url,
            video_url=args.video_url,
        )
        if args.mode == "stream":
            await _run_streaming(
                bundle,
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            await _run_blocking(
                bundle,
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        return 0
    finally:
        if args.keep_loaded:
            print("\n[shutdown] --keep-loaded specified; skipping keep_alive=0 unload.")
        else:
            print("\n[shutdown] unloading model from daemon (keep_alive=0)...")
            try:
                shutdown_ollama_text_bundle(bundle)
            except Exception as e:
                print(f"[shutdown] warning: {e}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Direct smoke test for inference/text/runtime/ollama_bridge.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--service-yaml",
        default=str(REPO_ROOT / "inference" / "config" / "service_text_qwen.yaml"),
        help="Path to a service YAML whose config.runtime.backend=ollama.",
    )
    p.add_argument(
        "--model-name",
        required=True,
        help="For ollama.model_source=local_gguf: model directory name under {models_dir} "
             "(e.g. Qwen3.6-35B-A3B-GGUF) or absolute path. "
             "For ollama.model_source=tag: direct Ollama model tag, e.g. qwen3.6:35b.",
    )
    p.add_argument(
        "--models-dir",
        default=None,
        help="Override models_dir (default: service YAML `models_dir` or "
             "<repo_root>/resources/models).",
    )
    p.add_argument("--prompt", default="你好，介绍一下你自己")
    p.add_argument("--system", default=None, help="Optional system prompt.")
    p.add_argument("--image-url", default=None, help="Optional image URL / local path for vision smoke test.")
    p.add_argument("--video-url", default=None, help="Optional video URL / local path for frame-sampled vision smoke test.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--mode", choices=("stream", "blocking"), default="stream")
    p.add_argument(
        "--load-only",
        action="store_true",
        help="Only trigger tag registration / blob upload; do not issue a chat request.",
    )
    p.add_argument(
        "--keep-loaded",
        action="store_true",
        help="Skip keep_alive=0 unload at exit (useful if you plan to run multiple "
             "back-to-back tests and don't want to pay reload cost).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\n[interrupt] user aborted.")
        return 130
    except Exception as e:
        logging.getLogger(__name__).exception("smoke test failed")
        print(f"\n[error] {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
