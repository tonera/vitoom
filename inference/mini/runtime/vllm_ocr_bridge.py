"""
GLM-OCR 的 vLLM 薄封装

职责：
- 加载 vLLM bundle（同步 LLM 引擎 + processor/tokenizer 组合）
- 多模态 chat 模板渲染（image-text-to-text）
- 同步 generate（OCR 不需要流式，不需要中断，一次请求一次结果）
- 资源释放（shutdown + torch cuda 清理）

和 text 服务 vllm_bridge 的差异（刻意收敛范围）：
- 不使用 AsyncLLMEngine（OCR 没有流式需求；同步 LLM 更简单且支持 chat API）
- 不处理 tool-calls / enable_thinking / 视频后端选择
- 只处理 image 多模态（PDF 输入由调用方预先渲染成 image 或交给 GLM-OCR SDK；这里永远看到 image）
- 只支持 MIT 风格的 messages：[{role: user, content: [{type:image,image:...},{type:text,text:...}]}]

只有当真正需要更多能力（流式 / 视频 / tool）时再抽取共享代码到 common/，避免过早抽象。
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# pyright: reportMissingImports=false

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VllmOcrPolicy:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.35
    max_model_len: Optional[int] = 8192
    trust_remote_code: bool = True
    dtype: str = "auto"
    # 每个 prompt 允许的最大图片数（GLM-OCR 技术报告里一页即一图，留余量）
    max_images_per_prompt: int = 4
    # 采样参数
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 8192
    # 额外透传到 LLM() 构造器的 kwargs（engine_kwargs）
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        return (
            f"tp={self.tensor_parallel_size}|"
            f"gpu_mem={self.gpu_memory_utilization:.3f}|"
            f"max_len={self.max_model_len or 0}|"
            f"trc={int(self.trust_remote_code)}|"
            f"dtype={self.dtype}|"
            f"max_img={self.max_images_per_prompt}|"
            f"temp={self.temperature:.3f}|top_p={self.top_p:.3f}|"
            f"max_new={self.max_new_tokens}|"
            f"engine={sorted((self.engine_kwargs or {}).items())}"
        )


@dataclass
class VllmOcrBundle:
    """单个 GLM-OCR 的运行时句柄。"""
    model_ref: str
    policy: VllmOcrPolicy
    llm: Any            # vllm.LLM
    processor: Any      # transformers.AutoProcessor
    tokenizer: Any      # processor.tokenizer 或 AutoTokenizer
    # 记录 stop_token_ids，供 SamplingParams 使用
    stop_token_ids: List[int] = field(default_factory=list)

    # ---------- OcrBundleLike ----------

    def generate_from_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """统一接口：渲染多模态 messages 并同步生成。

        等价于先 `render_prompt_and_mm(bundle, messages)` 再 `generate_ocr_text(bundle, prompt=..., multi_modal_data=...)`；
        这里封一层，让 handler/doc_pipeline 和 transformers runtime 使用同一个调用形态。
        """
        prompt, mm = render_prompt_and_mm(self, messages)
        return generate_ocr_text(
            self,
            prompt=prompt,
            multi_modal_data=mm,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def shutdown(self) -> None:
        """释放 vLLM engine + cuda 显存。供 MiniInferrer._release_bundle 调用。"""
        _shutdown_bundle(self)


def _normalize_model_ref(load_name: str, *, models_dir: Optional[str]) -> str:
    """把 load_name（可能是绝对路径、相对路径或仓库 id）归一化为真实路径或仓库 id。"""
    name = str(load_name or "").strip()
    if not name:
        raise ValueError("mini OCR requires a non-empty load_name")

    candidate = Path(name).expanduser()
    if candidate.is_absolute():
        if not candidate.exists():
            raise ValueError(f"OCR model path not found: {candidate}")
        return str(candidate.resolve())

    if models_dir:
        rooted = (Path(models_dir).expanduser().resolve() / name).resolve()
        if rooted.exists():
            return str(rooted)

    # 留一条"直接当 HF repo id"的口，便于开发环境未预下载时也能跑通
    logger.info(
        "OCR load_name=%s not found under models_dir=%s; "
        "will pass as-is to vLLM (HF repo id mode)",
        name, models_dir,
    )
    return name


def load_vllm_ocr_bundle(model_ref: str, policy: VllmOcrPolicy) -> VllmOcrBundle:
    """同步加载 GLM-OCR bundle。由 MiniInferrer 放到 run_blocking 线程调用。"""
    try:
        from vllm import LLM  # type: ignore
    except Exception as e:
        logger.exception("Failed to import vllm.LLM (underlying cause below)")
        raise RuntimeError(
            f"Failed to import vllm.LLM: {type(e).__name__}: {e}. "
            "Please ensure a recent vllm version is installed "
            "(GLM-OCR requires vllm with glm_ocr model support). "
            "If you just upgraded transformers to v5, note that older vllm versions "
            "rely on transformers v4 internals and may fail to import."
        ) from e

    try:
        from transformers import AutoProcessor, AutoTokenizer
    except Exception as e:
        logger.exception("Failed to import transformers for OCR processor")
        raise RuntimeError(
            f"Failed to import transformers for OCR processor: {type(e).__name__}: {e}. "
            "GLM-OCR requires a recent transformers version."
        ) from e

    logger.info(
        "Loading GLM-OCR vLLM bundle model_ref=%s tp=%s gpu_mem=%.3f max_len=%s dtype=%s",
        model_ref,
        policy.tensor_parallel_size,
        policy.gpu_memory_utilization,
        policy.max_model_len,
        policy.dtype,
    )

    # 先尝试 AutoProcessor（多模态必备）；失败时回退 AutoTokenizer 以保证至少能渲染文本
    #
    # 兼容性说明：
    # - transformers v4.x：processor 通常带 .tokenizer 子对象。
    # - transformers v5.x：对部分模型（如 GLM-OCR/Falcon-OCR），AutoProcessor 可能直接返回
    #   TokenizersBackend，本身就像一个 tokenizer（有 apply_chat_template / eos_token_id），
    #   此时没有 .tokenizer 属性。我们把 processor 自身当 tokenizer 使用即可。
    processor: Any = None
    tokenizer: Any = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_ref,
            trust_remote_code=policy.trust_remote_code,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None and processor is not None and hasattr(processor, "apply_chat_template"):
            tokenizer = processor
    except Exception as e:
        logger.info("AutoProcessor.from_pretrained failed for %s: %s", model_ref, e)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=policy.trust_remote_code,
        )

    engine_kwargs: Dict[str, Any] = {
        "model": model_ref,
        "tensor_parallel_size": policy.tensor_parallel_size,
        "gpu_memory_utilization": policy.gpu_memory_utilization,
        "trust_remote_code": policy.trust_remote_code,
        "dtype": policy.dtype,
        "limit_mm_per_prompt": {"image": policy.max_images_per_prompt},
    }
    if policy.max_model_len:
        engine_kwargs["max_model_len"] = policy.max_model_len
    if policy.engine_kwargs:
        engine_kwargs.update(policy.engine_kwargs)

    llm = LLM(**engine_kwargs)

    # 收集 eos/extra stop token ids（OCR 模型有时会有自定义 end token）
    stop_token_ids: List[int] = []
    try:
        eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            stop_token_ids.append(eos)
    except Exception:
        pass

    return VllmOcrBundle(
        model_ref=model_ref,
        policy=policy,
        llm=llm,
        processor=processor if processor is not None else tokenizer,
        tokenizer=tokenizer,
        stop_token_ids=stop_token_ids,
    )


def _shutdown_bundle(bundle: VllmOcrBundle) -> None:
    """尽力释放 vLLM 引擎与 cuda 显存。"""
    try:
        # vllm 0.6+ 的 LLM 对外暴露的 llm_engine；不同版本 API 名不一，best-effort
        engine = getattr(bundle.llm, "llm_engine", None)
        if engine is not None:
            for attr in ("shutdown", "shutdown_background_loop", "stop", "close"):
                fn = getattr(engine, attr, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
    except Exception:
        logger.warning("llm_engine shutdown best-effort failed", exc_info=True)

    try:
        del bundle.llm
    except Exception:
        pass
    bundle.llm = None  # type: ignore[assignment]

    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 多模态消息构造与渲染
# ---------------------------------------------------------------------------


def build_ocr_messages(
    *,
    image_paths_or_urls: List[str],
    text_prompt: str,
) -> List[Dict[str, Any]]:
    """构造 GLM-OCR 期望的单轮多模态 chat messages。

    参考官方 README：
        [{ "role": "user", "content": [
              {"type": "image", "url": "..."} ,
              {"type": "text",  "text": "Text Recognition:"}
        ]}]
    """
    content: List[Dict[str, Any]] = []
    for item in image_paths_or_urls:
        content.append({"type": "image", "image": item})
    content.append({"type": "text", "text": text_prompt})
    return [{"role": "user", "content": content}]


def render_prompt_and_mm(
    bundle: VllmOcrBundle,
    messages: List[Dict[str, Any]],
) -> tuple[str, Dict[str, Any]]:
    """把 messages 渲染为 prompt 字符串 + multi_modal_data（供 vLLM LLM.generate 使用）。

    步骤：
    1. 用 processor.apply_chat_template 生成带占位符的 prompt（保留 <image> 等特殊 token）
    2. 用 PIL 打开每个图片（本地路径；PDF 不应出现在这里——调用方上游必须已展开为图片）
    3. 返回 (prompt, {"image": [PIL.Image, ...]})

    注意：本函数只处理 "image" 这一种 mm 类型。若未来要接视频/音频小模型，请在对应的 handler 里另走一条渲染路径。
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError("PIL is required for OCR image inputs") from e

    adapter = bundle.processor or bundle.tokenizer
    if adapter is None or not hasattr(adapter, "apply_chat_template"):
        raise RuntimeError(
            "vLLM OCR bundle has no chat-capable processor/tokenizer; "
            "check model_ref and transformers version."
        )

    try:
        prompt = adapter.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        raise RuntimeError(f"apply_chat_template failed for OCR messages: {e}") from e

    # 从 messages 里抽取 image 引用并加载为 PIL
    pil_images: List[Any] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type") or "").lower() != "image":
                continue
            src = part.get("image") or part.get("url") or part.get("path")
            if not src:
                continue
            img = _load_pil_image(src)
            pil_images.append(img)

    mm_data: Dict[str, Any] = {}
    if pil_images:
        # vLLM 对 multi_modal_data.image：单图传单对象，多图传列表
        mm_data["image"] = pil_images[0] if len(pil_images) == 1 else pil_images

    return prompt, mm_data


def _load_pil_image(src: Any) -> Any:
    from PIL import Image  # type: ignore

    if hasattr(src, "convert"):
        # 已是 PIL.Image
        img = src
    elif isinstance(src, (str, Path)):
        p = str(src)
        if p.startswith(("http://", "https://")):
            import io

            import requests  # type: ignore

            resp = requests.get(p, timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
        else:
            img = Image.open(p)
    else:
        raise ValueError(f"unsupported image source type: {type(src)}")

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# 同步 generate
# ---------------------------------------------------------------------------


def generate_ocr_text(
    bundle: VllmOcrBundle,
    *,
    prompt: str,
    multi_modal_data: Optional[Dict[str, Any]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop_token_ids: Optional[List[int]] = None,
) -> str:
    """同步执行一次生成。返回拼接后的文本。"""
    try:
        from vllm import SamplingParams  # type: ignore
    except Exception as e:
        raise RuntimeError("vllm.SamplingParams import failed") from e

    sp_kwargs: Dict[str, Any] = {
        "temperature": float(temperature if temperature is not None else bundle.policy.temperature),
        "top_p": float(top_p if top_p is not None else bundle.policy.top_p),
        "max_tokens": int(max_new_tokens if max_new_tokens is not None else bundle.policy.max_new_tokens),
    }
    final_stop_ids = list(stop_token_ids) if stop_token_ids else list(bundle.stop_token_ids or [])
    if final_stop_ids:
        sp_kwargs["stop_token_ids"] = final_stop_ids

    sampling = SamplingParams(**sp_kwargs)

    inputs: Dict[str, Any] = {"prompt": prompt}
    if multi_modal_data:
        inputs["multi_modal_data"] = multi_modal_data

    outputs = bundle.llm.generate([inputs], sampling_params=sampling)
    if not outputs:
        return ""

    # outputs[0].outputs 是 List[CompletionOutput]；取首个即可（我们没开 n>1）
    try:
        first = outputs[0].outputs[0]
    except Exception:
        return ""

    text = getattr(first, "text", "") or ""
    return str(text)
