"""
GLM-OCR 的 transformers 薄封装（默认 runtime）

职责：
- 加载 HF bundle（AutoProcessor + AutoModelForImageTextToText）
- 按官方示例用 `processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")`
  一步到位拿到 input_ids / pixel_values 等，然后 `model.generate`
- 提供与 `OcrBundleLike` 相同的 `generate_from_messages` / `shutdown` 接口，供 handler/doc_pipeline 共用
- 冷启动远快于 vLLM：只做权重加载 + CUDA 驻留，无 KV 预分配 / 图编译等开销

和 vllm_ocr_bridge 的差异（刻意缩小表面）：
- messages 不先渲染成 prompt 字符串再配 mm_data；而是直接交给 processor，由它内部分词 + 打图像张量
- 没有 tensor_parallel_size / gpu_memory_utilization 概念（不适用）
- 显存稳态低很多：只有权重 + 生成时的单请求 KV，没有全局 KV 池
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# pyright: reportMissingImports=false

from common.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Policy / Bundle
# ---------------------------------------------------------------------------


@dataclass
class HfOcrPolicy:
    """transformers 后端的运行时策略。"""

    dtype: str = "auto"              # "auto" / "bfloat16" / "float16" / "float32"
    device_map: str = "auto"         # 交给 accelerate 分配
    trust_remote_code: bool = True
    # 采样参数（OCR 默认贪心）
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 8192
    # 透传给 AutoModelForImageTextToText.from_pretrained 的额外 kwargs
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        return (
            f"runtime=hf|dtype={self.dtype}|device_map={self.device_map}|"
            f"trc={int(self.trust_remote_code)}|"
            f"temp={self.temperature:.3f}|top_p={self.top_p:.3f}|"
            f"max_new={self.max_new_tokens}|"
            f"model_kw={sorted((self.model_kwargs or {}).items())}"
        )


@dataclass
class HfOcrBundle:
    """transformers 版本的 OCR 运行时句柄。"""

    model_ref: str
    policy: HfOcrPolicy
    processor: Any                   # AutoProcessor
    model: Any                       # AutoModelForImageTextToText
    device: Any = None               # model.device（torch.device 或 str）

    # ---------- OcrBundleLike ----------

    def generate_from_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        return _hf_generate(
            self,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate_batch_from_messages(
        self,
        messages_list: List[List[Dict[str, Any]]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        """批量版本：一次 processor + 一次 model.generate 完成多条请求。

        上层（doc_pipeline）把同一页同一类（共用 prompt 的 text/title/caption/list）
        的 block 合并成一个 batch 送进来，能把单块串行的 prefill/解码摊到 GPU 并行，
        小模型 + 消费卡上 batch=4~8 通常能带来 2~3x 吞吐提升。
        """
        return _hf_generate_batch(
            self,
            messages_list=messages_list,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def shutdown(self) -> None:
        _shutdown_bundle(self)


# ---------------------------------------------------------------------------
# Normalize model_ref
# ---------------------------------------------------------------------------


def _normalize_model_ref(load_name: str, *, models_dir: Optional[str]) -> str:
    """把 load_name（可能是绝对路径 / 相对路径 / HF 仓库 id）归一化为路径或 repo id。

    规则和 vllm_ocr_bridge 完全一致，保证切 runtime 时 cache key 一致性行为。
    """
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

    logger.info(
        "OCR load_name=%s not found under models_dir=%s; "
        "will pass as-is to transformers (HF repo id mode)",
        name, models_dir,
    )
    return name


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _resolve_torch_dtype(dtype: str) -> Any:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    s = str(dtype or "auto").strip().lower()
    if s in ("auto", ""):
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(s, "auto")


def load_hf_ocr_bundle(model_ref: str, policy: HfOcrPolicy) -> HfOcrBundle:
    """同步加载 transformers 版 GLM-OCR bundle。由 MiniInferrer 放到 run_blocking 线程调用。"""
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText  # type: ignore
    except Exception as e:
        logger.exception("Failed to import transformers for HF OCR bridge")
        raise RuntimeError(
            f"Failed to import transformers: {type(e).__name__}: {e}. "
            "GLM-OCR (HF runtime) requires a recent transformers version "
            "with AutoModelForImageTextToText (>=4.44)."
        ) from e

    logger.info(
        "Loading GLM-OCR HF bundle model_ref=%s dtype=%s device_map=%s",
        model_ref, policy.dtype, policy.device_map,
    )

    processor = AutoProcessor.from_pretrained(
        model_ref,
        trust_remote_code=policy.trust_remote_code,
    )

    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_ref,
        "torch_dtype": _resolve_torch_dtype(policy.dtype),
        "device_map": policy.device_map,
        "trust_remote_code": policy.trust_remote_code,
    }
    if policy.model_kwargs:
        # 调用方透传（如 attn_implementation / low_cpu_mem_usage 等）
        model_kwargs.update(policy.model_kwargs)

    model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)
    try:
        model.eval()
    except Exception:
        pass

    device = getattr(model, "device", None)
    logger.info("HF OCR bundle ready model_ref=%s device=%s", model_ref, device)

    return HfOcrBundle(
        model_ref=model_ref,
        policy=policy,
        processor=processor,
        model=model,
        device=device,
    )


def _shutdown_bundle(bundle: HfOcrBundle) -> None:
    """释放 HF 模型 + CUDA 显存。"""
    try:
        model = bundle.model
    except Exception:
        model = None

    try:
        bundle.model = None  # type: ignore[assignment]
    except Exception:
        pass
    try:
        bundle.processor = None  # type: ignore[assignment]
    except Exception:
        pass

    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass

    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Message normalization (reuse the same schema as vllm bridge)
# ---------------------------------------------------------------------------


def _normalize_messages_for_hf(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    GLM-OCR 官方示例里 image 部分用 {"type":"image","url":"..."}；
    我们项目内部统一用 {"type":"image","image":<path|PIL|URL>}。
    这里兼容两种写法，并把 PIL 对象直接保留，processor 会自己处理。

    注意：
    - transformers 的 image processor 会按 "url" 字段从本地/URL 自动加载；
      如果我们传的是 PIL，会兜到 "image" 字段上，也有 processor 支持。
    - 为了避免不同版本差异，这里若识别到非 URL/路径字符串（比如 PIL）就改成
      "image" 字段，其余（路径 / http(s)）一律改成 "url"，与官方 README 对齐。
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue

        new_content: List[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue
            if str(part.get("type") or "").lower() != "image":
                new_content.append(part)
                continue

            src = part.get("image", part.get("url", part.get("path")))
            if src is None:
                new_content.append(part)
                continue

            if hasattr(src, "convert"):
                # PIL 对象 → 统一放到 "image" 字段
                new_content.append({"type": "image", "image": src})
            elif isinstance(src, (str, Path)):
                s = str(src)
                if s.startswith(("http://", "https://", "file://")):
                    new_content.append({"type": "image", "url": s})
                else:
                    # 本地路径：HF processor 接受 "image" 字段直接传路径，
                    # 也可以借 "url": "file://..."；用 "image" 最通用
                    new_content.append({"type": "image", "image": s})
            else:
                new_content.append(part)

        out.append({**msg, "content": new_content})

    return out


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def _hf_generate(
    bundle: HfOcrBundle,
    *,
    messages: List[Dict[str, Any]],
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    """对一次多模态 messages 调用 transformers 的 generate，返回解码后的文本。"""
    processor = bundle.processor
    model = bundle.model
    if processor is None or model is None:
        raise RuntimeError("HfOcrBundle has been shutdown; no model/processor available.")

    hf_messages = _normalize_messages_for_hf(messages)

    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover - torch 必存在
        raise RuntimeError("PyTorch is required for HF OCR runtime") from e

    try:
        inputs = processor.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        raise RuntimeError(
            f"apply_chat_template failed for HF OCR messages: {type(e).__name__}: {e}"
        ) from e

    # 有些 tokenizer 返回 token_type_ids 但 model.generate 不识别，移除即可
    try:
        inputs.pop("token_type_ids", None)
    except Exception:
        pass

    # 搬到模型设备
    device = getattr(model, "device", bundle.device)
    try:
        inputs = inputs.to(device)
    except Exception:
        # 部分版本是纯 dict，退回到手动搬运
        moved: Dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                try:
                    moved[k] = v.to(device)
                    continue
                except Exception:
                    pass
            moved[k] = v
        inputs = moved  # type: ignore[assignment]

    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    _temp = float(temperature if temperature is not None else bundle.policy.temperature)
    _top_p = float(top_p if top_p is not None else bundle.policy.top_p)
    _max_new = int(max_new_tokens if max_new_tokens is not None else bundle.policy.max_new_tokens)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": _max_new,
    }
    if _temp > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = _temp
        gen_kwargs["top_p"] = _top_p
    else:
        gen_kwargs["do_sample"] = False

    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs) if isinstance(inputs, dict) else model.generate(**inputs, **gen_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"model.generate failed for HF OCR bundle: {type(e).__name__}: {e}"
        ) from e

    # generated: (batch, seq_len)；batch=1
    try:
        new_tokens = generated[0][prompt_len:]
    except Exception:
        new_tokens = generated[0]

    try:
        text = processor.decode(new_tokens, skip_special_tokens=True)
    except Exception:
        # 部分 processor 没有 decode，退回 tokenizer
        tok = getattr(processor, "tokenizer", None) or processor
        text = tok.decode(new_tokens, skip_special_tokens=True)

    return str(text or "").strip()


def _hf_generate_batch(
    bundle: HfOcrBundle,
    *,
    messages_list: List[List[Dict[str, Any]]],
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> List[str]:
    """批量 generate。参见 HfOcrBundle.generate_batch_from_messages。"""
    if not messages_list:
        return []
    if len(messages_list) == 1:
        # 退化到单条路径：避免 padding / batch 开销。
        return [
            _hf_generate(
                bundle,
                messages=messages_list[0],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        ]

    processor = bundle.processor
    model = bundle.model
    if processor is None or model is None:
        raise RuntimeError("HfOcrBundle has been shutdown; no model/processor available.")

    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for HF OCR runtime") from e

    hf_batch = [_normalize_messages_for_hf(m) for m in messages_list]

    # 生成阶段必须左 padding：HF 的 tokenizer 默认一般是 right padding，会让自回归起点
    # 落在 pad 上导致乱码。只在本次调用期间切换，不污染进程全局。
    tokenizer = getattr(processor, "tokenizer", None)
    prev_padding_side = None
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        prev_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

    try:
        try:
            inputs = processor.apply_chat_template(
                hf_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"apply_chat_template(batch) failed: {type(e).__name__}: {e}"
            ) from e

        try:
            inputs.pop("token_type_ids", None)
        except Exception:
            pass

        device = getattr(model, "device", bundle.device)
        try:
            inputs = inputs.to(device)
        except Exception:
            moved: Dict[str, Any] = {}
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    try:
                        moved[k] = v.to(device)
                        continue
                    except Exception:
                        pass
                moved[k] = v
            inputs = moved  # type: ignore[assignment]

        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs["input_ids"]
        prompt_len = int(input_ids.shape[1])   # 左 padding 后所有样本共享同一"起点"

        _temp = float(temperature if temperature is not None else bundle.policy.temperature)
        _top_p = float(top_p if top_p is not None else bundle.policy.top_p)
        _max_new = int(max_new_tokens if max_new_tokens is not None else bundle.policy.max_new_tokens)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": _max_new}
        if _temp > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = _temp
            gen_kwargs["top_p"] = _top_p
        else:
            gen_kwargs["do_sample"] = False

        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, **gen_kwargs) if isinstance(inputs, dict) else model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"model.generate(batch) failed: {type(e).__name__}: {e}"
            ) from e

        decoder = getattr(processor, "decode", None)
        if decoder is None:
            tok = tokenizer or processor
            decoder = tok.decode

        results: List[str] = []
        for i in range(generated.shape[0]):
            new_tokens = generated[i][prompt_len:]
            try:
                text = decoder(new_tokens, skip_special_tokens=True)
            except Exception:
                tok = tokenizer or processor
                text = tok.decode(new_tokens, skip_special_tokens=True)
            results.append(str(text or "").strip())
        return results
    finally:
        if tokenizer is not None and prev_padding_side is not None:
            try:
                tokenizer.padding_side = prev_padding_side
            except Exception:
                pass


__all__ = [
    "HfOcrPolicy",
    "HfOcrBundle",
    "load_hf_ocr_bundle",
    "_normalize_model_ref",
]
