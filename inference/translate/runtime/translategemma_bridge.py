# pyright: reportMissingImports=false

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranslateGemmaPolicy:
    dtype: str = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 768
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        return (
            f"runtime=translategemma|dtype={self.dtype}|device_map={self.device_map}|"
            f"trc={int(self.trust_remote_code)}|"
            f"temp={self.temperature:.3f}|top_p={self.top_p:.3f}|"
            f"max_new={self.max_new_tokens}|"
            f"model_kw={sorted((self.model_kwargs or {}).items())}"
        )


@dataclass
class TranslateGemmaBundle:
    model_ref: str
    policy: TranslateGemmaPolicy
    processor: Any
    model: Any
    device: Any = None

    def translate(self, messages: List[Dict[str, Any]], *, max_new_tokens: Optional[int] = None) -> str:
        return _generate(self, messages=messages, max_new_tokens=max_new_tokens)

    def shutdown(self) -> None:
        _shutdown_bundle(self)


def normalize_model_ref(load_name: str, *, models_dir: Optional[str]) -> str:
    name = str(load_name or "").strip()
    if not name:
        raise ValueError("translate task requires a non-empty load_name")

    candidate = Path(name).expanduser()
    if candidate.is_absolute():
        if not candidate.exists():
            raise ValueError(f"Translate model path not found: {candidate}")
        return str(candidate.resolve())

    if models_dir:
        rooted = (Path(models_dir).expanduser().resolve() / name).resolve()
        if rooted.exists():
            return str(rooted)

    logger.info(
        "translate load_name=%s not found under models_dir=%s; "
        "will pass as-is to transformers (HF repo id mode)",
        name,
        models_dir,
    )
    return name


def build_translate_messages(
    *,
    source_lang: str,
    target_lang: str,
    text: Optional[str] = None,
    image_ref: Optional[str] = None,
) -> List[Dict[str, Any]]:
    source = str(source_lang or "").strip()
    target = str(target_lang or "").strip()
    if not source or not target:
        raise ValueError("source_lang and target_lang are required for TranslateGemma")

    if image_ref:
        image_url = str(image_ref).strip()
        if image_url and not image_url.startswith(("http://", "https://", "file://")):
            image_url = Path(image_url).expanduser().resolve().as_uri()
        content_item: Dict[str, Any] = {
            "type": "image",
            "source_lang_code": source,
            "target_lang_code": target,
            "url": image_url,
        }
    else:
        content_item = {
            "type": "text",
            "source_lang_code": source,
            "target_lang_code": target,
            "text": str(text or ""),
        }

    return [{"role": "user", "content": [content_item]}]


def _resolve_torch_dtype(dtype: str) -> Any:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    normalized = str(dtype or "auto").strip().lower()
    if normalized in {"auto", ""}:
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
    return mapping.get(normalized, "auto")


def load_translategemma_bundle(model_ref: str, policy: TranslateGemmaPolicy) -> TranslateGemmaBundle:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "TranslateGemma requires transformers with AutoModelForImageTextToText support."
        ) from e

    logger.info(
        "Loading TranslateGemma bundle model_ref=%s dtype=%s device_map=%s",
        model_ref,
        policy.dtype,
        policy.device_map,
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
        model_kwargs.update(policy.model_kwargs)

    model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)
    try:
        model.eval()
    except Exception:
        pass

    device = getattr(model, "device", None)
    logger.info("TranslateGemma bundle ready model_ref=%s device=%s", model_ref, device)
    return TranslateGemmaBundle(
        model_ref=model_ref,
        policy=policy,
        processor=processor,
        model=model,
        device=device,
    )


def _generate(
    bundle: TranslateGemmaBundle,
    *,
    messages: List[Dict[str, Any]],
    max_new_tokens: Optional[int] = None,
) -> str:
    processor = bundle.processor
    model = bundle.model
    if processor is None or model is None:
        raise RuntimeError("TranslateGemmaBundle has been shutdown")

    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("PyTorch is required for TranslateGemma runtime") from e

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        raise RuntimeError(
            f"TranslateGemma apply_chat_template failed: {type(e).__name__}: {e}"
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
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        inputs = moved

    model_dtype = getattr(model, "dtype", None)
    if model_dtype is not None:
        try:
            inputs = inputs.to(dtype=model_dtype)
        except Exception:
            pass

    max_new = int(max_new_tokens or bundle.policy.max_new_tokens)
    temperature = float(bundle.policy.temperature)
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = float(bundle.policy.top_p)

    input_len = 0
    try:
        input_len = int(inputs["input_ids"].shape[-1])
    except Exception:
        pass

    with torch.inference_mode():
        generation = model.generate(**inputs, **generation_kwargs)

    try:
        trimmed = generation[0][input_len:]
        return processor.decode(trimmed, skip_special_tokens=True)
    except Exception:
        return processor.decode(generation[0], skip_special_tokens=True)


def _shutdown_bundle(bundle: TranslateGemmaBundle) -> None:
    model = getattr(bundle, "model", None)
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
