"""加载并运行 flux-3-image-klein-pup-nunchaku 扩写器。"""

from __future__ import annotations

import base64
import gc
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import Any

from prompt.runtime.request import PromptRequestError

MODEL_NAME = "flux-3-image-klein-pup-nunchaku"
QUANT_FILENAME = "svdq-int4-flux-3-image-klein-pup-nunchaku.safetensors"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
# 来自 https://www.modelscope.cn/models/tonera/flux-3-image-klein-pup-nunchaku
# 不读取模型目录里的 system_prompts.json。
BUNDLED_SYSTEM_PROMPTS = Path(__file__).with_name("system_prompts.json")


@dataclass
class PromptUpsamplerBundle:
    model_ref: str
    pup_dir: str
    quant_path: str
    device: str
    processor: Any
    model: Any
    system_prompts: dict[str, str]


def resolve_model_paths(models_dir: str, weights_dir: str, load_name: str) -> tuple[Path, Path]:
    name = str(load_name or MODEL_NAME).strip() or MODEL_NAME
    pup = _find_pup_dir(models_dir, weights_dir, name)
    quant = _find_quant_file(pup, models_dir, weights_dir, name)
    return pup, quant


def load_upsampler_bundle(
    *,
    models_dir: str,
    weights_dir: str,
    load_name: str,
    device: str = "cuda",
    dtype_name: str = "bfloat16",
) -> PromptUpsamplerBundle:
    import torch

    resolved_device = str(device or "cuda").strip() or "cuda"
    if not resolved_device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Prompt upsampler stays on CUDA and needs a usable GPU.")

    pup, quant = resolve_model_paths(models_dir, weights_dir, load_name)
    system_prompts = _load_system_prompts(BUNDLED_SYSTEM_PROMPTS)

    from transformers import AutoProcessor
    from nunchaku import NunchakuQwen3_5ForConditionalGeneration
    from common.torch_transfer_utils import pretouch_pipeline_cpu_tensors, should_pretouch

    processor = AutoProcessor.from_pretrained(str(pup))
    model = None
    try:
        model = NunchakuQwen3_5ForConditionalGeneration.from_pretrained(
            str(quant),
            dtype=_torch_dtype(dtype_name),
        ).eval()
        if should_pretouch(resolved_device):
            pretouch_pipeline_cpu_tensors(
                type("_Holder", (), {"text_encoder": model})(),
                component_attrs=("text_encoder",),
            )
        model.to(resolved_device)
    except Exception:
        if model is not None:
            _drop_module(model)
        raise

    return PromptUpsamplerBundle(
        model_ref=str(load_name or MODEL_NAME),
        pup_dir=str(pup),
        quant_path=str(quant),
        device=resolved_device,
        processor=processor,
        model=model,
        system_prompts=system_prompts,
    )


def load_image(source: str, *, max_bytes: int = MAX_IMAGE_BYTES) -> Any:
    from PIL import Image, ImageOps

    raw = _fetch_image_bytes(str(source or "").strip(), max_bytes=max_bytes)
    try:
        with Image.open(BytesIO(raw)) as image:
            converted = ImageOps.exif_transpose(image).convert("RGB").copy()
            converted.load()
            return converted
    except PromptRequestError:
        raise
    except Exception as exc:
        raise PromptRequestError(f"Cannot decode image: {exc}") from exc


def upsample(
    bundle: PromptUpsamplerBundle,
    *,
    prompt: str,
    images: list[Any],
    max_new_tokens: int,
    cancel_event: Event | None = None,
    task_id: str = "",
) -> tuple[str, int, int, str]:
    """返回 (caption, prompt_tokens, completion_tokens, finish_reason)。取消时抛出 TaskCancelledError。"""
    import torch
    from transformers.generation import StoppingCriteria, StoppingCriteriaList

    from common.task_cancel import TaskCancelledError

    if cancel_event is not None and cancel_event.is_set():
        raise TaskCancelledError(task_id, "before upsample")

    messages = _build_messages(bundle, prompt=prompt, images=images)
    inputs = bundle.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(bundle.model.device)

    class _CancelStoppingCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
            return bool(cancel_event is not None and cancel_event.is_set())

    try:
        with torch.no_grad():
            output = bundle.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([_CancelStoppingCriteria()]),
            )
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    if cancel_event is not None and cancel_event.is_set():
        raise TaskCancelledError(task_id, "during upsample")

    prompt_tokens = int(inputs["input_ids"].shape[1])
    new_tokens = output[0, prompt_tokens:]
    caption = bundle.processor.decode(new_tokens, skip_special_tokens=True).strip()
    completion_tokens = int(new_tokens.shape[0])
    finish = "length" if completion_tokens >= int(max_new_tokens) else "stop"
    return caption, prompt_tokens, completion_tokens, finish


def release_upsampler_bundle(bundle: PromptUpsamplerBundle | None) -> None:
    if bundle is None:
        return
    model = getattr(bundle, "model", None)
    bundle.model = None
    bundle.processor = None
    bundle.system_prompts = {}
    _drop_module(model)
    try:
        from common.runtime_cleanup import cleanup_after_release

        cleanup_after_release(trim_malloc=True)
    except Exception:
        gc.collect()


def _build_messages(bundle: PromptUpsamplerBundle, *, prompt: str, images: list[Any]) -> list[dict[str, Any]]:
    if images:
        text = (
            "Reference images follow in order (the first image, the second image, ...).\n"
            f"User request: {prompt}\nOutput the caption."
        )
        content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": text}]
        return [
            {"role": "system", "content": bundle.system_prompts["i2i"]},
            {"role": "user", "content": content},
        ]
    return [
        {"role": "system", "content": bundle.system_prompts["t2i"]},
        {"role": "user", "content": prompt},
    ]


def _load_system_prompts(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Bundled system prompts not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise FileNotFoundError(f"system_prompts.json must be an object: {path}")
    prompts = {str(key): str(value) for key, value in raw.items() if isinstance(value, str) and str(value).strip()}
    missing = [key for key in ("t2i", "i2i") if key not in prompts]
    if missing:
        raise FileNotFoundError(f"system_prompts.json missing {missing}: {path}")
    return prompts


def _find_pup_dir(models_dir: str, weights_dir: str, load_name: str) -> Path:
    raw = Path(load_name)
    candidates: list[Path] = []
    if raw.is_absolute() or raw.exists():
        candidates.append(raw)
    else:
        candidates.append(Path(models_dir) / load_name)
        candidates.append(Path(weights_dir) / load_name)
    for root in candidates:
        if root.is_dir():
            return root
    searched = ", ".join(str(path) for path in candidates) or load_name
    raise FileNotFoundError(
        f"Prompt upsampler repo not found for {load_name}. "
        f"Expected processor directory under: {searched}"
    )


def _find_quant_file(pup: Path, models_dir: str, weights_dir: str, load_name: str) -> Path:
    names = [QUANT_FILENAME]
    candidates = [
        pup / QUANT_FILENAME,
        Path(weights_dir) / load_name / QUANT_FILENAME,
        Path(models_dir) / load_name / QUANT_FILENAME,
        Path(weights_dir) / QUANT_FILENAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(pup.glob("svdq-*.safetensors"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        preferred = [path for path in matches if path.name in names]
        if preferred:
            return preferred[0]
        return matches[0]
    raise FileNotFoundError(
        f"Quantized upsampler not found for {load_name}. "
        f"Expected {QUANT_FILENAME} under {pup} or {weights_dir}"
    )


def _fetch_image_bytes(source: str, *, max_bytes: int) -> bytes:
    if not source:
        raise PromptRequestError("Image source is empty")
    if source.startswith("data:"):
        try:
            _, encoded = source.split(",", 1)
        except ValueError as exc:
            raise PromptRequestError("Malformed data URI image") from exc
        try:
            raw = base64.b64decode(encoded)
        except Exception as exc:
            raise PromptRequestError("Data URI is not valid base64") from exc
        if len(raw) > max_bytes:
            raise PromptRequestError(f"Each image must be at most {max_bytes // (1024 * 1024)}MB")
        return raw

    if source.startswith("file:"):
        path = Path(urllib.request.url2pathname(urllib.parse.urlparse(source).path))
        return _read_file(path, max_bytes=max_bytes)

    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "flux3-pup"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return _read_capped(iter(lambda: response.read(1024 * 1024), b""), max_bytes=max_bytes)
        except PromptRequestError:
            raise
        except Exception as exc:
            raise PromptRequestError(f"Failed to download image: {exc}") from exc

    return _read_file(Path(source).expanduser(), max_bytes=max_bytes)


def _read_file(path: Path, *, max_bytes: int) -> bytes:
    if not path.is_file():
        raise PromptRequestError(f"Image not found: {path}")
    if path.stat().st_size > max_bytes:
        raise PromptRequestError(f"Each image must be at most {max_bytes // (1024 * 1024)}MB")
    return path.read_bytes()


def _read_capped(chunks, *, max_bytes: int) -> bytes:
    total = 0
    parts: list[bytes] = []
    for block in chunks:
        if not block:
            break
        total += len(block)
        if total > max_bytes:
            raise PromptRequestError(f"Each image must be at most {max_bytes // (1024 * 1024)}MB")
        parts.append(block)
    if not parts:
        raise PromptRequestError("Image is empty")
    return b"".join(parts)


def _torch_dtype(name: str):
    import torch

    key = str(name or "bfloat16").strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported prompt upsampler dtype: {name}")
    return mapping[key]


def _drop_module(model: Any) -> None:
    if model is None:
        return
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
