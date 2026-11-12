"""在模型根目录里按子目录查找 MiniMax-H3 组件，不把文件名写死。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class H3Checkpoints:
    root: Path
    dit: Path
    text_encoder: Path
    video_vae: Path
    audio_vae: Path
    lora: Path | None
    tokenizer_dir: Path


def resolve_model_root(models_base_dir: str, model_ref: str) -> Path:
    ref = str(model_ref or "").strip()
    if not ref:
        raise ValueError("H3 需要 load_name（models_dir 下的目录名，或绝对路径）")
    path = Path(ref)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"模型目录不存在: {path}")
        return path
    if "/" in ref or "\\" in ref:
        raise ValueError(f"load_name 必须是 models_dir 下的目录名，不能包含斜杠: {model_ref}")
    root = (Path(models_base_dir).expanduser().resolve() / ref).resolve()
    if not root.exists():
        raise FileNotFoundError(f"模型目录不存在: {root}")
    return root


def _component_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "minimax_h3" in name and name.endswith((".safetensors", ".sft", ".pt", ".pth")):
            found.append(path)
    return found


def _pick(files: list[Path], predicate, label: str, searched: Path) -> Path:
    matches = [path for path in files if predicate(path.name.lower())]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        newest = max(matches, key=lambda path: path.stat().st_mtime)
        return newest
    names = [path.name for path in files]
    raise FileNotFoundError(f"缺少{label}: 在 {searched} 中没有匹配文件。已有文件: {names or '（目录空或不存在）'}")


def resolve_checkpoints(
    root: Path,
    *,
    dit_variant: str = "fl2va",
    turbo: bool = True,
    tokenizer_dir: Path | None = None,
) -> H3Checkpoints:
    """dit_variant 选 DiT 文件：t2va 和 fl2va 共用 fl2va 的 DiT，ref2va 用自己的。"""
    missing: list[str] = []
    dit_dir = root / "diffusion_models"
    text_dir = root / "text_encoders"
    vae_dir = root / "vae"
    lora_dir = root / "loras"
    for folder in (dit_dir, text_dir, vae_dir):
        if not folder.is_dir():
            missing.append(str(folder.relative_to(root)) + "/")

    dit_files = _component_files(dit_dir)
    text_files = _component_files(text_dir)
    vae_files = _component_files(vae_dir)
    token_dir = tokenizer_dir if tokenizer_dir is not None else (root / "tokenizer")
    for name in ("tokenizer.json", "preprocessor_config.json"):
        if not (token_dir / name).is_file():
            missing.append(str(token_dir / name) if tokenizer_dir is not None else f"tokenizer/{name}")

    dit_token = "ref2va" if dit_variant == "ref2va" else "fl2va"
    try:
        dit = _pick(dit_files, lambda name: dit_token in name, f"{dit_token} DiT", dit_dir)
        text_encoder = _pick(text_files, lambda name: "nvfp4" in name, "nvfp4 文本编码器", text_dir)
        video_vae = _pick(vae_files, lambda name: "video" in name, "视频 VAE", vae_dir)
        audio_vae = _pick(vae_files, lambda name: "audio" in name, "音频 VAE", vae_dir)
    except FileNotFoundError as exc:
        missing.append(str(exc))
        dit = text_encoder = video_vae = audio_vae = Path()

    lora: Path | None = None
    if turbo:
        if not lora_dir.is_dir():
            missing.append("loras/")
        else:
            lora_token = "ref2" if dit_variant == "ref2va" else "fl2"
            try:
                lora = _pick(
                    _component_files(lora_dir),
                    lambda name: "turbo" in name and lora_token in name,
                    "Turbo LoRA",
                    lora_dir,
                )
            except FileNotFoundError as exc:
                missing.append(str(exc))

    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"H3 模型目录缺少必需文件（{root}）:\n{detail}")

    return H3Checkpoints(
        root=root,
        dit=dit,
        text_encoder=text_encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
        lora=lora,
        tokenizer_dir=token_dir,
    )
