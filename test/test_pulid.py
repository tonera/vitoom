import argparse
from os import PathLike
from pathlib import Path
from types import MethodType
from typing import Any, Optional, Union

import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image

from nunchaku.models.pulid.pulid_forward import pulid_forward
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline, PuLIDPipeline
from nunchaku.utils import get_precision


class PuLIDFluxPipelineWithPaths(PuLIDFluxPipeline):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | PathLike, *args: Any, **kwargs: Any):
        pulid_device = kwargs.pop("pulid_device", "cuda")
        onnx_provider = kwargs.pop("onnx_provider", "gpu")

        torch_dtype = kwargs.get("torch_dtype", None)
        weight_dtype = kwargs.pop("weight_dtype", torch_dtype if torch_dtype is not None else torch.bfloat16)

        pulid_path = kwargs.pop("pulid_path", "guozinan/PuLID/pulid_flux_v0.9.1.safetensors")
        eva_clip_path = kwargs.pop("eva_clip_path", "QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt")
        insightface_dirpath = kwargs.pop("insightface_dirpath", None)
        facexlib_dirpath = kwargs.pop("facexlib_dirpath", None)

        base = FluxPipeline.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        pipe = cls(
            scheduler=base.scheduler,
            vae=base.vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            text_encoder_2=getattr(base, "text_encoder_2", None),
            tokenizer_2=getattr(base, "tokenizer_2", None),
            transformer=base.transformer,
            image_encoder=getattr(base, "image_encoder", None),
            feature_extractor=getattr(base, "feature_extractor", None),
            pulid_device=pulid_device,
            weight_dtype=weight_dtype,
            onnx_provider=onnx_provider,
            pulid_path=pulid_path,
            eva_clip_path=eva_clip_path,
            insightface_dirpath=insightface_dirpath,
            facexlib_dirpath=facexlib_dirpath,
        )

        del base
        return pipe

    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
        image_encoder=None,
        feature_extractor=None,
        pulid_device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
        onnx_provider: str = "gpu",
        pulid_path: Union[str, PathLike] = "guozinan/PuLID/pulid_flux_v0.9.1.safetensors",
        eva_clip_path: Union[str, PathLike] = "QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt",
        insightface_dirpath: Optional[Union[str, PathLike]] = None,
        facexlib_dirpath: Optional[Union[str, PathLike]] = None,
    ):
        FluxPipeline.__init__(
            self,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

        self.pulid_device = torch.device(pulid_device)
        self.weight_dtype = weight_dtype
        self.onnx_provider = onnx_provider

        self.pulid_model = PuLIDPipeline(
            dit=self.transformer,
            device=self.pulid_device,
            weight_dtype=self.weight_dtype,
            onnx_provider=self.onnx_provider,
            pulid_path=pulid_path,
            eva_clip_path=eva_clip_path,
            insightface_dirpath=insightface_dirpath,
            facexlib_dirpath=facexlib_dirpath,
        )


def build_args():
    parser = argparse.ArgumentParser(description="PuLID standalone script")
    parser.add_argument("--id-image", required=True, help="ID参考图路径或URL")
    parser.add_argument("--prompt", default="A woman holding a sign that says 'SVDQuant is fast!'")
    parser.add_argument("--output", default="/home/tonera/website/output/flux.1-dev-pulid.png")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--id-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights-dir", default="/home/tonera/project/aiservice/diffusers/weights", help="本地权重目录")
    parser.add_argument("--models-dir", default="/home/tonera/project/aiservice/diffusers/models", help="本地模型目录")
    parser.add_argument("--pulid-path", default=None, help="PuLID权重路径（可选）")
    parser.add_argument("--eva-clip-path", default=None, help="EVA-CLIP权重路径（可选）")
    parser.add_argument("--insightface-dir", default=None, help="insightface目录（可选）")
    parser.add_argument("--facexlib-dir", default=None, help="facexlib目录（可选）")
    parser.add_argument(
        "--transformer-path",
        default=None,
        help="nunchaku transformer路径（为空时根据weights-dir或HF默认）",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="FLUX.1-dev模型路径或HF仓库名（为空时根据models-dir或HF默认）",
    )
    return parser.parse_args()


def main():
    args = build_args()
    device = args.device
    precision = get_precision()
    weights_dir = Path(args.weights_dir).expanduser().resolve() if args.weights_dir else None
    models_dir = Path(args.models_dir).expanduser().resolve() if args.models_dir else None

    transformer_path = args.transformer_path
    if transformer_path is None:
        if weights_dir is not None:
            transformer_path = str(
                weights_dir / f"nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
            )
        else:
            transformer_path = f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"

    model_path = args.model_path
    if model_path is None:
        if models_dir is not None:
            model_path = str(models_dir / "FLUX.1-dev")
        else:
            model_path = "black-forest-labs/FLUX.1-dev"

    pulid_path = args.pulid_path
    if pulid_path is None:
        pulid_path = str(weights_dir / "PuLID/pulid_flux_v0.9.1.safetensors") if weights_dir else None

    eva_clip_path = args.eva_clip_path
    if eva_clip_path is None:
        eva_clip_path = str(weights_dir / "EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt") if weights_dir else None

    insightface_dir = args.insightface_dir
    if insightface_dir is None and models_dir is not None:
        insightface_dir = str(models_dir / "roop")

    facexlib_dir = args.facexlib_dir
    if facexlib_dir is None and models_dir is not None:
        facexlib_dir = str(models_dir / "roop")

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(transformer_path)
    # transformer.update_lora_params(
    #     "/home/tonera/project/aiservice/diffusers/loras/FLUX-dev-lora-children-simple-sketch.safetensors"
    # ) 
    # transformer.set_lora_strength(1) 
    from nunchaku.lora.flux.compose import compose_lora
    composed_lora = compose_lora(
        [
            ("/home/tonera/project/aiservice/diffusers/loras/FLUX-dev-lora-children-simple-sketch.safetensors", 1),
            ("/home/tonera/project/aiservice/diffusers/loras/FluxThouS40k.safetensors", 1),
        ]
    ) 
    transformer.update_lora_params(composed_lora)

    torch_dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    pipeline = PuLIDFluxPipelineWithPaths.from_pretrained(
        model_path,
        transformer=transformer,
        torch_dtype=torch_dtype,
        pulid_path=pulid_path or "guozinan/PuLID/pulid_flux_v0.9.1.safetensors",
        eva_clip_path=eva_clip_path or "QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt",
        insightface_dirpath=insightface_dir,
        facexlib_dirpath=facexlib_dir,
    ).to(device)

    pipeline.transformer.forward = MethodType(pulid_forward, pipeline.transformer)

    

    id_image = load_image(args.id_image)

    generator = None
    if args.seed and args.seed > 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    image = pipeline(
        args.prompt,
        id_image=id_image,
        id_weight=args.id_weight,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"[OK] saved: {output_path}")


if __name__ == "__main__":
    main()

