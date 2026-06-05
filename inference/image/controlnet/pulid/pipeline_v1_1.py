import os
import shutil
from pathlib import Path
from typing import Optional, Union

import cv2
import insightface
import numpy as np
import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from image.controlnet.eva_clip import create_model_and_transforms
from image.controlnet.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .encoders_transformer import IDFormer
from .utils import img2tensor, tensor2img, is_torch2_available, sample_dpmpp_2m, sample_dpmpp_sde

if is_torch2_available():
    from .attention_processor import AttnProcessor2_0 as AttnProcessor
    from .attention_processor import IDAttnProcessor2_0 as IDAttnProcessor
else:
    from .attention_processor import AttnProcessor, IDAttnProcessor


class PuLIDPipeline:
    def __init__(
        self,
        sdxl_repo: str = "Lykon/dreamshaper-xl-lightning",
        sampler: str = "dpmpp_sde",
        *args,
        # Local-path mode (strategy A): prefer local files and avoid unexpected downloads.
        models_dir: Optional[Union[str, os.PathLike]] = None,
        weights_dir: Optional[Union[str, os.PathLike]] = None,
        pulid_path: Optional[Union[str, os.PathLike]] = None,
        eva_clip_path: Optional[Union[str, os.PathLike]] = None,
        antelope_root: Optional[Union[str, os.PathLike]] = None,
        allow_download: bool = True,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        onnx_provider: str = "gpu",
        facexlib_dirpath: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        super().__init__()
        # 可选：上层（如 IdHandler）判定 v-prediction 后透传进来，
        # 由这里统一合并到 scheduler.from_config（避免在 pipeline 内部读取权重文件做判定）。
        scheduler_args = kwargs.pop("scheduler_args", None)
        self.allow_download = bool(allow_download)
        self.models_dir = Path(models_dir).expanduser().resolve() if models_dir else None
        self.weights_dir = Path(weights_dir).expanduser().resolve() if weights_dir else None
        self.onnx_provider = str(onnx_provider or "gpu").lower().strip()
        self.facexlib_dirpath = Path(facexlib_dirpath).expanduser().resolve() if facexlib_dirpath else None

        # Device & dtype
        self.device = torch.device(device)
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        # load base model (directory/repo OR single-file .safetensors/.ckpt)
        repo = str(sdxl_repo or "").strip()
        repo_path = Path(repo).expanduser()

        def _resolve_project_path(rel: str) -> str:
            """
            Resolve a project-relative path robustly (works even if cwd isn't project root).
            """
            try:
                # .../inference/image/controlnet/pulid/pipeline_v1_1.py -> parents[3]=inference, parents[4]=project root
                project_root = Path(__file__).resolve().parents[4]
                p = (project_root / rel).resolve()
                return str(p)
            except Exception:
                return rel

        if repo_path.is_file() or repo.lower().endswith((".safetensors", ".ckpt")):
            # single-file: must use from_single_file
            original_config = _resolve_project_path("inference/config/sdxl/sd_xl_base.yaml")
            config_dir = _resolve_project_path("inference/config/sdxl")
            cache_dir = _resolve_project_path("resources/cache")
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            use_safetensors = repo_path.suffix.lower() == ".safetensors"

            # diffusers 不同版本参数名可能为 original_config 或 original_config_file；这里双写兼容
            try:
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    repo,
                    use_safetensors=use_safetensors,
                    local_files_only=not self.allow_download,
                    original_config=original_config,
                    config=config_dir,
                    cache_dir=cache_dir,
                    torch_dtype=self.torch_dtype,
                ).to(self.device)
            except TypeError:
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    repo,
                    use_safetensors=use_safetensors,
                    local_files_only=not self.allow_download,
                    original_config_file=original_config,  # type: ignore[call-arg]
                    config=config_dir,
                    cache_dir=cache_dir,
                    torch_dtype=self.torch_dtype,
                ).to(self.device)
        else:
            variant = "fp16" if self.torch_dtype == torch.float16 else None
            pipe_kwargs = {"torch_dtype": self.torch_dtype, "local_files_only": (not self.allow_download)}
            # diffusers 对 variant 的要求较“死板”：会认为需要存在形如 `*.fp16.safetensors` 的文件名。
            # 但现实里很多本地模型即便是 fp16 权重，文件名仍是 `diffusion_pytorch_model.safetensors`。
            # 因此：本地目录模式下先探测变体文件是否真实存在；不存在则不传 variant。
            def _has_variant_files(local_repo: str, v: str) -> bool:
                try:
                    rp = Path(str(local_repo)).expanduser()
                    if not rp.exists() or not rp.is_dir():
                        return True  # 非本地目录（或不可访问），交给 diffusers 处理
                    for f in rp.rglob("*"):
                        try:
                            if f.is_file() and f".{v}." in f.name:
                                return True
                        except Exception:
                            continue
                    return False
                except Exception:
                    return True

            if variant and _has_variant_files(repo, variant):
                pipe_kwargs["variant"] = variant

            try:
                self.pipe = StableDiffusionXLPipeline.from_pretrained(repo, **pipe_kwargs).to(self.device)
            except ValueError as e:
                # 兜底：即使预检测误判，也在 “variant 不存在” 的特定报错下自动重试不带 variant
                msg = str(e)
                if variant and ("variant=fp16" in msg or f"variant={variant}" in msg) and "no such modeling files" in msg:
                    try:
                        self.logger.warning(
                            f"SDXL from_pretrained failed with variant={variant} (no variant files). Retrying without variant."
                        )
                    except Exception:
                        pass
                    pipe_kwargs.pop("variant", None)
                    self.pipe = StableDiffusionXLPipeline.from_pretrained(repo, **pipe_kwargs).to(self.device)
                else:
                    raise
        self.pipe.watermark = None
        self.hack_unet_attn_layers(self.pipe.unet)

        # scheduler
        sa = scheduler_args if isinstance(scheduler_args, dict) else {}
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, **sa)

        # ID adapters
        self.id_adapter = IDFormer().to(self.device)

        # preprocessors
        # face align and parsing
        self._ensure_facexlib_weights()
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # clip-vit backbone
        eva_clip_path = self._resolve_eva_clip_path(eva_clip_path)
        model, _, _ = self._create_eva_model(eva_clip_path)
        model = model.visual
        self.clip_vision_model = model.to(self.device)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        # antelopev2 (prefer local files; only download if allow_download=True)
        root_dir = self._resolve_antelope_root(antelope_root)
        providers = ["CPUExecutionProvider"] if self.onnx_provider == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(name="antelopev2", root=str(root_dir), providers=providers)
        ctx_id = -1 if self.onnx_provider == "cpu" else 0
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model(
            str(Path(root_dir) / "models" / "antelopev2" / "glintr100.onnx"),
            providers=providers,
        )
        self.handler_ante.prepare(ctx_id=ctx_id)

        self.load_pretrain(pulid_path=pulid_path)

        # other configs
        self.debug_img_list = []

        # karras schedule related code, borrow from lllyasviel/Omost
        linear_start = 0.00085
        linear_end = 0.012
        timesteps = 1000
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, timesteps, dtype=torch.float64) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)

        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()
        self.sigma_data = 1.0

        if sampler == 'dpmpp_sde':
            self.sampler = sample_dpmpp_sde
        elif sampler == 'dpmpp_2m':
            self.sampler = sample_dpmpp_2m
        else:
            raise NotImplementedError(f'sampler {sampler} not implemented')

    def _ensure_facexlib_weights(self) -> None:
        """
        facexlib 默认会把权重下载到 site-packages/facexlib/weights/。
        你本地已有权重时，如果不在它默认目录，就会触发 GitHub 下载。

        这里策略是：
        - 优先在 {models_dir}/roop（或显式 facexlib_dirpath）下寻找已存在的权重文件
        - 找到后把它 link/copy 到 facexlib 期望的 weights 目录
        - allow_download=False 且找不到时，直接报错（避免偷偷联网/下载）
        """
        try:
            import facexlib  # type: ignore
        except Exception:
            return

        # facexlib 的默认权重目录：<site-packages>/facexlib/weights
        try:
            dst_dir = Path(getattr(facexlib, "__file__", "")).resolve().parent / "weights"
        except Exception:
            return
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # 这些是 facexlib 常见会自动下载的文件名（至少要覆盖你日志里的 detection_Resnet50_Final.pth）
        need = [
            "detection_Resnet50_Final.pth",
            # parsing(bisenet) 常见权重名（不同版本可能略有差异，先覆盖主流）
            "parsing_bisenet.pth",
            "parsing_parsenet.pth",
        ]

        # 优先搜索你项目里常见的 roop 目录
        candidates: list[Path] = []
        if self.facexlib_dirpath is not None:
            candidates.append(self.facexlib_dirpath)
        if self.models_dir is not None:
            candidates.append((self.models_dir / "roop").resolve())

        # 扩展一些常见子目录形态
        expanded: list[Path] = []
        for base in candidates:
            expanded.extend(
                [
                    base,
                    base / "weights",
                    base / "facexlib",
                    base / "facexlib" / "weights",
                    base / "models",
                    base / "models" / "facexlib",
                    base / "models" / "facexlib" / "weights",
                ]
            )
        candidates = expanded

        def _find_local(fname: str) -> Optional[Path]:
            for d in candidates:
                try:
                    p = (d / fname)
                    if p.exists():
                        return p
                except Exception:
                    continue
            return None

        for fname in need:
            dst = dst_dir / fname
            if dst.exists():
                continue
            src = _find_local(fname)
            if src is None:
                if not self.allow_download:
                    raise FileNotFoundError(
                        f"facexlib 权重缺失且禁止下载: {fname}. "
                        f"请将其放到 {self.models_dir}/roop/weights 或 {self.models_dir}/roop/facexlib/weights 等目录，"
                        f"或显式传 facexlib_dirpath。"
                    )
                # allow_download=True：留给 facexlib 自己下载
                continue
            try:
                # 优先软链接（不复制大文件）
                try:
                    os.symlink(str(src), str(dst))
                except Exception:
                    shutil.copy2(str(src), str(dst))
            except Exception:
                # best-effort：失败就让 facexlib 自己处理（可能下载）
                pass

    def release(self) -> None:
        """
        best-effort 释放显存/句柄：
        - 将 torch 模型搬回 CPU 并断开引用
        - 清理 insightface/onnxruntime 相关对象引用（显存是否完全归还取决于 ORT 内部 allocator）
        """
        try:
            self.debug_img_list = []
        except Exception:
            pass

        # 先尽量把 torch 模型搬回 CPU（减少 allocator 里“仍被引用”的显存）
        try:
            if getattr(self, "pipe", None) is not None:
                # accelerate/offload hooks 可能持有模块引用，先解除更稳
                try:
                    if hasattr(self.pipe, "maybe_free_model_hooks"):
                        self.pipe.maybe_free_model_hooks()
                except Exception:
                    pass
                try:
                    if hasattr(self.pipe, "_remove_all_hooks"):
                        self.pipe._remove_all_hooks()  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    self.pipe.to("cpu")
                except Exception:
                    pass
        except Exception:
            pass

        for attr in ("id_adapter", "clip_vision_model"):
            try:
                m = getattr(self, attr, None)
                if m is not None and hasattr(m, "to"):
                    try:
                        m.to("cpu")
                    except Exception:
                        pass
            except Exception:
                pass

        # facexlib 相关（FaceRestoreHelper 持有 detector/parser）
        try:
            fh = getattr(self, "face_helper", None)
            if fh is not None:
                for sub in ("face_det", "face_parse"):
                    try:
                        sm = getattr(fh, sub, None)
                        if sm is not None and hasattr(sm, "to"):
                            try:
                                sm.to("cpu")
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

        # 断开对重对象的引用
        for attr in (
            "pipe",
            "id_adapter",
            "clip_vision_model",
            "face_helper",
            "app",
            "handler_ante",
            "id_adapter_attn_layers",
        ):
            try:
                setattr(self, attr, None)
            except Exception:
                pass
        # 注意：真正的 gc/empty_cache 由上层 PipelineLifecycle.release_pipeline 统一执行。
        # 这里不再重复做，避免项目里“到处都是在释放显存”的割裂感。

    def _resolve_eva_clip_path(self, eva_clip_path: Optional[Union[str, os.PathLike]]) -> Optional[Path]:
        if eva_clip_path is not None:
            p = Path(eva_clip_path).expanduser().resolve()
            if p.exists():
                return p
            if not self.allow_download:
                raise FileNotFoundError(f"EVA-CLIP 权重不存在: {p}")
        if self.weights_dir is not None:
            p = (self.weights_dir / "EVA-CLIP" / "EVA02_CLIP_L_336_psz14_s6B.pt").resolve()
            if p.exists():
                return p
            if not self.allow_download:
                raise FileNotFoundError(
                    f"EVA-CLIP 权重不存在: {p}（请放置到 weights_dir/EVA-CLIP/ 下，或显式传 eva_clip_path）"
                )
        # allow_download=True 时，回退到 eva_clip 自己的默认下载/加载行为
        return None

    def _create_eva_model(self, eva_clip_path: Optional[Path]):
        """
        eva_clip 的 create_model_and_transforms 第二参数通常是 pretrained 标识；
        这里优先传入本地权重路径，避免触发下载。
        """
        if eva_clip_path is not None:
            try:
                return create_model_and_transforms("EVA02-CLIP-L-14-336", str(eva_clip_path), force_custom_clip=True)
            except TypeError:
                # 兼容不同版本 eva_clip 的签名
                try:
                    return create_model_and_transforms(
                        "EVA02-CLIP-L-14-336",
                        "eva_clip",
                        pretrained=str(eva_clip_path),
                        force_custom_clip=True,
                    )
                except Exception as e:
                    if not self.allow_download:
                        raise RuntimeError(f"使用本地 EVA-CLIP 权重加载失败: {eva_clip_path}: {e}") from e
        return create_model_and_transforms("EVA02-CLIP-L-14-336", "eva_clip", force_custom_clip=True)

    def _resolve_antelope_root(self, antelope_root: Optional[Union[str, os.PathLike]]) -> Path:
        """
        返回一个 root 目录，使得 `root/models/antelopev2` 存在（insightface FaceAnalysis 的约定）。
        """
        candidates: list[Path] = []
        if antelope_root is not None:
            candidates.append(Path(antelope_root).expanduser().resolve())
        if self.models_dir is not None:
            # 优先复用项目已有 roop 目录（历史上用于 insightface/facexlib）
            candidates.append((self.models_dir / "roop").resolve())
            candidates.append(self.models_dir.resolve())
        candidates.append(Path(".").resolve())

        for root in candidates:
            if (root / "models" / "antelopev2").exists():
                return root

        if self.allow_download:
            # 默认下载到 models_dir/roop/models/antelopev2（若有 models_dir），否则下载到 ./models/antelopev2
            root = (self.models_dir / "roop").resolve() if self.models_dir is not None else Path(".").resolve()
            local_dir = root / "models" / "antelopev2"
            local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download("DIAMONIK7777/antelopev2", local_dir=str(local_dir))
            return root

        raise FileNotFoundError(
            "未找到 antelopev2 资源目录（需要 root/models/antelopev2）。"
            "请把 antelopev2 放到 models_dir/roop/models/antelopev2 或指定 antelope_root。"
        )

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def get_sigmas_karras(self, n, rho=7.0):
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def hack_unet_attn_layers(self, unet):
        id_adapter_attn_procs = {}
        for name, _ in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is not None:
                id_adapter_attn_procs[name] = IDAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
            else:
                id_adapter_attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(id_adapter_attn_procs)
        self.id_adapter_attn_layers = nn.ModuleList(unet.attn_processors.values())

    def load_pretrain(self, *, pulid_path: Optional[Union[str, os.PathLike]] = None):
        ckpt_path: Optional[Path] = None
        if pulid_path is not None:
            p = Path(pulid_path).expanduser().resolve()
            if p.exists():
                ckpt_path = p
            elif not self.allow_download:
                raise FileNotFoundError(f"PuLID v1.1 权重不存在: {p}")

        if ckpt_path is None and self.weights_dir is not None:
            p = (self.weights_dir / "PuLID" / "pulid_v1.1.safetensors").resolve()
            if p.exists():
                ckpt_path = p
            elif not self.allow_download:
                raise FileNotFoundError(
                    f"PuLID v1.1 权重不存在: {p}（请放置到 weights_dir/PuLID/ 下，或显式传 pulid_path）"
                )

        if ckpt_path is None:
            if not self.allow_download:
                raise FileNotFoundError("PuLID v1.1 权重未提供且不允许下载（allow_download=False）")
            # 兼容原始行为：下载到 ./models/
            hf_hub_download("guozinan/PuLID", "pulid_v1.1.safetensors", local_dir="models")
            ckpt_path = Path("models/pulid_v1.1.safetensors").expanduser().resolve()

        state_dict = load_file(str(ckpt_path))
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    def get_id_embedding(self, image_list):
        """
        Args:
            image in image_list: numpy rgb image, range [0, 255]
        """
        id_cond_list = []
        id_vit_hidden_list = []
        for ii, image in enumerate(image_list):
            self.face_helper.clean_all()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # get antelopev2 embedding
            face_info = self.app.get(image_bgr)
            if len(face_info) > 0:
                face_info = sorted(
                    face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
                )[
                    -1
                ]  # only use the maximum face
                id_ante_embedding = face_info['embedding']
                self.debug_img_list.append(
                    image[
                        int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                        int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                    ]
                )
            else:
                id_ante_embedding = None

            # using facexlib to detect and align face
            self.face_helper.read_image(image_bgr)
            self.face_helper.get_face_landmarks_5(only_center_face=True)
            self.face_helper.align_warp_face()
            if len(self.face_helper.cropped_faces) == 0:
                raise RuntimeError('facexlib align face fail')
            align_face = self.face_helper.cropped_faces[0]
            # incase insightface didn't detect face
            if id_ante_embedding is None:
                print('fail to detect face using insightface, extract embedding on align face')
                id_ante_embedding = self.handler_ante.get_feat(align_face)

            id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device)
            if id_ante_embedding.ndim == 1:
                id_ante_embedding = id_ante_embedding.unsqueeze(0)

            # parsing
            input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
            input = input.to(self.device)
            parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[
                0
            ]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(input)
            # only keep the face features
            face_features_image = torch.where(bg, white_image, self.to_gray(input))
            self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

            # transform img before sending to eva-clip-vit
            face_features_image = resize(
                face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC
            )
            face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
            id_cond_vit, id_vit_hidden = self.clip_vision_model(
                face_features_image, return_all_features=False, return_hidden=True, shuffle=False
            )
            id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
            id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

            id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

            id_cond_list.append(id_cond)
            id_vit_hidden_list.append(id_vit_hidden)

        id_uncond = torch.zeros_like(id_cond_list[0])
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden_list[0])):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden_list[0][layer_idx]))

        id_cond = torch.stack(id_cond_list, dim=1)
        id_vit_hidden = id_vit_hidden_list[0]
        for i in range(1, len(image_list)):
            for j, x in enumerate(id_vit_hidden_list[i]):
                id_vit_hidden[j] = torch.cat([id_vit_hidden[j], x], dim=1)
        id_embedding = self.id_adapter(id_cond, id_vit_hidden)
        uncond_id_embedding = self.id_adapter(id_uncond, id_vit_hidden_uncond)

        # return id_embedding
        return uncond_id_embedding, id_embedding

    def __call__(self, x, sigma, **extra_args):
        x_ddim_space = x / (sigma[:, None, None, None] ** 2 + self.sigma_data**2) ** 0.5
        t = self.timestep(sigma)
        cfg_scale = extra_args['cfg_scale']
        eps_positive = self.pipe.unet(x_ddim_space, t, return_dict=False, **extra_args['positive'])[0]
        eps_negative = self.pipe.unet(x_ddim_space, t, return_dict=False, **extra_args['negative'])[0]
        noise_pred = eps_negative + cfg_scale * (eps_positive - eps_negative)
        return x - noise_pred * sigma[:, None, None, None]

    def inference(
        self,
        prompt,
        size,
        prompt_n='',
        id_embedding=None,
        uncond_id_embedding=None,
        id_scale=1.0,
        guidance_scale=1.2,
        steps=4,
        seed=-1,
    ):

        # sigmas
        sigmas = self.get_sigmas_karras(steps).to(self.device)

        # latents
        noise = torch.randn((size[0], 4, size[1] // 8, size[2] // 8), device="cpu", generator=torch.manual_seed(seed))
        noise = noise.to(dtype=self.pipe.unet.dtype, device=self.device)
        latents = noise * sigmas[0].to(noise)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=prompt_n,
        )

        add_time_ids = list((size[1], size[2]) + (0, 0) + (size[1], size[2]))
        add_time_ids = torch.tensor([add_time_ids], dtype=self.pipe.unet.dtype, device=self.device)
        add_neg_time_ids = add_time_ids.clone()

        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                cross_attention_kwargs={'id_embedding': id_embedding, 'id_scale': id_scale},
            ),
            negative=dict(
                encoder_hidden_states=negative_prompt_embeds,
                added_cond_kwargs={"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_neg_time_ids},
                cross_attention_kwargs={'id_embedding': uncond_id_embedding, 'id_scale': id_scale},
            ),
        )

        latents = self.sampler(self, latents, sigmas, extra_args=sampler_kwargs, disable=False)
        latents = latents.to(dtype=self.pipe.vae.dtype, device=self.device) / self.pipe.vae.config.scaling_factor
        images = self.pipe.vae.decode(latents).sample
        images = self.pipe.image_processor.postprocess(images, output_type='pil')

        return images
