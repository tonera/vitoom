"""
调度器选择与加载
根据用户传入的 schedulerName 和模型信息为 pipe 绑定合适的 scheduler
"""
import math
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    TCDScheduler,
    UniPCMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    PixArtAlphaPipeline,
)

from common.logger import get_logger
from common.config_loader import load_inference_config
from schemas import InferenceRequestParams
from common.Constant import MODEL_SD3
from common.model_registry import MODEL_REGISTRY

logger = get_logger(__name__)


def _sdxl_single_file_config_kwargs() -> dict[str, str]:
    """Return local SDXL config paths for diffusers single-file loading."""
    inference_dir = Path(__file__).resolve().parents[2]
    config_dir = inference_dir / "config" / "sdxl"
    original_config = config_dir / "sd_xl_base.yaml"

    kwargs: dict[str, str] = {}
    if config_dir.is_dir():
        kwargs["config"] = str(config_dir)
    if original_config.is_file():
        kwargs["original_config"] = str(original_config)
    return kwargs


def _load_sdxl_single_file_for_prediction_test(
    pipeline_cls: Any,
    model_path: Path,
    *,
    local_files_only: bool | None,
    common_kwargs: dict[str, Any],
):
    single_file_kwargs = {**common_kwargs, **_sdxl_single_file_config_kwargs()}

    def _call(kwargs: dict[str, Any]):
        if local_files_only is None:
            return pipeline_cls.from_single_file(str(model_path), **kwargs)
        return pipeline_cls.from_single_file(str(model_path), local_files_only=local_files_only, **kwargs)

    try:
        return _call(single_file_kwargs)
    except TypeError as exc:
        # diffusers 历史版本有的使用 original_config_file 参数名。
        if "original_config" not in single_file_kwargs:
            raise
        legacy_kwargs = dict(single_file_kwargs)
        legacy_kwargs["original_config_file"] = legacy_kwargs.pop("original_config")
        try:
            return _call(legacy_kwargs)
        except TypeError:
            raise exc


def _is_v_prediction_model(params: InferenceRequestParams) -> bool:
    """
    仅对 SDXL 家族自动判定是否为 v_prediction（其余模型一律返回 False）。

    判定流程（从快到慢）：
    1) 文件名判定（vpred/v-pred/v_pred/v_prediction）
    2) 缓存判定（inference/cache/prediction_type/{sha1(key)}.json；key = {models_dir}+load_name）
    3) quick-latent-test（StableDiffusionXLPipeline：steps=4, 128x128, seed=0；失败直接 False 且不写缓存）
    4) 写缓存（仅当 3 成功得出结论）
    """
    mv = MODEL_REGISTRY.to_family(getattr(params, "family", None))
    if mv != "sdxl":
        return False

    # 0) 显式配置优先：允许通过 model_cfg.prediction_type 指定（schemas.py 也约定了该字段）
    # - 值为 "v_prediction" / "epsilon" 时直接返回，跳过 quick-latent-test（避免额外加载与小推理）
    try:
        cfg = getattr(params, "model_cfg", None)
        if isinstance(cfg, dict) and "prediction_type" in cfg:
            v = str(cfg.get("prediction_type") or "").strip().lower()
            if v in {"v_prediction", "v-prediction", "vpred", "v_pred", "vprediction"}:
                return True
            if v in {"epsilon", "eps"}:
                return False
    except Exception:
        pass

    load_name = str(getattr(params, "load_name", "") or "").strip()
    if not load_name:
        return False

    inference_config = load_inference_config()
    # models_dir 在配置里通常是 str，但下面既要参与 cache_key（字符串拼接、保持原语义）
    # 也要参与文件路径拼接（Path / "cache" / ...）。
    # 这里做一次性归一：models_dir_str 用于 key；models_dir_path 用于路径计算。
    models_dir_str = str(getattr(inference_config, "models_dir", "") or "")
    if not models_dir_str:
        # 无 models_dir 时无法定位缓存目录，也不应尝试触发 quick-latent-test
        return False
    models_dir_path = Path(models_dir_str)

    # 1) 文件名快速判定（支持 vpred/v-pred/v_pred/v_prediction）
    name_lower = load_name.lower()
    if re.search(r"(?:^|[^a-z0-9])v[\-_ ]?pred(?:iction)?(?:$|[^a-z0-9])", name_lower):
        return True
    if re.search(r"(?:^|[^a-z0-9])eps(?:ilon)?(?:$|[^a-z0-9])", name_lower):
        return False

    # key = {model_dir}+load_name（按你的要求，不做额外规范化）
    cache_key = f"{models_dir_str}|{load_name}"
    cache_dir = models_dir_path / "cache" / "prediction_type"
    cache_path = cache_dir / f"{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()}.json"

    # 2) 缓存命中直接返回
    try:
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if data.get("key") != cache_key:
                    # 文件存在但 key 不一致：常见于 models_dir 变更 / load_name 表达形式变更（相对/绝对路径）
                    logger.warning(
                        f"prediction_type cache key mismatch, ignoring cache file: path={str(cache_path)!r} "
                        f"stored_key={data.get('key')!r} current_key={cache_key!r}"
                    )
                else:
                    v = data.get("is_v_prediction")
                    if isinstance(v, bool):
                        logger.info(
                            f"prediction_type cache HIT: model={load_name!r} is_v_prediction={v} path={str(cache_path)!r}"
                        )
                        return v
    except Exception as e:
        logger.warning(f"prediction_type cache read failed (ignored): path={str(cache_path)!r} err={e}")

    # 3) UNet 单步 MSE 自一致判定（加载失败/测试失败 -> False 且不写缓存）
    try:
        logger.info(
            f"prediction_type cache MISS -> running unet-mse test (slow, one-time): model={load_name!r} "
            f"cache_path={str(cache_path)!r}"
        )
        # 解析模型路径：允许绝对路径；否则从 models_dir 拼接
        model_path = Path(load_name)
        if not model_path.is_absolute():
            model_path = models_dir_path / load_name

        # 延迟导入：避免非 SDXL 请求引入额外开销
        import torch
        from diffusers import StableDiffusionXLPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # 离线加载：不允许联网（缺组件直接报错）
        common_kwargs = {"torch_dtype": dtype, "use_safetensors": True}
        try:
            if model_path.is_dir():
                pipe = StableDiffusionXLPipeline.from_pretrained(str(model_path), local_files_only=True, **common_kwargs)
            else:
                pipe = _load_sdxl_single_file_for_prediction_test(
                    StableDiffusionXLPipeline,
                    model_path,
                    local_files_only=True,
                    common_kwargs=common_kwargs,
                )
        except TypeError:
            # 兼容旧 diffusers：可能不支持 local_files_only 参数；此时依赖部署环境本身无网
            if model_path.is_dir():
                pipe = StableDiffusionXLPipeline.from_pretrained(str(model_path), **common_kwargs)
            else:
                pipe = _load_sdxl_single_file_for_prediction_test(
                    StableDiffusionXLPipeline,
                    model_path,
                    local_files_only=None,
                    common_kwargs=common_kwargs,
                )
        pipe = pipe.to(device)

        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        # 核心：UNet 单步自一致（MSE）判定 prediction_type
        # - 构造 x0、eps，并合成 xt = add_noise(x0, eps, t)
        # - 只跑一次 UNet 前向得到 y
        # - 比较 MSE(y, eps_true) 与 MSE(y, v_true)；小者更像对应 prediction_type
        import inspect

        prompt = "a photo of a cat, high quality"
        height = 128
        width = 128
        trials = 3
        ratio_undetermined_th = 0.85

        train_sched = DDPMScheduler.from_config(pipe.scheduler.config)
        num_train = int(getattr(getattr(train_sched, "config", None), "num_train_timesteps", 1000) or 1000)
        timestep = min(500, max(0, num_train - 1))
        t = torch.tensor([timestep], device=device, dtype=torch.long)

        def _filter_kwargs(fn, kwargs: dict) -> dict:
            try:
                allowed = set(inspect.signature(fn).parameters.keys())
                return {k: v for k, v in kwargs.items() if k in allowed}
            except Exception:
                return kwargs

        if not hasattr(pipe, "encode_prompt"):
            raise RuntimeError("pipeline has no encode_prompt()")
        last_err = None
        enc = None
        for kw in (
            dict(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt="",
                negative_prompt_2="",
            ),
            dict(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt="",
            ),
            dict(prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False),
        ):
            try:
                enc = pipe.encode_prompt(**_filter_kwargs(pipe.encode_prompt, kw))  # type: ignore[attr-defined]
                break
            except Exception as e:
                last_err = e
                enc = None
        if enc is None or not isinstance(enc, tuple):
            raise RuntimeError(f"encode_prompt failed: {type(last_err).__name__}: {last_err}")
        if len(enc) == 2:
            prompt_embeds, pooled = enc
        elif len(enc) == 4:
            prompt_embeds, pooled = enc[0], enc[2]
        else:
            prompt_embeds, pooled = enc[0], None
            for cand in enc[1:]:
                if isinstance(cand, torch.Tensor) and getattr(cand, "ndim", 0) == 2:
                    pooled = cand
                    break
            if pooled is None:
                pooled = prompt_embeds.mean(dim=1)

        if not hasattr(pipe, "_get_add_time_ids"):
            raise RuntimeError("pipeline missing _get_add_time_ids (not SDXL?)")
        requires_aes = bool(
            getattr(getattr(pipe, "config", None), "requires_aesthetics_score", False)
            or getattr(pipe, "requires_aesthetics_score", False)
        )
        tid_kwargs = dict(
            original_size=(int(height), int(width)),
            crops_coords_top_left=(0, 0),
            target_size=(int(height), int(width)),
            dtype=prompt_embeds.dtype,
        )
        if requires_aes:
            tid_kwargs["aesthetic_score"] = 6.0
            tid_kwargs["negative_aesthetic_score"] = 2.5
        try:
            add_time_ids = pipe._get_add_time_ids(**_filter_kwargs(pipe._get_add_time_ids, tid_kwargs)).to(device)  # type: ignore[attr-defined]
        except Exception:
            vals = [int(height), int(width), 0, 0, int(height), int(width)]
            if requires_aes:
                vals += [6.0, 2.5]
            add_time_ids = torch.tensor([vals], device=device, dtype=prompt_embeds.dtype)

        h = int(height) // 8
        w = int(width) // 8
        added = {"text_embeds": pooled, "time_ids": add_time_ids}

        mses = []
        with torch.no_grad():
            for i in range(int(trials)):
                g = torch.Generator(device="cpu").manual_seed(1234 + i)
                x0 = torch.randn((1, 4, h, w), generator=g, device="cpu", dtype=torch.float32).to(
                    device=device, dtype=dtype
                )
                eps = torch.randn((1, 4, h, w), generator=g, device="cpu", dtype=torch.float32).to(
                    device=device, dtype=dtype
                )
                xt = train_sched.add_noise(x0, eps, t)
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        out = pipe.unet(
                            xt, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added, return_dict=True
                        )
                else:
                    out = pipe.unet(xt, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added, return_dict=True)

                if isinstance(out, torch.Tensor):
                    y = out.float()
                else:
                    sample = getattr(out, "sample", None)
                    y = (sample if sample is not None else out[0]).float()

                alpha_prod_t = train_sched.alphas_cumprod[timestep].to(device=device, dtype=torch.float32)
                alpha = alpha_prod_t.sqrt()
                sigma = (1.0 - alpha_prod_t).sqrt()
                v_true = (alpha * eps.float()) - (sigma * x0.float())

                mse_eps = float(((y - eps.float()) ** 2).mean().item())
                mse_v = float(((y - v_true) ** 2).mean().item())
                mses.append((mse_eps, mse_v))

        mse_eps_avg = sum(a for a, _ in mses) / len(mses)
        mse_v_avg = sum(b for _, b in mses) / len(mses)
        ratio = min(mse_eps_avg, mse_v_avg) / max(1e-12, max(mse_eps_avg, mse_v_avg))

        undetermined = bool(ratio > ratio_undetermined_th)
        if undetermined:
            is_v = False
            logger.warning(
                f"unet-mse undetermined (treat as epsilon, will cache): model={load_name!r} "
                f"mse_eps={mse_eps_avg:.6g} mse_v={mse_v_avg:.6g} ratio={ratio:.6g}"
            )
        else:
            is_v = bool(mse_v_avg < mse_eps_avg)
    except Exception as e:
        # 关键：这会导致“没有写缓存且每次都慢”；提升到 warning 方便排障
        logger.warning(f"unet-mse failed for v_prediction detect (no cache written): {e}")
        return False

    # 4) 写缓存（仅成功判定后）
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "key": cache_key,
                    "is_v_prediction": bool(is_v),
                    # 额外信息：不影响旧读取逻辑（仍只读 is_v_prediction）
                    "undetermined": bool(locals().get("undetermined", False)),
                    "method": "unet_mse",
                    "mse": {
                        "epsilon": float(locals().get("mse_eps_avg", 0.0)),
                        "v_prediction": float(locals().get("mse_v_avg", 0.0)),
                        "ratio": float(locals().get("ratio", 1.0)),
                        "timestep": int(locals().get("timestep", -1)),
                        "trials": int(locals().get("trials", 0)),
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            f"prediction_type cache WRITE: model={load_name!r} is_v_prediction={bool(is_v)} path={str(cache_path)!r}"
        )
    except Exception as e:
        # 关键：以前这里静默吞掉，导致“实际没落盘但看起来像有缓存逻辑”
        logger.warning(
            f"prediction_type cache write failed (ignored): dir={str(cache_dir)!r} path={str(cache_path)!r} err={e}"
        )
    return bool(is_v)


def load_scheduler_from_pipe(pipe: Any, params: InferenceRequestParams):
    """
    根据用户指定的 schedulerName 生成调度器并绑定到 pipe
    若未指定 schedulerName，则沿用默认 scheduler
    """
    item = params.schedulerName.strip() if params.schedulerName else None

    # 按你的要求：只有 SDXL 才允许根据用户选择更换 Scheduler；
    # 其它模型（flux/zimage/qwen/...）直接跳过用户的 schedulerName，保持默认 pipeline scheduler。
    try:
        mv = MODEL_REGISTRY.to_family(getattr(params, "family", None))
        if item and mv != "sdxl":
            logger.debug(
                f"non-sdxl model ignores schedulerName: family={getattr(params, 'family', None)!r} "
                f"family={mv!r} schedulerName={item!r}"
            )
            return pipe
    except Exception:
        # 出现异常时保持原行为（不阻断推理）
        pass

    scheduler_args = {}
    # 从模型文件键名判断是否为 v-prediction 模型
    if _is_v_prediction_model(params):
        logger.debug(f"V预测模型设定: {params.load_name} ...")
        scheduler_args = {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True}

    # 没有指定 schedulerName：默认沿用原 scheduler，但若是 Vpred 模型需要覆写 prediction_type
    if not item:
        if scheduler_args:
            try:
                base = getattr(pipe, "scheduler", None)
                if base is not None and hasattr(base, "config") and hasattr(base, "__class__"):
                    pipe.scheduler = base.__class__.from_config(base.config, **scheduler_args)
                    logger.debug("默认 scheduler 已按 Vpred 覆写 prediction_type=v_prediction")
            except Exception as exc:
                logger.warning(f"默认 scheduler 覆写 prediction_type 失败，保持原有 scheduler。reason={exc}")
        return pipe

    try:
        if isinstance(pipe, PixArtAlphaPipeline):
            scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif MODEL_REGISTRY.is_flowmatch_family(getattr(params, "family", None)) or (
            (params.family or "").lower() in {m.lower() for m in MODEL_SD3}
        ):
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif item == "DPM++ 2M":
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "DPM++ 2M Karras":
            scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, begin_index=1, **scheduler_args
            )
        elif item == "DPM++ 2M SDE":
            scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="sde-dpmsolver++", **scheduler_args
            )
        elif item == "DPM++ 2M SDE Karras":
            scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", **scheduler_args
            )
        elif item == "DPM++ SDE":
            scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "DPM++ SDE Karras":
            scheduler = DPMSolverSinglestepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, **scheduler_args
            )
        elif item == "DPM2":
            scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "DPM2 Karras":
            scheduler = KDPM2DiscreteScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, **scheduler_args
            )
        elif item == "DPM2 a":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "DPM2 a Karras":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, **scheduler_args
            )
        elif item == "Euler":
            scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "Euler a":
            scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "Heun":
            scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "LMS":
            scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "LMS Karras":
            scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, **scheduler_args)
        elif item == "DDPM":
            scheduler = DDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "DDIM":
            scheduler = DDIMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "PNDM":
            scheduler = PNDMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "TCD":
            scheduler = TCDScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "UniPC":
            scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        elif item == "qwen_scheduler_lightning":
            base_cfg = getattr(pipe, "scheduler", None)
            base_cfg = getattr(base_cfg, "config", {}) if base_cfg else {}
            # diffusers 的 scheduler.config 可能是 FrozenDict（不可变），不能对其原地 update；
            # 这里强制拷贝成普通 dict 后再合并自定义配置。
            try:
                scheduler_config = dict(base_cfg)
            except Exception:
                scheduler_config = {}

            scheduler_config.update(
                {
                    "base_image_seq_len": 256,
                    "base_shift": math.log(3),
                    "invert_sigmas": False,
                    "max_image_seq_len": 8192,
                    "max_shift": math.log(3),
                    "num_train_timesteps": 1000,
                    "shift": 1.0,
                    "shift_terminal": None,
                    "stochastic_sampling": False,
                    "time_shift_type": "exponential",
                    "use_beta_sigmas": False,
                    "use_dynamic_shifting": True,
                    "use_exponential_sigmas": False,
                    "use_karras_sigmas": False,
                }
            )
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        else:
            scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                **scheduler_args,
            )
            scheduler.config._class_name = "DDIMScheduler"

        pipe.scheduler = scheduler
        logger.debug(f"载入Scheduler: 指定scheduler = {item}")
    except Exception as exc:
        logger.warning(f"载入自定义 scheduler 失败，使用原有 scheduler。reason={exc}")

    return pipe

