"""
推理器侧通用常量（download 等多个模块共享）
"""

# 文件共享/复用：仅索引与复用大文件（默认 50MB）
DOWNLOAD_SHARE_MIN_SIZE_BYTES = 50 * 1024 * 1024

# 允许参与共享/复用的文件后缀
DOWNLOAD_SHARE_EXTS = (".safetensors", ".pth", ".pt", ".ckpt")

MODEL_15 = ['sd15', 'SD15']
MODEL_SDXL = ['sdxl', 'SDXL 1.0', 'Illustrious', 'NoobAI', 'Pony']
MODEL_SD3 = ['sd3', 'presd3', 'SDXL 3.0']
MODEL_FLUX = ['flux', 'flux-d', 'flux-s', 'Flux.1 D', 'Flux.1 S']
MODEL_QWEN = ['qwen', 'Qwen']
MODEL_QWEN_EDIT = ['qwen.edit', 'qwen-edit', 'Qwen-edit']
MODEL_FLUX2 = ['flux2', 'Flux.2 D']
MODEL_FLUX2_KLEIN = ['flux2_klein', 'Flux.2 Klein']
MODEL_Z_IMAGE = ['zimage', 'ZImageTurbo','z-image','z-image-turbo']
MODEL_FLUX_KONTEXT = ['flux_kontext', 'Flux.1 Kontext']
MODEL_CHROMA = ['chroma', 'Chroma']
MODEL_WAN = ['wan', 'Wan', 'Wan-edit', 'Wan-painting', 'Wan-sketch']
MODEL_WAN_VIDEO = ['wan_video', 'Wan-video']

# Anima（非 diffusers 目录格式；由 third_party/anima_runtime 提供推理）
MODEL_ANIMA = ["anima", "Anima"]

# 补齐历史模型族（新设计下也按“可兼容多值”的列表方式维护）
MODEL_AURA = ["aura"]
MODEL_HUNYUAN = ["hunyuan"]

# ===== Model component capability config =====
# 用于 pipeline_component_injector：定义各模型家族在自动装配/补件阶段支持的组件槽位。
# 注意：该表不再表示“允许通过 model_config 传哪些 key”，而是后端内部能力表。
FAMILY_ALLOWED_KEYS: dict[str, set[str]] = {
    # sd15
    "sd15": {"text_encoder", "unet", "vae"},
    # sdxl
    "sdxl": {"text_encoder", "text_encoder_2", "unet", "nunchaku_unet", "vae"},
    # flux
    "flux": {"text_encoder", "text_encoder_2", "nunchaku_text_encoder_2", "transformer", "nunchaku_transformer", "vae"},
    "flux_kontext": {"text_encoder", "text_encoder_2", "nunchaku_text_encoder_2", "transformer", "nunchaku_transformer", "vae"},
    # flux2
    "flux2": {"text_encoder", "nunchaku_text_encoder", "transformer", "nunchaku_transformer", "vae"},
    "flux2_klein": {"text_encoder", "nunchaku_text_encoder", "transformer", "nunchaku_transformer", "vae"},
    # qwen
    "qwen": {"text_encoder", "nunchaku_text_encoder", "transformer", "nunchaku_transformer", "vae"},
    "qwen.edit": {"text_encoder", "nunchaku_text_encoder", "transformer", "nunchaku_transformer", "vae"},
    # zimage / chroma
    "zimage": {"text_encoder", "transformer", "nunchaku_transformer", "vae"},
    "chroma": {"text_encoder", "nunchaku_text_encoder", "transformer", "nunchaku_transformer", "vae"},
}

# ===== from_single_file assembly policy =====
# 说明：
# - 该表用于 from_single_file 场景的“运行时必选组件补齐”（与 FAMILY_ALLOWED_KEYS 完全不同职责）
# - key 使用 canonical family（to_model_family 输出）
# - base_path：相对 {models_dir} 的基模目录（用于兜底缺失组件）
# - main_component：单文件主权重对应的运行时组件槽位（通常总是存在于 ckpt；不应从 base 覆盖）
# - required_components：运行时管道必须具备的组件槽位（均为标准槽位名，不包含 nunchaku_*）
# - presence_prefixes：用于扫描 safetensors key 判断“单文件是否包含该组件”（保守策略：只作为是否需要 base 兜底的依据）
FAMILY_SINGLEFILE_POLICY: dict[str, dict[str, object]] = {
    "sd15": {
        "base_path": "stable-diffusion-v1-5",
        "main_component": "unet",
        "required_components": {"text_encoder", "vae", "unet"},
        "presence_prefixes": {
            "text_encoder": ("text_encoder.", "cond_stage_model."),
            "vae": ("vae.", "first_stage_model."),
        },
    },
    "sdxl": {
        "base_path": "stable-diffusion-xl-base-1.0",
        "main_component": "unet",
        "required_components": {"text_encoder", "text_encoder_2", "vae", "unet"},
        "presence_prefixes": {
            # 常见单文件（A1111/Comfy）权重：conditioner/cond_stage_model 风格；diffusers 风格仍可能是 text_encoder.*
            "text_encoder": ("text_encoder.", "cond_stage_model.", "conditioner.embedders.0"),
            "text_encoder_2": ("text_encoder_2.", "cond_stage_model_2.", "conditioner.embedders.1"),
            "vae": ("vae.", "first_stage_model."),
        },
    },
    "flux": {
        "base_path": "FLUX.1-dev",
        "main_component": "transformer",
        "required_components": {"text_encoder", "text_encoder_2", "vae", "transformer"},
        "presence_prefixes": {
            # 常见 flux 单文件：text_encoders.clip_l / text_encoders.t5xxl / vae / model.diffusion_model.*
            "text_encoder": ("text_encoder.", "text_encoders.clip_l."),
            "text_encoder_2": ("text_encoder_2.", "text_encoders.t5xxl."),
            "vae": ("vae.",),
        },
    },
    "flux_kontext": {
        "base_path": "FLUX.1-Kontext-dev",
        "main_component": "transformer",
        "required_components": {"text_encoder", "text_encoder_2", "vae", "transformer"},
        "presence_prefixes": {
            "text_encoder": ("text_encoder.", "text_encoders.clip_l."),
            "text_encoder_2": ("text_encoder_2.", "text_encoders.t5xxl."),
            "vae": ("vae.",),
        },
    },
    "qwen": {
        "base_path": "Qwen-Image",
        "main_component": "transformer",
        "required_components": {"text_encoder", "vae", "transformer"},
        "presence_prefixes": {
            # 常见 qwen 单文件仅包含 model.diffusion_model.transformer_blocks（主权重）
            "text_encoder": ("text_encoder.", "text_encoders."),
            "vae": ("vae.",),
        },
    },
    "qwen.edit": {
        "base_path": "Qwen-Image-Edit",
        "main_component": "transformer",
        "required_components": {"text_encoder", "vae", "transformer"},
        "presence_prefixes": {
            "text_encoder": ("text_encoder.", "text_encoders."),
            "vae": ("vae.",),
        },
    },
    "zimage": {
        "base_path": "Z-Image-Turbo",
        "main_component": "transformer",
        "required_components": {"text_encoder", "vae", "transformer"},
        # zimage 单文件的 TE/VAE 可能是非标准 key；presence_prefixes 仅用于兜底决策，
        # 实际从单文件加载需要 converter（见 common/single_file_component_converter.py）
        "presence_prefixes": {
            # 常见 zimage 单文件：text_encoders.qwen3_4b / vae / model.diffusion_model.*
            "text_encoder": ("text_encoder.", "text_encoders.qwen3_4b.", "text_encoders."),
            "vae": ("vae.", "first_stage_model."),
        },
        "converter": "zimage",
    },
    "chroma": {
        "base_path": "Chroma",
        "main_component": "transformer",
        "required_components": {"text_encoder", "vae", "transformer"},
        "presence_prefixes": {
            "text_encoder": ("text_encoder.",),
            "vae": ("vae.",),
        },
    },
}

# 暂时未分类
# Framepack
# Gemini
# Imagen
# Omnihuman
# Seedance
# seededit
# Seedream
# Voxcpm


JT_ED = 'ED'
# 身份保持（PuLID / ID）
JT_ID = "ID"
# 姿态控制（ControlNet / POSE）
JT_POSE = "POSE"
# 单图编辑，用于批处理，处理时不按image_num循环，而是按tpl_list循环
JT_SED = 'SED'
JT_SR = 'SR'
JT_MK = 'MK'
JT_RBG  = "RBG"
JT_FS = "FS"

# ===== Mini service job types =====
# Mini 服务用于承载轻量、多种类、按需加载的小模型工具集。
# 所有 mini 任务共享同一服务进程（LRU=1 + TTL 驱逐策略），按 **family** 分发到对应 handler
# （详见 inference/mini/inferrer.py::_HANDLER_REGISTRY）。
#
# 下面的 JT_* 常量只是"任务语义标签"，用来在日志 / 统计 / 未来业务分组里归类，
# 不参与 handler 分发选择。新增小模型无需新增这里的常量。
JT_OCR = "OCR"

# ===== Translate service job types =====
JT_TRANSLATE = "TRANSLATE"

# ===== Video job types (Wan2 专用推理器的规划常量；后续可扩展更多家族/类型) =====
# 视频生成
JT_MKV = 'MKV'
# 语音生成视频
JT_S2V = 'S2V'
# 镜头控制
JT_CCV = 'CCV'
# 首尾帧视频补全
JT_INP = 'INP'

# ===== Video output settings =====
# diffsynth.utils.data.save_video 的 quality 参数（推荐 0~10-ish，越大质量越好）
VIDEO_SAVE_QUALITY = 9

# 视频推理显存阀值：当探测到 CUDA 可用显存（free VRAM，单位 GiB）低于该值时，
# 强制启用低显存模式（cpu offload / low_vram / force_offload），以降低峰值显存占用。
VIDEO_FORCE_OFFLOAD_FREE_VRAM_GIB = 40.0


KEEY_SIZE_USER = 'user'
KEEY_SIZE_INIT = 'init_images'
KEEY_SIZE_FILE2 = 'image_file2'
