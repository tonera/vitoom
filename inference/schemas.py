"""
推理请求参数定义
基于ReqMessage类的参数结构
"""
from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices
from pydantic.config import ConfigDict
from typing import Optional, List, Literal, Union, Any, Dict
import json


class InferenceRequestParams(BaseModel):
    """推理请求参数（基于ReqMessage类）"""

    # Pydantic v2：模型配置必须使用 model_config（类配置），因此字段名不能叫 model_config
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # ========== 1. 消息参数 ==========
    type: str = Field(..., description="任务大类:image/video/audio/text/translate/mini/upload")
    job_type: str = Field(..., description="任务执行分类:MK,RBG...")
    storage: Literal["local", "oss", "s3", "server"] = Field(
        default="local", 
        description="存储方式"
    )
    reference_id: str = Field(default="", description="参考ID")
    index: Optional[int] = Field(default=None, description="当前任务子序号")
    total: Optional[int] = Field(default=None, description="任务总数")
    
    # ========== 2. 任务参数 ==========
    id: str = Field(..., description="消息标识")
    big: Optional[str] = Field(default=None, description="大图路径")
    thumb: Optional[str] = Field(default=None, description="缩略图路径")
    user_id: str = Field(..., description="用户ID")
    task_id: str = Field(..., description="任务ID")
    res_channel: str = Field(default="", description="结果返回的消息队列channel")
    
    # ========== 3. 生成图片参数 ==========
    prompt: str = Field(..., description="提示词")
    negative_prompt: str = Field(default="", description="负面提示词")
    width: int = Field(default=1024, ge=64, le=4096, description="图片宽度")
    height: int = Field(default=1024, ge=64, le=4096, description="图片高度")
    # guidance_scale 允许为 0（例如 Turbo/LCM 等模型）
    guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0, description="引导比例")
    # 允许前端传 null（None）；缺省仍为 0（保持历史兼容）
    # 约定（与 video/image 侧统一）：seed is None 或 seed < 0 表示随机；seed >= 0 则按其值使用（包括 0）
    seed: Optional[int] = Field(default=0, description="随机种子；传 null 或负数表示随机")
    # 说明：
    # - 对扩散类任务，num_inference_steps 需要 >= 1
    # - 对非扩散后处理类任务（如 RBG/SR/FS），该字段可能为 0（上游兼容字段），不应导致任务直接失败
    num_inference_steps: int = Field(default=30, ge=0, le=50, description="推理步数（扩散类任务需>=1）")
    # strength 在文生图场景可不传；图生图/编辑场景使用时仍会校验范围
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="去噪强度")
    # 注意：该字段同时用于 image/video/audio 三类输出的扩展名选择。
    # - image: jpeg/png/webp
    # - video: mp4/webm/mov/avi
    # - audio: mp3/wav/ogg/flac
    # - text: txt
    # 默认值会在 from_task_dict 中按 task.type 自动选择（image->jpeg, video->mp4, audio->mp3）
    file_type: Literal[
        "jpeg", "png", "webp",
        "mp4", "webm", "mov", "avi",
        "mp3", "wav", "ogg", "flac",
        "txt",
        # mini 服务（OCR 等小模型工具集）输出的文本/结构化结果
        "md", "json",
        # mini/OCR 的图文混排打包产物（document.md + images/）
        "zip",
    ] = Field(default="jpeg", description="返回的文件类型（image/video/audio/mini 通用）")
    url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("url", "init_images"),
        description="图生图所需的图片URL",
    )
    generate_num: int = Field(default=1, ge=1, le=10, description="生成图片数量")
    load_name: Optional[str] = Field(default=None, description="模型加载名")
    family: Optional[str] = Field(default=None, description="模型家族稳定 key")
    # 注意：family 是推理侧内部 canonical family：
    # 统一在 ImageInferrer.inference_callback 入口完成决策与归一化；
    # 若未提供则触发 PipelineDetector 自动侦测，侦测失败则任务失败。
    family: Optional[str] = Field(
        default=None,
        description="模型分类（例如 sdxl/sd15/flux/qwen 等）。若未提供则由推理器自动侦测；侦测失败任务将失败。",
    )
    third_transformer_path:Optional[str] = Field(default=None, description="第三方transformer路径")
    schedulerName: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("schedulerName", "sampler_name"),
        description="调度器名称（兼容客户端传 sampler_name）",
    )
    keep_size: Literal["user", "init_images", "image_file2", "ref_image"] = Field(
        default="user",
        description="编辑图片时尺寸保持方式"
    )
    remove_bg: bool = Field(default=False, description="是否去背景")
    low_vram: bool = Field(
        default=False,
        validation_alias=AliasChoices("low_vram", "low_vram"),
        description="是否启用低显存模式（兼容 low_vram）",
    )
    fast_mode: bool = Field(default=True, description="快速模式")
    
    # ========== 4. 人脸增强超分参数 ==========
    upscale: int = Field(default=0, description="超分倍数：仅 2/4 生效；0/1 表示不超分")
    face_enhance: bool = Field(default=False, description="是否人脸增强")
    arch: str = Field(default="clean", description="架构")
    
    # ========== 5. 视频参数 ==========
    duration: int = Field(default=5, ge=1, le=60, description="视频时长（秒）")
    resolution: Optional[str] = Field(default=None, description="视频分辨率")
    aspect_ratio: Optional[str] = Field(default=None, description="视频宽高比（例如 16:9 / 9:16）")
    # ref_video：视频相关通用参考/控制输入
    # - MKV: VICV/IVV2V 分支中作为 control/pose video
    # - S2V: 复用为 pose_video（等价示例中的 s2v_pose_video）
    ref_video: Optional[str] = Field(default=None, description="参考/控制视频URL（video任务通用）")
    # face_video：IVV2V 分支中作为 animate_face_video
    face_video: Optional[str] = Field(default=None, description="人脸/表情驱动视频URL（IVV2V）")
    # camera control：CCV 分支参数
    direction: Optional[str] = Field(default=None, description="镜头控制方向（例如 Left/Right/Up/Down）")
    speed: Optional[float] = Field(default=None, description="镜头控制速度（例如 0.01）")
    
    # ========== 6. 语音生成参数 ==========
    audio_mode: Literal["tts", "asr", "realtime_tts"] = Field(
        default="tts",
        description="音频任务模式：tts/asr/realtime_tts"
    )
    input_audio_url: Optional[str] = Field(default=None, description="ASR 输入音频 URL 或本地路径")
    prompt_wav_path: Optional[str] = Field(default=None, description="提示语音路径url")
    prompt_text: Optional[str] = Field(default=None, description="参考文本")
    instruct: Optional[str] = Field(default=None, description="Qwen-TTS 等模型的风格/情感控制指令")
    voice_preset: Optional[str] = Field(default=None, description="预置音色名称或缓存提示名称")
    speaker_name: Optional[str] = Field(default=None, description="说话人/音色名称")

    # ---- Qwen3-TTS 多模式（仅 audio_mode=tts 时生效）----
    # custom_voice        : 使用 CustomVoice 权重 + 9 个预置 speaker（可叠 instruct）
    # voice_design        : 使用 VoiceDesign 权重 + instruct 描述声线（无 speaker）
    # voice_clone         : 使用 Base 权重 + 用户录音(ref_audio) + 参考文本(ref_text)
    # voice_design_then_clone : 两步（VoiceDesign 合成参考音 -> Base 克隆）
    tts_mode: Literal[
        "custom_voice", "voice_design", "voice_clone", "voice_design_then_clone"
    ] = Field(
        default="custom_voice",
        description="Qwen3-TTS 的合成子模式；仅 audio_mode=tts 时生效",
    )
    ref_audio: Optional[str] = Field(
        default=None,
        description="voice_clone 的参考音频；接受 URL 或本地路径。"
        "voice_design_then_clone 模式中该字段不使用（参考音由 VoiceDesign 现场合成）。",
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="voice_clone 的参考文本，要求与 ref_audio 内容一致。"
        "x_vector_only=True 时可省略，但克隆质量会下降。",
    )
    clone_base_load_name: Optional[str] = Field(
        default=None,
        description="voice_design_then_clone 第 2 阶段使用的 Base 克隆权重名称；"
        "后端工具层自动填充，前端一般无需传入。"
        "注意：此模式下 load_name 指向 VoiceDesign 权重（第 1 阶段）。",
    )
    design_seed_text: Optional[str] = Field(
        default=None,
        description="voice_design_then_clone 第 1 阶段让 VoiceDesign 念的那句种子文本；"
        "第 2 阶段会把这句话作为 Base 克隆的 ref_text。",
    )
    design_instruct: Optional[str] = Field(
        default=None,
        description="voice_design_then_clone 第 1 阶段的 VoiceDesign 声线指令；"
        "若未提供则回退到 instruct。",
    )
    x_vector_only: bool = Field(
        default=False,
        description="voice_clone 可选：仅使用 speaker embedding，允许省略 ref_text；质量略降。",
    )
    response_format: Literal["audio_file", "text_file", "both"] = Field(
        default="audio_file",
        description="音频任务最终产物类型"
    )
    stream: bool = Field(default=False, description="是否开启流式返回")
    sample_rate: Optional[int] = Field(default=None, ge=8000, le=96000, description="采样率")
    language: Optional[str] = Field(default=None, description="语言代码")
    timestamps: bool = Field(default=False, description="是否要求时间戳")
    speaker_diarization: bool = Field(default=False, description="是否要求说话人分离")
    drama: Optional[Dict[str, Any]] = Field(
        default=None,
        description="多角色广播剧 TTS 扩展：{characters:[], dialogues:[]}",
    )
    
    # ========== 7. ControlNet 参数 ==========
    image_file2: str = Field(default="", description="控制图url")
    edit_act: str = Field(default="", description="编辑动作（如'canny'）")
    
    # ========== 8. 多参考图图片url列表 ==========
    tpl_list: List[str] = Field(default_factory=list, description="模板列表")

    # ========== 9. LoRA 参数 ==========
    # 兼容两种来源：
    # 1) prompt 内 <lora:xxx:0.8>
    # 2) request_params.loras：推荐为 JSON 字符串 '[{"name":"xxx","weight":0.8}]'（也兼容 list/dict）
    loras: Optional[Any] = Field(default=None, description="LoRA 参数 JSON 字符串（例如：'[{\"name\":\"xxx\",\"weight\":0.8}]'）")

    # ========== 9.1 运行时预处理缓存（内部字段） ==========
    # 说明：
    # - prompt sanitize 会移除 <lora:...> 标签，因此必须在 sanitize 之前解析并缓存 LoRA 列表
    # - 该字段仅供后端运行时内部使用，不要求前端传入
    parsed_loras: Optional[Any] = Field(default=None, description="（内部）预解析的 LoRA 列表，供加载 LoRA 使用")
    # ========== 9. 模型推理配置 ==========
    runtime_config: Optional[dict] = Field(
        default=None,
        description="模型运行配置",
    )

    # ========== 10. Mini / Translate 服务扩展参数 ==========
    # extract 是 mini（OCR 等）与 translate（语言对等）的通用扩展参数槽位。
    # 设计动机：避免为每个小模型往主 schema 里加字段；所有专属参数统一塞进 extract。
    #
    # 当前约定（OCR）：
    #   extract.task   ∈ {"text", "table", "formula", "extract"}，默认 "text"
    #   extract.schema ：仅当 task=="extract" 时必填，严格 JSON 对象（用于信息抽取）
    #
    # 未来扩展（示例）：
    #   rerank:  extract = {"query": "...", "top_k": 10}
    #   embed:   extract = {"normalize": true}
    #   layout:  extract = {"return_blocks": true}
    extract: Optional[Dict[str, Any]] = Field(
        default=None,
        description="mini/translate 扩展参数；OCR: task/schema；translate: source_lang/target_lang",
    )
    
    @field_validator('tpl_list', mode='before')
    @classmethod
    def parse_tpl_list(cls, v):
        """解析tpl_list，支持JSON字符串"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return []
        return v if isinstance(v, list) else []

    @field_validator("loras", mode="before")
    @classmethod
    def parse_loras(cls, v):
        """解析 loras，支持 JSON 字符串 / list / dict / None。新协议不再使用 {json: ...} 包装。"""
        if v is None:
            return None
        # 允许直接传 json 字符串（推荐：字符串代表一个 list）
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                parsed = json.loads(s)
                # 若能解析成 list/dict，直接返回解析结果；否则保留原字符串
                return parsed if isinstance(parsed, (list, dict)) else s
            except Exception:
                # 解析失败：保留原字符串，让下游自行 best-effort 处理
                return s
        # list/dict 直接透传（兼容历史输入）
        if isinstance(v, (list, dict)):
            return v
        # 其他类型直接丢弃，避免污染
        return None

    @field_validator("low_vram", "fast_mode", "stream", "timestamps", "speaker_diarization", mode="before")
    @classmethod
    def parse_bool_flags(cls, v):
        """兼容消息侧常见布尔表达：0/1/false/true。"""
        if isinstance(v, bool):
            return v
        if isinstance(v, int) and v in (0, 1):
            return bool(v)
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in ("0", "false"):
                return False
            if normalized in ("1", "true"):
                return True
        raise ValueError("must be a boolean-like value: 0, 1, false, true")

    @field_validator("audio_mode", mode="before")
    @classmethod
    def normalize_audio_mode(cls, v):
        raw = str(v or "").strip().lower()
        aliases = {
            "": "tts",
            "tts": "tts",
            "asr": "asr",
            "realtime_tts": "realtime_tts",
            "realtime": "realtime_tts",
            "streaming_tts": "realtime_tts",
        }
        normalized = aliases.get(raw)
        if not normalized:
            raise ValueError("audio_mode must be one of: tts, asr, realtime_tts")
        return normalized

    @field_validator("tts_mode", mode="before")
    @classmethod
    def normalize_tts_mode(cls, v):
        raw = str(v or "").strip().lower().replace("-", "_")
        aliases = {
            "": "custom_voice",
            "custom_voice": "custom_voice",
            "customvoice": "custom_voice",
            "custom": "custom_voice",
            "voice_design": "voice_design",
            "voicedesign": "voice_design",
            "design": "voice_design",
            "voice_clone": "voice_clone",
            "voiceclone": "voice_clone",
            "clone": "voice_clone",
            "voice_design_then_clone": "voice_design_then_clone",
            "voicedesignthenclone": "voice_design_then_clone",
            "design_then_clone": "voice_design_then_clone",
            "designclone": "voice_design_then_clone",
            "dtc": "voice_design_then_clone",
        }
        normalized = aliases.get(raw)
        if not normalized:
            raise ValueError(
                "tts_mode must be one of: custom_voice, voice_design, voice_clone, voice_design_then_clone"
            )
        return normalized

    @field_validator("x_vector_only", mode="before")
    @classmethod
    def parse_x_vector_only(cls, v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        if isinstance(v, int) and v in (0, 1):
            return bool(v)
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in ("", "0", "false", "no"):
                return False
            if normalized in ("1", "true", "yes"):
                return True
        raise ValueError("x_vector_only must be a boolean-like value")

    @field_validator("upscale")
    @classmethod
    def validate_upscale(cls, v: int):
        """upscale 仅允许 0/1/2/4；其余值直接报错，避免出现“默认就超分”等语义问题。"""
        if v in (0, 1, 2, 4):
            return v
        raise ValueError("upscale must be one of 0, 1, 2, 4")

    @model_validator(mode="after")
    def normalize_num_inference_steps(self):
        """
        兼容上游把 num_inference_steps 置为 0 的情况：
        - RBG/SR/FS：允许为 0（不走扩散 steps）
        - 其他任务：若 < 1，则回退到默认 30，避免后续 pipeline 因 steps=0 异常
        """
        jt = str(getattr(self, "job_type", "") or "").strip().upper()
        steps = int(getattr(self, "num_inference_steps", 0) or 0)
        # RBG/SR/FS 走图像后处理；OCR/TRANSLATE 等 job 不走扩散 steps；均允许 0
        if jt in ("RBG", "SR", "FS", "OCR", "TRANSLATE"):
            if steps < 0:
                self.num_inference_steps = 0
            return self

        if steps < 1:
            self.num_inference_steps = 30
        return self
    
    @classmethod
    def from_task_dict(cls, task_dict: dict) -> "InferenceRequestParams":
        """
        从任务字典创建参数对象
        
        Args:
            task_dict: 任务字典（包含id, user_id, type, prompt, params等）
        
        Returns:
            InferenceRequestParams实例
        """
        params = task_dict.get("params", {}) or {}
        model_meta = task_dict.get("model", {}) or {}
        
        # 基础字段映射
        type_value = task_dict.get("type")
        if not type_value:
            raise ValueError("task.type is required and cannot be empty")
        
        # 不同任务大类的默认输出扩展名
        # mini 默认按 OCR 常用形态落 md；extract 模式会由 handler 改写为 json
        default_file_type = {
            "image": "jpeg",
            "video": "mp4",
            "audio": "mp3",
            "mini": "md",
            "translate": "txt",
        }.get(type_value, "bin")

        data = {
            "type": type_value,
            "job_type": params.get("job_type", "MK") if params else "MK",  # job_type统一使用job_type字段
            "storage": task_dict.get("storage") or "local",
            "reference_id": task_dict.get("reference_id", ""),
            "id": task_dict.get("id"),
            "big": task_dict.get("big"),
            "thumb": task_dict.get("thumb"),
            "user_id": task_dict.get("user_id"),
            "task_id": task_dict.get("id"),
            "res_channel": task_dict.get("res_channel", ""),
            "prompt": task_dict.get("prompt", ""),
        }
        
        # 从params中提取参数（支持多种命名方式）
        if params:
            # 图片生成参数
            data.update({
                "negative_prompt": params.get("negative_prompt", ""),
                "width": params.get("width", params.get("w", 1024)),
                "height": params.get("height", params.get("h", 1024)),
                "guidance_scale": params.get("guidance_scale", params.get("cfg_value", 7.5)),
                "seed": params.get("seed", 0),
                "num_inference_steps": params.get(
                    "num_inference_steps", 
                    params.get("steps", params.get("inference_timesteps", 30))
                ),
                "strength": params.get("strength", params.get("denoising_strength", 0.5)),
                "file_type": params.get("file_type", default_file_type),
                # 兼容：部分业务用 init_images 表示图生图输入（通常为单张 URL）
                "url": params.get("url", params.get("init_images")),
                "generate_num": params.get("generate_num", params.get("num", 1)),
                "load_name": params.get("load_name") or (
                    task_dict.get("model", {}).get("load_name") if task_dict.get("model") else None
                ),
                "family": None,
                "runtime_config": task_dict.get("model", {}).get("runtime_config") if task_dict.get("model") else None,
                "third_transformer_path": "",
                "schedulerName": params.get(
                    "schedulerName", params.get("sampler_name", params.get("scheduler"))
                ),
                "keep_size": params.get("keep_size", "user"),
                "remove_bg": params.get("remove_bg", False),
                "low_vram": params.get("low_vram", False),
                "fast_mode": params.get("fast_mode", True),
            })
            
            # 人脸增强超分参数
            data.update({
                "upscale": params.get("upscale", 2),
                "face_enhance": params.get("face_enhance", False),
                "arch": params.get("arch", "clean"),
            })
            
            # 视频参数
            data.update({
                "duration": params.get("duration", 5),
                "resolution": params.get("resolution"),
                "aspect_ratio": params.get("aspect_ratio"),
                # 新增：视频控制/参考参数
                "ref_video": params.get("ref_video", params.get("pose_video")),
                "face_video": params.get("face_video"),
                "direction": params.get("direction"),
                "speed": params.get("speed"),
            })
            
            # 语音生成参数
            data.update({
                "audio_mode": params.get("audio_mode", "tts"),
                "input_audio_url": params.get("input_audio_url", params.get("audio_url", params.get("prompt_wav_path"))),
                "prompt_wav_path": params.get("prompt_wav_path"),
                "prompt_text": params.get("prompt_text"),
                "instruct": params.get("instruct"),
                "voice_preset": params.get("voice_preset"),
                "speaker_name": params.get("speaker_name"),
                "response_format": params.get("response_format", "audio_file"),
                "stream": params.get("stream", False),
                "sample_rate": params.get("sample_rate"),
                "language": params.get("language"),
                "timestamps": params.get("timestamps", False),
                "speaker_diarization": params.get("speaker_diarization", False),
                "drama": params.get("drama"),
                # Qwen3-TTS 多模式
                "tts_mode": params.get("tts_mode", "custom_voice"),
                "ref_audio": params.get("ref_audio"),
                "ref_text": params.get("ref_text"),
                "clone_base_load_name": params.get("clone_base_load_name"),
                "design_seed_text": params.get("design_seed_text"),
                "design_instruct": params.get("design_instruct"),
                "x_vector_only": params.get("x_vector_only", False),
            })
            
            # ControlNet参数
            data.update({
                "image_file2": params.get("image_file2", ""),
                "edit_act": params.get("edit_act", ""),
            })
            
            # 多参考图
            tpl_list = params.get("tpl_list", [])
            if isinstance(tpl_list, str):
                try:
                    tpl_list = json.loads(tpl_list)
                except:
                    tpl_list = []
            data["tpl_list"] = tpl_list if isinstance(tpl_list, list) else []

            # LoRA 参数
            if "loras" in params:
                data["loras"] = params.get("loras")

            # mini 服务扩展参数（OCR 等小模型）
            if "extract" in params:
                data["extract"] = params.get("extract")
        
        # ===== family：统一使用 model_catalog.family =====
        raw = ""
        try:
            if isinstance(model_meta, dict):
                raw = str(model_meta.get("family") or "").strip()
        except Exception:
            raw = ""
        data["family"] = raw or None

        return cls(**data)
    
    def to_inference_image_kwargs(self) -> dict:
        """
        转换为图片生成模型推理所需的参数字典
        
        Returns:
            图片生成推理参数字典
        """
        kwargs = {
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }
        
        # 可选参数
        if self.negative_prompt:
            kwargs["negative_prompt"] = self.negative_prompt
        
        if self.seed > 0:
            kwargs["generator"] = self.seed  # 实际使用时需要转换为torch.Generator
        
        # 图生图参数
        if self.url:
            kwargs["strength"] = self.strength
        
        return kwargs


class FileInfo:
    """文件信息（用于InferenceResponseParams）"""
    
    def __init__(self, file_id: str, storage_path: str, file_name: str, 
                 file_size: int, mime_type: str, seed: Optional[int] = None,
                 index: int = 0, thumbnail_path: Optional[str] = None,
                 http_url: Optional[str] = None, width: Optional[int] = None,
                 height: Optional[int] = None, thumb_url: Optional[str] = None,
                 url: Optional[str] = None):
        """
        初始化文件信息
        
        Args:
            file_id: 文件ID（UUID）
            storage_path: 存储路径（相对路径）
            file_name: 文件名
            file_size: 文件大小（字节）
            mime_type: MIME类型
            seed: 当前文件使用的随机种子（可选，多文件时每个文件可能有不同的seed）
            index: 当前文件在任务中的序号（从0开始）
            thumbnail_path: 缩略图路径（可选）
            http_url: HTTP URL（可选）
            width: 图片/视频宽度（可选）
            height: 图片/视频高度（可选）
            thumb_url: 缩略图URL（可选）
            url: 原图URL（可选）
        """
        self.file_id = file_id
        self.storage_path = storage_path
        self.file_name = file_name
        self.file_size = file_size
        self.mime_type = mime_type
        self.seed = seed
        self.index = index
        self.thumbnail_path = thumbnail_path
        self.http_url = http_url
        self.width = width
        self.height = height
        self.thumb_url = thumb_url
        self.url = url
    
    def to_dict(self) -> dict:
        """转换为字典"""
        result = {
            "file_id": self.file_id,
            "storage_path": self.storage_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "seed": self.seed,
            "index": self.index,
        }
        if self.thumbnail_path:
            result["thumbnail_path"] = self.thumbnail_path
        if self.http_url:
            result["http_url"] = self.http_url
        if self.width is not None:
            result["width"] = self.width
        if self.height is not None:
            result["height"] = self.height
        if self.thumb_url:
            result["thumb_url"] = self.thumb_url
        if self.url:
            result["url"] = self.url
        return result


class InferenceResponseParams:
    """
    推理结果响应参数（重构版）
    
    设计原则：
    1. 文件信息统一在files数组中，每个文件包含完整元数据
    2. 任务级别信息单独存储
    3. 性能指标单独存储
    4. 包含创建数据库文件记录所需的所有信息
    """
    
    def __init__(self, message: dict, service_id: str, files: List[FileInfo]):
        """
        初始化结果消息
        
        Args:
            message: 消息字典（包含InferenceRequestParams的字段）
            service_id: 当前推理器的service_id
            files: 文件信息列表（必需）
        """
        if not files:
            raise ValueError("files list cannot be empty")
        
        # 推理器信息
        self.queue_length = 0  # 当前推理器剩余队列长度
        self.service_id = service_id  # 当前推理器的service_id
        
        # 任务基本信息
        self.task_id = message.get('task_id')
        self.user_id = message.get('user_id')
        self.task_type = message.get('type')  # 任务大类：image/video/audio/text
        self.job_type = message.get('job_type', 'MK')  # 任务执行分类：MK, RBG...
        self.status = message.get('status', 'completed')
        self.progress = message.get('progress', 100)
        
        # 任务产物存储目标
        self.storage = message.get('storage', 'local')  # local oss s3 server
        
        # 推理参数（用于记录和展示）
        self.seed = message.get('seed')
        self.load_name = message.get('load_name')
        self.reference_id = message.get('reference_id', '')
        self.duration = message.get('duration', 0)  # 音视频时长
        
        # 性能指标
        self.generate_time = message.get('generate_time', -1)  # 文件生成耗时（秒）
        self.upload_time = message.get('upload_time', -1)  # 文件存储耗时（秒）
        gen_time = self.generate_time if isinstance(self.generate_time, (int, float)) and self.generate_time >= 0 else 0
        up_time = self.upload_time if isinstance(self.upload_time, (int, float)) and self.upload_time >= 0 else 0
        self.used_time = gen_time + up_time  # 总耗时
        
        # 文件列表（核心数据结构）
        self.files = files
        
        # 当前结果序号与任务总数（用于多图任务）
        self.index = message.get('index')
        message_total = message.get('total')
        self.total = message_total if isinstance(message_total, int) and message_total > 0 else len(files)
    
    def to_dict(self) -> dict:
        """
        转换为字典（用于WebSocket消息）
        
        Returns:
            包含所有信息的字典，可以直接用于创建数据库文件记录
        """
        return {
            # 消息类型
            "type": "result",
            
            # 推理器信息
            "queue_length": self.queue_length,
            "service_id": self.service_id,
            
            # 任务基本信息
            "task_id": self.task_id,
            "user_id": self.user_id,
            "task_type": self.task_type,  # 任务大类：image/video/audio/text
            "job_type": self.job_type,  # 任务执行分类：MK, RBG...
            "status": self.status,
            "progress": self.progress,
            
            # 任务产物存储目标
            "storage": self.storage,
            
            # 推理参数
            "seed": self.seed,  # 任务级别的seed（如果所有文件使用相同seed）
            "load_name": self.load_name,
            "reference_id": self.reference_id,
            "duration": self.duration,
            
            # 性能指标
            "generate_time": self.generate_time,
            "upload_time": self.upload_time,
            "used_time": self.used_time,
            
            # 文件信息
            "total": self.total,  # 文件总数
            "index": self.index,  # 当前结果序号（从0开始）
            
            # 文件列表（核心数据）
            "files": [file_info.to_dict() for file_info in self.files],
        }

