"""
后端推理API路由
提供图片、视频、音频、文字生成等推理任务接口
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from backend.auth import get_current_user_id
from backend.database import Task
from backend.core.logger import get_app_logger
from backend.utils import generate_uuid
from backend.core.response import ok

logger = get_app_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Tasks"])


# ==================== 请求/响应模型 ====================

class TaskCreateRequest(BaseModel):
    """统一的任务创建请求"""

    task_type: str = Field(..., description="Task type: image/video/audio/text/translate/mini")
    job_type: str = Field("MK", description="Task execution category: MK/RBG/ED/POSE/SED/SR/FS")
    prompt: Optional[str] = Field(None, description="Prompt (required for image/video/audio tasks)")
    
    # 图片生成参数
    negative_prompt: Optional[str] = Field("", description="Negative prompt")
    width: int = Field(1024, ge=64, le=4096, description="Image width")
    height: int = Field(1024, ge=64, le=4096, description="Image height")
    # guidance_scale 允许为 0（例如 Turbo/LCM 等模型）
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0, description="Guidance scale")
    # 约定：seed < 0（例如 -1）表示随机；seed >= 0 表示固定种子（包括 0）
    seed: int = Field(0, description="Random seed (<0 for random; >=0 for fixed)")
    num_inference_steps: Optional[int] = Field(None, ge=1, le=50, description="Number of inference steps")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Denoising strength")
    file_type: str = Field("jpeg", description="Output file type")
    url: Optional[str] = Field(None, description="Source image URL for image-to-image")
    generate_num: int = Field(1, ge=1, le=10, description="Number of images to generate")
    family: str = Field("sdxl", description="Model family")
    schedulerName: Optional[str] = Field(None, description="Scheduler name")
    keep_size: str = Field("user", description="Size preservation mode when editing images")
    remove_bg: bool = Field(False, description="Whether to remove background")
    fast_mode: bool = Field(True, description="Fast mode")
    upscale: int = Field(0, ge=0, le=4, description="Upscale factor; 0/1 = no upscale, 2/4 = upscale")
    face_enhance: bool = Field(False, description="Whether to enhance faces")
    arch: str = Field("clean", description="Architecture")
    image_file2: str = Field("", description="Second input image URL (deprecated in ED/POSE new protocol)")
    edit_act: str = Field("", description="Edit action")
    tpl_list: List[str] = Field(default_factory=list, description="Reference image list; ED/POSE uses this field only")
    
    # 视频生成参数
    duration: int = Field(5, ge=1, le=60, description="Video duration in seconds")
    fps: Optional[int] = Field(None, ge=1, le=60, description="Video FPS override; takes precedence over model default when set")
    resolution: Optional[str] = Field(None, description="Video resolution")
    aspect_ratio: Optional[str] = Field(None, description="Video aspect ratio (e.g. 16:9 / 9:16)")
    ref_video: Optional[str] = Field(None, description="Reference/control video URL (common for video tasks)")
    face_video: Optional[str] = Field(None, description="Face/expression driving video URL (IVV2V)")
    direction: Optional[str] = Field(None, description="Camera control direction (CCV)")
    speed: Optional[float] = Field(None, description="Camera control speed (CCV)")
    
    # 音频生成参数
    audio_mode: Optional[str] = Field("tts", description="Audio mode: tts/asr/realtime_tts")
    input_audio_url: Optional[str] = Field(None, description="ASR input audio URL")
    prompt_wav_path: Optional[str] = Field(None, description="Prompt speech path URL")
    prompt_text: Optional[str] = Field(None, description="Reference text")
    instruct: Optional[str] = Field(None, description="Style/emotion control instruction for Qwen-TTS and similar models")
    voice_preset: Optional[str] = Field(None, description="Preset voice name")
    speaker_name: Optional[str] = Field(None, description="Speaker/voice name")
    response_format: Optional[str] = Field("audio_file", description="Audio task output type: audio_file/text_file/both")
    sample_rate: Optional[int] = Field(None, ge=8000, le=96000, description="Sample rate")
    language: Optional[str] = Field(None, description="Language code")
    timestamps: bool = Field(False, description="Whether to return timestamps")
    speaker_diarization: bool = Field(False, description="Whether to return speaker diarization results")
    drama: Optional[Dict[str, Any]] = Field(
        None,
        description="Multi-character drama TTS extension: {characters:[], dialogues:[]}",
    )
    tts_mode: Optional[str] = Field(
        "custom_voice",
        description="Qwen3-TTS synthesis sub-mode: custom_voice/voice_design/voice_clone/voice_design_then_clone",
    )
    ref_audio: Optional[str] = Field(None, description="Reference audio URL or local path for voice_clone")
    ref_text: Optional[str] = Field(None, description="Reference text for voice_clone (aligned with ref_audio)")
    clone_base_load_name: Optional[str] = Field(
        None,
        description="Base weight load_name for stage 2 of voice_design_then_clone; auto-filled by backend",
    )
    design_seed_text: Optional[str] = Field(
        None,
        description="Seed text for VoiceDesign in stage 1 of voice_design_then_clone; "
        "also used as ref_text for stage 2 Base cloning",
    )
    design_instruct: Optional[str] = Field(
        None,
        description="Voice design instruction for stage 1 of voice_design_then_clone; falls back to instruct if omitted",
    )
    x_vector_only: bool = Field(False, description="voice_clone option: use speaker embedding only")
    
    # 文字生成参数（OpenAI兼容格式）
    model: Optional[str] = Field(None, description="OpenAI-compatible field: model load name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum token count")
    stream: bool = Field(False, description="Whether to stream the response")
    
    # 通用参数
    model_key: Optional[str] = Field(None, description="Optional stable model key; server looks up model_catalog by this field first when set")
    load_name: Optional[str] = Field(None, description="Model load name; backfilled from model_catalog by server")
    agent_run_id: Optional[str] = Field(
        None,
        description="Internal field: binds agent_run_id when task is spawned by chat/master agent",
    )
    # LoRA 参数（可选）：前端提交 JSON 字符串，如 '[{"name":"xxx.safetensors","weight":0.8}]'
    # （历史兼容：推理侧仍可解析 list/dict/json-string，但新协议不再使用 {json: "..."} 包装）
    loras: Optional[Any] = Field(default=None, description="LoRA parameters as JSON string (e.g. '[{\"name\":\"xxx\",\"weight\":0.8}]')")

    # mini / translate 服务扩展参数
    # mini OCR: {"task": "text|table|formula|extract", "schema": {...}}
    # translate: {"source_lang": "zh", "target_lang": "en"}
    extract: Optional[Dict[str, Any]] = Field(default=None, description="Mini/translate extension parameters")


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    progress: int
    message: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskListResponse(BaseModel):
    """任务列表响应"""
    tasks: List[TaskStatusResponse]
    total: int


# ==================== 辅助函数 ====================

def _extract_params_from_request(request: TaskCreateRequest, task_type: str) -> Dict[str, Any]:
    """
    从请求对象中提取任务参数
    
    Args:
        request: 请求对象
        task_type: 任务类型
    
    Returns:
        参数字典
    """
    params = {
        "job_type": request.job_type,
        "load_name": request.load_name,
        # family 由 model_catalog.family 回填，作为推理侧模型家族字段。
        "family": request.family,
    }
    
    if task_type == "image":
        params.update({
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
            "strength": request.strength,
            "file_type": request.file_type,
            "url": request.url,
            "generate_num": request.generate_num,
            "family": request.family,
            "schedulerName": request.schedulerName,
            "keep_size": request.keep_size,
            "remove_bg": request.remove_bg,
            "fast_mode": request.fast_mode,
            "upscale": request.upscale,
            "face_enhance": request.face_enhance,
            "arch": request.arch,
            "image_file2": request.image_file2,
            "edit_act": request.edit_act,
            "tpl_list": request.tpl_list,
            "loras": request.loras,
        })
        if request.guidance_scale is not None:
            params["guidance_scale"] = request.guidance_scale
        if request.num_inference_steps is not None:
            params["num_inference_steps"] = request.num_inference_steps
    elif task_type == "video":
        params.update({
            # Keep consistency with frontend and WS result schema.
            # NOTE: Task.prompt is stored in DB field `prompt` already; we still keep a copy in params for inference/backward compatibility.
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
            "generate_num": request.generate_num,
            "url": request.url,
            "image_file2": request.image_file2,
            "edit_act": request.edit_act,
            "tpl_list": request.tpl_list,
            "duration": request.duration,
            "fps": request.fps,
            "resolution": request.resolution,
            "aspect_ratio": request.aspect_ratio,
            "ref_video": request.ref_video,
            "face_video": request.face_video,
            # S2V uses video task_type, so audio-related inputs must also be forwarded here.
            "prompt_wav_path": request.prompt_wav_path,
            "prompt_text": request.prompt_text,
            "direction": request.direction,
            "speed": request.speed,
            "fast_mode": request.fast_mode,
            "loras": request.loras,
        })
        if request.guidance_scale is not None:
            params["guidance_scale"] = request.guidance_scale
        if request.num_inference_steps is not None:
            params["num_inference_steps"] = request.num_inference_steps
    elif task_type == "audio":
        params.update({
            "prompt": request.prompt,
            "audio_mode": request.audio_mode,
            "input_audio_url": request.input_audio_url,
            "prompt_wav_path": request.prompt_wav_path,
            "prompt_text": request.prompt_text,
            "instruct": request.instruct,
            "voice_preset": request.voice_preset,
            "speaker_name": request.speaker_name,
            "response_format": request.response_format,
            "stream": request.stream,
            "sample_rate": request.sample_rate,
            "language": request.language,
            "timestamps": request.timestamps,
            "speaker_diarization": request.speaker_diarization,
            "drama": request.drama,
            "file_type": request.file_type,
            # Qwen3-TTS 多模式
            "tts_mode": request.tts_mode,
            "ref_audio": request.ref_audio,
            "ref_text": request.ref_text,
            "clone_base_load_name": request.clone_base_load_name,
            "design_seed_text": request.design_seed_text,
            "design_instruct": request.design_instruct,
            "x_vector_only": request.x_vector_only,
        })
    elif task_type == "text":
        params.update({
            "model": request.load_name or request.model,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        })
    elif task_type == "mini":
        # mini 服务（OCR / rerank / embed / layout 等小模型工具集）统一走这里。
        # 约定：
        #   - 输入文件（图片/PDF）统一放在 tpl_list，允许多个
        #   - 专属参数统一塞到 extract dict（OCR 的 task/schema、rerank 的 query 等）
        #   - 不涉及扩散类参数；prompt 可选（OCR 未使用）
        #
        # file_type 策略（InferenceRequestParams.file_type 是 Literal，不允许 None）：
        #   - 调用方传了合法值（jpeg/空以外）→ 尊重其值
        #   - 否则按 extract.task 给默认：
        #       extract → "json"
        #       text    → "zip"（图文混排默认输出）
        #       其它    → "md"（table/formula/未知）
        #   - 注：显式传 "md" 会触发推理端"强制纯文字 md 旁路"（跳过图文混排）
        extract_dict = request.extract if isinstance(request.extract, dict) else {}
        ocr_task = str(extract_dict.get("task") or "text").strip().lower()
        if ocr_task == "extract":
            mini_default_ft = "json"
        elif ocr_task == "text":
            mini_default_ft = "zip"
        else:
            mini_default_ft = "md"

        raw_ft = str(request.file_type or "").strip().lower()
        mini_ft = raw_ft if raw_ft and raw_ft != "jpeg" else mini_default_ft

        params.update({
            "tpl_list": request.tpl_list,
            "extract": request.extract,
            "file_type": mini_ft,
        })
    elif task_type == "translate":
        extract_dict = request.extract if isinstance(request.extract, dict) else {}
        params.update({
            "tpl_list": request.tpl_list,
            "extract": extract_dict,
            "file_type": "txt" if not request.file_type or request.file_type == "jpeg" else request.file_type,
            "max_tokens": request.max_tokens,
        })

    return params


def _audio_dispatch_capability_from_request(request: TaskCreateRequest) -> str:
    audio_mode = str(getattr(request, "audio_mode", None) or "").strip().lower()
    job_type = str(getattr(request, "job_type", None) or "").strip().upper()
    if audio_mode == "asr" or job_type == "ASR":
        return "asr"
    if audio_mode in {"tts", "realtime_tts"} or job_type in {"TTS", "REALTIME_TTS", "RTT", "RTTS"}:
        return "tts"
    return ""


def _will_dispatch_task_to_inference(
    *,
    task_type: str,
    requires_model: bool,
    resolved_model: Optional[Dict[str, Any]],
) -> bool:
    if task_type == "audio":
        return True
    if resolved_model is not None:
        return str(resolved_model.get("storage_mode") or "").strip().lower() == "local"
    return task_type == "image" and not requires_model


async def _assert_inference_dispatch_available_for_request(
    request: TaskCreateRequest,
    *,
    task_type: str,
    requires_model: bool,
    resolved_model: Optional[Dict[str, Any]],
) -> None:
    if not _will_dispatch_task_to_inference(
        task_type=task_type,
        requires_model=requires_model,
        resolved_model=resolved_model,
    ):
        return

    from backend.services.chat.dispatch_feedback import assert_inference_dispatch_available

    await assert_inference_dispatch_available(
        service_type=task_type,
        load_name=str(request.load_name or "").strip(),
        capability=_audio_dispatch_capability_from_request(request) if task_type == "audio" else "",
    )


def _resolve_request_model(request: TaskCreateRequest, *, task_type: str, requires_model: bool) -> Optional[Dict[str, Any]]:
    """
    统一按 model_key/load_name 查库并回填内部模型信息。

    说明：
      - model_key 和 load_name 都传且非空时，以 model_key 为准；
      - 只传 load_name 时，按 `model_catalog.load_name` 查库；
      - 查到后统一回填 model_key / load_name / family，后续派发可拿到完整 catalog 记录。
    """
    if not requires_model:
        return None

    if task_type == "text" and not request.load_name and request.model:
        request.load_name = str(request.model).strip()

    model_key = str(request.model_key or "").strip()
    load_name = str(request.load_name or "").strip()
    if not model_key and not load_name:
        raise HTTPException(
            status_code=400,
            detail=f"model_key or load_name is required for {task_type} tasks",
        )

    from backend.database import Model

    model_dict = Model.get_by_model_key(model_key) if model_key else Model.get_by_load_name(load_name)
    if not model_dict:
        if task_type == "translate" and not model_key:
            from backend.services.agent.settings import (
                get_translate_default_family,
                get_translate_default_model_name,
            )

            default_name = str(get_translate_default_model_name() or "").strip()
            if default_name and load_name.lower() == default_name.lower():
                request.family = str(request.family or get_translate_default_family() or "").strip()
                if not request.family:
                    raise HTTPException(
                        status_code=400,
                        detail="family is required for translate tasks",
                    )
                return None
        if model_key:
            raise HTTPException(status_code=400, detail=f"Invalid model_key: {model_key}")
        raise HTTPException(status_code=400, detail=f"Invalid load_name: {load_name}")

    request.model_key = str(model_dict.get("model_key") or model_key).strip() or None
    request.load_name = str(model_dict.get("load_name") or load_name).strip() or None
    if task_type == "text":
        request.model = request.load_name

    family = str(model_dict.get("family") or "").strip()
    if task_type != "text":
        if not family:
            raise HTTPException(
                status_code=400,
                detail=f"family is required for load_name={request.load_name}"
            )
        request.family = family

    return model_dict


async def _create_task(request: TaskCreateRequest, user_id: str) -> TaskResponse:
    """
    创建任务的通用函数
    
    Args:
        request: 请求对象（包含task_type字段）
        user_id: 用户ID
    
    Returns:
        TaskResponse对象
    
    Raises:
        HTTPException: 如果创建任务失败
    """
    task_type = request.task_type
    job_type = (request.job_type or "MK").upper().strip()
    request.job_type = job_type
    
    # 验证任务类型（与 config/model_catalog_meta.yaml modalities 一致）
    from backend.models.catalog_meta import is_valid_modality, modality_ids_description

    if not is_valid_modality(task_type):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_type: {task_type}. Must be one of: {modality_ids_description()}",
        )

    if task_type == "audio":
        raw_audio_mode = str(request.audio_mode or "").strip().lower()
        if not raw_audio_mode:
            raw_audio_mode = {
                "ASR": "asr",
                "REALTIME_TTS": "realtime_tts",
                "RTT": "realtime_tts",
                "RTTS": "realtime_tts",
                "TTS": "tts",
            }.get(job_type, "tts")
        alias_map = {
            "tts": "tts",
            "asr": "asr",
            "realtime_tts": "realtime_tts",
            "realtime": "realtime_tts",
            "streaming_tts": "realtime_tts",
        }
        audio_mode = alias_map.get(raw_audio_mode)
        if not audio_mode:
            raise HTTPException(
                status_code=400,
                detail="audio_mode must be one of: tts, asr, realtime_tts"
            )
        request.audio_mode = audio_mode
        request.job_type = audio_mode.upper()

        response_format = str(request.response_format or "").strip().lower() or "audio_file"
        if response_format not in {"audio_file", "text_file", "both"}:
            raise HTTPException(
                status_code=400,
                detail="response_format must be one of: audio_file, text_file, both"
            )
        request.response_format = response_format

        if audio_mode == "tts":
            if not request.prompt:
                raise HTTPException(status_code=400, detail="prompt is required for audio_mode=tts")
            tts_mode_alias = {
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
            raw_tts_mode = str(request.tts_mode or "").strip().lower().replace("-", "_")
            tts_mode = tts_mode_alias.get(raw_tts_mode)
            if not tts_mode:
                raise HTTPException(
                    status_code=400,
                    detail="tts_mode must be one of: custom_voice, voice_design, voice_clone, voice_design_then_clone",
                )
            request.tts_mode = tts_mode

            if tts_mode == "voice_clone":
                if not (request.ref_audio or "").strip():
                    raise HTTPException(
                        status_code=400,
                        detail="ref_audio is required for tts_mode=voice_clone",
                    )
                if not request.x_vector_only and not (request.ref_text or "").strip():
                    raise HTTPException(
                        status_code=400,
                        detail="ref_text is required for tts_mode=voice_clone (or set x_vector_only=True)",
                    )
            elif tts_mode == "voice_design_then_clone":
                if not (request.design_seed_text or "").strip():
                    raise HTTPException(
                        status_code=400,
                        detail="design_seed_text is required for tts_mode=voice_design_then_clone",
                    )
                if not (request.design_instruct or request.instruct or "").strip():
                    raise HTTPException(
                        status_code=400,
                        detail="design_instruct (or instruct) is required for tts_mode=voice_design_then_clone",
                    )
                if not (request.clone_base_load_name or "").strip():
                    raise HTTPException(
                        status_code=400,
                        detail="clone_base_load_name is required for tts_mode=voice_design_then_clone",
                    )
            elif tts_mode == "voice_design":
                drama = request.drama if isinstance(request.drama, dict) else {}
                dialogue_lines = drama.get("dialogues") if isinstance(drama.get("dialogues"), list) else []
                character_rows = drama.get("characters") if isinstance(drama.get("characters"), list) else []
                character_instructs = {
                    str(item.get("id") or "").strip(): str(item.get("instruct") or "").strip()
                    for item in character_rows
                    if isinstance(item, dict) and str(item.get("id") or "").strip()
                }
                valid_dialogue_lines = [
                    line for line in dialogue_lines
                    if isinstance(line, dict) and str(line.get("text") or "").strip()
                ]
                missing_dialogue_instruct = any(
                    not (
                        str(line.get("instruct") or "").strip()
                        or character_instructs.get(str(line.get("speaker_id") or "").strip())
                    )
                    for line in valid_dialogue_lines
                )
                if valid_dialogue_lines:
                    if missing_dialogue_instruct:
                        raise HTTPException(
                            status_code=400,
                            detail="drama.dialogues[].instruct or matching drama.characters[].instruct is required for tts_mode=voice_design",
                        )
                elif not (request.design_instruct or request.instruct or "").strip():
                    raise HTTPException(
                        status_code=400,
                        detail="instruct is required for tts_mode=voice_design",
                    )
        elif audio_mode == "asr":
            request.input_audio_url = request.input_audio_url or request.prompt_wav_path
            if not request.input_audio_url:
                raise HTTPException(status_code=400, detail="input_audio_url is required for audio_mode=asr")
            if request.response_format == "audio_file":
                request.response_format = "text_file"
        elif audio_mode == "realtime_tts":
            if not request.prompt:
                raise HTTPException(status_code=400, detail="prompt is required for audio_mode=realtime_tts")
            if not request.stream:
                raise HTTPException(status_code=400, detail="stream=true is required for audio_mode=realtime_tts")
            if request.response_format == "text_file":
                raise HTTPException(status_code=400, detail="response_format=text_file is invalid for realtime_tts")

    if task_type == "translate":
        extract = request.extract if isinstance(request.extract, dict) else {}
        source_lang = str(
            extract.get("source_lang") or extract.get("source_lang_code") or ""
        ).strip()
        target_lang = str(
            extract.get("target_lang") or extract.get("target_lang_code") or ""
        ).strip()
        if not source_lang or not target_lang:
            raise HTTPException(
                status_code=400,
                detail="translate task requires extract.source_lang and extract.target_lang",
            )
        prompt_text = str(request.prompt or "").strip()
        tpl_list = [str(item).strip() for item in (request.tpl_list or []) if str(item or "").strip()]
        if not prompt_text and not tpl_list:
            raise HTTPException(
                status_code=400,
                detail="translate task requires prompt (text) or tpl_list (image URL/path)",
            )
        if not request.job_type or request.job_type.upper() == "MK":
            request.job_type = "TRANSLATE"
        if not str(request.load_name or "").strip() and not str(request.model_key or "").strip():
            from backend.services.agent.settings import get_translate_default_model_name

            default_model = get_translate_default_model_name()
            if not default_model:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "load_name is required for translate tasks when "
                        "agents.tools.translate.default_model_name is not configured"
                    ),
                )
            request.load_name = default_model
    
    # -------- Ensure model fields are present and consistent (only when required) --------
    # Some image job types (e.g. RBG/SR/FS) do not rely on diffusion models, so they can run without model lookup.
    requires_model = True
    if task_type == "image" and job_type in ("RBG", "SR", "FS"):
        requires_model = False

    # 音频任务新协议：load_name 允许为空。
    # - 若目标推理服务声明了 config.fixed_model（pin 模式），dispatch 会把
    #   任务路由到 pinned 服务，由推理侧用 fixed_model / fixed_family 自我纠正；
    # - 若提供了 model_key/load_name，按统一模型解析规则查库并回填 family。
    audio_requires_model_lookup = task_type == "audio" and bool(
        str(request.model_key or "").strip() or str(request.load_name or "").strip()
    )
    effective_requires_model = requires_model and not (
        task_type == "audio" and not audio_requires_model_lookup
    )

    resolved_model = _resolve_request_model(
        request,
        task_type=task_type,
        requires_model=effective_requires_model,
    )

    # For model-based tasks we require load_name lookup to provide normalized fields.
    if effective_requires_model:
        if not request.load_name:
            raise HTTPException(status_code=400, detail=f"load_name is required for {task_type} tasks")
        if task_type != "text" and not request.family:
            if task_type == "translate":
                from backend.services.agent.settings import get_translate_default_family

                request.family = get_translate_default_family()
            if not request.family:
                raise HTTPException(status_code=400, detail=f"family is required for {task_type} tasks")

    # S2V is modeled as a video task, but it requires both a reference image and an audio input.
    if task_type == "video" and job_type == "S2V":
        if not request.url:
            raise HTTPException(status_code=400, detail="url is required for S2V video tasks")
        if not request.prompt_wav_path:
            raise HTTPException(status_code=400, detail="prompt_wav_path is required for S2V video tasks")

    await _assert_inference_dispatch_available_for_request(
        request,
        task_type=task_type,
        requires_model=requires_model,
        resolved_model=resolved_model,
    )

    if task_type == "image" and request.num_inference_steps is None:
        try:
            from backend.services.agent.settings import (
                get_image_generator_default_num_inference_steps,
            )

            request.num_inference_steps = get_image_generator_default_num_inference_steps()
        except Exception:
            request.num_inference_steps = 30
    if task_type == "image" and request.guidance_scale is None:
        try:
            from backend.services.agent.settings import (
                get_image_generator_default_guidance_scale,
            )

            request.guidance_scale = get_image_generator_default_guidance_scale()
        except Exception:
            request.guidance_scale = 7.5

    task_id = generate_uuid()
    
    # 提取prompt（DB 字段不允许 NULL）
    prompt = request.prompt or ""
    
    # 提取参数
    params = _extract_params_from_request(request, task_type)
    
    # 创建任务记录（产物 storage 仅由 config storage.default 决定，不接受请求体覆盖）
    task_dict = Task.create(
        id=task_id,
        user_id=user_id,
        task_type=task_type,
        prompt=prompt,
        params=params,
        model_key=request.model_key,
        agent_run_id=(str(request.agent_run_id or "").strip() or None),
        status="pending",
        storage=None,
        priority=5
    )
    
    if not task_dict:
        raise HTTPException(status_code=500, detail="Failed to create task")
    
    task_type_name = {
        "image": "Image generation",
        "video": "Video generation",
        "audio": "Audio generation",
        "text": "Chat completion",
        "mini": "Mini inference",
        "translate": "Translation",
    }.get(task_type, "Task")
    
    logger.info(f"{task_type_name} task created: {task_id} (user: {user_id})")
    
    # 通过 WebSocket 发送任务给推理器。
    # - audio：按 service_type=audio 派发，具体 runtime 由推理服务内部基于 family 决定
    # - 其他类型：沿用现有“本地模型/本地任务 -> inference service”的策略
    from backend.websocket.manager import get_websocket_manager
    # 输出创建任务日志
    logger.info(f"创建任务 {task_id} 并转发给推理器")
    
    ws_manager = get_websocket_manager()
    
    should_send = False
    if task_type == "audio":
        should_send = True
        audio_model_label = (request.load_name or "").strip() or "<unset; will route to pinned service>"
        logger.info(
            f"Task {task_id} is audio task with load_name={audio_model_label}, "
            "will send to inference service"
        )
    elif resolved_model is not None:
        if str(resolved_model.get("storage_mode") or "").strip().lower() == "local":
            should_send = True
            logger.info(f"Task {task_id} uses local model {request.load_name}, will send to inference service")
        else:
            logger.info(f"Task {task_id} uses cloud model {request.load_name}, will not send to inference service")
    else:
        # For non-model image jobs (RBG/SR/FS), always send to local inference service.
        if task_type == "image" and not requires_model:
            should_send = True
            logger.info(f"Task {task_id} is non-model image job_type={job_type}, will send to inference service")
        else:
            raise HTTPException(status_code=400, detail=f"load_name is required for {task_type} tasks")
    
    if should_send:
        try:
            dispatched = await ws_manager.send_task_to_inference_service(task_id, task_type)
            if dispatched:
                logger.info(f"Task {task_id} sent to inference service successfully (type: {task_type})")
            else:
                logger.warning(
                    f"Task {task_id} could not be dispatched to inference service (type: {task_type})"
                )
        except Exception as e:
            logger.error(
                f"Failed to send task {task_id} to inference service: {e}",
                exc_info=True
            )
            # 即使发送失败，任务仍然创建成功，推理器可以从数据库轮询任务
    
    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="Task created successfully"
    )


# ==================== API端点 ====================

@router.post("/tasks", status_code=201)
async def create_task(
    request: TaskCreateRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    创建推理任务（统一接口）
    
    支持四种任务类型：
    - **image**: 图片生成/编辑
    - **video**: 视频生成
    - **audio**: 音频生成
    - **text**: 文字生成（OpenAI兼容格式）
    
    **请求参数**：
    - **task_type** (必需): 任务类型，必须是 image/video/audio/text 之一
    - **prompt** : 提示词
    - **job_type** (可选): 任务执行分类，默认 "MK"，可选值：MK/RBG/ED/SED/SR/FS
    - **model_key / load_name** (模型类任务二选一): 同时传入时 model_key 优先；服务端统一查库回填模型信息
    - 产物 **storage** 由配置 `storage.default` 决定（不接受请求体字段）
    
    **图片生成参数**（task_type=image时）：
    - width, height, guidance_scale, seed, num_inference_steps 等
    
    **视频生成参数**（task_type=video时）：
    - duration, resolution 等
    
    **音频生成参数**（task_type=audio时）：
    - prompt_wav_path, prompt_text 等
    
    **文字生成参数**（task_type=text时）：
    - model, temperature, max_tokens, stream 等
    
    返回task_id，前端应通过WebSocket连接 /ws/task/{task_id} 获取实时更新
    
    ```
    """
    resp = await _create_task(request, user_id)
    return ok(data=resp, msg="created")


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    获取任务状态
    
    - **task_id**: 任务ID
    """
    task_dict = Task.get_by_id(task_id)
    
    if not task_dict:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 检查用户权限
    if task_dict["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    resp = TaskStatusResponse(
        task_id=task_dict["id"],
        status=task_dict["status"],
        progress=task_dict.get("progress", 0),
        message=None,
        error=task_dict.get("error"),
        created_at=task_dict["created_at"],
        started_at=task_dict.get("started_at"),
        completed_at=task_dict.get("completed_at")
    )
    return ok(data=resp, msg="ok")


@router.get("/tasks")
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id)
):
    """
    获取任务列表
    
    - **status**: 任务状态（可选）
    - **limit**: 返回数量（默认50）
    - **offset**: 偏移量（默认0）
    """
    # 获取用户的任务列表
    tasks = Task.list_by_user(user_id, limit=limit, offset=offset)
    
    # 过滤状态
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    
    task_responses = [
        TaskStatusResponse(
            task_id=t["id"],
            status=t["status"],
            progress=t.get("progress", 0),
            message=None,
            error=t.get("error"),
            created_at=t["created_at"],
            started_at=t.get("started_at"),
            completed_at=t.get("completed_at")
        )
        for t in tasks
    ]
    
    resp = TaskListResponse(
        tasks=task_responses,
        total=len(task_responses)
    )
    return ok(data=resp, msg="ok")


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    取消任务
    
    - **task_id**: 任务ID
    
    注意：此接口会通过WebSocket向推理器发送中断信号
    """
    task_dict = Task.get_by_id(task_id)
    
    if not task_dict:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 检查用户权限
    if task_dict["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # 检查任务状态
    if task_dict["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Task already {task_dict['status']}")
    
    # 更新任务状态为cancelled
    Task.update(task_id, status="cancelled")
    
    # 通过WebSocket向推理器发送中断信号
    from backend.websocket.manager import get_websocket_manager
    ws_manager = get_websocket_manager()
    
    # 发送中断消息给推理器
    await ws_manager.send_cancel_signal_to_inference_service(task_id)
    
    logger.info(f"Task cancelled: {task_id} (user: {user_id})")
    
    return ok(data={"task_id": task_id}, msg="cancelled")

