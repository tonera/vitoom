"""统一 dispatch/router。

设计纠偏后：

    - 不再维护 ``supports_session + 旧场景绑定`` 或独立 capability registry；
    - 服务注册只保留最小事实：内容类型字段 ``service_type``（派发时不使用 ``type``）/
      ``supports_task`` / 运行状态 / 心跳 / websocket 连接状态；
    - 请求侧（chat / openai / tool/task）决定自己要什么，再交给这里
      统一做"选一台 running 服务实例"。

本模块因此承担两件事：

    1. ``DispatchRouter``：按 ``service_type`` 统一选择推理服务实例；
    2. ``LoadNameRouter``：按 ``load_name -> service_type -> DispatchRouter`` 选择服务。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from backend.core.logger import get_app_logger
from backend.database import InferenceService

logger = get_app_logger(__name__)


# 任务派发只认 inference_services.service_type（内容类型：image/video/audio/text/mini 等），
# 不回退到 type 列（type 表示引擎/实现，如 vllm、diffusers）。
_TEXT_SERVICE_TYPES = {"text", "llm", "chat"}
_AUDIO_SERVICE_TYPES = {"audio", "asr", "tts"}


class ModelNotAvailableError(RuntimeError):
    """没有找到可承载该 load_name 的 running 推理服务。"""


class DispatchSelectionError(RuntimeError):
    """统一 dispatch/router 无法选出目标服务。"""


@dataclass(frozen=True)
class DispatchSpec:
    """一次派发请求的最小执行规格。"""

    service_type: str
    require_supports_task: bool = False
    reason: str = ""
    load_name: str = ""
    # 音频子能力过滤：``tts`` / ``asr`` 等。非空时只保留声明了该 capability 的候选
    # （读库表 config.capabilities）。留空表示不做 capability 维度过滤。由请求侧
    # 根据 role 填入：chat TTS 走 "tts"、ASR 走 "asr"。
    capability: str = ""


def _service_type_for_model(load_name: str) -> str:
    """根据 load_name 粗略推断需要哪一类推理服务。

    目前 chat/LLM 场景都走 text 服务；音频类会话会走 audio。
    这里故意保守：未知时返回 ``"text"``。
    """
    n = (load_name or "").lower()
    if any(k in n for k in ("asr", "whisper", "tts", "voice", "audio")):
        return "audio"
    return "text"


def _service_type_aliases(service_type: str) -> Set[str]:
    normalized = str(service_type or "").strip().lower()
    if not normalized:
        return set()
    if normalized in _TEXT_SERVICE_TYPES:
        return set(_TEXT_SERVICE_TYPES)
    if normalized in _AUDIO_SERVICE_TYPES:
        return set(_AUDIO_SERVICE_TYPES)
    return {normalized}


def _normalized_service_type(service: Dict[str, Any]) -> str:
    """仅使用库表 ``service_type``（内容类型），与 ``tasks.type`` / 派发 ``task_type`` 对齐。"""
    return str(service.get("service_type") or "").strip().lower()


def _normalize_load_name(load_name: str) -> str:
    return str(load_name or "").strip().casefold()


def _supported_models(service: Dict[str, Any]) -> Set[str]:
    """服务声明的静态能力清单（同 family 下理论可运行的模型集合）。"""
    config = service.get("config") if isinstance(service.get("config"), dict) else {}
    raw_models = config.get("supported_models")
    if not isinstance(raw_models, list):
        return set()
    return {
        _normalize_load_name(item)
        for item in raw_models
        if _normalize_load_name(str(item))
    }


def _service_fixed_model(service: Dict[str, Any]) -> str:
    """读取服务 config.fixed_model（pin 模式）。非 pin 返回空串。"""
    config = service.get("config") if isinstance(service.get("config"), dict) else {}
    return str(config.get("fixed_model") or "").strip()


def _service_capabilities(service: Dict[str, Any]) -> Set[str]:
    """读取服务 ``config.capabilities``，返回规范化小写集合（可能为空集）。"""
    config = service.get("config") if isinstance(service.get("config"), dict) else {}
    raw = config.get("capabilities")
    if not isinstance(raw, list):
        return set()
    return {
        str(item).strip().lower()
        for item in raw
        if str(item).strip()
    }


def _effective_served_models(service: Dict[str, Any]) -> Set[str]:
    """当前实际在服务的模型集合。

    - pin 模式（``fixed_model`` 非空）：只服务 ``fixed_model``；
      ``supported_models`` 里其它模型虽然属于类能力，但这台实例当前不提供；
    - 非 pin 模式：等价于 ``supported_models``。
    """
    fixed = _service_fixed_model(service)
    if fixed:
        normalized = _normalize_load_name(fixed)
        return {normalized} if normalized else set()
    return _supported_models(service)


class DispatchRouter:
    """统一按 ``service_type`` 选择一台 running 推理服务实例。

    音频派发规则：
      - ASR (``capability='asr'``)：严格匹配。``load_name`` 非空时必须命中
        ``_effective_served_models``；为空时必须落到 pinned 服务。匹配不到直接失败。
      - TTS (``capability='tts'``)：放宽匹配。``load_name`` 仍优先匹配 pin/supported_models，
        但若没有任何服务声明它，则回落到"任意 tts capability 在线服务"——由推理侧 handler
        自行决定加载哪个具体权重（含 qwen-tts 的默认规则：有 instruct → VoiceDesign，
        否则 → CustomVoice）。``load_name`` 为空时也是同样的"先 pinned 后任意"。
        放宽的代价：调用方若指名了一个不在线的 load_name，不会在 dispatch 阶段失败，
        但会在推理侧得到 capability 不足的明确错误（例如 pin=Base 的服务被要求做
        voice_design 时）。
    """

    def list_services(
        self,
        spec: DispatchSpec,
        *,
        connected_service_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        wanted = _service_type_aliases(spec.service_type)
        requested_load_name = _normalize_load_name(spec.load_name)
        required_capability = str(spec.capability or "").strip().lower()
        audio_dispatch = bool(wanted & _AUDIO_SERVICE_TYPES)
        is_tts_dispatch = audio_dispatch and required_capability == "tts"
        connected_set = {
            str(item).strip() for item in (connected_service_ids or []) if str(item).strip()
        }

        audio_strict: List[Dict[str, Any]] = []
        audio_pinned: List[Dict[str, Any]] = []
        # 用作 tts 路径"放宽兜底"：所有已通过 service_type / capability 过滤的 audio 服务
        # （即都有 capability='tts'），不再要求 load_name 命中或 pin 存在。
        audio_capability_only: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        for svc in InferenceService.list_all():
            if (svc.get("status") or "").lower() != "running":
                continue
            service_id = str(svc.get("id") or "").strip()
            if connected_set and service_id not in connected_set:
                continue
            if spec.require_supports_task and not bool(svc.get("supports_task", True)):
                continue
            if _normalized_service_type(svc) not in wanted:
                continue
            # 音频子能力过滤：只在 spec 指定 capability 时生效。未声明 capabilities 的
            # audio 服务在注册期已被拒绝（sync_service_registration），理论上不会到这里。
            if audio_dispatch and required_capability:
                if required_capability not in _service_capabilities(svc):
                    continue
            if audio_dispatch:
                audio_capability_only.append(svc)
                if requested_load_name:
                    if requested_load_name in _effective_served_models(svc):
                        audio_strict.append(svc)
                else:
                    if _service_fixed_model(svc):
                        audio_pinned.append(svc)
                continue
            candidates.append(svc)

        if audio_dispatch:
            if requested_load_name:
                if audio_strict:
                    return audio_strict
                # tts 路径：模型没人声明也允许任意 tts 服务接单，让推理侧决定权重。
                # asr 路径维持严格匹配。
                if is_tts_dispatch and audio_capability_only:
                    logger.info(
                        "[dispatch] loose tts fallback: requested load_name=%r not declared by any "
                        "running service; falling back to any tts-capable service (count=%d)",
                        spec.load_name,
                        len(audio_capability_only),
                    )
                    return audio_capability_only
                return []
            # load_name 为空：优先 pinned；tts 路径下若无 pinned 也允许任意 tts 服务。
            if audio_pinned:
                return audio_pinned
            if is_tts_dispatch and audio_capability_only:
                logger.info(
                    "[dispatch] loose tts fallback: no pinned tts service; "
                    "falling back to any tts-capable service (count=%d)",
                    len(audio_capability_only),
                )
                return audio_capability_only
            return []
        return candidates

    def pick_service(
        self,
        spec: DispatchSpec,
        *,
        connected_service_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        candidates = self.list_services(spec, connected_service_ids=connected_service_ids)
        if not candidates:
            detail = spec.reason or f"service_type={spec.service_type}"
            audio_dispatch = bool(
                _service_type_aliases(spec.service_type) & _AUDIO_SERVICE_TYPES
            )
            if spec.capability:
                detail = f"{detail} capability={spec.capability}"
            if spec.load_name:
                if audio_dispatch:
                    detail = (
                        f"{detail} (requested load_name={spec.load_name} is not served by any "
                        "connected+running audio service; check supported_models / fixed_model / capabilities "
                        "on the online instances)"
                    )
                else:
                    detail = f"{detail} (requested load_name={spec.load_name})"
            elif audio_dispatch:
                if spec.capability:
                    detail = (
                        f"{detail} (no audio service declares config.fixed_model with capability="
                        f"{spec.capability!r}; pin an audio service that provides this capability)"
                    )
                else:
                    detail = (
                        f"{detail} (no audio service declares config.fixed_model; "
                        "caller must either set load_name explicitly, or pin an audio service)"
                    )
            raise DispatchSelectionError(
                f"No running inference service available for {detail}"
            )
        return random.choice(candidates)

    def try_pick_service(
        self,
        spec: DispatchSpec,
        *,
        connected_service_ids: Optional[Iterable[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            return self.pick_service(spec, connected_service_ids=connected_service_ids)
        except DispatchSelectionError:
            return None

    def pick_service_for_model(
        self,
        load_name: str,
        *,
        connected_service_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        target_type = _service_type_for_model(load_name)
        return self.pick_service(
            DispatchSpec(
                service_type=target_type,
                reason=f"service_type matching '{target_type}'",
                load_name=load_name,
            ),
            connected_service_ids=connected_service_ids,
        )


class LoadNameRouter:
    """按 load_name 选择可承载的推理服务。"""

    def __init__(self, *, dispatch_router: Optional[DispatchRouter] = None):
        self._dispatch_router = dispatch_router or get_dispatch_router()

    def list_services_for_model(
        self,
        load_name: str,
        *,
        connected_service_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        target_type = _service_type_for_model(load_name)
        return self._dispatch_router.list_services(
            DispatchSpec(
                service_type=target_type,
                reason=f"service_type matching '{target_type}'",
                load_name=load_name,
            ),
            connected_service_ids=connected_service_ids,
        )

    def pick_service(
        self,
        load_name: str,
        *,
        connected_service_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        try:
            return self._dispatch_router.pick_service_for_model(
                load_name,
                connected_service_ids=connected_service_ids,
            )
        except DispatchSelectionError as exc:
            raise ModelNotAvailableError(str(exc)) from exc

    def try_pick_service(
        self,
        load_name: str,
        *,
        connected_service_ids: Optional[Iterable[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            return self.pick_service(
                load_name,
                connected_service_ids=connected_service_ids,
            )
        except ModelNotAvailableError:
            return None


_dispatch_router: Optional[DispatchRouter] = None
_router: Optional[LoadNameRouter] = None


def get_dispatch_router() -> DispatchRouter:
    global _dispatch_router
    if _dispatch_router is None:
        _dispatch_router = DispatchRouter()
    return _dispatch_router


def get_load_name_router() -> LoadNameRouter:
    global _router
    if _router is None:
        _router = LoadNameRouter(dispatch_router=get_dispatch_router())
    return _router


__all__ = [
    "DispatchRouter",
    "DispatchSelectionError",
    "DispatchSpec",
    "LoadNameRouter",
    "ModelNotAvailableError",
    "get_dispatch_router",
    "get_load_name_router",
]
