"""
启动配置加载模块
读取 config/{service_id}.yaml 配置文件
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .logger import get_logger

logger = get_logger(__name__)


class StartupConfig:
    """启动配置类"""
    
    def __init__(self, config_dict: Dict[str, Any], inference_config: 'InferenceConfig'):
        """
        初始化启动配置
        
        Args:
            config_dict: 配置字典
            inference_config: 全局推理配置
        """
        self.service_id: str = config_dict.get("service_id", "")
        self.service_type: str = config_dict.get("service_type", "")  # image/video/audio/text
        self.port: Optional[int] = config_dict.get("port")
        self.host: str = config_dict.get("host", "127.0.0.1")
        self.name: str = config_dict.get("name", "")
        self.type: str = config_dict.get("type", "")  # vllm/ollama/diffusers等
        self.config: Dict[str, Any] = config_dict.get("config", {})
        
        # 从全局配置读取
        self.api_base_url: str = inference_config.api_base_url
        self.ws_url: str = inference_config.ws_url

        # 直接暴露全局配置，便于 download 等新 worker 使用 models_dir 等信息
        self.inference_config: 'InferenceConfig' = inference_config
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "service_id": self.service_id,
            "service_type": self.service_type,
            "port": self.port,
            "host": self.host,
            "name": self.name,
            "type": self.type,
            "config": self.config,
            "api_base_url": self.api_base_url,
            "ws_url": self.ws_url,
            "models_dir": getattr(self.inference_config, "models_dir", None),
        }


def load_startup_config(service_id: str) -> StartupConfig:
    """
    加载启动配置文件
    
    Args:
        service_id: 服务ID
    
    Returns:
        StartupConfig对象
    
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误
    """
    # 加载全局配置（可被 service 配置覆盖：api_base_url/ws_url/transport）
    inference_config = load_inference_config(service_id=service_id)
    
    # 获取配置文件路径
    config_dir = Path(__file__).parent.parent / "config"
    config_file = config_dir / f"{service_id}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Startup config file not found: {config_file}")
    
    logger.info(f"Loading startup config from: {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict:
            raise ValueError("Config file is empty")
        
        # 确保service_id一致
        if "service_id" not in config_dict:
            config_dict["service_id"] = service_id
        elif config_dict["service_id"] != service_id:
            logger.warning(
                f"service_id mismatch: config has '{config_dict['service_id']}', "
                f"expected '{service_id}'. Using provided service_id."
            )
            config_dict["service_id"] = service_id
        
        config = StartupConfig(config_dict, inference_config)
        logger.info(f"Startup config loaded successfully: service_id={config.service_id}, type={config.type}, service_type={config.service_type}")
        
        return config
    
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in config file: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load config file: {e}")


class InferenceConfig:
    """推理服务全局配置类"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化推理配置
        
        Args:
            config_dict: 配置字典
        """
        # 目录类配置统一解析为绝对路径：
        # - 若为绝对路径：保持不变
        # - 若为相对路径：以项目根目录为基准，避免受当前工作目录影响
        repo_root = Path(__file__).resolve().parents[2]

        def _resolve_dir(p: Optional[str], default: str) -> str:
            raw = str(p) if p not in (None, "") else default
            path = Path(raw)
            if path.is_absolute():
                return str(path)
            return str((repo_root / path).resolve())

        self.models_dir: str = _resolve_dir(config_dict.get("models_dir"), "resources/models")
        # 权重/LoRA 目录（用于量化权重、LoRA 等资源定位）
        self.weights_dir: str = _resolve_dir(config_dict.get("weights_dir"), "resources/weights")
        self.loras_dir: str = _resolve_dir(config_dict.get("loras_dir"), "resources/loras")
        # 统一输出目录，不区分类型
        self.outputs_dir: str = _resolve_dir(config_dict.get("outputs_dir"), "resources/outputs")

        # ========== pipeline 缓存 ==========
        # 为空或 0：每次任务结束释放；>0：启用 LRU=1 pipeline 缓存并按 TTL 驱逐
        try:
            self.pipeline_cache_ttl_seconds: int = int(config_dict.get("pipeline_cache_ttl_seconds") or 0)
        except Exception:
            self.pipeline_cache_ttl_seconds = 0

        # ========== 存储相关配置 ==========
        # storage 值域固定：local | server | s3 | oss
        storage_cfg: Dict[str, Any] = config_dict.get("storage", {}) if isinstance(config_dict.get("storage", {}), dict) else {}
        self.storage_default: str = storage_cfg.get("default", "local")

        # server 直传配置（推理侧只负责上传调用，后端实现另写）
        server_cfg: Dict[str, Any] = storage_cfg.get("server", {}) if isinstance(storage_cfg.get("server", {}), dict) else {}
        self.server_upload_path: str = server_cfg.get("upload_path", "/api/inference/upload")
        self.server_upload_timeout_seconds: float = float(server_cfg.get("timeout_seconds", 60.0))
        # server 上传签权（可选）：用于给 /api/inference/upload 增加最基本的鉴权
        # 约定：
        # - 未配置/为空：不启用签权（保持历史兼容）
        # - 配置后：推理侧会在上传请求中加入签名 header（后端需使用同一 secret 校验）
        secret = ""
        try:
            auth_cfg = server_cfg.get("auth", {}) if isinstance(server_cfg.get("auth", {}), dict) else {}
            secret = str(auth_cfg.get("secret") or "")
        except Exception:
            secret = ""
        if not secret:
            # 兼容扁平字段名
            secret = str(server_cfg.get("upload_auth_secret") or server_cfg.get("auth_secret") or "")
        self.server_upload_auth_secret: str = secret.strip()

        # s3 直传配置（AK/SK 从配置读取）
        s3_cfg: Dict[str, Any] = storage_cfg.get("s3", {}) if isinstance(storage_cfg.get("s3", {}), dict) else {}
        self.s3_endpoint: Optional[str] = s3_cfg.get("endpoint")
        self.s3_region: Optional[str] = s3_cfg.get("region")
        self.s3_bucket: str = s3_cfg.get("bucket", "")
        self.s3_access_key_id: str = s3_cfg.get("access_key_id", "")
        self.s3_secret_access_key: str = s3_cfg.get("secret_access_key", "")
        self.s3_public_base_url: Optional[str] = s3_cfg.get("public_base_url")

        # oss 直传配置（AK/SK 从配置读取）
        oss_cfg: Dict[str, Any] = storage_cfg.get("oss", {}) if isinstance(storage_cfg.get("oss", {}), dict) else {}
        self.oss_endpoint: str = oss_cfg.get("endpoint", "")
        self.oss_bucket: str = oss_cfg.get("bucket", "")
        self.oss_access_key_id: str = oss_cfg.get("access_key_id", "")
        self.oss_access_key_secret: str = oss_cfg.get("access_key_secret", "")
        self.oss_public_base_url: Optional[str] = oss_cfg.get("public_base_url")
        
        # API后端配置
        self.api_base_url: str = config_dict.get("api_base_url", "http://127.0.0.1:8888")
        
        # WebSocket Server配置
        self.ws_url: str = config_dict.get("ws_url", "ws://127.0.0.1:8888")

        # Backend 访问本推理容器 Supervisor Agent 的地址。
        # 多机部署时必须是 Backend 可达的地址，而不是浏览器地址或推理侧连接 Backend 的地址。
        self.supervisor_url: str = str(config_dict.get("supervisor_url") or "").strip().rstrip("/")

        # ========== 消息传输（Ingress/Egress）==========
        # 可选配置。若未配置 transport，则默认使用 WS（保持历史兼容）。
        transport_cfg = config_dict.get("transport", {})
        self.transport: Dict[str, Any] = transport_cfg if isinstance(transport_cfg, dict) else {}


_inference_config_cache: Dict[str, 'InferenceConfig'] = {}


def _infer_service_id(service_id: Optional[str]) -> Optional[str]:
    sid = (service_id or "").strip()
    if sid:
        return sid
    sid = str(os.environ.get("VITOOM_SERVICE_ID") or "").strip()
    return sid or None


def _apply_service_overrides(base_cfg: Dict[str, Any], service_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 inference.yaml 与 {service_id}.yaml 进行合并：
    - 始终使用 inference.yaml 作为 base
    - service 配置对 base 做覆盖（同名键覆盖）
    - dict：递归合并；list/标量：直接覆盖
    - 默认忽略值为 None 的覆盖（避免把有效值覆盖成 null）
    """
    def _deep_merge(a: Any, b: Any) -> Any:
        if b is None:
            return a
        if isinstance(a, dict) and isinstance(b, dict):
            out: Dict[str, Any] = dict(a)
            for k, bv in b.items():
                av = out.get(k)
                out[k] = _deep_merge(av, bv)
            return out
        # list / 标量：直接覆盖
        return b

    return _deep_merge(base_cfg or {}, service_cfg or {})  # type: ignore[return-value]


def _read_yaml_dict(path: Path) -> Dict[str, Any]:
    """
    读取 YAML 文件为 dict。
    - 文件不存在：返回 {}
    - 空文件：返回 {}
    - 非 dict 顶层：返回 {}
    - YAML 解析异常：抛出（由上层统一兜底并打日志）
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _build_inference_config_dict(*, service_id: Optional[str]) -> Dict[str, Any]:
    """
    构建最终用于 InferenceConfig 的配置 dict：
    inference.yaml 作为 base，{service_id}.yaml 作为覆盖层（深度合并）。
    """
    config_dir = Path(__file__).parent.parent / "config"
    base_path = config_dir / "inference.yaml"
    base_cfg = _read_yaml_dict(base_path)

    sid = _infer_service_id(service_id)
    if not sid:
        return base_cfg

    service_path = config_dir / f"{sid}.yaml"
    service_cfg = _read_yaml_dict(service_path)
    return _apply_service_overrides(base_cfg, service_cfg)


def load_inference_config(service_id: Optional[str] = None) -> InferenceConfig:
    """
    加载推理服务全局配置文件
    
    Returns:
        InferenceConfig对象
    
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误
    """
    global _inference_config_cache

    sid = _infer_service_id(service_id)
    cache_key = sid or "__global__"

    # 使用缓存，避免重复加载
    cached = _inference_config_cache.get(cache_key)
    if cached is not None:
        return cached

    config_dir = Path(__file__).parent.parent / "config"
    base_path = config_dir / "inference.yaml"
    logger.info(f"Loading inference config from: {base_path}" + (f" (service_id={sid})" if sid else ""))

    try:
        config_dict = _build_inference_config_dict(service_id=service_id)
        if not base_path.exists():
            logger.warning(f"Inference config file not found: {base_path}, using defaults")
        cfg = InferenceConfig(config_dict)
        _inference_config_cache[cache_key] = cfg
        logger.info(
            f"Inference config loaded: models_dir={cfg.models_dir}, "
            f"outputs_dir={cfg.outputs_dir}, "
            f"api_base_url={cfg.api_base_url}, "
            f"ws_url={cfg.ws_url}"
        )
        return cfg
    
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML format in inference config file: {e}, using defaults")
        cfg = InferenceConfig({})
        _inference_config_cache[cache_key] = cfg
        return cfg
    except Exception as e:
        logger.error(f"Failed to load inference config file: {e}, using defaults", exc_info=True)
        cfg = InferenceConfig({})
        _inference_config_cache[cache_key] = cfg
        return cfg

