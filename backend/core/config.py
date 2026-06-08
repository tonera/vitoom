"""
配置管理模块
支持YAML配置文件、环境变量，实现配置优先级处理
"""
import os
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import lru_cache

logger = logging.getLogger(__name__)

# 配置目录路径
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "default.yaml"
APP_CONFIG_FILE = CONFIG_DIR / "app.yaml"

class ConfigError(Exception):
    """配置错误异常"""
    pass


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """加载配置（优先级：环境变量 > 用户配置 > 默认配置）"""
        # 1. 加载默认配置
        default_config = self._load_yaml_file(DEFAULT_CONFIG_FILE)
        if not default_config:
            raise ConfigError(f"Failed to load default config from {DEFAULT_CONFIG_FILE}")
        
        # 2. 加载用户配置文件（如果存在）
        app_config = self._load_yaml_file(APP_CONFIG_FILE) or {}
        
        # 3. 合并配置（用户配置覆盖默认配置）
        self._config = self._deep_merge(default_config, app_config)
        
        # 4. 应用环境变量（环境变量覆盖所有配置）
        self._apply_env_vars()
        
        # 5. 处理向后兼容和规范化
        self._normalize_config()
        
        logger.info(f"Configuration loaded from {CONFIG_DIR}")
    
    def _load_yaml_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """加载YAML配置文件"""
        if not file_path.exists():
            logger.debug(f"Config file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}", exc_info=True)
            return None
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_vars(self):
        """应用环境变量覆盖配置"""
        # 环境变量命名规则：VITOOM_<SECTION>_<KEY>（大写）
        # 例如：VITOOM_SERVER_PORT=9999
        
        env_prefix = "VITOOM_"
        
        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue
            
            # 移除前缀并转换为配置路径
            config_key = key[len(env_prefix):].lower()
            
            # 支持嵌套配置，例如：VITOOM_SERVER_PORT -> server.port
            # 但环境变量使用下划线，配置使用点号
            parts = config_key.split('_')
            
            # 尝试解析为嵌套配置
            if len(parts) >= 2:
                section = parts[0]
                sub_key = '_'.join(parts[1:])
                
                if section in self._config and isinstance(self._config[section], dict):
                    # 尝试转换值类型
                    typed_value = self._convert_type(value, self._config[section].get(sub_key))
                    self._config[section][sub_key] = typed_value
                    logger.debug(f"Set {section}.{sub_key} = {typed_value} from env {key}")
        
        # 特殊处理一些常用的环境变量
        self._apply_special_env_vars()
    
    def _apply_special_env_vars(self):
        """应用特殊环境变量"""
        # VITOOM_BACKEND_URL：部署统一变量（.env / compose），映射到 server.public_base_url
        backend_url = os.environ.get("VITOOM_BACKEND_URL", "").strip().rstrip("/")
        if backend_url:
            server = self._config.setdefault("server", {})
            if not isinstance(server, dict):
                server = {}
                self._config["server"] = server
            server["public_base_url"] = backend_url
            logger.info("server.public_base_url set from VITOOM_BACKEND_URL environment variable")

        # DATABASE_URL (ignore empty values injected by compose when unset in .env)
        database_url = os.environ.get("DATABASE_URL", "").strip()
        if database_url:
            if "database" not in self._config:
                self._config["database"] = {}
            self._config["database"]["url"] = database_url
            logger.info("Database URL set from DATABASE_URL environment variable")
        
        # LOG_LEVEL
        if "LOG_LEVEL" in os.environ:
            if "logging" not in self._config:
                self._config["logging"] = {}
            self._config["logging"]["level"] = os.environ["LOG_LEVEL"]
            logger.info(f"Log level set to {os.environ['LOG_LEVEL']} from LOG_LEVEL environment variable")
        
        # DEBUG
        if "DEBUG" in os.environ:
            debug_value = os.environ["DEBUG"].lower() in ("true", "1", "yes")
            if "server" not in self._config:
                self._config["server"] = {}
            self._config["server"]["debug"] = debug_value
            logger.info(f"Debug mode set to {debug_value} from DEBUG environment variable")

        # Knowledge base Elasticsearch settings. These names avoid ambiguity with
        # the generic VITOOM_<SECTION>_<KEY> parser because the top-level key is
        # "knowledge_base".
        kb_es_url = os.environ.get("KNOWLEDGE_BASE_ES_URL")
        kb_es_username = os.environ.get("KNOWLEDGE_BASE_ES_USERNAME")
        kb_es_password = os.environ.get("KNOWLEDGE_BASE_ES_PASSWORD")
        if any(value is not None for value in (kb_es_url, kb_es_username, kb_es_password)):
            knowledge_base = self._config.setdefault("knowledge_base", {})
            if not isinstance(knowledge_base, dict):
                knowledge_base = {}
                self._config["knowledge_base"] = knowledge_base
            es_config = knowledge_base.setdefault("es", {})
            if not isinstance(es_config, dict):
                es_config = {}
                knowledge_base["es"] = es_config
            if kb_es_url is not None:
                es_config["url"] = kb_es_url
                logger.info("Knowledge base ES URL set from KNOWLEDGE_BASE_ES_URL")
            if kb_es_username is not None:
                es_config["username"] = kb_es_username
            if kb_es_password is not None:
                es_config["password"] = kb_es_password

        # Optional OpenClaw integration. OpenClaw keeps its own IM/model/channel
        # configuration; Vitoom only needs the gateway connection settings.
        openclaw_enabled = os.environ.get("OPENCLAW_ENABLED")
        openclaw_base_url = os.environ.get("OPENCLAW_BASE_URL")
        openclaw_token = os.environ.get("OPENCLAW_TOKEN")
        if any(value is not None for value in (openclaw_enabled, openclaw_base_url, openclaw_token)):
            agents = self._config.setdefault("agents", {})
            if not isinstance(agents, dict):
                agents = {}
                self._config["agents"] = agents
            openclaw = agents.setdefault("openclaw", {})
            if not isinstance(openclaw, dict):
                openclaw = {}
                agents["openclaw"] = openclaw
            if openclaw_enabled is not None:
                openclaw["enabled"] = openclaw_enabled.strip().lower() in ("true", "1", "yes", "on")
                logger.info("OpenClaw enabled flag set from OPENCLAW_ENABLED")
            if openclaw_base_url is not None:
                openclaw["base_url"] = openclaw_base_url
                logger.info("OpenClaw base URL set from OPENCLAW_BASE_URL")
            if openclaw_token is not None:
                openclaw["token"] = openclaw_token
    
    def _normalize_config(self):
        """规范化配置（处理向后兼容等）"""
        # storage 配置已统一为 storage.default（不做向后兼容）
    
    def _convert_type(self, value: str, default_value: Any = None) -> Any:
        """转换环境变量值为合适的类型"""
        if default_value is None:
            # 尝试推断类型
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        
        # 根据默认值类型转换
        if isinstance(default_value, bool):
            return value.lower() in ("true", "1", "yes")
        elif isinstance(default_value, int):
            try:
                return int(value)
            except ValueError:
                return default_value
        elif isinstance(default_value, float):
            try:
                return float(value)
            except ValueError:
                return default_value
        elif isinstance(default_value, list):
            # 尝试解析为JSON列表
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # 尝试按逗号分割
                return [item.strip() for item in value.split(",")]
        elif isinstance(default_value, dict):
            # 尝试解析为JSON字典
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        支持点号分隔的嵌套键，例如：server.port
        
        Args:
            key: 配置键，支持嵌套，如 "server.port"
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置值（仅内存中，不会保存到文件）
        支持点号分隔的嵌套键
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取整个配置节"""
        return self._config.get(section, {})
    
    def reload(self):
        """重新加载配置"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def to_dict(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return self._config.copy()
    
    def save_app_config(self, config: Dict[str, Any]):
        """保存应用配置到app.yaml"""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(APP_CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            logger.info(f"App config saved to {APP_CONFIG_FILE}")
            # 重新加载配置
            self.reload()
        except Exception as e:
            logger.error(f"Failed to save app config: {e}", exc_info=True)
            raise ConfigError(f"Failed to save app config: {e}")


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


@lru_cache(maxsize=1)
def get_config_manager() -> ConfigManager:
    """获取配置管理器实例（单例）"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """
    获取配置值（便捷函数）
    
    Args:
        key: 配置键，支持嵌套，如 "server.port"
        default: 默认值
    
    Returns:
        配置值
    
    Example:
        >>> port = get_config("server.port", 8888)
        >>> debug = get_config("server.debug", False)
    """
    return get_config_manager().get(key, default)


def get_section(section: str) -> Dict[str, Any]:
    """
    获取配置节（便捷函数）
    
    Args:
        section: 配置节名称，如 "server"
    
    Returns:
        配置节字典
    
    Example:
        >>> server_config = get_section("server")
    """
    return get_config_manager().get_section(section)


def reload_config():
    """重新加载配置（便捷函数）"""
    get_config_manager().reload()


# 常用配置的便捷访问函数
def get_server_config() -> Dict[str, Any]:
    """获取服务器配置"""
    return get_section("server")


def get_database_config() -> Dict[str, Any]:
    """获取数据库配置"""
    return get_section("database")


def get_logging_config() -> Dict[str, Any]:
    """获取日志配置"""
    return get_section("logging")


def get_models_config() -> Dict[str, Any]:
    """获取模型配置"""
    return get_section("models")


def get_storage_config() -> Dict[str, Any]:
    """获取存储配置"""
    return get_section("storage")


def get_i18n_config() -> Dict[str, Any]:
    """获取国际化配置"""
    return get_section("i18n")


def get_security_config() -> Dict[str, Any]:
    """获取安全配置"""
    return get_section("security")


