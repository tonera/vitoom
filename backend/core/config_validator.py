"""
配置验证工具
用于验证配置的有效性
"""
import logging
from typing import Any, Dict, List, Tuple, Optional
from .config import get_config_manager, ConfigError

logger = logging.getLogger(__name__)


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        验证所有配置
        
        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        config = get_config_manager()
        
        # 验证各个配置节
        self._validate_server(config)
        self._validate_database(config)
        self._validate_logging(config)
        self._validate_models(config)
        self._validate_storage(config)
        self._validate_security(config)
        self._validate_i18n(config)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_server(self, config):
        """验证服务器配置"""
        server = config.get_section("server")
        
        # 验证端口
        port = server.get("port")
        if port is None:
            self.errors.append("server.port is required")
        elif not isinstance(port, int) or port < 1 or port > 65535:
            self.errors.append(f"server.port must be an integer between 1 and 65535, got: {port}")
        
        # 验证主机
        host = server.get("host")
        if host is None:
            self.errors.append("server.host is required")
        elif not isinstance(host, str):
            self.errors.append(f"server.host must be a string, got: {type(host)}")
        
        # 验证调试模式
        debug = server.get("debug")
        if debug is not None and not isinstance(debug, bool):
            self.warnings.append(f"server.debug should be a boolean, got: {type(debug)}")
    
    def _validate_database(self, config):
        """验证数据库配置"""
        database = config.get_section("database")
        
        url = database.get("url")
        if url and not isinstance(url, str):
            self.errors.append(f"database.url must be a string, got: {type(url)}")
    
    def _validate_logging(self, config):
        """验证日志配置"""
        logging_config = config.get_section("logging")
        
        level = logging_config.get("level")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level and level not in valid_levels:
            self.warnings.append(f"logging.level should be one of {valid_levels}, got: {level}")
    
    def _validate_models(self, config):
        """验证模型配置"""
        models = config.get_section("models")
        
        storage_path = models.get("storage_path")
        if storage_path and not isinstance(storage_path, str):
            self.errors.append(f"models.storage_path must be a string, got: {type(storage_path)}")
    
    def _validate_storage(self, config):
        """验证存储配置"""
        storage = config.get_section("storage")
        
        storage_mode = storage.get("default")
        valid_types = ["local", "s3", "oss"]
        if storage_mode and storage_mode not in valid_types:
            self.errors.append(f"storage.default must be one of {valid_types}, got: {storage_mode}")
        
        # 验证云存储配置（如果类型不是local）
        if storage_mode in ["s3", "oss"]:
            if storage_mode == "s3":
                s3_config = storage.get("s3", {})
                required_keys = ["bucket", "region", "access_key", "secret_key"]
                for key in required_keys:
                    if not s3_config.get(key):
                        self.errors.append(f"storage.s3.{key} is required when storage.default is s3")
            elif storage_mode == "oss":
                oss_config = storage.get("oss", {})
                required_keys = ["bucket", "endpoint", "access_key_id", "access_key_secret"]
                for key in required_keys:
                    if not oss_config.get(key):
                        self.errors.append(f"storage.oss.{key} is required when storage.default is oss")
    
    def _validate_security(self, config):
        """验证安全配置"""
        security = config.get_section("security")
        
        jwt = security.get("jwt", {})
        secret_key = jwt.get("secret_key")
        if not secret_key:
            self.warnings.append("security.jwt.secret_key is not set, JWT tokens may not be secure")
        
        access_token_expire = jwt.get("access_token_expire")
        if access_token_expire and (not isinstance(access_token_expire, int) or access_token_expire < 0):
            self.errors.append(f"security.jwt.access_token_expire must be a positive integer, got: {access_token_expire}")
    
    def _validate_i18n(self, config):
        """验证国际化配置"""
        i18n = config.get_section("i18n")
        
        default_language = i18n.get("default_language")
        supported_languages = i18n.get("supported_languages", [])
        
        if default_language and default_language not in supported_languages:
            self.errors.append(f"i18n.default_language '{default_language}' must be in supported_languages: {supported_languages}")
        
        if not supported_languages:
            self.warnings.append("i18n.supported_languages is empty")


def validate_config() -> Tuple[bool, List[str], List[str]]:
    """
    验证配置（便捷函数）
    
    Returns:
        (is_valid, errors, warnings)
    """
    validator = ConfigValidator()
    return validator.validate()


def check_config() -> bool:
    """
    检查配置是否有效（便捷函数）
    如果配置无效，会记录错误日志
    
    Returns:
        配置是否有效
    """
    is_valid, errors, warnings = validate_config()
    
    if warnings:
        for warning in warnings:
            logger.warning(f"Config warning: {warning}")
    
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        return False
    
    return True

