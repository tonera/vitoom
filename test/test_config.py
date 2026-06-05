"""
配置管理模块测试脚本
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.config import (
    get_config,
    get_section,
    get_config_manager,
    get_server_config,
    get_database_config,
    ConfigError,
)
from backend.core.config_validator import validate_config, check_config


def test_basic_config():
    """测试基本配置读取"""
    print("=" * 50)
    print("Testing Basic Config Access")
    print("=" * 50)
    
    # 测试获取配置值
    port = get_config("server.port")
    print(f"✓ Server port: {port}")
    assert port == 8888, f"Expected 8888, got {port}"
    
    host = get_config("server.host")
    print(f"✓ Server host: {host}")
    assert host == "0.0.0.0", f"Expected '0.0.0.0', got {host}"
    
    debug = get_config("server.debug")
    print(f"✓ Server debug: {debug}")
    assert debug == False, f"Expected False, got {debug}"
    
    # 测试获取配置节
    server_config = get_server_config()
    print(f"✓ Server config section: {list(server_config.keys())}")
    assert "port" in server_config, "Server config should contain 'port'"
    
    # 测试默认值
    non_existent = get_config("non.existent.key", "default_value")
    print(f"✓ Non-existent key with default: {non_existent}")
    assert non_existent == "default_value", "Should return default value"
    
    print("Basic config test passed!\n")


def test_nested_config():
    """测试嵌套配置"""
    print("=" * 50)
    print("Testing Nested Config")
    print("=" * 50)
    
    # 测试嵌套配置访问
    log_level = get_config("logging.level")
    print(f"✓ Log level: {log_level}")
    
    log_file_path = get_config("logging.file.path")
    print(f"✓ Log file path: {log_file_path}")
    
    max_bytes = get_config("logging.file.max_bytes")
    print(f"✓ Log max bytes: {max_bytes}")
    
    # 测试获取整个节
    logging_config = get_section("logging")
    print(f"✓ Logging config keys: {list(logging_config.keys())}")
    assert "level" in logging_config, "Logging config should contain 'level'"
    assert "file" in logging_config, "Logging config should contain 'file'"
    
    print("Nested config test passed!\n")


def test_config_sections():
    """测试配置节访问"""
    print("=" * 50)
    print("Testing Config Sections")
    print("=" * 50)
    
    # 测试各个配置节
    sections = [
        ("server", get_server_config),
        ("database", get_database_config),
    ]
    
    for section_name, get_func in sections:
        section = get_func()
        print(f"✓ {section_name} config: {len(section)} keys")
        assert isinstance(section, dict), f"{section_name} should be a dict"
    
    print("Config sections test passed!\n")


def test_env_override():
    """测试环境变量覆盖"""
    print("=" * 50)
    print("Testing Environment Variable Override")
    print("=" * 50)
    
    # 设置环境变量
    original_port = get_config("server.port")
    print(f"Original port: {original_port}")
    
    os.environ["VITOOM_SERVER_PORT"] = "9999"
    os.environ["VITOOM_INFERENCE_UPLOAD_AUTH_SECRET"] = "test-upload-secret"
    os.environ["DEBUG"] = "true"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
    
    # 重新加载配置
    config_manager = get_config_manager()
    config_manager.reload()
    
    # 检查环境变量是否生效
    new_port = get_config("server.port")
    print(f"Port after env override: {new_port}")
    assert new_port == 9999, f"Expected 9999 from env, got {new_port}"
    
    debug = get_config("server.debug")
    print(f"Debug after env override: {debug}")
    assert debug == True, f"Expected True from env, got {debug}"
    
    db_url = get_config("database.url")
    print(f"Database URL from env: {db_url}")
    assert db_url == "postgresql://test:test@localhost/test", "Database URL should be set from env"

    upload_auth_secret = get_config("inference.upload_auth_secret")
    print(f"Inference upload auth secret from env: {upload_auth_secret}")
    assert upload_auth_secret == "test-upload-secret", "Inference upload auth secret should be set from env"
    
    # 清理环境变量
    del os.environ["VITOOM_SERVER_PORT"]
    del os.environ["VITOOM_INFERENCE_UPLOAD_AUTH_SECRET"]
    del os.environ["DEBUG"]
    del os.environ["DATABASE_URL"]
    
    # 重新加载配置
    config_manager.reload()
    
    # 验证恢复
    restored_port = get_config("server.port")
    print(f"Port after cleanup: {restored_port}")
    assert restored_port == original_port, "Port should be restored"
    
    print("Environment variable override test passed!\n")


def test_config_validation():
    """测试配置验证"""
    print("=" * 50)
    print("Testing Config Validation")
    print("=" * 50)
    
    is_valid, errors, warnings = validate_config()
    
    print(f"Config valid: {is_valid}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    
    # 配置应该是有效的（使用默认配置）
    assert is_valid, f"Config should be valid, but got errors: {errors}"
    
    print("Config validation test passed!\n")


def test_config_manager():
    """测试配置管理器"""
    print("=" * 50)
    print("Testing Config Manager")
    print("=" * 50)
    
    config_manager = get_config_manager()
    
    # 测试设置配置值
    config_manager.set("test.key", "test_value")
    value = config_manager.get("test.key")
    print(f"✓ Set and get config: {value}")
    assert value == "test_value", "Should get the value we set"
    
    # 测试获取完整配置
    full_config = config_manager.to_dict()
    print(f"✓ Full config keys: {len(full_config)} top-level keys")
    assert isinstance(full_config, dict), "Should return a dict"
    
    print("Config manager test passed!\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("Configuration Module Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_basic_config()
        test_nested_config()
        test_config_sections()
        test_env_override()
        test_config_validation()
        test_config_manager()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

