"""
错误处理模块测试脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.exceptions import (
    BaseAppException,
    AuthRequiredException,
    UserNotFoundException,
    InvalidParameterException,
    MissingParameterException,
    ResourceNotFoundException,
    TaskNotFoundException,
    ModelNotFoundException,
    ModelLoadFailedException,
    TimeoutException,
    FileNotFoundException,
)
from backend.core.error_codes import ErrorCode, get_http_status, get_error_type
from backend.core.error_handler import create_error_response


def test_error_codes():
    """测试错误码"""
    print("=" * 50)
    print("Testing Error Codes")
    print("=" * 50)
    
    # 测试错误码值
    assert ErrorCode.AUTH_REQUIRED == 2000
    assert ErrorCode.USER_NOT_FOUND == 2005
    assert ErrorCode.INVALID_PARAMETER == 3000
    assert ErrorCode.MODEL_NOT_FOUND == 6000
    
    # 测试HTTP状态码映射
    assert get_http_status(ErrorCode.AUTH_REQUIRED) == 401
    assert get_http_status(ErrorCode.NOT_FOUND) == 404
    assert get_http_status(ErrorCode.INTERNAL_ERROR) == 500
    
    # 测试错误类型
    assert get_error_type(ErrorCode.AUTH_REQUIRED) == "auth_error"
    assert get_error_type(ErrorCode.INVALID_PARAMETER) == "user_error"
    assert get_error_type(ErrorCode.MODEL_NOT_FOUND) == "model_error"
    
    print("✓ Error codes test passed!\n")


def test_exceptions():
    """测试异常类"""
    print("=" * 50)
    print("Testing Exception Classes")
    print("=" * 50)
    
    # 测试认证异常
    auth_exc = AuthRequiredException()
    assert auth_exc.error_code == ErrorCode.AUTH_REQUIRED
    assert auth_exc.http_status == 401
    assert auth_exc.error_type == "auth_error"
    print(f"✓ Auth exception: {auth_exc.message}")
    
    # 测试用户不存在异常
    user_exc = UserNotFoundException("user_123")
    assert user_exc.error_code == ErrorCode.USER_NOT_FOUND
    assert "user_123" in user_exc.details.get("user_id", "")
    print(f"✓ User not found exception: {user_exc.message}")
    
    # 测试参数异常
    param_exc = InvalidParameterException("task_id")
    assert param_exc.error_code == ErrorCode.INVALID_PARAMETER
    assert param_exc.details.get("parameter") == "task_id"
    print(f"✓ Invalid parameter exception: {param_exc.message}")
    
    # 测试资源不存在异常
    resource_exc = ResourceNotFoundException("task", "task_123")
    assert resource_exc.error_code == ErrorCode.RESOURCE_NOT_FOUND
    assert resource_exc.details.get("resource_type") == "task"
    print(f"✓ Resource not found exception: {resource_exc.message}")
    
    # 测试任务异常
    task_exc = TaskNotFoundException("task_123")
    assert task_exc.error_code == ErrorCode.TASK_NOT_FOUND
    print(f"✓ Task not found exception: {task_exc.message}")
    
    # 测试模型异常
    model_exc = ModelNotFoundException("model_123")
    assert model_exc.error_code == ErrorCode.MODEL_NOT_FOUND
    print(f"✓ Model not found exception: {model_exc.message}")
    
    # 测试超时异常
    timeout_exc = TimeoutException(30)
    assert timeout_exc.error_code == ErrorCode.TIMEOUT_ERROR
    assert timeout_exc.details.get("timeout") == 30
    print(f"✓ Timeout exception: {timeout_exc.message}")
    
    print("Exception classes test passed!\n")


def test_exception_to_dict():
    """测试异常转字典"""
    print("=" * 50)
    print("Testing Exception to Dict")
    print("=" * 50)
    
    exc = UserNotFoundException("user_123", "Custom message")
    exc_dict = exc.to_dict()
    
    assert exc_dict["error_code"] == ErrorCode.USER_NOT_FOUND.value
    assert exc_dict["error_type"] == "auth_error"
    assert exc_dict["message"] == "Custom message"
    assert exc_dict["http_status"] == 404
    assert "user_id" in exc_dict.get("details", {})
    
    print(f"✓ Exception dict: {exc_dict}")
    print("Exception to dict test passed!\n")


def test_error_response():
    """测试错误响应创建"""
    print("=" * 50)
    print("Testing Error Response Creation")
    print("=" * 50)
    
    response = create_error_response(
        error_code=ErrorCode.USER_NOT_FOUND,
        message="User not found",
        details={"user_id": "user_123"}
    )
    
    assert response["success"] == False
    assert "error" in response
    assert response["error"]["code"] == ErrorCode.USER_NOT_FOUND.value
    assert response["error"]["message"] == "User not found"
    assert response["error"]["details"]["user_id"] == "user_123"
    
    print(f"✓ Error response: {response}")
    print("Error response creation test passed!\n")


def test_exception_chaining():
    """测试异常链"""
    print("=" * 50)
    print("Testing Exception Chaining")
    print("=" * 50)
    
    from backend.core.error_codes import ErrorCode
    
    try:
        raise ValueError("Original error")
    except ValueError as e:
        # 使用BaseAppException测试异常链
        exc = BaseAppException(
            ErrorCode.MODEL_LOAD_FAILED,
            "Model load failed",
            details={"model_id": "model_123"},
            cause=e
        )
        assert exc.cause == e
        print(f"✓ Exception chaining: {exc.message}, cause: {exc.cause}")
    
    print("Exception chaining test passed!\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("Error Handling Module Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_error_codes()
        test_exceptions()
        test_exception_to_dict()
        test_error_response()
        test_exception_chaining()
        
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

