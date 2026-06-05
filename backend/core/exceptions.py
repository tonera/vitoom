"""
异常基类和具体异常类
"""
from typing import Optional, Dict, Any
from .error_codes import ErrorCode, get_http_status, get_error_type


class BaseAppException(Exception):
    """应用异常基类"""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        初始化异常
        
        Args:
            error_code: 错误码
            message: 错误消息
            details: 错误详情（可选）
            cause: 原始异常（可选）
        """
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.error_type = get_error_type(error_code)
        self.http_status = get_http_status(error_code)
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "error_code": self.error_code.value,
            "error_type": self.error_type,
            "message": self.message,
            "http_status": self.http_status,
        }
        
        if self.details:
            result["details"] = self.details
        
        return result


# ==================== 认证相关异常 ====================

class AuthException(BaseAppException):
    """认证异常基类"""
    pass


class AuthRequiredException(AuthException):
    """需要认证"""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(ErrorCode.AUTH_REQUIRED, message)


class AuthFailedException(AuthException):
    """认证失败"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(ErrorCode.AUTH_FAILED, message)


class InvalidTokenException(AuthException):
    """无效的Token"""
    def __init__(self, message: str = "Invalid token"):
        super().__init__(ErrorCode.INVALID_TOKEN, message)


class TokenExpiredException(AuthException):
    """Token已过期"""
    def __init__(self, message: str = "Token expired"):
        super().__init__(ErrorCode.TOKEN_EXPIRED, message)


class PermissionDeniedException(AuthException):
    """权限不足"""
    def __init__(self, message: str = "Permission denied"):
        super().__init__(ErrorCode.PERMISSION_DENIED, message)


class UserNotFoundException(AuthException):
    """用户不存在"""
    def __init__(self, user_id: Optional[str] = None, message: Optional[str] = None):
        if message is None:
            message = f"User not found" + (f": {user_id}" if user_id else "")
        super().__init__(ErrorCode.USER_NOT_FOUND, message, details={"user_id": user_id} if user_id else {})


class UserAlreadyExistsException(AuthException):
    """用户已存在"""
    def __init__(self, email: Optional[str] = None, message: Optional[str] = None):
        if message is None:
            message = f"User already exists" + (f": {email}" if email else "")
        super().__init__(ErrorCode.USER_ALREADY_EXISTS, message, details={"email": email} if email else {})


class InvalidCredentialsException(AuthException):
    """无效的凭据"""
    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(ErrorCode.INVALID_CREDENTIALS, message)


# ==================== 参数相关异常 ====================

class ParameterException(BaseAppException):
    """参数异常基类"""
    pass


class InvalidParameterException(ParameterException):
    """无效参数"""
    def __init__(self, parameter: str, message: Optional[str] = None):
        if message is None:
            message = f"Invalid parameter: {parameter}"
        super().__init__(ErrorCode.INVALID_PARAMETER, message, details={"parameter": parameter})


class MissingParameterException(ParameterException):
    """缺少必需参数"""
    def __init__(self, parameter: str, message: Optional[str] = None):
        if message is None:
            message = f"Missing required parameter: {parameter}"
        super().__init__(ErrorCode.MISSING_PARAMETER, message, details={"parameter": parameter})


class InvalidFormatException(ParameterException):
    """无效格式"""
    def __init__(self, field: str, expected_format: str, message: Optional[str] = None):
        if message is None:
            message = f"Invalid format for {field}, expected: {expected_format}"
        super().__init__(
            ErrorCode.INVALID_FORMAT,
            message,
            details={"field": field, "expected_format": expected_format}
        )


class FileTooLargeException(ParameterException):
    """文件过大"""
    def __init__(self, max_size: int, message: Optional[str] = None):
        if message is None:
            message = f"File too large, maximum size: {max_size} bytes"
        super().__init__(ErrorCode.FILE_TOO_LARGE, message, details={"max_size": max_size})


class InvalidFileTypeException(ParameterException):
    """无效的文件类型"""
    def __init__(self, file_type: str, allowed_types: list, message: Optional[str] = None):
        if message is None:
            message = f"Invalid file type: {file_type}, allowed types: {allowed_types}"
        super().__init__(
            ErrorCode.INVALID_FILE_TYPE,
            message,
            details={"file_type": file_type, "allowed_types": allowed_types}
        )


# ==================== 资源相关异常 ====================

class ResourceException(BaseAppException):
    """资源异常基类"""
    pass


class ResourceNotFoundException(ResourceException):
    """资源不存在"""
    def __init__(self, resource_type: str, resource_id: str, message: Optional[str] = None):
        if message is None:
            message = f"{resource_type} not found: {resource_id}"
        super().__init__(
            ErrorCode.RESOURCE_NOT_FOUND,
            message,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ResourceAlreadyExistsException(ResourceException):
    """资源已存在"""
    def __init__(self, resource_type: str, resource_id: str, message: Optional[str] = None):
        if message is None:
            message = f"{resource_type} already exists: {resource_id}"
        super().__init__(
            ErrorCode.RESOURCE_ALREADY_EXISTS,
            message,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class InsufficientStorageException(ResourceException):
    """存储空间不足"""
    def __init__(self, required: int, available: int, message: Optional[str] = None):
        if message is None:
            message = f"Insufficient storage: required {required}, available {available}"
        super().__init__(
            ErrorCode.INSUFFICIENT_STORAGE,
            message,
            details={"required": required, "available": available}
        )


class InsufficientMemoryException(ResourceException):
    """内存不足"""
    def __init__(self, message: str = "Insufficient memory"):
        super().__init__(ErrorCode.INSUFFICIENT_MEMORY, message)


# ==================== 任务相关异常 ====================

class TaskException(BaseAppException):
    """任务异常基类"""
    pass


class TaskNotFoundException(TaskException):
    """任务不存在"""
    def __init__(self, task_id: str, message: Optional[str] = None):
        if message is None:
            message = f"Task not found: {task_id}"
        super().__init__(ErrorCode.TASK_NOT_FOUND, message, details={"task_id": task_id})


class TaskExecutionFailedException(TaskException):
    """任务执行失败"""
    def __init__(self, task_id: str, reason: str, message: Optional[str] = None):
        if message is None:
            message = f"Task execution failed: {reason}"
        super().__init__(
            ErrorCode.TASK_EXECUTION_FAILED,
            message,
            details={"task_id": task_id, "reason": reason}
        )


class TaskTimeoutException(TaskException):
    """任务超时"""
    def __init__(self, task_id: str, timeout: int, message: Optional[str] = None):
        if message is None:
            message = f"Task timeout after {timeout}s"
        super().__init__(
            ErrorCode.TASK_TIMEOUT,
            message,
            details={"task_id": task_id, "timeout": timeout}
        )


class TaskQueueFullException(TaskException):
    """任务队列已满"""
    def __init__(self, max_size: int, message: Optional[str] = None):
        if message is None:
            message = f"Task queue is full, maximum size: {max_size}"
        super().__init__(ErrorCode.TASK_QUEUE_FULL, message, details={"max_size": max_size})


# ==================== 模型相关异常 ====================

class ModelException(BaseAppException):
    """模型异常基类"""
    pass


class ModelNotFoundException(ModelException):
    """模型不存在"""
    def __init__(self, model_key: str, message: Optional[str] = None):
        if message is None:
            message = f"Model not found: {model_key}"
        super().__init__(ErrorCode.MODEL_NOT_FOUND, message, details={"model_key": model_key})


class ModelLoadFailedException(ModelException):
    """模型加载失败"""
    def __init__(self, model_key: str, reason: str, message: Optional[str] = None):
        if message is None:
            message = f"Model load failed: {reason}"
        super().__init__(
            ErrorCode.MODEL_LOAD_FAILED,
            message,
            details={"model_key": model_key, "reason": reason}
        )


class ModelInferenceFailedException(ModelException):
    """模型推理失败"""
    def __init__(self, model_key: str, reason: str, message: Optional[str] = None):
        if message is None:
            message = f"Model inference failed: {reason}"
        super().__init__(
            ErrorCode.MODEL_INFERENCE_FAILED,
            message,
            details={"model_key": model_key, "reason": reason}
        )


class ModelFileCorruptedException(ModelException):
    """模型文件损坏"""
    def __init__(self, model_key: str, file_path: str, message: Optional[str] = None):
        if message is None:
            message = f"Model file corrupted: {file_path}"
        super().__init__(
            ErrorCode.MODEL_FILE_CORRUPTED,
            message,
            details={"model_key": model_key, "file_path": file_path}
        )


# ==================== 网络相关异常 ====================

class NetworkException(BaseAppException):
    """网络异常基类"""
    pass


class ConnectionErrorException(NetworkException):
    """连接错误"""
    def __init__(self, url: str, message: Optional[str] = None):
        if message is None:
            message = f"Connection error: {url}"
        super().__init__(ErrorCode.CONNECTION_ERROR, message, details={"url": url})


class TimeoutException(NetworkException):
    """超时错误"""
    def __init__(self, timeout: int, message: Optional[str] = None):
        if message is None:
            message = f"Request timeout after {timeout}s"
        super().__init__(ErrorCode.TIMEOUT_ERROR, message, details={"timeout": timeout})


class APICallFailedException(NetworkException):
    """API调用失败"""
    def __init__(self, url: str, status_code: Optional[int] = None, message: Optional[str] = None):
        if message is None:
            message = f"API call failed: {url}"
            if status_code:
                message += f" (status: {status_code})"
        super().__init__(
            ErrorCode.API_CALL_FAILED,
            message,
            details={"url": url, "status_code": status_code} if status_code else {"url": url}
        )


class ServiceUnavailableException(NetworkException):
    """服务不可用"""
    def __init__(self, service_name: str, message: Optional[str] = None):
        if message is None:
            message = f"Service unavailable: {service_name}"
        super().__init__(ErrorCode.SERVICE_UNAVAILABLE, message, details={"service_name": service_name})


# ==================== 推理服务相关异常 ====================

class InferenceServiceException(BaseAppException):
    """推理服务异常基类"""
    pass


class InferenceServiceNotFoundException(InferenceServiceException):
    """推理服务不存在"""
    def __init__(self, service_id: str, message: Optional[str] = None):
        if message is None:
            message = f"Inference service not found: {service_id}"
        super().__init__(
            ErrorCode.INFERENCE_SERVICE_NOT_FOUND,
            message,
            details={"service_id": service_id}
        )


class InferenceServiceNotRunningException(InferenceServiceException):
    """推理服务未运行"""
    def __init__(self, service_id: str, message: Optional[str] = None):
        if message is None:
            message = f"Inference service not running: {service_id}"
        super().__init__(
            ErrorCode.INFERENCE_SERVICE_NOT_RUNNING,
            message,
            details={"service_id": service_id}
        )


class InferenceServiceConnectionFailedException(InferenceServiceException):
    """推理服务连接失败"""
    def __init__(self, service_id: str, message: Optional[str] = None):
        if message is None:
            message = f"Inference service connection failed: {service_id}"
        super().__init__(
            ErrorCode.INFERENCE_SERVICE_CONNECTION_FAILED,
            message,
            details={"service_id": service_id}
        )


class InferenceDispatchUnavailableException(InferenceServiceNotRunningException):
    """推理任务无法派发到任何在线推理服务。"""

    def __init__(
        self,
        message: str,
        *,
        message_code: str,
        message_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("dispatch", message)
        self.message_code = message_code
        self.message_params = message_params or {}


# ==================== 文件相关异常 ====================

class FileException(BaseAppException):
    """文件异常基类"""
    pass


class FileNotFoundException(FileException):
    """文件不存在"""
    def __init__(self, file_id: str, message: Optional[str] = None):
        if message is None:
            message = f"File not found: {file_id}"
        super().__init__(ErrorCode.FILE_NOT_FOUND, message, details={"file_id": file_id})


class FileUploadFailedException(FileException):
    """文件上传失败"""
    def __init__(self, reason: str, message: Optional[str] = None):
        if message is None:
            message = f"File upload failed: {reason}"
        super().__init__(ErrorCode.FILE_UPLOAD_FAILED, message, details={"reason": reason})


# ==================== 配置相关异常 ====================

class ConfigException(BaseAppException):
    """配置异常基类"""
    pass


class ConfigNotFoundException(ConfigException):
    """配置不存在"""
    def __init__(self, config_key: str, message: Optional[str] = None):
        if message is None:
            message = f"Config not found: {config_key}"
        super().__init__(ErrorCode.CONFIG_NOT_FOUND, message, details={"config_key": config_key})


class InvalidConfigException(ConfigException):
    """无效配置"""
    def __init__(self, config_key: str, reason: str, message: Optional[str] = None):
        if message is None:
            message = f"Invalid config {config_key}: {reason}"
        super().__init__(
            ErrorCode.INVALID_CONFIG,
            message,
            details={"config_key": config_key, "reason": reason}
        )

