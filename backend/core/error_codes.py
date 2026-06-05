"""
统一错误码定义
"""
from enum import IntEnum
from typing import Dict, Any


class ErrorCode(IntEnum):
    """错误码枚举"""
    
    # ==================== 通用错误 (1000-1999) ====================
    # 约定：code=1 表示成功（不考虑向下兼容）
    SUCCESS = 1
    UNKNOWN_ERROR = 1000
    INTERNAL_ERROR = 1001
    INVALID_REQUEST = 1002
    METHOD_NOT_ALLOWED = 1003
    NOT_FOUND = 1004
    
    # ==================== 认证错误 (2000-2999) ====================
    AUTH_REQUIRED = 2000
    AUTH_FAILED = 2001
    INVALID_TOKEN = 2002
    TOKEN_EXPIRED = 2003
    PERMISSION_DENIED = 2004
    USER_NOT_FOUND = 2005
    USER_ALREADY_EXISTS = 2006
    INVALID_CREDENTIALS = 2007
    
    # ==================== 参数错误 (3000-3999) ====================
    INVALID_PARAMETER = 3000
    MISSING_PARAMETER = 3001
    INVALID_FORMAT = 3002
    PARAMETER_TOO_LARGE = 3003
    PARAMETER_TOO_SMALL = 3004
    INVALID_FILE_TYPE = 3005
    FILE_TOO_LARGE = 3006
    
    # ==================== 资源错误 (4000-4999) ====================
    RESOURCE_NOT_FOUND = 4000
    RESOURCE_ALREADY_EXISTS = 4001
    RESOURCE_IN_USE = 4002
    RESOURCE_LIMIT_EXCEEDED = 4003
    INSUFFICIENT_STORAGE = 4004
    INSUFFICIENT_MEMORY = 4005
    
    # ==================== 任务错误 (5000-5999) ====================
    TASK_NOT_FOUND = 5000
    TASK_ALREADY_EXISTS = 5001
    TASK_CREATION_FAILED = 5002
    TASK_EXECUTION_FAILED = 5003
    TASK_TIMEOUT = 5004
    TASK_CANCELLED = 5005
    TASK_QUEUE_FULL = 5006
    
    # ==================== 模型错误 (6000-6999) ====================
    MODEL_NOT_FOUND = 6000
    MODEL_LOAD_FAILED = 6001
    MODEL_INFERENCE_FAILED = 6002
    MODEL_FILE_CORRUPTED = 6003
    MODEL_NOT_ACTIVATED = 6004
    MODEL_DOWNLOAD_FAILED = 6005
    MODEL_DELETE_FAILED = 6006
    
    # ==================== 网络错误 (7000-7999) ====================
    NETWORK_ERROR = 7000
    CONNECTION_ERROR = 7001
    TIMEOUT_ERROR = 7002
    API_CALL_FAILED = 7003
    SERVICE_UNAVAILABLE = 7004
    
    # ==================== 推理服务错误 (8000-8999) ====================
    INFERENCE_SERVICE_NOT_FOUND = 8000
    INFERENCE_SERVICE_NOT_RUNNING = 8001
    INFERENCE_SERVICE_CONNECTION_FAILED = 8002
    INFERENCE_SERVICE_TIMEOUT = 8003
    INFERENCE_SERVICE_ERROR = 8004
    
    # ==================== 文件错误 (9000-9999) ====================
    FILE_NOT_FOUND = 9000
    FILE_UPLOAD_FAILED = 9001
    FILE_DELETE_FAILED = 9002
    FILE_READ_FAILED = 9003
    FILE_WRITE_FAILED = 9004
    INVALID_FILE_PATH = 9005
    
    # ==================== 配置错误 (10000-10999) ====================
    CONFIG_ERROR = 10000
    CONFIG_NOT_FOUND = 10001
    INVALID_CONFIG = 10002


# 错误码到HTTP状态码的映射
ERROR_CODE_TO_HTTP_STATUS: Dict[ErrorCode, int] = {
    # 通用错误
    ErrorCode.SUCCESS: 200,
    ErrorCode.UNKNOWN_ERROR: 500,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.METHOD_NOT_ALLOWED: 405,
    ErrorCode.NOT_FOUND: 404,
    
    # 认证错误
    ErrorCode.AUTH_REQUIRED: 401,
    ErrorCode.AUTH_FAILED: 401,
    ErrorCode.INVALID_TOKEN: 401,
    ErrorCode.TOKEN_EXPIRED: 401,
    ErrorCode.PERMISSION_DENIED: 403,
    ErrorCode.USER_NOT_FOUND: 404,
    ErrorCode.USER_ALREADY_EXISTS: 409,
    ErrorCode.INVALID_CREDENTIALS: 401,
    
    # 参数错误
    ErrorCode.INVALID_PARAMETER: 400,
    ErrorCode.MISSING_PARAMETER: 400,
    ErrorCode.INVALID_FORMAT: 400,
    ErrorCode.PARAMETER_TOO_LARGE: 400,
    ErrorCode.PARAMETER_TOO_SMALL: 400,
    ErrorCode.INVALID_FILE_TYPE: 400,
    ErrorCode.FILE_TOO_LARGE: 413,
    
    # 资源错误
    ErrorCode.RESOURCE_NOT_FOUND: 404,
    ErrorCode.RESOURCE_ALREADY_EXISTS: 409,
    ErrorCode.RESOURCE_IN_USE: 409,
    ErrorCode.RESOURCE_LIMIT_EXCEEDED: 429,
    ErrorCode.INSUFFICIENT_STORAGE: 507,
    ErrorCode.INSUFFICIENT_MEMORY: 507,
    
    # 任务错误
    ErrorCode.TASK_NOT_FOUND: 404,
    ErrorCode.TASK_ALREADY_EXISTS: 409,
    ErrorCode.TASK_CREATION_FAILED: 500,
    ErrorCode.TASK_EXECUTION_FAILED: 500,
    ErrorCode.TASK_TIMEOUT: 504,
    ErrorCode.TASK_CANCELLED: 200,
    ErrorCode.TASK_QUEUE_FULL: 503,
    
    # 模型错误
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.MODEL_LOAD_FAILED: 500,
    ErrorCode.MODEL_INFERENCE_FAILED: 500,
    ErrorCode.MODEL_FILE_CORRUPTED: 500,
    ErrorCode.MODEL_NOT_ACTIVATED: 400,
    ErrorCode.MODEL_DOWNLOAD_FAILED: 500,
    ErrorCode.MODEL_DELETE_FAILED: 500,
    
    # 网络错误
    ErrorCode.NETWORK_ERROR: 502,
    ErrorCode.CONNECTION_ERROR: 502,
    ErrorCode.TIMEOUT_ERROR: 504,
    ErrorCode.API_CALL_FAILED: 502,
    ErrorCode.SERVICE_UNAVAILABLE: 503,
    
    # 推理服务错误
    ErrorCode.INFERENCE_SERVICE_NOT_FOUND: 404,
    ErrorCode.INFERENCE_SERVICE_NOT_RUNNING: 503,
    ErrorCode.INFERENCE_SERVICE_CONNECTION_FAILED: 502,
    ErrorCode.INFERENCE_SERVICE_TIMEOUT: 504,
    ErrorCode.INFERENCE_SERVICE_ERROR: 500,
    
    # 文件错误
    ErrorCode.FILE_NOT_FOUND: 404,
    ErrorCode.FILE_UPLOAD_FAILED: 500,
    ErrorCode.FILE_DELETE_FAILED: 500,
    ErrorCode.FILE_READ_FAILED: 500,
    ErrorCode.FILE_WRITE_FAILED: 500,
    ErrorCode.INVALID_FILE_PATH: 400,
    
    # 配置错误
    ErrorCode.CONFIG_ERROR: 500,
    ErrorCode.CONFIG_NOT_FOUND: 404,
    ErrorCode.INVALID_CONFIG: 400,
}


def get_http_status(error_code: ErrorCode) -> int:
    """获取错误码对应的HTTP状态码"""
    return ERROR_CODE_TO_HTTP_STATUS.get(error_code, 500)


def get_error_type(error_code: ErrorCode) -> str:
    """根据错误码获取错误类型"""
    code_value = error_code.value
    
    if 2000 <= code_value < 3000:
        return "auth_error"
    elif 3000 <= code_value < 4000:
        return "user_error"
    elif 4000 <= code_value < 5000:
        return "system_error"
    elif 5000 <= code_value < 6000:
        return "task_error"
    elif 6000 <= code_value < 7000:
        return "model_error"
    elif 7000 <= code_value < 8000:
        return "network_error"
    elif 8000 <= code_value < 9000:
        return "inference_error"
    elif 9000 <= code_value < 10000:
        return "file_error"
    elif 10000 <= code_value < 11000:
        return "config_error"
    else:
        return "system_error"

