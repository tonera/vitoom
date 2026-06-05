"""
FastAPI错误处理中间件和异常处理器
"""
import traceback
from typing import Dict, Any, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .exceptions import BaseAppException
from .error_codes import ErrorCode
from .response import fail
from .logger import get_error_logger
from .config import get_config

# 导入数据库模型（用于记录错误日志）
try:
    from ..database import ErrorLog
    from ..database.db import get_db_context
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

logger = get_error_logger(__name__)


def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    http_status: Optional[int] = None,
    request_id: Optional[str] = None,
    request: Optional[Request] = None,
    message_code: Optional[str] = None,
    message_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    创建统一错误响应：{code, data, msg, message_code?, message_params?}
    - code: ErrorCode 的数值
    - msg: 错误消息（按 locale 翻译后的 fallback 文案）
    - data: 可选的错误详情/上下文
    """
    from backend.i18n.error_codes import get_message_code
    from backend.i18n.locale import get_locale_from_request
    from backend.i18n.translator import build_message_params, t
    from .error_codes import get_error_type

    locale = get_locale_from_request(request)
    resolved_message_code = message_code or get_message_code(error_code)
    resolved_message_params = (
        message_params
        if message_params is not None
        else build_message_params(details)
    )
    translated_message = t(resolved_message_code, locale, **resolved_message_params)
    if translated_message == resolved_message_code:
        translated_message = message

    data: Dict[str, Any] = {"type": get_error_type(error_code)}
    if details:
        data["details"] = details
    if http_status is not None:
        data["http_status"] = http_status
    if request_id:
        data["request_id"] = request_id

    return fail(
        error_code,
        translated_message,
        data=data if data else None,
        message_code=resolved_message_code,
        message_params=resolved_message_params or None,
    )


async def app_exception_handler(request: Request, exc: BaseAppException) -> JSONResponse:
    """
    处理应用自定义异常
    
    Args:
        request: FastAPI请求对象
        exc: 应用异常实例
    
    Returns:
        JSON错误响应
    """
    # 记录错误日志
    logger.error(
        f"Application error: {exc.message}",
        extra={
            "error_code": exc.error_code.value,
            "error_type": exc.error_type,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
        exc_info=exc.cause
    )
    
    # 记录到数据库（如果可用）
    if DB_AVAILABLE:
        try:
            import uuid
            task_id = exc.details.get("task_id") if exc.details else None
            
            ErrorLog.create(
                id=str(uuid.uuid4()),
                error_type=exc.error_type,
                message=exc.message,
                stack_trace=traceback.format_exc() if exc.cause else None,
                task_id=task_id,
                severity="error" if exc.error_code.value >= 5000 else "warning"
            )
        except Exception as e:
            logger.warning(f"Failed to save error log to database: {e}")
    
    # 创建错误响应
    response = create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        http_status=exc.http_status,
        request=request,
        message_code=getattr(exc, "message_code", None),
        message_params=getattr(exc, "message_params", None),
    )
    
    return JSONResponse(
        status_code=exc.http_status,
        content=response
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    处理HTTP异常（404, 405等）
    
    Args:
        request: FastAPI请求对象
        exc: HTTP异常实例
    
    Returns:
        JSON错误响应
    """
    # 映射HTTP状态码到错误码
    status_to_error_code = {
        400: ErrorCode.INVALID_REQUEST,
        404: ErrorCode.NOT_FOUND,
        405: ErrorCode.METHOD_NOT_ALLOWED,
        413: ErrorCode.FILE_TOO_LARGE,
        429: ErrorCode.RESOURCE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        502: ErrorCode.NETWORK_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
        504: ErrorCode.TIMEOUT_ERROR,
    }

    detail_msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    error_code = status_to_error_code.get(exc.status_code, ErrorCode.UNKNOWN_ERROR)

    response = create_error_response(
        error_code=error_code,
        message=detail_msg,
        http_status=exc.status_code,
        request=request,
    )
    # 4xx 优先展示 HTTPException.detail，避免被 common.unknownError 等泛化文案覆盖
    if 400 <= exc.status_code < 500 and detail_msg:
        response["msg"] = detail_msg
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    处理请求验证异常（参数验证失败）
    
    Args:
        request: FastAPI请求对象
        exc: 验证异常实例
    
    Returns:
        JSON错误响应
    """
    errors = exc.errors()
    
    # 提取第一个错误作为主要错误消息
    if errors:
        first_error = errors[0]
        field = ".".join(str(loc) for loc in first_error.get("loc", []))
        message = first_error.get("msg", "Validation error")
        error_message = f"Validation error for {field}: {message}"
    else:
        error_message = "Validation error"
    
    # 记录警告日志
    logger.warning(
        f"Validation error: {error_message}",
        extra={
            "errors": errors,
            "path": request.url.path,
            "method": request.method,
        }
    )
    
    response = create_error_response(
        error_code=ErrorCode.INVALID_PARAMETER,
        message=error_message,
        details={"validation_errors": errors},
        http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        request=request,
        message_code="validation.invalidParameter",
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    处理未预期的异常
    
    Args:
        request: FastAPI请求对象
        exc: 异常实例
    
    Returns:
        JSON错误响应
    """
    # 记录错误日志（包含完整堆栈跟踪）
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
        },
        exc_info=exc
    )
    
    # 记录到数据库（如果可用）
    if DB_AVAILABLE:
        try:
            import uuid
            ErrorLog.create(
                id=str(uuid.uuid4()),
                error_type="system_error",
                message=str(exc),
                stack_trace=traceback.format_exc(),
                severity="critical"
            )
        except Exception as e:
            logger.warning(f"Failed to save error log to database: {e}")
    
    # 根据配置决定是否返回详细错误信息
    debug_mode = get_config("server.debug", False)
    
    if debug_mode:
        # 调试模式：返回详细错误信息
        response = create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=str(exc),
            details={"traceback": traceback.format_exc()},
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request=request,
        )
    else:
        # 生产模式：返回通用错误信息
        response = create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="An internal error occurred. Please contact support.",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request=request,
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response
    )


def register_error_handlers(app):
    """
    注册错误处理器到FastAPI应用
    
    Args:
        app: FastAPI应用实例
    
    Example:
        >>> from fastapi import FastAPI
        >>> from core.error_handler import register_error_handlers
        >>> app = FastAPI()
        >>> register_error_handlers(app)
    """
    # 注册应用自定义异常处理器
    app.add_exception_handler(BaseAppException, app_exception_handler)
    
    # 注册HTTP异常处理器
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # 注册请求验证异常处理器
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # 注册通用异常处理器（捕获所有未处理的异常）
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Error handlers registered")

