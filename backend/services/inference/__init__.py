"""
AI推理服务模块
提供推理服务管理功能
"""
from .service import InferenceServiceManager, get_inference_service_manager
from .routes import router

__all__ = [
    "InferenceServiceManager",
    "get_inference_service_manager",
    "router",
]
