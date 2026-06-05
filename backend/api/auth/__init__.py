"""
用户认证API模块
"""

__all__ = ["router"]


def __getattr__(name):  # PEP 562: 延迟加载，便于仅使用 service 的 CLI/脚本不依赖 FastAPI
    if name == "router":
        from .routes import router as _router

        globals()["router"] = _router
        return _router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
