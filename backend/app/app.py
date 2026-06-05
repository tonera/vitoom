"""
FastAPI应用初始化
集成中间件、静态文件服务、健康检查等
"""
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from backend.core.response import ok

from backend.core.config import get_server_config, get_security_config, get_config
from backend.core.logger import setup_logging, get_app_logger
from backend.core.error_handler import register_error_handlers
from backend.core.version import get_version

logger = get_app_logger(__name__)

# 全局应用实例
_app: Optional[FastAPI] = None


class SPAStaticFiles(StaticFiles):
    """Serve Vue history routes from index.html while preserving API 404s."""

    excluded_prefixes = ("api/", "v1/", "outputs/", "ws/")

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            method = scope.get("method")
            normalized_path = path.lstrip("/")
            if (
                exc.status_code == 404
                and method in ("GET", "HEAD")
                and not normalized_path.startswith(self.excluded_prefixes)
                and not normalized_path.startswith("assets/")
            ):
                return await super().get_response("index.html", scope)
            raise


def create_app(
    title: str = "Vitoom API",
    description: str = "AIGC应用系统API",
    version: Optional[str] = None,
    enable_cors: Optional[bool] = None,
    enable_static_files: bool = True,
    static_files_dir: Optional[Path] = None,
    enable_health_check: bool = True
) -> FastAPI:
    """
    创建FastAPI应用实例
    
    Args:
        title: 应用标题
        description: 应用描述
        version: 应用版本
        enable_cors: 是否启用CORS，如果为None则从配置读取
        enable_static_files: 是否启用静态文件服务
        static_files_dir: 静态文件目录，如果为None则使用默认目录
        enable_health_check: 是否启用健康检查端点
    
    Returns:
        FastAPI应用实例
    
    Example:
        >>> app = create_app()
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=8888)
    """
    global _app

    if version is None:
        version = get_version()
    
    # 初始化日志系统
    setup_logging()
    logger.info("Initializing FastAPI application...")
    
    # 创建FastAPI应用
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )
    
    # 注册错误处理器（最先注册，确保所有异常都被处理）
    register_error_handlers(app)
    logger.info("Error handlers registered")
    
    # 配置CORS
    if enable_cors is None:
        security_config = get_security_config()
        enable_cors = security_config.get("cors", {}).get("enabled", True)
    
    if enable_cors:
        security_config = get_security_config()
        cors_config = security_config.get("cors", {})
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get("allow_origins", ["*"]),
            allow_credentials=True,
            allow_methods=cors_config.get("allow_methods", ["*"]),
            allow_headers=cors_config.get("allow_headers", ["*"]),
        )
        logger.info("CORS middleware enabled")
    
    # 注册请求日志中间件
    @app.middleware("http")
    async def log_requests(request, call_next):
        """记录HTTP请求日志"""
        import time
        start_time = time.time()
        
        # 记录请求
        logger.info(
            f"{request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
            }
        )
        
        # 处理请求
        response = await call_next(request)
        
        # 记录响应
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": process_time,
            }
        )
        
        return response
    
    logger.info("Request logging middleware registered")
    
    # 健康检查端点（必须在静态文件之前注册）
    if enable_health_check:
        @app.get("/api/health", tags=["System"])
        async def health_check():
            """健康检查端点"""
            return ok(
                data={
                    "status": "healthy",
                    "service": title,
                    "version": version,
                },
                msg="ok",
            )
        
        logger.info("Health check endpoint registered at /api/health")
    
    # 注册API路由（必须在静态文件之前注册）
    try:
        from backend.api.admin import router as admin_router
        app.include_router(admin_router)
        logger.info("Admin API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register admin routes: {e}")

    try:
        from backend.api.auth import router as auth_router
        app.include_router(auth_router)
        logger.info("Auth API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register auth routes: {e}")

    try:
        from backend.api.api_keys import router as api_keys_router
        app.include_router(api_keys_router)
        logger.info("API key management routes registered")
    except Exception as e:
        logger.warning(f"Failed to register API key management routes: {e}")
    
    try:
        from backend.models.routes import router as models_router
        app.include_router(models_router)
        logger.info("Models API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register models routes: {e}")
    
    try:
        from backend.websocket import router as websocket_router
        app.include_router(websocket_router)
        logger.info("WebSocket routes registered")
    except Exception as e:
        logger.warning(f"Failed to register websocket routes: {e}")
    
    try:
        from backend.services.inference.routes import router as inference_router
        app.include_router(inference_router)
        logger.info("Inference services API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register inference services routes: {e}")
    
    try:
        from backend.api.tasks import router as tasks_router
        app.include_router(tasks_router)
        logger.info("Tasks API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register tasks routes: {e}")

    try:
        from backend.api.chat import router as chat_router
        app.include_router(chat_router)
        logger.info("Chat API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register chat routes: {e}")

    try:
        from backend.api.audio import router as audio_router
        app.include_router(audio_router)
        logger.info("Audio API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register audio routes: {e}")

    try:
        from backend.api.openai import router as openai_router
        app.include_router(openai_router)
        logger.info("OpenAI compatible routes registered")
    except Exception as e:
        logger.warning(f"Failed to register OpenAI compatible routes: {e}")

    try:
        from backend.api.avatar import router as avatar_router
        app.include_router(avatar_router)
        logger.info("Avatar (LiveTalking) reverse proxy routes registered")
    except Exception as e:
        logger.warning(f"Failed to register avatar routes: {e}")

    try:
        from backend.api.uploads import router as uploads_router
        app.include_router(uploads_router)
        logger.info("Uploads API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register uploads routes: {e}")

    try:
        from backend.api.documents.routes import router as documents_router
        app.include_router(documents_router)
        logger.info("Documents API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register documents routes: {e}")

    try:
        from backend.api.user import router as user_router
        app.include_router(user_router)
        logger.info("User API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register user routes: {e}")

    try:
        from backend.api.agents import router as agents_router
        app.include_router(agents_router)
        logger.info("Agents API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register agents routes: {e}")

    try:
        from backend.api.channel_ingress import router as channel_ingress_router
        app.include_router(channel_ingress_router)
        logger.info("Channel ingress API routes registered")
    except Exception as e:
        logger.warning(f"Failed to register channel ingress routes: {e}")

    # -------- Serve outputs as static files (for frontend direct URL access) --------
    # Expose: GET /outputs/<relative_key>
    # Where relative_key is the key used by storage.local.base_path (default: resources/outputs)
    try:
        outputs_dir = Path(get_config("storage.local.base_path", "resources/outputs"))
        if not outputs_dir.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            outputs_dir = (project_root / outputs_dir).resolve()
        outputs_dir.mkdir(parents=True, exist_ok=True)

        app.mount(
            "/outputs",
            StaticFiles(directory=str(outputs_dir), check_dir=False),
            name="outputs",
        )
        logger.info(f"Outputs static files mounted at /outputs -> {outputs_dir}")
    except Exception as e:
        logger.warning(f"Failed to mount outputs static files: {e}")
    
    # ---------------- Frontend static files (SPA) ----------------
    # 目标：FastAPI 一体化托管 frontend/dist
    # 关键点：
    # - 必须在所有 API 路由注册完成后，再挂载静态文件到 "/"
    # - Vue Router 使用 createWebHistory()，需要 SPA fallback：任意前端路由都返回 index.html
    if enable_static_files:
        if static_files_dir is None:
            # 默认静态文件目录：frontend/dist
            static_files_dir = Path(__file__).parent.parent.parent / "frontend" / "dist"
        
        if static_files_dir.exists():
            logger.info(f"Static files directory found: {static_files_dir}")

            # 静态资源 + SPA fallback：放在最后，避免覆盖 /api /v1 /outputs /ws 等路由。
            app.mount(
                "/",
                SPAStaticFiles(directory=str(static_files_dir), html=True, check_dir=False),
                name="frontend",
            )
        else:
            logger.warning(f"Static files directory not found: {static_files_dir}")
    else:
        # 根路径重定向到API文档（如果静态文件未启用）
        @app.get("/", include_in_schema=False)
        async def root():
            """根路径重定向到API文档"""
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/api/docs")
    
    logger.info("FastAPI application created successfully")

    # 启动前补齐 schema（create_all 不会给已有表加列）
    try:
        from backend.database.migrations import migrate

        migrate()
    except Exception as e:
        logger.warning(f"Failed to run database migration: {e}")
    
    # 首次启动初始化（创建默认管理员）
    try:
        from backend.api.auth.init import check_and_init
        check_and_init()
    except Exception as e:
        logger.warning(f"Failed to run initialization: {e}")
    
    # 重置所有推理服务状态为stopped（系统启动时）
    try:
        from backend.services.inference.service import get_inference_service_manager
        manager = get_inference_service_manager()
        manager.reset_all_status_on_startup()
        logger.info("All inference services status reset to stopped")
    except Exception as e:
        logger.warning(f"Failed to reset inference services status: {e}")

    @app.on_event("startup")
    async def startup_agent_workers():
        try:
            from backend.workers import startup_agent_runtime

            await startup_agent_runtime()
        except Exception as e:
            logger.warning(f"Failed to start agent runtime: {e}")

    @app.on_event("startup")
    async def register_ws_event_loop():
        """把运行中的事件循环注入给 WebSocketManager，供同步代码通过
        `asyncio.run_coroutine_threadsafe` 访问进程内订阅机制。"""
        try:
            import asyncio
            from backend.websocket.manager import get_websocket_manager

            get_websocket_manager().set_event_loop(asyncio.get_running_loop())
            logger.info("WebSocketManager event loop registered")
        except Exception as e:
            logger.warning(f"Failed to register event loop on websocket manager: {e}")

    @app.on_event("startup")
    async def warm_tool_selection_index_on_startup():
        try:
            import asyncio
            from backend.services.agent.tool_selection import (
                warm_tool_selection_embedding_model,
                warm_tool_selection_index,
            )

            try:
                await asyncio.to_thread(warm_tool_selection_embedding_model)
            except Exception as e:
                logger.warning(f"Failed to warm tool selection embedding model: {e}")
            try:
                await asyncio.to_thread(warm_tool_selection_index)
            except Exception as e:
                logger.warning(f"Failed to warm tool selection index: {e}")
        except Exception as e:
            logger.warning(f"Failed to schedule tool selection warmup: {e}")

    @app.on_event("startup")
    async def warm_knowledge_base_embedding_model_on_startup():
        try:
            import asyncio
            from backend.services.agent.embeddings import warm_knowledge_base_embedding_model

            warmed = await asyncio.to_thread(warm_knowledge_base_embedding_model)
            if warmed:
                logger.info("Knowledge base embedding model warmed")
            else:
                logger.info("Knowledge base embedding warmup skipped")
        except Exception as e:
            logger.warning(f"Failed to warm knowledge base embedding model: {e}")

    @app.on_event("shutdown")
    async def shutdown_agent_workers():
        try:
            from backend.workers import shutdown_agent_runtime

            await shutdown_agent_runtime()
        except Exception as e:
            logger.warning(f"Failed to stop agent runtime: {e}")
    
    # 保存全局实例
    _app = app
    
    return app


def get_app() -> Optional[FastAPI]:
    """
    获取全局FastAPI应用实例
    
    Returns:
        FastAPI应用实例，如果未创建则返回None
    
    Example:
        >>> app = create_app()
        >>> app_instance = get_app()
        >>> app == app_instance
        True
    """
    return _app

