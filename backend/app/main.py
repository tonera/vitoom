"""
FastAPI应用启动脚本
"""
import uvicorn
from pathlib import Path
from backend.app import create_app
from backend.core.config import get_server_config
from backend.core.logger import get_app_logger
from backend.core.version import get_version

logger = get_app_logger(__name__)


def main():
    """启动FastAPI应用"""
    # 获取服务器配置
    server_config = get_server_config()
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8888)
    debug = server_config.get("debug", False)
    reload = server_config.get("reload", False)
    
    # 创建应用
    app = create_app(
        title="Vitoom API",
        description="AIGC application system API",
        version=get_version()
    )
    
    # 启动服务器
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Debug mode: {debug}, Reload: {reload}")
    logger.info(f"API docs: http://{host}:{port}/api/docs")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info" if not debug else "debug",
        reload=reload
    )


if __name__ == "__main__":
    main()

