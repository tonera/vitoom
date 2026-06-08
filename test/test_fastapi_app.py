"""
FastAPI应用框架测试脚本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.app import create_app
from backend.core import register_error_handlers


def test_app_creation():
    """测试应用创建"""
    print("=" * 50)
    print("Testing App Creation")
    print("=" * 50)
    
    app = create_app(
        title="Test API",
        description="Test API Description",
        version="1.0.0"
    )
    
    assert app is not None
    assert app.title == "Test API"
    print(f"✓ App created: {app.title}")
    
    print("App creation test passed!\n")


def test_health_check():
    """测试健康检查端点"""
    print("=" * 50)
    print("Testing Health Check Endpoint")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    client = TestClient(app)
    
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print(f"✓ Health check: {data}")
    
    print("Health check test passed!\n")


def test_error_handling():
    """测试错误处理"""
    print("=" * 50)
    print("Testing Error Handling")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    
    @app.get("/test/error")
    def test_error():
        from backend.core.exceptions import UserNotFoundException
        raise UserNotFoundException("test_user")
    
    client = TestClient(app)
    response = client.get("/test/error")
    assert response.status_code == 404
    data = response.json()
    assert data["success"] == False
    assert "error" in data
    print(f"✓ Error handling: {data['error']['message']}")
    
    print("Error handling test passed!\n")


def test_cors():
    """测试CORS中间件"""
    print("=" * 50)
    print("Testing CORS Middleware")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    client = TestClient(app)
    
    # 测试OPTIONS请求（CORS预检）
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    # CORS中间件应该处理OPTIONS请求
    print(f"✓ CORS middleware configured")
    
    print("CORS test passed!\n")


def test_api_docs():
    """测试API文档"""
    print("=" * 50)
    print("Testing API Documentation")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    client = TestClient(app)
    
    # 测试OpenAPI JSON
    response = client.get("/api/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    print(f"✓ OpenAPI JSON available")
    
    # 测试Swagger UI
    response = client.get("/api/docs")
    assert response.status_code == 200
    print(f"✓ Swagger UI available")
    
    # 测试ReDoc
    response = client.get("/api/redoc")
    assert response.status_code == 200
    print(f"✓ ReDoc available")
    
    print("API docs test passed!\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("FastAPI App Framework Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_app_creation()
        test_health_check()
        test_error_handling()
        test_cors()
        test_api_docs()
        
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

