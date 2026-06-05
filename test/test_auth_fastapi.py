"""
测试FastAPI认证集成
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from backend.auth import (
    create_access_token,
    get_current_user_id,
    get_optional_user_id,
)
from backend.core import register_error_handlers


def test_fastapi_auth_integration():
    """测试FastAPI认证集成"""
    print("=" * 50)
    print("Testing FastAPI Auth Integration")
    print("=" * 50)
    
    # 创建FastAPI应用
    app = FastAPI()
    
    # 注册错误处理器
    register_error_handlers(app)
    
    # 创建测试用户Token
    test_user_id = "test_user_123"
    test_token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    
    # 定义需要认证的路由
    @app.get("/protected")
    async def protected_route(user_id: str = Depends(get_current_user_id)):
        return {"user_id": user_id, "message": "This is a protected route"}
    
    # 定义可选认证的路由
    @app.get("/public")
    async def public_route(user_id: str = Depends(get_optional_user_id)):
        if user_id:
            return {"user_id": user_id, "authenticated": True}
        return {"authenticated": False}
    
    # 创建测试客户端
    client = TestClient(app)
    
    # 测试未认证访问受保护路由
    response = client.get("/protected")
    assert response.status_code == 401
    print(f"✓ Unauthenticated access correctly rejected")
    
    # 测试认证访问受保护路由
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {test_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == test_user_id
    print(f"✓ Authenticated access works: {data}")
    
    # 测试可选认证路由（未认证）
    response = client.get("/public")
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] == False
    print(f"✓ Public route without auth: {data}")
    
    # 测试可选认证路由（已认证）
    response = client.get(
        "/public",
        headers={"Authorization": f"Bearer {test_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] == True
    assert data["user_id"] == test_user_id
    print(f"✓ Public route with auth: {data}")
    
    print("\nFastAPI auth integration test passed!\n")


if __name__ == "__main__":
    test_fastapi_auth_integration()

