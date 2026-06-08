"""
用户认证API测试脚本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.app import create_app
from backend.api.auth.service import register_user, login_user, get_user_by_id
from backend.database import User
from backend.database.db import get_db_context
from backend.utils import generate_uuid


def test_user_register():
    """测试用户注册"""
    print("=" * 50)
    print("Testing User Registration")
    print("=" * 50)
    
    # 清理测试数据
    test_email = f"test_{generate_uuid()}@example.com"
    
    # 注册用户
    user_dict = register_user(
        email=test_email,
        password="test_password_123",
        nickname="Test User"
    )
    
    assert user_dict is not None
    assert user_dict["email"] == test_email.lower()
    assert user_dict["nickname"] == "Test User"
    print(f"✓ User registered: {user_dict['email']}")
    
    # 测试重复注册
    try:
        register_user(email=test_email, password="password")
        assert False, "Should raise UserAlreadyExistsException"
    except Exception as e:
        assert "already exists" in str(e).lower()
        print(f"✓ Duplicate registration correctly rejected")
    
    # 清理
    with get_db_context() as db:
        db.query(User).filter(User.email == test_email.lower()).delete()
        db.commit()
    
    print("User registration test passed!\n")


def test_user_login():
    """测试用户登录"""
    print("=" * 50)
    print("Testing User Login")
    print("=" * 50)
    
    # 创建测试用户
    test_email = f"test_{generate_uuid()}@example.com"
    test_password = "test_password_123"
    
    register_user(
        email=test_email,
        password=test_password,
        nickname="Test User"
    )
    
    # 测试正确登录
    result = login_user(test_email, test_password)
    assert "access_token" in result
    assert "refresh_token" in result
    assert result["token_type"] == "bearer"
    assert result["user"]["email"] == test_email.lower()
    print(f"✓ Login successful: {result['user']['email']}")
    
    # 测试错误密码
    try:
        login_user(test_email, "wrong_password")
        assert False, "Should raise InvalidCredentialsException"
    except Exception as e:
        assert "invalid" in str(e).lower() or "password" in str(e).lower()
        print(f"✓ Wrong password correctly rejected")
    
    # 测试不存在的用户
    try:
        login_user("nonexistent@example.com", "password")
        assert False, "Should raise UserNotFoundException"
    except Exception as e:
        assert "not found" in str(e).lower()
        print(f"✓ Non-existent user correctly rejected")
    
    # 清理
    with get_db_context() as db:
        db.query(User).filter(User.email == test_email.lower()).delete()
        db.commit()
    
    print("User login test passed!\n")


def test_get_user_by_id():
    """测试获取用户信息"""
    print("=" * 50)
    print("Testing Get User by ID")
    print("=" * 50)
    
    # 创建测试用户
    test_email = f"test_{generate_uuid()}@example.com"
    user_dict = register_user(
        email=test_email,
        password="test_password_123"
    )
    
    # 获取用户信息
    user_info = get_user_by_id(user_dict["id"])
    assert user_info is not None
    assert user_info["email"] == test_email.lower()
    assert "password_hash" not in user_info  # 密码哈希不应返回
    print(f"✓ User info retrieved: {user_info['email']}")
    
    # 清理
    with get_db_context() as db:
        db.query(User).filter(User.email == test_email.lower()).delete()
        db.commit()
    
    print("Get user by ID test passed!\n")


def test_api_endpoints():
    """测试API端点"""
    print("=" * 50)
    print("Testing API Endpoints")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    client = TestClient(app)
    
    # 测试注册端点
    test_email = f"test_{generate_uuid()}@example.com"
    response = client.post(
        "/api/auth/register",
        json={
            "email": test_email,
            "password": "test_password_123",
            "nickname": "Test User"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == test_email.lower()
    print(f"✓ Register endpoint works: {data['email']}")
    
    # 测试登录端点
    response = client.post(
        "/api/auth/login",
        json={
            "email": test_email,
            "password": "test_password_123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    access_token = data["access_token"]
    print(f"✓ Login endpoint works")
    
    # 测试获取当前用户端点
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_email.lower()
    print(f"✓ Get current user endpoint works: {data['email']}")
    
    # 测试未认证访问
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    print(f"✓ Unauthenticated access correctly rejected")
    
    # 测试刷新Token端点
    login_response = client.post(
        "/api/auth/login",
        json={
            "email": test_email,
            "password": "test_password_123"
        }
    )
    if login_response.status_code == 200:
        refresh_token = login_response.json().get("refresh_token")
        if refresh_token:
            response = client.post(
                "/api/auth/refresh",
                json={"refresh_token": refresh_token}
            )
            assert response.status_code == 200
            assert "access_token" in response.json()
            print(f"✓ Refresh token endpoint works")
    
    # 清理
    with get_db_context() as db:
        db.query(User).filter(User.email == test_email.lower()).delete()
        db.commit()
    
    print("API endpoints test passed!\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("User Authentication API Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_user_register()
        test_user_login()
        test_get_user_by_id()
        test_api_endpoints()
        
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

