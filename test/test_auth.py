"""
认证基础模块测试脚本
"""
import sys
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    decode_token,
    get_user_id_from_token,
    refresh_access_token,
    hash_password,
    verify_password,
)
from backend.core.exceptions import InvalidTokenException, TokenExpiredException


def test_jwt_token():
    """测试JWT Token生成和验证"""
    print("=" * 50)
    print("Testing JWT Token")
    print("=" * 50)
    
    # 创建访问Token
    user_data = {"sub": "user_123", "email": "user@example.com"}
    access_token = create_access_token(user_data)
    print(f"✓ Access token created: {access_token[:50]}...")
    
    # 验证Token
    payload = verify_token(access_token)
    assert payload["sub"] == "user_123"
    assert payload["email"] == "user@example.com"
    assert payload["type"] == "access"
    print(f"✓ Token verified: user_id={payload['sub']}")
    
    # 创建刷新Token
    refresh_token = create_refresh_token({"sub": "user_123"})
    print(f"✓ Refresh token created: {refresh_token[:50]}...")
    
    # 验证刷新Token
    refresh_payload = verify_token(refresh_token, token_type="refresh")
    assert refresh_payload["sub"] == "user_123"
    assert refresh_payload["type"] == "refresh"
    print(f"✓ Refresh token verified")
    
    # 从Token获取用户ID
    user_id = get_user_id_from_token(access_token)
    assert user_id == "user_123"
    print(f"✓ User ID extracted: {user_id}")
    
    # 测试Token刷新
    new_access_token = refresh_access_token(refresh_token)
    new_payload = verify_token(new_access_token)
    assert new_payload["sub"] == "user_123"
    print(f"✓ Token refreshed successfully")
    
    # 测试无效Token
    try:
        verify_token("invalid_token")
        assert False, "Should raise InvalidTokenException"
    except InvalidTokenException:
        print(f"✓ Invalid token correctly rejected")
    
    print("JWT token test passed!\n")


def test_token_expiration():
    """测试Token过期"""
    print("=" * 50)
    print("Testing Token Expiration")
    print("=" * 50)
    
    # 创建立即过期的Token
    expired_token = create_access_token(
        {"sub": "user_123"},
        expires_delta=timedelta(seconds=-1)  # 已过期
    )
    
    # 解码Token（不验证过期）
    payload = decode_token(expired_token)
    assert payload["sub"] == "user_123"
    print(f"✓ Expired token decoded (without verification)")
    
    # 验证Token（应该失败）
    try:
        verify_token(expired_token)
        assert False, "Should raise TokenExpiredException"
    except TokenExpiredException:
        print(f"✓ Expired token correctly rejected")
    
    print("Token expiration test passed!\n")


def test_password_hashing():
    """测试密码哈希"""
    print("=" * 50)
    print("Testing Password Hashing")
    print("=" * 50)
    
    password = "my_secure_password_123"
    
    # 哈希密码
    hashed = hash_password(password)
    assert hashed != password
    assert len(hashed) > 0
    print(f"✓ Password hashed: {hashed[:30]}...")
    
    # 验证密码
    assert verify_password(password, hashed)
    assert not verify_password("wrong_password", hashed)
    print(f"✓ Password verification works")
    
    # 测试相同密码生成不同的哈希（salt）
    hashed2 = hash_password(password)
    assert hashed != hashed2  # 应该不同（因为salt不同）
    assert verify_password(password, hashed2)  # 但都能验证通过
    print(f"✓ Different salts produce different hashes")
    
    print("Password hashing test passed!\n")


def test_token_types():
    """测试Token类型验证"""
    print("=" * 50)
    print("Testing Token Types")
    print("=" * 50)
    
    # 创建访问Token
    access_token = create_access_token({"sub": "user_123"})
    
    # 验证访问Token类型
    payload = verify_token(access_token, token_type="access")
    assert payload["type"] == "access"
    print(f"✓ Access token type verified")
    
    # 使用刷新Token类型验证应该失败
    try:
        verify_token(access_token, token_type="refresh")
        assert False, "Should raise InvalidTokenException"
    except InvalidTokenException:
        print(f"✓ Wrong token type correctly rejected")
    
    print("Token types test passed!\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("Authentication Module Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_jwt_token()
        test_token_expiration()
        test_password_hashing()
        test_token_types()
        
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

