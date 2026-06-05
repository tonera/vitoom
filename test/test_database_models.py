"""
数据库模型测试脚本
用于验证数据库模型功能
"""
import sys
import uuid
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import User, Task, Model, Config, init_db


def test_user_model():
    """测试用户模型"""
    print("Testing User model...")
    
    # 创建用户（使用唯一email）
    user_id = str(uuid.uuid4())
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    user = User.create(
        id=user_id,
        email=email,
        password_hash="hashed_password_123",
        nickname="Test User"
    )
    assert user is not None, "User creation failed"
    print(f"✓ Created user: {user['email']}")
    
    # 查询用户
    found_user = User.get_by_email(email)
    assert found_user is not None
    assert found_user["email"] == email
    print(f"✓ Found user by email: {found_user['email']}")
    
    # 更新用户
    updated_user = User.update(user_id, nickname="Updated User")
    assert updated_user["nickname"] == "Updated User"
    print(f"✓ Updated user nickname: {updated_user['nickname']}")
    
    print("User model test passed!\n")


def test_task_model():
    """测试任务模型"""
    print("Testing Task model...")
    
    # 创建用户（用于关联）
    user_id = str(uuid.uuid4())
    email = f"task_user_{uuid.uuid4().hex[:8]}@example.com"
    user = User.create(
        id=user_id,
        email=email,
        password_hash="hashed_password",
    )
    assert user is not None, "User creation failed"
    
    # 创建任务
    task_id = str(uuid.uuid4())
    task = Task.create(
        id=task_id,
        user_id=user_id,
        task_type="image",
        prompt="A beautiful landscape",
        params={"width": 1024, "height": 1024},
        priority=8
    )
    assert task is not None, "Task creation failed"
    print(f"✓ Created task: {task['id']}")
    
    # 查询任务
    found_task = Task.get_by_id(task_id)
    assert found_task is not None
    assert found_task["type"] == "image"
    assert found_task["params"]["width"] == 1024
    print(f"✓ Found task with params: {found_task['params']}")
    
    # 更新任务
    updated_task = Task.update(task_id, status="processing", progress=50)
    assert updated_task["status"] == "processing"
    assert updated_task["progress"] == 50
    print(f"✓ Updated task status: {updated_task['status']}, progress: {updated_task['progress']}")
    
    print("Task model test passed!\n")


def test_config_model():
    """测试配置模型"""
    print("Testing Config model...")
    
    # 设置配置
    Config.set("test.key1", "value1", description="Test config", category="test")
    Config.set("test.key2", {"nested": "value"}, category="test")
    
    # 获取配置
    value1 = Config.get("test.key1")
    assert value1 == "value1"
    print(f"✓ Got config value: {value1}")
    
    value2 = Config.get("test.key2")
    assert value2["nested"] == "value"
    print(f"✓ Got config dict: {value2}")
    
    # 获取分类配置
    all_test_configs = Config.get_all_by_category("test")
    assert len(all_test_configs) >= 2
    print(f"✓ Got all configs in category: {len(all_test_configs)} items")
    
    print("Config model test passed!\n")


def main():
    """主测试函数"""
    print("=" * 50)
    print("Database Models Test")
    print("=" * 50)
    print()
    
    # 确保数据库已初始化
    init_db()
    
    try:
        test_user_model()
        test_task_model()
        test_config_model()
        
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

