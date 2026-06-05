"""
文件存储模块测试脚本
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage import StorageManager, LocalStorageAdapter, get_storage_manager
from backend.database import File, User
from backend.database.db import get_db_context
from backend.utils import generate_uuid
from backend.auth import hash_password


async def create_test_user():
    """创建测试用户"""
    test_user_id = generate_uuid()
    test_email = f"test_{test_user_id}@example.com"
    
    # 检查用户是否已存在
    existing_user = User.get_by_email(test_email)
    if existing_user:
        return existing_user["id"]
    
    # 创建用户
    user_dict = User.create(
        id=test_user_id,
        email=test_email,
        password_hash=hash_password("test_password"),
        nickname="Test User"
    )
    
    return user_dict["id"] if user_dict else test_user_id


async def cleanup_test_user(user_id: str):
    """清理测试用户"""
    with get_db_context() as db:
        from backend.database.models import User as UserModel
        db.query(UserModel).filter(UserModel.id == user_id).delete()
        db.commit()


async def test_local_storage_adapter():
    """测试本地存储适配器"""
    print("=" * 50)
    print("Testing Local Storage Adapter")
    print("=" * 50)
    
    adapter = LocalStorageAdapter(
        base_path="test_outputs",
        http_base_url="/files"
    )
    
    # 测试保存文件
    test_data = b"Hello, World! This is a test file."
    result = await adapter.save_file(
        file_data=test_data,
        category="test",
        filename="test.txt"
    )
    
    assert "storage_path" in result
    assert "http_url" in result
    assert result["file_size"] == len(test_data)
    print(f"✓ File saved: {result['storage_path']}")
    print(f"✓ HTTP URL: {result['http_url']}")
    
    # 测试文件存在
    exists = await adapter.file_exists(result["storage_path"])
    assert exists
    print(f"✓ File exists check passed")
    
    # 测试读取文件
    read_data = await adapter.read_file(result["storage_path"])
    assert read_data == test_data
    print(f"✓ File read test passed")
    
    # 测试获取文件大小
    file_size = await adapter.get_file_size(result["storage_path"])
    assert file_size == len(test_data)
    print(f"✓ File size check passed")
    
    # 测试获取URL
    url = await adapter.get_file_url(result["storage_path"])
    assert url == result["http_url"]
    print(f"✓ URL generation test passed")
    
    # 测试删除文件
    deleted = await adapter.delete_file(result["storage_path"])
    assert deleted
    print(f"✓ File deletion test passed")
    
    # 清理测试目录
    import shutil
    if Path("test_outputs").exists():
        shutil.rmtree("test_outputs")
    
    print("Local storage adapter test passed!\n")


async def test_storage_manager():
    """测试存储管理器"""
    print("=" * 50)
    print("Testing Storage Manager")
    print("=" * 50)
    
    # 创建测试用户
    test_user_id = await create_test_user()
    
    manager = StorageManager()
    
    # 测试保存文件
    test_data = b"Test file content for storage manager"
    file_dict = await manager.save_file(
        file_data=test_data,
        user_id=test_user_id,
        category="test",
        filename="manager_test.txt"
    )
    
    assert file_dict is not None
    assert file_dict["user_id"] == test_user_id
    assert file_dict["category"] == "test"
    assert file_dict["file_name"] == "manager_test.txt"
    print(f"✓ File saved via manager: {file_dict['id']}")
    
    # 测试获取文件
    retrieved_file = await manager.get_file(file_dict["id"])
    assert retrieved_file is not None
    assert retrieved_file["id"] == file_dict["id"]
    print(f"✓ File retrieved: {retrieved_file['id']}")
    
    # 测试获取文件URL
    url = await manager.get_file_url(file_dict["id"])
    assert url is not None
    assert url.startswith("/files")
    print(f"✓ File URL: {url}")
    
    # 测试读取文件
    read_data = await manager.read_file(file_dict["id"])
    assert read_data == test_data
    print(f"✓ File read via manager passed")
    
    # 测试文件存在
    exists = await manager.file_exists(file_dict["id"])
    assert exists
    print(f"✓ File exists check passed")
    
    # 测试列出文件
    files = await manager.list_files(user_id=test_user_id)
    assert len(files) >= 1
    print(f"✓ Listed {len(files)} files")
    
    # 测试删除文件
    deleted = await manager.delete_file(file_dict["id"])
    assert deleted
    
    # 验证文件已删除
    retrieved_file = await manager.get_file(file_dict["id"])
    assert retrieved_file is None
    print(f"✓ File deletion test passed")
    
    await cleanup_test_user(test_user_id)
    
    print("Storage manager test passed!\n")


async def test_storage_manager_with_task():
    """测试带任务ID的文件存储"""
    print("=" * 50)
    print("Testing Storage Manager with Task")
    print("=" * 50)
    
    # 创建测试用户
    test_user_id = await create_test_user()
    
    manager = StorageManager()
    
    # 创建测试任务
    from backend.database import Task
    task_id = generate_uuid()
    task_dict = Task.create(
        id=task_id,
        user_id=test_user_id,
        task_type="test",
        prompt="Test task",
        status="completed"
    )
    
    # 保存文件（关联任务）
    test_data = b"Task-related file content"
    file_dict = await manager.save_file(
        file_data=test_data,
        user_id=test_user_id,
        category="test",
        filename="task_file.txt",
        task_id=task_id
    )
    
    assert file_dict["task_id"] == task_id
    print(f"✓ File saved with task ID: {file_dict['id']}")
    
    # 测试按任务列出文件
    task_files = await manager.list_files(task_id=task_id)
    assert len(task_files) >= 1
    assert task_files[0]["task_id"] == task_id
    print(f"✓ Listed {len(task_files)} files for task")
    
    # 清理
    await manager.delete_file(file_dict["id"])
    with get_db_context() as db:
        from backend.database.models import Task as TaskModel
        db.query(TaskModel).filter(TaskModel.id == task_id).delete()
        db.commit()
    
    await cleanup_test_user(test_user_id)
    
    print("Storage manager with task test passed!\n")


async def test_storage_manager_category_filter():
    """测试按类别过滤文件"""
    print("=" * 50)
    print("Testing Storage Manager Category Filter")
    print("=" * 50)
    
    # 创建测试用户
    test_user_id = await create_test_user()
    
    manager = StorageManager()
    
    # 保存不同类别的文件
    categories = ["image", "video", "audio"]
    file_ids = []
    
    for category in categories:
        test_data = f"Test {category} file".encode()
        file_dict = await manager.save_file(
            file_data=test_data,
            user_id=test_user_id,
            category=category,
            filename=f"test.{category}"
        )
        file_ids.append(file_dict["id"])
    
    print(f"✓ Created {len(categories)} files in different categories")
    
    # 测试按类别列出文件
    for category in categories:
        files = await manager.list_files(category=category)
        assert len(files) >= 1
        assert all(f["category"] == category for f in files)
        print(f"✓ Listed {len(files)} files in category '{category}'")
    
    # 清理
    for file_id in file_ids:
        await manager.delete_file(file_id)
    
    await cleanup_test_user(test_user_id)
    
    print("Category filter test passed!\n")


async def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("File Storage Module Test Suite")
    print("=" * 50 + "\n")
    
    try:
        await test_local_storage_adapter()
        await test_storage_manager()
        await test_storage_manager_with_task()
        await test_storage_manager_category_filter()
        
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
    sys.exit(asyncio.run(main()))

