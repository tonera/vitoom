"""
WebSocket模块测试脚本
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.app import create_app
from backend.websocket import WebSocketManager, get_websocket_manager
from backend.queue import TaskQueue
from backend.database import Task, User
from backend.database.db import get_db_context
from backend.utils import generate_uuid
from backend.auth import hash_password


def create_test_user():
    """创建测试用户"""
    test_user_id = generate_uuid()
    test_email = f"test_{test_user_id}@example.com"
    
    existing_user = User.get_by_email(test_email)
    if existing_user:
        return existing_user["id"]
    
    user_dict = User.create(
        id=test_user_id,
        email=test_email,
        password_hash=hash_password("test_password"),
        nickname="Test User"
    )
    
    return user_dict["id"] if user_dict else test_user_id


def cleanup_test_user(user_id: str):
    """清理测试用户"""
    with get_db_context() as db:
        from backend.database.models import User as UserModel
        db.query(UserModel).filter(UserModel.id == user_id).delete()
        db.commit()


async def test_websocket_manager():
    """测试WebSocket管理器"""
    print("=" * 50)
    print("Testing WebSocket Manager")
    print("=" * 50)
    
    manager = WebSocketManager()
    
    # 测试连接计数
    count = await manager.get_total_connections()
    assert count == 0
    print(f"✓ Initial connection count: {count}")
    
    # 测试任务连接计数
    task_id = generate_uuid()
    count = await manager.get_connection_count(task_id)
    assert count == 0
    print(f"✓ Task connection count: {count}")
    
    print("WebSocket manager test passed!\n")


async def test_websocket_progress_push():
    """测试WebSocket进度推送"""
    print("=" * 50)
    print("Testing WebSocket Progress Push")
    print("=" * 50)
    
    manager = WebSocketManager()
    task_id = generate_uuid()
    
    # 模拟发送进度（没有实际连接）
    await manager.send_progress(
        task_id=task_id,
        progress=50,
        message="Processing..."
    )
    
    print(f"✓ Progress push sent for task: {task_id}")
    
    # 测试任务更新（result参数已废弃，不再使用）
    await manager.send_task_update(
        task_id=task_id,
        status="completed",
        message="Task completed"
    )
    
    print(f"✓ Task update sent for task: {task_id}")
    
    print("WebSocket progress push test passed!\n")


async def test_websocket_integration():
    """测试WebSocket与任务队列集成"""
    print("=" * 50)
    print("Testing WebSocket Integration with Task Queue")
    print("=" * 50)
    
    # 创建测试用户
    test_user_id = create_test_user()
    
    # 创建队列并注册处理器
    queue = TaskQueue(max_size=10, max_workers=1)
    
    async def test_handler(task: dict) -> dict:
        task_id = task["id"]
        # 更新进度（会自动推送WebSocket）
        await queue.update_progress(task_id, 25, "Starting...")
        await asyncio.sleep(0.1)
        await queue.update_progress(task_id, 50, "Processing...")
        await asyncio.sleep(0.1)
        await queue.update_progress(task_id, 75, "Almost done...")
        await asyncio.sleep(0.1)
        await queue.update_progress(task_id, 100, "Complete")
        return {"result": "test_result"}
    
    queue.register_handler("test", test_handler)
    
    # 启动工作线程
    await queue.start_workers()
    
    # 添加任务（会自动创建任务记录）
    task_id = generate_uuid()
    await queue.add_task(
        task_id=task_id,
        user_id=test_user_id,
        task_type="test",
        prompt="Test task",
        priority=5
    )
    
    print(f"✓ Test task created: {task_id}")
    
    # 等待任务完成
    await asyncio.sleep(1)
    
    # 检查任务状态
    task = Task.get_by_id(task_id)
    assert task["status"] == "completed"
    assert task["progress"] == 100
    print(f"✓ Task completed: {task_id}")
    
    await queue.stop_workers()
    
    # 清理
    with get_db_context() as db:
        from backend.database.models import Task as TaskModel
        db.query(TaskModel).filter(TaskModel.id == task_id).delete()
        db.commit()
    
    cleanup_test_user(test_user_id)
    
    print("WebSocket integration test passed!\n")
    
    print("WebSocket integration test passed!\n")


async def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("WebSocket Module Test Suite")
    print("=" * 50 + "\n")
    
    try:
        await test_websocket_manager()
        await test_websocket_progress_push()
        await test_websocket_integration()
        
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
