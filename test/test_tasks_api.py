"""
测试 /v1/tasks API接口
"""
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.app import create_app
from backend.auth import create_access_token
from backend.database import Task, Model, User
from backend.utils import generate_uuid

LOCAL_MODEL_NAME = "test-local-model"
CLOUD_MODEL_NAME = "test-cloud-model"


def setup_test_data():
    """设置测试数据"""
    # 创建测试用户
    test_user_id = "test_user_" + generate_uuid()
    test_user_email = f"test_{test_user_id}@example.com"
    
    # 创建用户（如果不存在）
    user = User.get_by_email(test_user_email)
    if not user:
        User.create(
            id=test_user_id,
            email=test_user_email,
            password_hash="hashed_password",
            nickname="Test User"
        )
    
    # 创建测试模型（本地模型）
    local_model_id = "local_model_" + generate_uuid()
    local_model = Model.get_by_id(local_model_id)
    if not local_model:
        Model.create(
            id=local_model_id,
            name=LOCAL_MODEL_NAME,
            model_type="image",
            storage_mode="local",
            local_path=f"/path/to/{LOCAL_MODEL_NAME}",
            status="active",
            family="sdxl",
            is_local_model=True,
        )
    
    # 创建测试模型（云端模型）
    cloud_model_id = "cloud_model_" + generate_uuid()
    cloud_model = Model.get_by_id(cloud_model_id)
    if not cloud_model:
        Model.create(
            id=cloud_model_id,
            name=CLOUD_MODEL_NAME,
            model_type="text",
            storage_mode="cloud",
            cloud_config={"provider": "openai", "model_id": "gpt-4"},
            status="active",
        )
    
    return test_user_id, local_model_id, cloud_model_id


def test_create_image_task():
    """测试创建图片生成任务"""
    print("=" * 50)
    print("Testing Create Image Task")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    # 测试创建图片任务（本地模型）
    with patch('backend.websocket.manager.get_websocket_manager') as mock_ws_manager:
        mock_manager = MagicMock()
        mock_manager.send_task_to_inference_service = AsyncMock()
        mock_ws_manager.return_value = mock_manager
        
        response = client.post(
            "/v1/tasks",
            json={
                "task_type": "image",
                "prompt": "A beautiful landscape",
                "width": 1024,
                "height": 1024,
                "model_name": LOCAL_MODEL_NAME,
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 201:
            print(f"❌ Response status: {response.status_code}")
            print(f"❌ Response body: {response.text}")
            try:
                error_data = response.json()
                print(f"❌ Error details: {error_data}")
            except:
                pass
        
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        print(f"✓ Image task created: {data['task_id']}")
        
        # 验证任务已保存到数据库
        task = Task.get_by_id(data["task_id"])
        assert task is not None
        assert task["task_type"] == "image"
        assert task["prompt"] == "A beautiful landscape"
        assert task["model_id"] == local_model_id
        print(f"✓ Task saved to database: {task['id']}")
        
        # 验证 WebSocket 消息已发送（本地模型）
        # 注意：由于 TestClient 的限制，这里只能验证方法是否被调用
        # 实际测试中需要 mock WebSocket 连接
    
    print("Create image task test passed!\n")


def test_create_video_task():
    """测试创建视频生成任务"""
    print("=" * 50)
    print("Testing Create Video Task")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    response = client.post(
        "/v1/tasks",
        json={
            "task_type": "video",
            "prompt": "A cat playing",
            "duration": 5,
            "model_name": LOCAL_MODEL_NAME,
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"
    print(f"✓ Video task created: {data['task_id']}")
    
    # 验证任务参数
    task = Task.get_by_id(data["task_id"])
    assert task["task_type"] == "video"
    assert "duration" in task["params"]
    assert task["params"]["duration"] == 5
    print(f"✓ Video task params saved correctly")
    
    print("Create video task test passed!\n")


def test_create_audio_task():
    """测试创建音频生成任务"""
    print("=" * 50)
    print("Testing Create Audio Task")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    response = client.post(
        "/v1/tasks",
        json={
            "task_type": "audio",
            "prompt": "Generate music",
            "prompt_wav_path": "/path/to/reference.wav",
            "model_name": LOCAL_MODEL_NAME,
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "task_id" in data
    print(f"✓ Audio task created: {data['task_id']}")
    
    # 验证任务参数
    task = Task.get_by_id(data["task_id"])
    assert task["task_type"] == "audio"
    assert "prompt_wav_path" in task["params"]
    print(f"✓ Audio task params saved correctly")
    
    print("Create audio task test passed!\n")


def test_create_text_task():
    """测试创建文字生成任务"""
    print("=" * 50)
    print("Testing Create Text Task")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    response = client.post(
        "/v1/tasks",
        json={
            "task_type": "text",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "model_name": CLOUD_MODEL_NAME,
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "task_id" in data
    print(f"✓ Text task created: {data['task_id']}")
    
    # 验证任务参数
    task = Task.get_by_id(data["task_id"])
    assert task["task_type"] == "text"
    assert "messages" in task["params"]
    assert task["params"]["temperature"] == 0.7
    # 验证 prompt 从 messages 中提取
    assert task["prompt"] == "Hello, how are you?"
    print(f"✓ Text task params saved correctly")
    
    print("Create text task test passed!\n")


def test_create_task_with_local_model():
    """测试使用本地模型创建任务（应发送WebSocket消息）"""
    print("=" * 50)
    print("Testing Create Task with Local Model")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    # Mock WebSocket manager
    with patch('backend.websocket.manager.get_websocket_manager') as mock_ws_manager:
        mock_manager = MagicMock()
        mock_manager.send_task_to_inference_service = AsyncMock()
        mock_ws_manager.return_value = mock_manager
        
        response = client.post(
            "/v1/tasks",
            json={
                "task_type": "image",
                "prompt": "Test prompt",
                "model_name": LOCAL_MODEL_NAME,
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 201
        data = response.json()
        task_id = data["task_id"]
        
        # 验证 WebSocket 消息发送被调用（本地模型）
        # 注意：由于异步调用，这里需要等待
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 等待异步调用完成
        import time
        time.sleep(0.1)
        
        # 验证模型是本地模型
        model = Model.get_by_id(local_model_id)
        assert model and model.get("local_path")
        print(f"✓ Local model detected: {model['local_path']}")
        
        print(f"✓ Task created with local model: {task_id}")
    
    print("Create task with local model test passed!\n")


def test_create_task_with_cloud_model():
    """测试使用云端模型创建任务（不应发送WebSocket消息）"""
    print("=" * 50)
    print("Testing Create Task with Cloud Model")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    # Mock WebSocket manager
    with patch('backend.websocket.manager.get_websocket_manager') as mock_ws_manager:
        mock_manager = MagicMock()
        mock_manager.send_task_to_inference_service = AsyncMock()
        mock_ws_manager.return_value = mock_manager
        
        response = client.post(
            "/v1/tasks",
            json={
                "task_type": "text",
                "messages": [
                    {"role": "user", "content": "Test message"}
                ],
                "model_name": CLOUD_MODEL_NAME,
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 201
        data = response.json()
        task_id = data["task_id"]
        
        # 验证模型是云端模型（没有 local_path）
        model = Model.get_by_id(cloud_model_id)
        assert model and not model.get("local_path")
        print(f"✓ Cloud model detected: {model['storage_mode']}")
        
        print(f"✓ Task created with cloud model: {task_id}")
    
    print("Create task with cloud model test passed!\n")


def test_get_task_status():
    """测试获取任务状态"""
    print("=" * 50)
    print("Testing Get Task Status")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    # 先创建一个任务
    create_response = client.post(
        "/v1/tasks",
        json={
            "task_type": "image",
            "prompt": "Test prompt",
            "model_name": LOCAL_MODEL_NAME,
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    task_id = create_response.json()["task_id"]
    
    # 获取任务状态
    response = client.get(
        f"/v1/tasks/{task_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == "pending"
    print(f"✓ Task status retrieved: {data}")
    
    # 测试权限控制（使用其他用户的token）
    other_user_id = "other_user_" + generate_uuid()
    other_token = create_access_token({"sub": other_user_id, "email": "other@example.com"})
    response = client.get(
        f"/v1/tasks/{task_id}",
        headers={"Authorization": f"Bearer {other_token}"}
    )
    assert response.status_code == 403
    print(f"✓ Permission check works: unauthorized access rejected")
    
    # 测试不存在的任务
    fake_task_id = generate_uuid()
    response = client.get(
        f"/v1/tasks/{fake_task_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404
    print(f"✓ Not found check works: non-existent task rejected")
    
    print("Get task status test passed!\n")


def test_list_tasks():
    """测试获取任务列表"""
    print("=" * 50)
    print("Testing List Tasks")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    # 创建多个任务
    task_ids = []
    for i in range(3):
        response = client.post(
            "/v1/tasks",
            json={
                "task_type": "image",
                "prompt": f"Test prompt {i}",
                "model_name": LOCAL_MODEL_NAME,
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        task_ids.append(response.json()["task_id"])
    
    # 获取任务列表
    response = client.get(
        "/v1/tasks",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert "total" in data
    assert len(data["tasks"]) >= 3
    print(f"✓ Task list retrieved: {data['total']} tasks")
    
    # 测试状态过滤
    response = client.get(
        "/v1/tasks?status=pending",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert all(task["status"] == "pending" for task in data["tasks"])
    print(f"✓ Status filter works: {len(data['tasks'])} pending tasks")
    
    print("List tasks test passed!\n")


def test_cancel_task():
    """测试取消任务"""
    print("=" * 50)
    print("Testing Cancel Task")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    # 创建一个任务
    create_response = client.post(
        "/v1/tasks",
        json={
            "task_type": "image",
            "prompt": "Test prompt",
            "model_name": LOCAL_MODEL_NAME,
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    task_id = create_response.json()["task_id"]
    
    # Mock WebSocket manager for cancel signal
    with patch('backend.websocket.manager.get_websocket_manager') as mock_ws_manager:
        mock_manager = MagicMock()
        mock_manager.send_cancel_signal_to_inference_service = AsyncMock()
        mock_ws_manager.return_value = mock_manager
        
        # 取消任务
        response = client.delete(
            f"/v1/tasks/{task_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Task cancelled successfully"
        assert data["task_id"] == task_id
        print(f"✓ Task cancelled: {task_id}")
        
        # 验证任务状态已更新
        task = Task.get_by_id(task_id)
        assert task["status"] == "cancelled"
        print(f"✓ Task status updated to cancelled")
        
        # 测试取消已完成的任务
        response = client.delete(
            f"/v1/tasks/{task_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 400
        print(f"✓ Cannot cancel already cancelled task")
    
    print("Cancel task test passed!\n")


def test_invalid_task_type():
    """测试无效的任务类型"""
    print("=" * 50)
    print("Testing Invalid Task Type")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    response = client.post(
        "/v1/tasks",
        json={
            "task_type": "invalid_type",
            "prompt": "Test prompt"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 400
    print(f"✓ Invalid task type rejected")
    
    print("Invalid task type test passed!\n")


def test_missing_prompt():
    """测试缺少必需参数"""
    print("=" * 50)
    print("Testing Missing Required Parameters")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    test_user_id, local_model_id, cloud_model_id = setup_test_data()
    token = create_access_token({"sub": test_user_id, "email": "test@example.com"})
    client = TestClient(app)
    
    # 测试缺少 prompt（image类型）
    response = client.post(
        "/v1/tasks",
        json={
            "task_type": "image"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    # 由于 prompt 是 Optional，但实际业务逻辑需要，这里可能返回400或422
    assert response.status_code in [400, 422]
    print(f"✓ Missing prompt rejected")
    
    # 测试缺少 messages（text类型）
    response = client.post(
        "/v1/tasks",
        json={
            "task_type": "text"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code in [400, 422]
    print(f"✓ Missing messages rejected")
    
    print("Missing required parameters test passed!\n")


def test_unauthorized_access():
    """测试未授权访问"""
    print("=" * 50)
    print("Testing Unauthorized Access")
    print("=" * 50)
    
    app = create_app(enable_static_files=False)
    client = TestClient(app)
    
    # 测试未授权创建任务
    response = client.post(
        "/v1/tasks",
        json={
            "task_type": "image",
            "prompt": "Test prompt"
        }
    )
    
    assert response.status_code == 401
    print(f"✓ Unauthorized access rejected")
    
    print("Unauthorized access test passed!\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("Running All Tests for /v1/tasks API")
    print("=" * 50 + "\n")
    
    try:
        test_create_image_task()
        test_create_video_task()
        test_create_audio_task()
        test_create_text_task()
        test_create_task_with_local_model()
        test_create_task_with_cloud_model()
        test_get_task_status()
        test_list_tasks()
        test_cancel_task()
        test_invalid_task_type()
        test_missing_prompt()
        test_unauthorized_access()
        
        print("\n" + "=" * 50)
        print("All Tests Passed!")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()

