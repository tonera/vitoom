"""
AI推理服务模块测试脚本
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.inference import (
    InferenceServiceManager,
    get_inference_service_manager,
    InferenceClient,
    get_inference_client
)
from backend.database import InferenceService
from backend.database.db import get_db_context


def test_create_service():
    """测试创建推理服务"""
    print("=" * 50)
    print("Testing Create Inference Service")
    print("=" * 50)
    
    manager = get_inference_service_manager()
    
    service_dict = manager.create_service(
        name="test-vllm",
        service_type="vllm",
        port=8001,
        config={"model_path": "resources/models/text/llama3-8b"},
        gpu_enabled=True,
        auto_start=False
    )
    
    assert service_dict is not None
    assert service_dict["name"] == "test-vllm"
    assert service_dict["type"] == "vllm"
    assert service_dict["status"] == "stopped"
    print(f"✓ Service created: {service_dict['id']}")
    
    # 清理
    with get_db_context() as db:
        from backend.database.models import InferenceService as InferenceServiceModel
        db.query(InferenceServiceModel).filter(InferenceServiceModel.id == service_dict["id"]).delete()
        db.commit()
    
    print("Create service test passed!\n")


def test_list_services():
    """测试列出推理服务"""
    print("=" * 50)
    print("Testing List Services")
    print("=" * 50)
    
    manager = get_inference_service_manager()
    
    # 创建几个测试服务
    service_ids = []
    for i in range(3):
        service_dict = manager.create_service(
            name=f"test-service-{i}",
            service_type="vllm" if i % 2 == 0 else "ollama",
            port=8001 + i
        )
        service_ids.append(service_dict["id"])
    
    # 列出所有服务
    all_services = manager.list_services()
    assert len(all_services) >= 3
    print(f"✓ Listed {len(all_services)} services")
    
    # 按类型列出服务
    vllm_services = manager.list_services(service_type="vllm")
    assert len(vllm_services) >= 2
    print(f"✓ Listed {len(vllm_services)} vllm services")
    
    # 按状态列出服务
    stopped_services = manager.list_services(status="stopped")
    assert len(stopped_services) >= 3
    print(f"✓ Listed {len(stopped_services)} stopped services")
    
    # 清理
    with get_db_context() as db:
        from backend.database.models import InferenceService as InferenceServiceModel
        db.query(InferenceServiceModel).filter(InferenceServiceModel.id.in_(service_ids)).delete()
        db.commit()
    
    print("List services test passed!\n")


def test_get_service():
    """测试获取推理服务"""
    print("=" * 50)
    print("Testing Get Service")
    print("=" * 50)
    
    manager = get_inference_service_manager()
    
    # 创建服务
    service_dict = manager.create_service(
        name="test-service",
        service_type="vllm"
    )
    
    # 获取服务
    retrieved = manager.get_service(service_dict["id"])
    assert retrieved is not None
    assert retrieved["id"] == service_dict["id"]
    print(f"✓ Service retrieved: {retrieved['id']}")
    
    # 清理
    with get_db_context() as db:
        from backend.database.models import InferenceService as InferenceServiceModel
        db.query(InferenceServiceModel).filter(InferenceServiceModel.id == service_dict["id"]).delete()
        db.commit()
    
    print("Get service test passed!\n")


async def test_start_stop_service():
    """测试启动/停止服务"""
    print("=" * 50)
    print("Testing Start/Stop Service")
    print("=" * 50)
    
    manager = get_inference_service_manager()
    
    # 创建服务
    service_dict = manager.create_service(
        name="test-service",
        service_type="vllm",
        port=8001
    )
    
    # 启动服务（只更新状态，不实际启动进程）
    started = manager.start_service(service_dict["id"])
    assert started["status"] == "starting"
    print(f"✓ Service started (status: {started['status']})")
    
    # 停止服务
    stopped = await manager.stop_service(service_dict["id"])
    assert stopped["status"] == "stopped"
    print(f"✓ Service stopped (status: {stopped['status']})")
    
    # 清理
    with get_db_context() as db:
        from backend.database.models import InferenceService as InferenceServiceModel
        db.query(InferenceServiceModel).filter(InferenceServiceModel.id == service_dict["id"]).delete()
        db.commit()
    
    print("Start/stop service test passed!\n")


def test_get_available_service():
    """测试获取可用服务"""
    print("=" * 50)
    print("Testing Get Available Service")
    print("=" * 50)
    
    manager = get_inference_service_manager()
    
    # 创建服务并设置为运行状态
    service_dict = manager.create_service(
        name="test-vllm",
        service_type="vllm",
        port=8001
    )
    
    # 更新状态为running（模拟服务已启动）
    InferenceService.update(service_dict["id"], status="running")
    
    # 查找可用服务
    available = manager.get_available_service("text")
    assert available is not None
    assert available["type"] in ["vllm", "ollama"]
    print(f"✓ Available service found: {available['id']}")
    
    # 清理
    with get_db_context() as db:
        from backend.database.models import InferenceService as InferenceServiceModel
        db.query(InferenceServiceModel).filter(InferenceServiceModel.id == service_dict["id"]).delete()
        db.commit()
    
    print("Get available service test passed!\n")


async def test_inference_client():
    """测试推理服务客户端"""
    print("=" * 50)
    print("Testing Inference Client")
    print("=" * 50)
    
    client = get_inference_client()
    
    # 创建服务
    manager = get_inference_service_manager()
    service_dict = manager.create_service(
        name="test-service",
        service_type="vllm",
        port=8001
    )
    
    # 注意：由于服务未实际运行，调用会失败，这是预期的
    # 这里只测试客户端初始化
    
    print(f"✓ Inference client initialized")
    
    # 清理
    with get_db_context() as db:
        from backend.database.models import InferenceService as InferenceServiceModel
        db.query(InferenceServiceModel).filter(InferenceServiceModel.id == service_dict["id"]).delete()
        db.commit()
    
    print("Inference client test passed!\n")


async def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("AI Inference Service Module Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_create_service()
        test_list_services()
        test_get_service()
        await test_start_stop_service()
        test_get_available_service()
        await test_inference_client()
        
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

