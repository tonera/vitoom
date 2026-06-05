"""
测试推理器消息接收
测试任务消息和取消消息是否能正确发送到推理器
"""
import asyncio
import sys
import argparse
from pathlib import Path
import httpx

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.database import InferenceService, User, Model
from backend.utils import generate_uuid
from backend.core.logger import get_app_logger
from backend.auth import create_access_token

logger = get_app_logger(__name__)

# API基础URL
API_BASE_URL = "http://127.0.0.1:8888"

# test/test_inference_messages.py -m model_123
async def create_test_user_and_token():
    """创建测试用户并返回token"""
    test_user_id = f"test_user_{generate_uuid()[:8]}"
    test_email = f"test_{test_user_id}@example.com"
    
    # 检查用户是否已存在
    user = User.get_by_email(test_email)
    if not user:
        User.create(
            id=test_user_id,
            email=test_email,
            password_hash="test_password_hash",
            nickname="Test User"
        )
    
    # 创建JWT token
    token = create_access_token({"sub": test_user_id, "email": test_email})
    return test_user_id, token


async def test_send_task_message(model_id: str = None):
    """
    测试发送任务消息到推理器（通过API创建任务）
    
    Args:
        model_id: 可选的模型ID，如果提供则使用此模型，否则从数据库查找
    """
    print("\n" + "="*60)
    print("测试1: 发送任务消息到推理器（通过API创建任务）")
    print("="*60)
    
    # 1. 检查是否有运行中的推理器
    all_services = InferenceService.list_all()
    running_services = [
        s for s in all_services 
        if s.get("status") == "running"
    ]
    
    if not running_services:
        print("❌ 错误: 没有运行中的推理器服务")
        print("   请先启动推理器（python inference/image/main.py service_123）")
        return False, None, None, None
    
    print(f"✓ 找到 {len(running_services)} 个运行中的推理器服务:")
    for service in running_services:
        print(f"  - {service['id']} (type: {service.get('service_type', 'unknown')})")
    
    # 选择一个image类型的推理器
    image_services = [s for s in running_services if s.get("service_type") == "image"]
    if not image_services:
        print("❌ 错误: 没有运行中的image类型推理器")
        return False, None, None, None
    
    service_id = image_services[0]["id"]
    print(f"\n✓ 使用推理器: {service_id}")
    
    # 2. 创建测试用户并获取token
    print(f"\n✓ 创建测试用户...")
    try:
        user_id, token = await create_test_user_and_token()
        print(f"✓ 测试用户已创建: {user_id}")
    except Exception as e:
        print(f"❌ 创建测试用户失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None
    
    # 3. 通过API创建任务
    print(f"\n✓ 通过API创建任务...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # 确定使用的模型ID
            test_model_id = model_id
            if test_model_id:
                # 验证模型是否存在
                model = Model.get_by_id(test_model_id)
                if not model:
                    print(f"❌ 错误: 指定的模型ID不存在: {test_model_id}")
                    return False, None, None, None
                if model.get("type") != "image":
                    print(f"⚠ 警告: 指定的模型类型不是image: {model.get('type')}")
                print(f"✓ 使用命令行指定的模型: {test_model_id}")
            else:
                # 从数据库查找一个测试用的image模型
                image_models = Model.list_by_type("image", status="active")
                if image_models:
                    # 优先使用local_path不为空的模型（本地模型）
                    local_models = [m for m in image_models if m.get("local_path")]
                    if local_models:
                        test_model_id = local_models[0]["id"]
                        print(f"✓ 使用数据库中的模型: {test_model_id}")
                    else:
                        test_model_id = image_models[0]["id"]
                        print(f"✓ 使用数据库中的模型 (no local_path): {test_model_id}")
                else:
                    print("⚠ 警告: 没有找到活跃的image模型，任务将不使用model_id")
            
            # 根据API日志，需要使用嵌套request字段格式
            request_data = {
                "task_type": "image",
                "prompt": "a beautiful sunset",
                "width": 1024,
                "height": 1024,
                "storage": "local"
            }
            
            # 如果有测试模型，添加到请求中
            if test_model_id:
                request_data["model_id"] = test_model_id
            
            response = await client.post(
                f"{API_BASE_URL}/v1/tasks",
                json={"request": request_data},  # 使用嵌套request字段格式
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code != 201:
                print(f"❌ API返回错误: {response.status_code}")
                print(f"❌ 响应内容: {response.text}")
                return False, None, None, None
            
            result = response.json()
            task_id = result.get("task_id")
            
            if not task_id:
                print(f"❌ API响应中没有task_id: {result}")
                return False, None, None, None
            
            print(f"✓ 任务已通过API创建: {task_id}")
            print(f"✓ 任务状态: {result.get('status', 'unknown')}")
            
            print("\n请检查推理器日志，应该看到类似以下内容:")
            print(f"  'Received task message: task_id={task_id}, task_type=image'")
            print(f"  'Starting inference for task {task_id}'")
            
            return True, task_id, user_id, token
            
        except httpx.TimeoutException:
            print(f"❌ API请求超时")
            return False, None, None, None
        except Exception as e:
            print(f"❌ API请求失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None, None, None


async def test_send_cancel_message(task_id: str = None, user_id: str = None, token: str = None, model_id: str = None):
    """
    测试发送取消消息到推理器（通过API取消任务）
    
    Args:
        task_id: 任务ID（可选，如果不提供则创建新任务）
        user_id: 用户ID（可选）
        token: JWT token（可选）
        model_id: 模型ID（可选，仅在创建新任务时使用）
    """
    print("\n" + "="*60)
    print("测试2: 发送取消消息到推理器（通过API取消任务）")
    print("="*60)
    
    # 1. 检查是否有运行中的推理器
    all_services = InferenceService.list_all()
    running_services = [
        s for s in all_services 
        if s.get("status") == "running"
    ]
    
    if not running_services:
        print("❌ 错误: 没有运行中的推理器服务")
        return False
    
    # 选择一个image类型的推理器
    image_services = [s for s in running_services if s.get("service_type") == "image"]
    if not image_services:
        print("❌ 错误: 没有运行中的image类型推理器")
        return False
    
    # 2. 如果没有提供task_id，先创建一个任务
    if not task_id:
        print(f"\n✓ 先创建一个任务用于取消测试...")
        # 如果没有提供user_id和token，创建新的测试用户
        if not user_id or not token:
            user_id, token = await create_test_user_and_token()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # 确定使用的模型ID
                test_model_id = model_id
                if test_model_id:
                    # 验证模型是否存在
                    model = Model.get_by_id(test_model_id)
                    if not model:
                        print(f"❌ 错误: 指定的模型ID不存在: {test_model_id}")
                        return False
                    if model.get("type") != "image":
                        print(f"⚠ 警告: 指定的模型类型不是image: {model.get('type')}")
                    print(f"✓ 使用命令行指定的模型: {test_model_id}")
                else:
                    # 从数据库查找一个测试用的image模型
                    image_models = Model.list_by_type("image", status="active")
                    if image_models:
                        # 优先使用local_path不为空的模型（本地模型）
                        local_models = [m for m in image_models if m.get("local_path")]
                        if local_models:
                            test_model_id = local_models[0]["id"]
                        else:
                            test_model_id = image_models[0]["id"]
                
                # 使用嵌套request字段格式（根据API日志，这是必需的）
                request_data = {
                    "task_type": "image",
                    "prompt": "Test cancel message - a beautiful sunset",
                    "width": 1024,
                    "height": 1024,
                    "storage": "local"
                }
                
                # 如果有测试模型，添加到请求中
                if test_model_id:
                    request_data["model_id"] = test_model_id
                
                response = await client.post(
                    f"{API_BASE_URL}/v1/tasks",
                    json={"request": request_data},
                    headers={"Authorization": f"Bearer {token}"}
                )
                
                if response.status_code != 201:
                    print(f"❌ 创建任务失败: {response.status_code}")
                    print(f"❌ 响应内容: {response.text}")
                    return False
                
                result = response.json()
                task_id = result.get("task_id")
                
                if not task_id:
                    print(f"❌ API响应中没有task_id: {result}")
                    return False
                
                print(f"✓ 任务已创建用于取消测试: {task_id}")
                # 等待一下，让任务可能进入processing状态
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ 创建任务失败: {e}")
                import traceback
                traceback.print_exc()
                return False
    else:
        # 如果提供了task_id但没有提供token，创建新的测试用户
        # 注意：这会导致403错误，因为任务属于不同的用户
        if not token:
            print("⚠ 警告: 提供了task_id但没有提供token，将创建新用户（可能导致403错误）")
            user_id, token = await create_test_user_and_token()
    
    # 3. 通过API取消任务
    print(f"\n✓ 通过API取消任务: {task_id}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.delete(
                f"{API_BASE_URL}/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ 任务已通过API取消: {task_id}")
                print(f"✓ API响应: {result.get('message', 'success')}")
                
                print("\n请检查推理器日志，应该看到类似以下内容:")
                print(f"  'Received cancel message: task_id={task_id}'")
                print(f"  'Task {task_id} was cancelled, aborting inference'")
                return True
            elif response.status_code == 404:
                print(f"⚠ 警告: 任务不存在或已被删除: {task_id}")
                return False
            elif response.status_code == 400:
                print(f"⚠ 警告: 任务状态不允许取消: {response.text}")
                return False
            elif response.status_code == 403:
                print(f"⚠ 警告: 权限不足（任务属于其他用户）: {response.text}")
                print(f"   这通常发生在使用不同的用户token删除任务时")
                print(f"   请确保使用创建任务时的相同用户token")
                return False
            else:
                print(f"❌ API返回错误: {response.status_code}")
                print(f"❌ 响应内容: {response.text}")
                return False
                
        except httpx.TimeoutException:
            print(f"❌ API请求超时")
            return False
        except Exception as e:
            print(f"❌ API请求失败: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="测试推理器消息接收",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型（从数据库查找）
  python test/test_inference_messages.py
  
  # 指定模型ID
  python test/test_inference_messages.py --model-id model_123
  
  # 使用短参数
  python test/test_inference_messages.py -m model_123
        """
    )
    parser.add_argument(
        "-m", "--model-id",
        type=str,
        default=None,
        help="指定要使用的模型ID（如果不指定，则从数据库查找活跃的image模型）"
    )
    
    args = parser.parse_args()
    model_id = args.model_key
    
    print("\n" + "="*60)
    print("推理器消息接收测试")
    print("="*60)
    print("\n前提条件:")
    print("  1. WebSocket Server已启动（python -m backend.app.main）")
    print("  2. 推理器已启动（python inference/image/main.py service_123）")
    print(f"  3. API服务地址: {API_BASE_URL}")
    if model_id:
        print(f"  4. 使用指定的模型ID: {model_id}")
    else:
        print(f"  4. 模型ID: 将从数据库查找活跃的image模型")
    print("\n开始测试...")
    
    # 测试1: 通过API创建任务（会自动发送任务消息到推理器）
    result1, task_id, user_id, token = await test_send_task_message(model_id=model_id)
    
    # 等待3秒，让推理器处理任务消息
    if result1:
        print("\n等待3秒，让推理器处理任务消息...")
        await asyncio.sleep(3)
    
    # 测试2: 通过API取消任务（会自动发送取消消息到推理器）
    # 如果测试1成功，使用测试1创建的任务和相同的用户token；否则创建新任务
    if result1 and task_id:
        result2 = await test_send_cancel_message(task_id, user_id, token, model_id=model_id)
    else:
        result2 = await test_send_cancel_message(None, None, None, model_id=model_id)
    
    # 等待2秒，让推理器处理取消消息
    if result2:
        print("\n等待2秒，让推理器处理取消消息...")
        await asyncio.sleep(2)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"测试1 (通过API创建任务): {'✓ 通过' if result1 else '❌ 失败'}")
    print(f"测试2 (通过API取消任务): {'✓ 通过' if result2 else '❌ 失败'}")
    
    if result1 and result2:
        print("\n✓ 所有测试通过！")
        print("\n请检查推理器日志确认消息是否已收到并处理。")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
    
    print("\n提示: 测试创建的任务记录在数据库中，可以手动清理。")


if __name__ == "__main__":
    asyncio.run(main())

