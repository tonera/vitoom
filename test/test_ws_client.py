"""
WebSocket测试客户端
通过API创建任务，然后连接WebSocket服务器接收任务状态更新
"""
import asyncio
import sys
import argparse
import json
from pathlib import Path
import httpx
import websockets
from websockets.exceptions import ConnectionClosed

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.database import User, Model
from backend.utils import generate_uuid
from backend.core.logger import get_app_logger
from backend.auth import create_access_token

logger = get_app_logger(__name__)

# API基础URL和WebSocket URL
API_BASE_URL = "http://127.0.0.1:8888"
WS_BASE_URL = "ws://127.0.0.1:8888"

# python test/test_ws_client.py -m model_88888889 -p "a beautiful dog"
# python test/test_ws_client.py --api-url http://127.0.0.1:8888 --ws-url ws://127.0.0.1:8888
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


async def create_task(token: str, model_id: str = None, prompt: str = "a beautiful sunset"):
    """
    通过API创建任务
    
    Args:
        token: JWT token
        model_id: 可选的模型ID
        prompt: 任务提示词
    
    Returns:
        task_id: 任务ID，如果创建失败返回None
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # 确定使用的模型ID
            test_model_id = model_id
            if not test_model_id:
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
                        print(f"✓ 使用数据库中的模型: {test_model_id}")
                else:
                    print("⚠ 警告: 没有找到活跃的image模型，任务将不使用model_id")
            
            # 构建请求数据
            request_data = {
                "task_type": "image",
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "storage": "local",
                "generate_num":1,
                "fast_mode":True,
            }
            
            # 如果有模型ID，添加到请求中
            if test_model_id:
                request_data["model_id"] = test_model_id
            
            # 发送创建任务请求
            response = await client.post(
                f"{API_BASE_URL}/v1/tasks",
                json={"request": request_data},
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code != 201:
                print(f"❌ API返回错误: {response.status_code}")
                print(f"❌ 响应内容: {response.text}")
                return None
            
            result = response.json()
            task_id = result.get("task_id")
            
            if not task_id:
                print(f"❌ API响应中没有task_id: {result}")
                return None
            
            print(f"✓ 任务已创建: {task_id}")
            print(f"✓ 任务状态: {result.get('status', 'unknown')}")
            return task_id
            
        except httpx.TimeoutException:
            print(f"❌ API请求超时")
            return None
        except Exception as e:
            print(f"❌ API请求失败: {e}")
            import traceback
            traceback.print_exc()
            return None


async def connect_websocket(task_id: str, token: str):
    """
    连接WebSocket服务器并接收任务状态更新
    
    Args:
        task_id: 任务ID
        token: JWT token
    """
    ws_url = f"{WS_BASE_URL}/ws/task/{task_id}?token={token}"
    
    print(f"\n✓ 正在连接WebSocket服务器: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"✓ WebSocket连接成功")
            print(f"\n等待任务状态更新...")
            print("=" * 60)
            
            # 接收消息
            message_count = 0
            while True:
                try:
                    # 接收消息（超时设置为None，表示一直等待）
                    message = await websocket.recv()
                    message_count += 1
                    
                    # 解析JSON消息
                    try:
                        data = json.loads(message)
                        print(f"\n[消息 #{message_count}]")
                        print(f"类型: {data.get('type', 'unknown')}")
                        print(f"任务ID: {data.get('task_id', 'unknown')}")
                        
                        # 根据消息类型显示不同信息
                        msg_type = data.get("type")
                        if msg_type == "task_status":
                            print(f"状态: {data.get('status', 'unknown')}")
                            if "started_at" in data:
                                print(f"开始时间: {data.get('started_at')}")
                            if "completed_at" in data:
                                print(f"完成时间: {data.get('completed_at')}")
                            if "error" in data:
                                print(f"错误: {data.get('error')}")
                        elif msg_type == "result":
                            print(f"状态: {data.get('status', 'unknown')}")
                            print(f"进度: {data.get('progress', 0)}%")
                            files = data.get("files", [])
                            print(f"文件数量: {len(files)}")
                            for i, file_info in enumerate(files):
                                print(f"  文件 {i+1}: {file_info.get('file_name', 'unknown')}")
                                print(f"    路径: {file_info.get('storage_path', 'unknown')}")
                                print(f"    大小: {file_info.get('file_size', 0)} bytes")
                        else:
                            # 显示所有字段
                            print(f"完整消息:")
                            print(json.dumps(data, indent=2, ensure_ascii=False))
                        
                        print("-" * 60)
                        
                        # 如果任务完成或失败，可以选择退出
                        status = data.get("status")
                        if status in ["completed", "failed", "cancelled"]:
                            print(f"\n✓ 任务已结束，状态: {status}")
                            print("可以按 Ctrl+C 退出，或等待更多消息...")
                    
                    except json.JSONDecodeError:
                        # 如果不是JSON消息，直接打印
                        print(f"\n[消息 #{message_count}] (非JSON)")
                        print(message)
                        print("-" * 60)
                
                except ConnectionClosed:
                    print(f"\n✓ WebSocket连接已关闭")
                    break
                except Exception as e:
                    print(f"\n❌ 接收消息时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    break
    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ WebSocket连接失败: HTTP {e.status_code}")
        print(f"   响应: {e.headers if hasattr(e, 'headers') else 'N/A'}")
    except Exception as e:
        print(f"❌ WebSocket连接失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主函数"""
    # 声明全局变量（必须在首次使用之前）
    global API_BASE_URL, WS_BASE_URL
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="WebSocket测试客户端 - 通过API创建任务并接收WebSocket消息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型和提示词
  python test/test_ws_client.py
  
  # 指定模型ID
  python test/test_ws_client.py --model-id model_123
  
  # 指定提示词
  python test/test_ws_client.py --prompt "a cat sitting on a chair"
  
  # 指定模型ID和提示词
  python test/test_ws_client.py -m model_123 -p "a beautiful landscape"
        """
    )
    parser.add_argument(
        "-m", "--model-id",
        type=str,
        default=None,
        help="指定要使用的模型ID（如果不指定，则从数据库查找活跃的image模型）"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default="a beautiful sunset",
        help="任务提示词（默认: 'a beautiful sunset'）"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=API_BASE_URL,
        help=f"API服务地址（默认: {API_BASE_URL}）"
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default=WS_BASE_URL,
        help=f"WebSocket服务地址（默认: {WS_BASE_URL}）"
    )
    
    args = parser.parse_args()
    
    # 更新全局变量
    API_BASE_URL = args.api_url
    WS_BASE_URL = args.ws_url.replace("http://", "ws://").replace("https://", "wss://")
    
    print("\n" + "=" * 60)
    print("WebSocket测试客户端")
    print("=" * 60)
    print("\n前提条件:")
    print("  1. WebSocket Server已启动（python -m backend.app.main）")
    print("  2. 推理器已启动（python inference/image/main.py service_123）")
    print(f"  3. API服务地址: {API_BASE_URL}")
    print(f"  4. WebSocket服务地址: {WS_BASE_URL}")
    if args.model_key:
        print(f"  5. 使用指定的模型ID: {args.model_key}")
    else:
        print(f"  5. 模型ID: 将从数据库查找活跃的image模型")
    print(f"  6. 提示词: {args.prompt}")
    print("\n开始测试...")
    
    try:
        # 1. 创建测试用户并获取token
        print(f"\n{'='*60}")
        print("步骤1: 创建测试用户")
        print("="*60)
        user_id, token = await create_test_user_and_token()
        print(f"✓ 测试用户已创建: {user_id}")
        
        # 2. 通过API创建任务
        print(f"\n{'='*60}")
        print("步骤2: 通过API创建任务")
        print("="*60)
        task_id = await create_task(token, args.model_key, args.prompt)
        
        if not task_id:
            print("\n❌ 任务创建失败，退出")
            return 1
        
        # 3. 连接WebSocket并接收消息
        print(f"\n{'='*60}")
        print("步骤3: 连接WebSocket服务器")
        print("="*60)
        await connect_websocket(task_id, token)
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        return 0
    
    except KeyboardInterrupt:
        print("\n\n✓ 用户中断，退出测试")
        return 0
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

