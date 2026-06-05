"""
手动模拟用户侧：创建图片任务并通过 /ws/task/{task_id} 等待结果。

运行示例：
    python test/manual_image_inference_client.py --model-id model_88888888 --prompt "a cute cat"

要求：
- 后端 API/WS 已启动（默认 http://127.0.0.1:8888）
- 已存在推理服务连接 /ws/inference/{service_id}
"""
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import websockets

# 将项目根目录加入路径，便于复用后端工具函数
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.auth import create_access_token
from backend.database import User
from backend.utils import generate_uuid


API_BASE = "http://127.0.0.1:8888"
WS_BASE = "ws://127.0.0.1:8888"


@dataclass
class ClientOptions:
    model_id: Optional[str]
    prompt: str
    width: int = 1024
    height: int = 1024
    storage: str = "local"
    timeout: float = 120.0  # 等待推理完成的最长时间（秒）


def create_test_user_and_token() -> tuple[str, str]:
    """创建测试用户并返回 (user_id, token)。如果邮箱已存在则复用。"""
    test_user_id = f"manual_user_{generate_uuid()[:8]}"
    test_email = f"{test_user_id}@example.com"

    # User.get_by_email 若不存在返回 None
    user = User.get_by_email(test_email)
    if not user:
        User.create(
            id=test_user_id,
            email=test_email,
            password_hash="test_password_hash",
            nickname="Manual Test User",
        )
        user_id = test_user_id
    else:
        user_id = user["id"]

    token = create_access_token({"sub": user_id, "email": test_email})
    return user_id, token


async def create_task(opts: ClientOptions, token: str) -> str:
    """调用后端 API 创建任务，返回 task_id。"""
    request_payload = {
        "task_type": "image",
        "prompt": opts.prompt,
        "width": opts.width,
        "height": opts.height,
        "storage": opts.storage,
    }
    if opts.model_key:
        request_payload["model_id"] = opts.model_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{API_BASE}/v1/tasks",
            json={"request": request_payload},
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 201:
            raise RuntimeError(f"创建任务失败: {resp.status_code} {resp.text}")
        data = resp.json()
        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"响应缺少 task_id: {data}")
        print(f"✓ 任务已创建 task_id={task_id} status={data.get('status')}")
        return task_id


async def listen_ws(task_id: str, token: str, timeout: float):
    """连接 /ws/task/{task_id}?token=... 并输出推理结果消息。"""
    ws_url = f"{WS_BASE}/ws/task/{task_id}?token={token}"
    print(f"✓ 连接 WS: {ws_url}")
    async with websockets.connect(ws_url, ping_interval=None) as ws:
        done = asyncio.get_event_loop().create_future()

        async def receiver():
            try:
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        print(f"[WS] 原始消息: {msg}")
                        continue
                    print(f"[WS] {data}")
                    status = data.get("status")
                    if status in {"completed", "failed", "cancelled"}:
                        done.set_result(data)
                        break
            except Exception as e:
                if not done.done():
                    done.set_exception(e)

        recv_task = asyncio.create_task(receiver())
        try:
            result = await asyncio.wait_for(done, timeout=timeout)
            print(f"\n✓ 推理结束，最终状态: {result.get('status')}")
            if "big" in result:
                print(f"大图路径: {result.get('big')}")
            if "thumb" in result:
                print(f"缩略图: {result.get('thumb')}")
            if "images_big" in result:
                print(f"多图: {result.get('images_big')}")
        finally:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="手动测试用户侧图片推理 WS 客户端")
    parser.add_argument("--model-id", default="model_88888888", help="模型ID")
    parser.add_argument("--prompt", default="a beautiful landscape", help="提示词")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0, help="等待推理结束超时")
    args = parser.parse_args()

    opts = ClientOptions(
        model_id=args.model_key,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        timeout=args.timeout,
    )

    print("✓ 创建测试用户与 token ...")
    user_id, token = create_test_user_and_token()
    print(f"  user_id={user_id}")

    print("✓ 通过 API 创建任务 ...")
    task_id = await create_task(opts, token)

    print("✓ 等待 WS 推理结果 ...\n")
    await listen_ws(task_id, token, timeout=opts.timeout)


if __name__ == "__main__":
    asyncio.run(main())

