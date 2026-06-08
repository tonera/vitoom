"""
手动模拟“另一个项目”的 Redis list 队列客户端：
- 向 req list（channel）推入一条 task JSON
- 从 res list（reschannle）BRPOP 输出 result/task_status

运行示例：
  python3 test/manual_redis_queue_inference_client.py \
    --host 127.0.0.1 --port 6379 --pwd "" \
    --req atz.req --res atz.res \
    --prompt "a cute cat" --job-type MK
"""

import argparse
import json
import time
from datetime import datetime
from typing import Any, Dict

import redis


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def build_task(task_id: str, prompt: str, job_type: str, width: int, height: int) -> Dict[str, Any]:
    # 尽量贴近本项目 InferenceRequestParams.from_task_dict() 期望的“全量 task_data”
    return {
        "id": task_id,
        "task_id": task_id,
        "type": "image",
        "user_id": "manual_user",
        "prompt": prompt,
        "storage": "local",
        "params": {
            "job_type": job_type,
            "width": width,
            "height": height,
            "generate_num": 1,
            "file_type": "png",
        },
        "timestamp": now_iso(),
    }


def main():
    p = argparse.ArgumentParser(description="Manual Redis queue inference client")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--pwd", default="")
    p.add_argument("--db", type=int, default=0)
    p.add_argument("--req", required=True, help="请求队列 key（channel）")
    p.add_argument("--res", required=True, help="响应队列 key（reschannle）")
    p.add_argument("--push", choices=["lpush", "rpush"], default="lpush", help="推入 req 队列的方向")
    p.add_argument("--prompt", default="a beautiful landscape")
    p.add_argument("--job-type", default="MK")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=300, help="等待结果超时（秒）")
    args = p.parse_args()

    r = redis.Redis(host=args.host, port=args.port, password=args.pwd or None, db=args.db, decode_responses=True)

    task_id = f"manual_{int(time.time())}"
    task = build_task(task_id, args.prompt, args.job_type, args.width, args.height)
    payload = json.dumps(task, ensure_ascii=False)

    if args.push == "rpush":
        r.rpush(args.req, payload)
    else:
        r.lpush(args.req, payload)

    print(f"✓ 已推入任务: task_id={task_id} -> {args.req}")
    print("✓ 等待响应（BRPOP）...\n")

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        item = r.brpop(args.res, timeout=5)
        if not item:
            continue
        _, s = item
        try:
            msg = json.loads(s)
        except Exception:
            print(f"[raw] {s}")
            continue

        print(msg)
        status = (msg.get("status") or "").lower()
        mtype = (msg.get("type") or "").lower()
        if mtype in ("result", "task_status") and status in ("completed", "failed", "cancelled"):
            print("\n✓ 终态收到，结束。")
            return

    raise SystemExit(f"Timeout after {args.timeout}s waiting on {args.res}")


if __name__ == "__main__":
    main()

