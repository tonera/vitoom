#!/usr/bin/env python3
"""从命令行注册用户（使用与 API 相同的数据库与 `register_user` 逻辑）。

示例:
  python scripts/create_user.py --email user@example.com --password 'your_password'
  python scripts/create_user.py -e user@example.com -p 'your_password' --nickname 昵称
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.api.auth.service import register_user  # noqa: E402
from backend.core.exceptions import UserAlreadyExistsException  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="注册 Vitoom 用户（邮箱 + 密码）")
    parser.add_argument("-e", "--email", required=True, help="用户邮箱")
    parser.add_argument("-p", "--password", required=True, help="密码（至少 6 位）")
    parser.add_argument("-n", "--nickname", default=None, help="可选昵称")
    args = parser.parse_args()

    if len(args.password) < 6:
        print("错误: 密码长度至少 6 位", file=sys.stderr)
        return 2

    try:
        user = register_user(
            email=args.email,
            password=args.password,
            nickname=args.nickname,
        )
    except UserAlreadyExistsException as exc:
        print(f"错误: {exc.message}", file=sys.stderr)
        return 1

    if not user:
        print("错误: 创建用户失败（详见日志）", file=sys.stderr)
        return 1

    print("已创建用户:")
    print(f"  id:       {user['id']}")
    print(f"  email:    {user['email']}")
    print(f"  nickname: {user.get('nickname')}")
    print(f"  is_admin: {user.get('is_admin', False)}")
    print(f"  status:   {user['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
