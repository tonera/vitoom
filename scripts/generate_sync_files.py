#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 rsync --files-from 可用的同步清单。

默认：
  扫描 /home/tonera/models
  输出到 /home/tonera/models/sync_files.txt

清单格式：
  - 相对于 BASE 的相对路径
  - 目录以 "/" 结尾
  - 每行一个条目

默认行为（重要）：
  - **非递归**：只列出 base 目录的第一层文件/目录本身
  - 跳过以 "." 开头的文件和目录
  - 如果你希望展开子目录生成"全量递归清单"，使用 --recursive

python scripts/generate_sync_files.py --base /home/tonera/models 

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _rel_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def generate_list(base_dir: Path, recursive: bool) -> list[str]:
    base_dir = base_dir.resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"base dir not found: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"base dir is not a directory: {base_dir}")

    items: list[str] = []

    dir_paths: list[Path] = []
    file_paths: list[Path] = []

    if recursive:
        # 递归：先收集目录（不含根目录本身），再收集文件；都做排序，输出稳定
        for root, dirs, files in os.walk(base_dir, topdown=True, followlinks=False):
            root_p = Path(root)

            # 排序保证稳定输出
            dirs.sort()
            files.sort()

            for d in dirs:
                p = root_p / d
                dir_paths.append(p)
            for f in files:
                p = root_p / f
                file_paths.append(p)
    else:
        # 非递归：只列出第一层文件/目录本身（目录以 / 结尾）
        for p in sorted(base_dir.iterdir(), key=lambda x: x.name):
            # 跳过以 . 开头的文件和目录
            if p.name.startswith("."):
                continue
            if p.is_dir():
                dir_paths.append(p)
            elif p.is_file():
                file_paths.append(p)
            else:
                # 跳过 socket/pipe 等特殊文件
                continue

    for p in sorted(dir_paths):
        rel = _rel_posix(p, base_dir)
        # 目录以 / 结尾，方便一眼看出是目录
        items.append(rel.rstrip("/") + "/")

    for p in sorted(file_paths):
        rel = _rel_posix(p, base_dir)
        items.append(rel)

    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate sync file list for rsync --files-from (relative paths)."
    )
    parser.add_argument(
        "--base",
        default="/home/tonera/models",
        help="base directory to scan (default: /home/tonera/models)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output file path (default: {base_dir}/sync_files.txt)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively scan all subdirectories (default: false)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base).resolve()
    if args.out is None:
        out_path = base_dir / "sync_files.txt"
    else:
        out_path = Path(args.out)

    items = generate_list(base_dir, recursive=bool(args.recursive))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for line in items:
            f.write(line)
            f.write("\n")

    print(f"Wrote {len(items)} lines to: {out_path.resolve()}")
    print(f"BASE: {base_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

