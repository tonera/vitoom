from __future__ import annotations

import json
import os
import time
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Optional



def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ShaIndex:
    """
    {models_dir}/models_sha256.json
    {
      "version": 1,
      "updated_at": "...",
      "files": { "sha": ["rel/path1", "rel/path2"] }
    }
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.index_path = self.models_dir / "models_sha256.json"
        self.data: Dict[str, List[str]] = {}
        self._loaded = False

    def load(self):
        # 单例运行：不做互斥；每次 load 都允许刷新到最新索引文件
        self._loaded = True
        if not self.index_path.exists():
            self.data = {}
            return
        try:
            obj = json.loads(self.index_path.read_text("utf-8"))
            files = obj.get("files") if isinstance(obj, dict) else {}
            self.data = files if isinstance(files, dict) else {}
        except Exception:
            self.data = {}

    async def save(self):
        obj = {
            "version": 1,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": self.data,
        }
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self.index_path)

    def add(self, sha256: str, rel_path: str):
        sha = str(sha256 or "").strip().lower()
        rp = str(rel_path or "").strip().lstrip("/\\")
        if not sha or not rp:
            return
        arr = self.data.get(sha) or []
        if rp not in arr:
            arr.append(rp)
        self.data[sha] = arr

    def remove(self, sha256: str, rel_path: str):
        """从索引中移除某个 sha 对应的指定路径；若该 sha 无剩余路径则删除 key。"""
        sha = str(sha256 or "").strip().lower()
        rp = str(rel_path or "").strip().lstrip("/\\")
        if not sha or not rp:
            return
        arr = self.data.get(sha) or []
        if rp in arr:
            arr = [x for x in arr if x != rp]
        if arr:
            self.data[sha] = arr
        else:
            self.data.pop(sha, None)

    def prune_missing(self, sha256: Optional[str] = None) -> int:
        """
        清理索引中的“缺失路径”：
        - 若 sha256=None：遍历整个索引
        - 否则只清理指定 sha
        返回：删除的路径数量
        """
        removed = 0
        if sha256 is not None:
            sha = str(sha256 or "").strip().lower()
            arr = list(self.data.get(sha) or [])
            if not arr:
                return 0
            kept = []
            for rp in arr:
                p = (self.models_dir / rp).resolve()
                try:
                    p.relative_to(self.models_dir.resolve())
                except Exception:
                    removed += 1
                    continue
                if p.exists():
                    kept.append(rp)
                else:
                    removed += 1
            if kept:
                self.data[sha] = kept
            else:
                self.data.pop(sha, None)
            return removed

        # 全量清理
        for sha in list(self.data.keys()):
            removed += self.prune_missing(sha)
        return removed

    def find_one(self, sha256: str) -> Optional[Path]:
        sha = str(sha256 or "").strip().lower()
        arr = self.data.get(sha) or []
        for rp in arr:
            p = (self.models_dir / rp).resolve()
            try:
                # 防目录穿越
                p.relative_to(self.models_dir.resolve())
            except Exception:
                continue
            if p.exists():
                return p
        return None

    def try_link(self, src_abs: Path, dst_abs: Path) -> bool:
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        if dst_abs.exists():
            return True
        # 防御：禁止对“不存在的源文件”创建软链接（否则会产生断链，看似成功实则坏）
        if not src_abs or not Path(src_abs).exists():
            return False
        try:
            os.link(src_abs, dst_abs)
            return True
        except Exception:
            try:
                os.symlink(src_abs, dst_abs)
                return True
            except Exception:
                return False

