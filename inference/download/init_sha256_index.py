"""
初始化 models_sha256.json

功能：
- 扫描 {models_dir}（含子目录）下所有“权重文件”
- 只处理大于指定常量阈值的文件（避免把小的配置/索引也塞进共享索引）
- 计算 sha256 并写入 {models_dir}/models_sha256.json，供下载器做去重/复用

用法（推荐从项目根目录执行）：
  python inference/download/init_sha256_index.py

可选：
  python inference/download/init_sha256_index.py --models-dir /abs/path/to/models
  
  python inference/download/init_sha256_index.py --files-from /abs/path/to/sync_files_image.txt
  
  python inference/download/init_sha256_index.py --min-size-bytes 104857600
  
  python inference/download/init_sha256_index.py --merge --dedupe --skip-indexed --prune-missing

python inference/download/init_sha256_index.py \
  --models-dir /home/tonera/ai/models \
  --files-from /home/tonera/ai/models/sync_files_image.txt \
  --merge --dedupe --skip-indexed --prune-missing

python inference/download/init_sha256_index.py \
  --models-dir /home/tonera/ai/models \
  --files-from /home/tonera/ai/models/sync_files_vv.txt \
  --merge --dedupe --skip-indexed --prune-missing

"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

# 只依赖本目录的 sha_index；不引入 inference/common，避免被本地环境依赖（psutil/aiohttp/PIL）卡住
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from sha_index import ShaIndex, file_sha256  # noqa: E402

logger = logging.getLogger("init_sha256_index")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ===== 可按需调整的“指定常量”阈值 =====
# 默认 50MB：通常权重文件远大于此；小文件（json/txt/索引）不进入共享 sha 索引。
MIN_WEIGHT_SIZE_BYTES: int = 50 * 1024 * 1024

# 常见权重后缀（按需扩展）
WEIGHT_EXTS = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".onnx",
)

def _try_replace_with_hardlink(existing: Path, current: Path) -> bool:
    """
    安全去重：将 current 原地替换为指向 existing 的硬链接。
    - 不先删除 current：先创建临时硬链接，再 os.replace 覆盖，避免失败丢文件
    - 若跨盘/权限问题导致硬链失败，返回 False
    """
    try:
        if not existing.exists() or not current.exists():
            return False

        # 已经是同一个文件（可能本来就是硬链接/同 inode），不需要处理
        try:
            if os.path.samefile(existing, current):
                return True
        except Exception:
            pass

        tmp = current.with_name(current.name + ".dedupe.tmp")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

        os.link(existing, tmp)  # 可能因跨设备/权限失败
        os.replace(tmp, current)  # 原子替换
        return True
    except Exception as e:
        logger.warning(f"[dedupe] hardlink replace failed: {current} -> {existing} ({e})")
        try:
            tmp = current.with_name(current.name + ".dedupe.tmp")
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def _resolve_repo_root() -> Path:
    # inference/download/init_sha256_index.py -> inference/download -> inference -> repo_root
    return Path(__file__).resolve().parents[2]


def _load_models_dir_from_inference_yaml() -> Path:
    """
    从 inference/config/inference.yaml 读取 models_dir，并按 InferenceConfig 规则解析为绝对路径：
    - 若 models_dir 是绝对路径：保持不变
    - 若是相对路径：以 repo_root 为基准
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 PyYAML：请安装 pyyaml 后再运行该初始化脚本") from e

    repo_root = _resolve_repo_root()
    cfg_path = repo_root / "inference" / "config" / "inference.yaml"
    if not cfg_path.exists():
        # 和 InferenceConfig 一致：找不到配置时用默认 resources/models（相对 repo_root）
        return (repo_root / "resources" / "models").resolve()

    raw = {}
    try:
        raw = yaml.safe_load(cfg_path.read_text("utf-8")) or {}
    except Exception as e:
        raise RuntimeError(f"读取/解析 inference.yaml 失败：{cfg_path} ({e})") from e

    models_dir = str(raw.get("models_dir") or "resources/models")
    p = Path(models_dir)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def _is_candidate_weight_file(p: Path) -> bool:
    if not p.is_file():
        return False
    name = p.name
    if name in ("models_sha256.json", "models_sha256.json.tmp"):
        return False
    if name.endswith(".tmp") or name.endswith(".part"):
        return False
    return p.suffix.lower() in WEIGHT_EXTS


def _iter_files(models_dir: Path, min_size_bytes: int) -> List[Tuple[Path, int]]:
    out: List[Tuple[Path, int]] = []
    # 使用 rglob 递归扫描
    for p in models_dir.rglob("*"):
        try:
            if not _is_candidate_weight_file(p):
                continue
            st = p.stat()
            if st.st_size < min_size_bytes:
                continue
            out.append((p, int(st.st_size)))
        except (FileNotFoundError, PermissionError):
            continue
        except Exception as e:
            logger.warning(f"Skip file due to error: {p} ({e})")
            continue
    # 稳定排序：先按路径，避免每次输出不同
    out.sort(key=lambda x: x[0].as_posix())
    return out


def _read_files_from_list(files_from: Path) -> List[str]:
    """
    读取 --files-from 清单（每行一条记录）：
    - 记录是以 models_dir 为根的相对路径
    - 忽略空行与以 # 开头的注释行
    """
    lines: List[str] = []
    for raw in files_from.read_text("utf-8", errors="ignore").splitlines():
        s = (raw or "").strip()
        if not s or s.startswith("#"):
            continue
        # 兼容可能的 Windows 行尾残留：strip 后一般已处理；再兜底去掉末尾 \r
        s = s.rstrip("\r")
        if s:
            lines.append(s)
    return lines


def _safe_join_under_root(root: Path, rel: str) -> Path | None:
    """
    将相对路径拼到 root 下，并确保最终 resolve 后仍在 root 内，避免越界（如 ../）。
    返回 None 表示非法/越界。
    """
    try:
        s = str(rel or "").strip()
        if not s:
            return None
        s = s.lstrip("/\\")
        # 统一分隔符（允许清单里写 windows 风格）
        s = s.replace("\\", "/")
        p = (root / s).resolve()
        p.relative_to(root.resolve())
        return p
    except Exception:
        return None


def _iter_files_from_rel_list(models_dir: Path, min_size_bytes: int, rel_paths: Iterable[str]) -> List[Tuple[Path, int]]:
    """
    仅扫描清单里列出的相对路径（文件或目录）：
    - 文件：若符合候选规则则纳入
    - 目录：递归扫描目录下候选文件
    - 不存在/无权限：跳过并记录 warning
    """
    out: List[Tuple[Path, int]] = []
    seen: Set[str] = set()  # 用 resolve 后的绝对路径去重

    for rel in rel_paths:
        abs_path = _safe_join_under_root(models_dir, rel)
        if abs_path is None:
            logger.warning(f"[files-from] illegal/out-of-root path skipped: {rel}")
            continue

        try:
            if not abs_path.exists():
                logger.warning(f"[files-from] missing path skipped: {rel}")
                continue

            targets: Iterable[Path]
            if abs_path.is_dir():
                targets = abs_path.rglob("*")
            else:
                targets = [abs_path]

            for p in targets:
                try:
                    if not _is_candidate_weight_file(p):
                        continue
                    st = p.stat()
                    if st.st_size < min_size_bytes:
                        continue
                    key = p.resolve().as_posix()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((p, int(st.st_size)))
                except (FileNotFoundError, PermissionError):
                    continue
                except Exception as e:
                    logger.warning(f"[files-from] Skip file due to error: {p} ({e})")
                    continue
        except (FileNotFoundError, PermissionError):
            logger.warning(f"[files-from] cannot access path skipped: {rel}")
            continue
        except Exception as e:
            logger.warning(f"[files-from] path scan failed skipped: {rel} ({e})")
            continue

    out.sort(key=lambda x: x[0].as_posix())
    return out


def _to_rel_posix(models_dir: Path, abs_path: Path) -> str:
    try:
        rel = abs_path.resolve().relative_to(models_dir.resolve())
        return rel.as_posix()
    except Exception:
        # 防御：不应发生，发生则跳过该文件
        return ""


async def _amain(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Initialize models_sha256.json by scanning models_dir.")
    parser.add_argument("--models-dir", default="", help="Override models_dir (absolute path preferred).")
    parser.add_argument(
        "--files-from",
        default="",
        help="Only hash files under these relative paths (one per line, relative to --models-dir). "
        "Supports file or directory entries; empty lines and lines starting with # are ignored.",
    )
    parser.add_argument(
        "--min-size-bytes",
        type=int,
        default=MIN_WEIGHT_SIZE_BYTES,
        help=f"Only include files >= this size (default={MIN_WEIGHT_SIZE_BYTES}).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge into existing index instead of rebuilding from scratch.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="If a file's sha256 already exists in index, replace it with a hardlink to save disk space (same filesystem only).",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Prune missing paths from existing models_sha256.json before/while saving (recommended with --merge).",
    )
    parser.add_argument(
        "--skip-indexed",
        action="store_true",
        help="Skip hashing files whose relative path is already present in existing index (only meaningful with --merge).",
    )
    args = parser.parse_args(argv)

    models_dir = Path(args.models_dir).expanduser().resolve() if args.models_dir else _load_models_dir_from_inference_yaml()
    files_from = str(args.files_from or "").strip()
    min_size = int(args.min_size_bytes or MIN_WEIGHT_SIZE_BYTES)
    merge = bool(args.merge)
    dedupe = bool(args.dedupe)
    prune_missing = bool(args.prune_missing)
    skip_indexed = bool(args.skip_indexed)

    if not models_dir.exists() or not models_dir.is_dir():
        logger.error(f"models_dir not found or not a directory: {models_dir}")
        return 2

    logger.info(f"models_dir={models_dir}")
    logger.info(f"files_from={files_from or '(none)'}")
    logger.info(f"min_size_bytes={min_size} (const_default={MIN_WEIGHT_SIZE_BYTES})")
    logger.info(f"merge={merge}")
    logger.info(f"dedupe={dedupe}")
    logger.info(f"prune_missing={prune_missing}")
    logger.info(f"skip_indexed={skip_indexed}")
    logger.info(f"weight_exts={WEIGHT_EXTS}")

    if files_from:
        ff = Path(files_from).expanduser().resolve()
        if not ff.exists() or not ff.is_file():
            logger.error(f"--files-from not found or not a file: {ff}")
            return 2
        rels = _read_files_from_list(ff)
        logger.info(f"[files-from] loaded {len(rels)} entries from: {ff}")
        candidates = _iter_files_from_rel_list(models_dir, min_size, rels)
    else:
        candidates = _iter_files(models_dir, min_size)
    logger.info(f"Found {len(candidates)} candidate weight files")

    index = ShaIndex(models_dir)
    if merge:
        index.load()
    else:
        index.data = {}

    if prune_missing and merge:
        removed = index.prune_missing()
        if removed:
            logger.info(f"[sha256] pruned missing paths from existing index: {removed}")

    indexed_paths = set()
    if merge and skip_indexed:
        # 索引结构：sha -> [rel_path...]
        for arr in (index.data or {}).values():
            if not isinstance(arr, list):
                continue
            for rp in arr:
                try:
                    s = str(rp or "").strip().lstrip("/\\")
                except Exception:
                    continue
                if s:
                    indexed_paths.add(s)
        logger.info(f"[sha256] loaded indexed rel_paths: {len(indexed_paths)}")

    done = 0
    deduped = 0
    skipped = 0
    for (p, size) in candidates:
        rp = _to_rel_posix(models_dir, p)
        if not rp:
            continue
        if merge and skip_indexed and rp in indexed_paths:
            skipped += 1
            continue
        try:
            # 可观测性：输出到终端
            logger.info(f"[sha256] hashing: {rp} ({size} bytes)")
            sha = file_sha256(p)

            # 若命中已有 sha：可选进行“去重-硬链接”，以节约磁盘
            if dedupe:
                existing = index.find_one(sha)
                if existing is not None:
                    ok = _try_replace_with_hardlink(existing, p)
                    if ok:
                        deduped += 1
                        existing_rp = _to_rel_posix(models_dir, existing)
                        logger.info(f"[删除重复文件] {rp} -> {existing_rp or existing.name}")

            index.add(sha, rp)
            done += 1
            # 每处理一定数量，刷一次磁盘，避免中途退出导致全丢
            if done % 50 == 0:
                if prune_missing:
                    removed = index.prune_missing()
                    if removed:
                        logger.info(f"[sha256] pruned missing paths: {removed}")
                await index.save()
                logger.info(f"[sha256] checkpoint saved: {done}/{len(candidates)}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(f"[sha256] failed: {rp} ({e})")
            continue

    if prune_missing:
        removed = index.prune_missing()
        if removed:
            logger.info(f"[sha256] pruned missing paths: {removed}")
    await index.save()
    logger.info(
        f"Done. Indexed files: {done}. Skipped (already indexed): {skipped}. "
        f"Deduped (hardlink replaced): {deduped}. Index written to: {index.index_path}"
    )
    return 0


def main():
    try:
        raise SystemExit(asyncio.run(_amain(sys.argv[1:])))
    except KeyboardInterrupt:
        # 允许中断：尽量保存当前进度
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()

