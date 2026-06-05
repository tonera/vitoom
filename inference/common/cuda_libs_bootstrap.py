"""
CUDA 运行时库 bootstrap

目的
----
PyTorch 在运行时会通过 nvrtc 即时编译部分 kernel（reduction/jit fusion 等），
需要配套的 `libnvrtc.so.<cuda_major>` 与 `libnvrtc-builtins.so.<cuda_major>.<cuda_minor>`。

但 torch wheel 本身在以下两种情况下**不会**自带这些 .so：
  1) cu130 wheel（NVIDIA 的惯例：nvrtc 被拆成独立 pip 包）
  2) 用户使用较新的 CUDA（如 13.x），而系统只装了老版 toolkit（如 12.4）

这会导致运行到 OCR 等带 GPU reduction 的代码时炸出：
    nvrtc: error: failed to open libnvrtc-builtins.so.13.0.

本模块做的事
------------
1) 探测当前 torch 所需的 cuda_major.minor；
2) 找 pip 安装的 `nvidia-cuda-nvrtc` wheel（或其他 nvidia/* 子包）里是否已经有对应 .so；
3) 若 `LD_LIBRARY_PATH` 尚未覆盖到它们，就在把路径拼好后
   **re-exec 自身一次**（Linux 下 LD_LIBRARY_PATH 必须在进程启动前生效）；
4) 如果根本就没装 wheel，给出明确安装指引后直接退出（而不是让用户自己看 nvrtc 报错）。

这样最终用户只需要一条命令：

    pip install nvidia-cuda-nvrtc

剩下的无需手工设置 LD_LIBRARY_PATH，也无需改启动脚本。

用法
----
在 `inference/<service>/main.py` 最顶部、其它 torch 相关 import **之前**：

    from common.cuda_libs_bootstrap import ensure_cuda_runtime_libs  # noqa: E402
    ensure_cuda_runtime_libs()

非 Linux（macOS / Windows）、非 CUDA torch、或 nvrtc 已经可见时，本函数是 no-op。
"""
from __future__ import annotations

import glob
import os
import re
import site
import sys
from typing import Iterable, List, Optional, Set, Tuple

_ENV_FLAG = "_VITOOM_CUDA_LIBS_BOOTSTRAPPED"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _torch_cuda_version() -> Optional[Tuple[int, int]]:
    """返回 torch.version.cuda 的 (major, minor)，CPU-only / 非 CUDA 返回 None。"""
    try:
        import torch  # type: ignore
    except Exception:
        return None
    v = getattr(torch.version, "cuda", None)
    if not v:
        return None
    m = re.match(r"^(\d+)\.(\d+)", str(v).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _candidate_site_roots() -> List[str]:
    roots: List[str] = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(user_site)
    except Exception:
        pass
    # conda / venv 下的 prefix/lib/pythonX.Y/site-packages
    for p in sys.path:
        if p and p.endswith("site-packages") and p not in roots:
            roots.append(p)
    return [r for r in roots if r and os.path.isdir(r)]


def _discover_nvidia_lib_dirs() -> List[str]:
    """扫描所有 `<site-packages>/nvidia/*/lib` 目录，返回包含 CUDA 运行时 so 的目录列表。"""
    dirs: Set[str] = set()
    for root in _candidate_site_roots():
        patterns = [
            os.path.join(root, "nvidia", "*", "lib", "libnvrtc*.so*"),
            os.path.join(root, "nvidia", "*", "lib64", "libnvrtc*.so*"),
            os.path.join(root, "nvidia", "*", "lib", "libcudart.so*"),
            os.path.join(root, "nvidia", "*", "lib64", "libcudart.so*"),
        ]
        for pattern in patterns:
            for p in glob.glob(pattern):
                dirs.add(os.path.dirname(p))
    return sorted(dirs)


def _discover_torch_lib_dirs() -> List[str]:
    """Return ``site-packages/torch/lib`` dirs that ship PyTorch CUDA shared objects."""
    dirs: List[str] = []
    for root in _candidate_site_roots():
        lib_dir = os.path.join(root, "torch", "lib")
        if not os.path.isdir(lib_dir):
            continue
        if _dir_has_any(
            lib_dir,
            (
                "libtorch.so",
                "libtorch_cuda.so",
                "libtorch_python.so",
            ),
        ):
            dirs.append(lib_dir)
    return list(dict.fromkeys(dirs))


def discover_native_lib_dirs() -> List[str]:
    """Collect native library dirs needed before importing vLLM/torch extensions."""
    return list(dict.fromkeys([*_discover_torch_lib_dirs(), *_discover_nvidia_lib_dirs()]))


def _required_nvrtc_names(cuda_ver: Tuple[int, int]) -> List[str]:
    """torch 运行期精确需要的文件名（torch 编译时 hard-code 了 minor 号）。"""
    major, minor = cuda_ver
    return [
        f"libnvrtc.so.{major}",
        f"libnvrtc-builtins.so.{major}.{minor}",
    ]


def _dir_has_required(lib_dir: str, required: List[str]) -> bool:
    """目录里同时包含所有 required 文件（允许是符号链接）。"""
    try:
        entries = set(os.listdir(lib_dir))
    except Exception:
        return False
    for name in required:
        if name not in entries:
            return False
    return True


def _dir_has_any(lib_dir: str, names: Iterable[str]) -> bool:
    try:
        entries = set(os.listdir(lib_dir))
    except Exception:
        return False
    return any(name in entries for name in names)


def _ensure_soname_in_dir(lib_dir: str, soname: str) -> Optional[str]:
    """Ensure a required soname is present in lib_dir, creating a local symlink if safe.

    Some CUDA wheels ship versioned files such as ``libcudart.so.13`` but not the
    unversioned ``libcudart.so`` that libraries loaded via ctypes may request.
    """
    try:
        entries = set(os.listdir(lib_dir))
    except Exception:
        return None

    if soname in entries:
        return lib_dir

    prefix = f"{soname}."
    candidates = sorted(
        (name for name in entries if name.startswith(prefix)),
        key=lambda name: [int(p) if p.isdigit() else p for p in name[len(prefix) :].split(".")],
        reverse=True,
    )
    for name in candidates:
        link = _ensure_versioned_symlink(os.path.join(lib_dir, name), soname)
        if link:
            return os.path.dirname(link)
    return None


def _resolve_required_soname_dirs(candidates: List[str], sonames: Tuple[str, ...]) -> Tuple[List[str], List[str]]:
    dirs: List[str] = []
    missing: List[str] = []
    for soname in sonames:
        if _sonames_loadable((soname,)):
            continue
        found = False
        for lib_dir in candidates:
            resolved = _ensure_soname_in_dir(lib_dir, soname)
            if not resolved:
                continue
            dirs.append(resolved)
            found = True
            break
        if not found:
            missing.append(soname)
    return list(dict.fromkeys(dirs)), missing


def _required_soname_install_hint(names: Iterable[str]) -> str:
    missing = ", ".join(str(name) for name in names)
    if "libcudart.so.12" in set(names):
        return (
            f"Missing CUDA runtime library required by vLLM: {missing}.\n"
            "Install the CUDA 12 runtime wheel in this Python environment:\n"
            "    pip install nvidia-cuda-runtime-cu12\n"
            "Then restart the inference service so LD_LIBRARY_PATH can be bootstrapped."
        )
    return (
        f"Missing required CUDA runtime libraries: {missing}.\n"
        "Install the matching nvidia-cuda-runtime wheel and restart the inference service."
    )


def _sonames_loadable(names: Iterable[str]) -> bool:
    try:
        import ctypes

        for name in names:
            ctypes.CDLL(str(name))
        return True
    except OSError:
        return False


def _find_compatible_builtins_minor(lib_dir: str, major: int) -> Optional[str]:
    """在目录里找 `libnvrtc-builtins.so.<major>.<x>`（任意次版本号），返回绝对路径。

    场景：新版 nvidia-cuda-nvrtc wheel 跟随 CUDA 13.2 发布，目录里只有
        libnvrtc-builtins.so.13.2
    但 torch cu13 wheel 编译时把需求写死成
        libnvrtc-builtins.so.13.0
    nvrtc 在 CUDA 13.x 里是 minor-ABI 兼容的（NVIDIA 官方推荐做法就是对
    次版本号做 symlink / ld.so cache），所以 13.2 可以直接给 torch 当 13.0 用。
    """
    try:
        entries = os.listdir(lib_dir)
    except Exception:
        return None
    pat = re.compile(rf"^libnvrtc-builtins\.so\.{major}\.(\d+)$")
    best: Optional[Tuple[int, str]] = None
    for name in entries:
        m = pat.match(name)
        if not m:
            continue
        minor = int(m.group(1))
        cand = os.path.join(lib_dir, name)
        if best is None or minor > best[0]:
            best = (minor, cand)
    return best[1] if best else None


def _ensure_versioned_symlink(src_path: str, link_name: str) -> Optional[str]:
    """确保 `dirname(src_path)/link_name` 是一个软链接指向 `src_path`。

    src_path 例如: /.../nvidia/cu13/lib/libnvrtc-builtins.so.13.2
    link_name 例如: libnvrtc-builtins.so.13.0
    """
    if not os.path.isfile(src_path):
        return None
    lib_dir = os.path.dirname(src_path)
    link_path = os.path.join(lib_dir, link_name)

    # 已经存在且指向同一个文件：直接返回
    if os.path.lexists(link_path):
        try:
            if os.path.samefile(link_path, src_path):
                return link_path
        except Exception:
            pass
        # 存在但指向别的东西：保险起见不动
        return link_path

    try:
        os.symlink(os.path.basename(src_path), link_path)
        return link_path
    except OSError as e:
        # 目录只读（比如 wheel 装到系统路径）时，换到用户目录兜底
        if e.errno in (13, 30):  # EACCES / EROFS
            user_link_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "vitoom", "cuda_libs"
            )
            try:
                os.makedirs(user_link_dir, exist_ok=True)
                # 为了让符号链接能解析到原文件，还得在用户目录里顺便把原文件也 link 过去
                dst_src = os.path.join(user_link_dir, os.path.basename(src_path))
                if not os.path.lexists(dst_src):
                    os.symlink(src_path, dst_src)
                user_link = os.path.join(user_link_dir, link_name)
                if not os.path.lexists(user_link):
                    os.symlink(os.path.basename(dst_src), user_link)
                return user_link
            except Exception:
                return None
        return None


def _ld_library_path_covers(lib_dirs: List[str]) -> bool:
    """当前 LD_LIBRARY_PATH 是否包含所有目标目录。

    注意：这里不只是 "contains"，还必须保证这些目录排在系统 cuda-12.x 之类的路径前面，
    否则 dlopen 仍会被系统 cu12 的 libnvrtc 截胡。
    """
    if not lib_dirs:
        return False
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if not current:
        return False
    parts = [p for p in current.split(":") if p]
    real_parts = {os.path.realpath(p) for p in parts}
    return all(os.path.realpath(d) in real_parts for d in lib_dirs)


def _install_hint(cuda_ver: Tuple[int, int]) -> str:
    major = cuda_ver[0]
    if major >= 13:
        return (
            "Missing CUDA nvrtc runtime libs required by torch "
            f"({cuda_ver[0]}.{cuda_ver[1]}). Please install the matching pip wheel:\n"
            "    pip install nvidia-cuda-nvrtc\n"
            "If that still fails, try: pip install nvidia-cuda-nvrtc-cu13\n"
            "(This ships libnvrtc.so.<major> and libnvrtc-builtins.so.<major>.<minor> "
            "into site-packages/nvidia/.)"
        )
    return (
        f"Missing CUDA nvrtc runtime libs required by torch ({cuda_ver[0]}.{cuda_ver[1]}). "
        "Install via: pip install nvidia-cuda-nvrtc"
    )


def ensure_cuda_runtime_libs(
    *,
    verbose: bool = True,
    require_nvrtc: bool = True,
    required_sonames: Iterable[str] = (),
) -> None:
    """入口函数：如果需要，就把 pip 装的 CUDA 运行时库目录前置到 LD_LIBRARY_PATH 并 re-exec。

    行为：
    - 非 Linux：直接返回（macOS / Windows 通过其它机制处理）。
    - 没有 CUDA torch：直接返回。
    - 已经 re-exec 过一次（`_VITOOM_CUDA_LIBS_BOOTSTRAPPED=1`）：直接返回，避免死循环。
    - 已经在 `LD_LIBRARY_PATH` 里覆盖到了目标目录：直接返回。
    - 找到需要的 .so 且路径尚未覆盖：更新环境变量后用 `os.execv` re-exec 自身。
    - `require_nvrtc=True` 且完全没找到 nvrtc：打印明确安装指引后 `sys.exit(1)`。

    `require_nvrtc=False` 用于 vLLM 等场景：它们可能只需要 pip wheel 里的
    `libcudart.so.*` 被动态链接器看到，不应因为缺 nvrtc 而阻止服务启动。

    `required_sonames` 用于显式要求某些动态库（例如 vLLM wheel 依赖的
    `libcudart.so.12`）。如果当前进程和候选 nvidia wheel 路径都找不到它们，
    会在启动阶段直接退出并给出安装提示。
    """
    if not _is_linux():
        return
    sonames = tuple(str(name) for name in required_sonames if str(name).strip())
    if os.environ.get(_ENV_FLAG) == "1":
        if sonames and not _sonames_loadable(sonames):
            sys.stderr.write("\n[vitoom] " + _required_soname_install_hint(sonames) + "\n\n")
            sys.exit(1)
        return

    candidates = discover_native_lib_dirs()
    soname_dirs, missing_sonames = _resolve_required_soname_dirs(candidates, sonames) if sonames else ([], [])
    if missing_sonames and not _sonames_loadable(missing_sonames):
        sys.stderr.write("\n[vitoom] " + _required_soname_install_hint(missing_sonames) + "\n\n")
        sys.exit(1)

    cuda_ver = _torch_cuda_version()
    if cuda_ver is None:
        # CPU-only torch 仍可能需要显式 soname（例如 cu130 上的 vLLM 0.14）。
        all_dirs = soname_dirs or candidates
        if not all_dirs:
            return
        if _ld_library_path_covers(all_dirs):
            return
        new_prefix = ":".join(all_dirs)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = new_prefix if not existing else f"{new_prefix}:{existing}"
        new_env = dict(os.environ)
        new_env["LD_LIBRARY_PATH"] = new_ld
        new_env[_ENV_FLAG] = "1"
        if verbose:
            sys.stderr.write(
                "[vitoom] injecting CUDA runtime libs into LD_LIBRARY_PATH and re-exec:\n"
                f"  added={new_prefix}\n"
            )
        os.execve(sys.executable, [sys.executable] + sys.argv, new_env)

    major, minor = cuda_ver
    required = _required_nvrtc_names(cuda_ver)

    # 第一轮：精确匹配（目录里同时有 libnvrtc.so.<major> 和 libnvrtc-builtins.so.<major>.<minor>）
    matched = [d for d in candidates if _dir_has_required(d, required)]
    compat_dirs: List[str] = []

    # 第二轮：同 major 的兼容 minor（新版 wheel 随 CUDA 次版本升 minor，
    # torch 编译时写死的 minor 不一定和 wheel 一致；nvrtc 在同 major 里 ABI 兼容，
    # 这里就地造一个符号链接让 torch 能 dlopen 成功）。
    if not matched:
        builtins_target = f"libnvrtc-builtins.so.{major}.{minor}"
        so_major_target = f"libnvrtc.so.{major}"
        for d in candidates:
            compat_builtin = _find_compatible_builtins_minor(d, major)
            if compat_builtin is None:
                continue

            # nvrtc.so.<major> 的次版本无关，优先直接存在；否则再找 .so.<major>.x 做 link
            entries = set()
            try:
                entries = set(os.listdir(d))
            except Exception:
                entries = set()

            need_link_dir: Optional[str] = None

            if so_major_target not in entries:
                # 找 libnvrtc.so.<major>.<any>，给它做个到 libnvrtc.so.<major> 的链接
                any_nvrtc = None
                for name in entries:
                    m = re.match(rf"^libnvrtc\.so\.{major}\.\d+$", name)
                    if m:
                        any_nvrtc = os.path.join(d, name)
                        break
                if any_nvrtc is not None:
                    link = _ensure_versioned_symlink(any_nvrtc, so_major_target)
                    if link:
                        need_link_dir = os.path.dirname(link)
                else:
                    # 真的没有 .so.<major>.* —— 跳过这个目录
                    continue

            # 建 builtins 精确 minor 的链接
            link2 = _ensure_versioned_symlink(compat_builtin, builtins_target)
            if link2 is None:
                continue
            need_link_dir = need_link_dir or os.path.dirname(link2)
            compat_dirs.append(os.path.dirname(link2))
            if need_link_dir and need_link_dir != os.path.dirname(link2):
                compat_dirs.append(need_link_dir)

    if matched:
        all_dirs = matched
    elif compat_dirs:
        # 去重并保序
        seen: Set[str] = set()
        all_dirs = []
        for d in compat_dirs:
            if d in seen:
                continue
            seen.add(d)
            all_dirs.append(d)
    else:
        all_dirs = []

    if not all_dirs and require_nvrtc:
        # 系统路径最后一搏：pip wheel 没装 / 目录里完全没有 libnvrtc.so.<major>.*。
        # 试一次 dlopen，万一用户已经手动设好了 LD_LIBRARY_PATH。
        try:
            import ctypes

            for name in required:
                ctypes.CDLL(name)
            return
        except OSError:
            sys.stderr.write("\n[vitoom] " + _install_hint(cuda_ver) + "\n\n")
            sys.exit(1)

    if not all_dirs:
        # 非强制 nvrtc 场景：把发现到的 nvidia CUDA runtime 目录尽量前置。
        all_dirs = candidates

    if soname_dirs:
        # 显式依赖的 soname 目录必须参与注入，即使 nvrtc 目录已经匹配。
        all_dirs = list(dict.fromkeys([*soname_dirs, *all_dirs]))

    if _ld_library_path_covers(all_dirs):
        return

    new_prefix = ":".join(all_dirs)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = new_prefix if not existing else f"{new_prefix}:{existing}"

    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = new_ld
    new_env[_ENV_FLAG] = "1"

    if verbose:
        sys.stderr.write(
            "[vitoom] injecting CUDA runtime libs into LD_LIBRARY_PATH and re-exec:\n"
            f"  cuda={cuda_ver[0]}.{cuda_ver[1]}\n"
            f"  added={new_prefix}\n"
        )

    # re-exec 当前 python 进程，让新的 LD_LIBRARY_PATH 在 loader 初始化前生效
    os.execve(sys.executable, [sys.executable] + sys.argv, new_env)


__all__ = ["discover_native_lib_dirs", "ensure_cuda_runtime_libs"]
