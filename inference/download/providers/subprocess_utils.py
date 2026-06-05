from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import List


def wrap_cmd_with_pdeathsig(cmd: List[str], *, sig: int = signal.SIGKILL) -> List[str]:
    """
    Linux-only: make the spawned process receive a signal automatically when its parent dies.

    Why:
    - If the parent process is SIGKILL'ed, Python can't run `finally` blocks, so normal cleanup can't happen.
    - With PR_SET_PDEATHSIG, the kernel delivers the signal to the child when parent exits, preventing orphans.

    Notes:
    - This guarantees the *direct* child process is killed when the parent dies.
    - If the child spawns grandchildren, killing the whole tree reliably typically requires cgroups/systemd/Docker.
    """
    if not cmd:
        return cmd
    if sys.platform != "linux":
        return cmd

    # Exec a tiny Python wrapper that sets PR_SET_PDEATHSIG then execvp() into the real command.
    py = r"""
import ctypes, ctypes.util, json, os, sys
PR_SET_PDEATHSIG = 1
sig = int(sys.argv[1])
cmd = json.loads(sys.argv[2])
libc_path = ctypes.util.find_library("c") or "libc.so.6"
libc = ctypes.CDLL(libc_path, use_errno=True)
libc.prctl(PR_SET_PDEATHSIG, sig)
if os.getppid() == 1:
    os.kill(os.getpid(), sig)
os.execvp(cmd[0], cmd)
""".strip()

    return [sys.executable, "-c", py, str(int(sig)), json.dumps(cmd)]


def _registry_file(models_dir: Path) -> Path:
    locks = (Path(models_dir).resolve() / ".locks")
    locks.mkdir(parents=True, exist_ok=True)
    return locks / "download_cli_procs.json"


def _load_registry(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_registry(path: Path, data: dict) -> None:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # best-effort
        return


def register_download_cli_proc(*, models_dir: Path, pid: int, kind: str, cmd: List[str]) -> None:
    """
    Record a spawned CLI subprocess PID, so that on next service startup we can clean up stale orphans
    without "kill by name" (which is dangerous on user machines).
    """
    try:
        path = _registry_file(models_dir)
        reg = _load_registry(path)
        service_id = str(os.environ.get("VITOOM_SERVICE_ID") or "")
        parent_pid = str(os.getpid())

        entries = reg.get(parent_pid)
        if not isinstance(entries, list):
            entries = []

        # de-dup by pid
        entries = [e for e in entries if not (isinstance(e, dict) and int(e.get("pid") or 0) == int(pid))]

        entry = {
            "pid": int(pid),
            "kind": str(kind or ""),
            "cmd": list(cmd or []),
            "service_id": service_id,
            "parent_pid": int(os.getpid()),
            "created_at": time.time(),
        }

        # pgid hint for posix group-kill
        if hasattr(os, "getpgid"):
            with contextlib.suppress(Exception):
                pgid = int(os.getpgid(int(pid)))
                if pgid == int(pid):
                    entry["pgid"] = pgid

        entries.append(entry)
        reg[parent_pid] = entries
        _save_registry(path, reg)
    except Exception:
        return


def unregister_download_cli_proc(*, models_dir: Path, pid: int) -> None:
    try:
        path = _registry_file(models_dir)
        reg = _load_registry(path)
        parent_pid = str(os.getpid())
        entries = reg.get(parent_pid)
        if not isinstance(entries, list):
            return
        new_entries = [e for e in entries if not (isinstance(e, dict) and int(e.get("pid") or 0) == int(pid))]
        if new_entries:
            reg[parent_pid] = new_entries
        else:
            reg.pop(parent_pid, None)
        _save_registry(path, reg)
    except Exception:
        return


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            kernel32.CloseHandle(h)
            return True
        except Exception:
            return False
    else:
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False


def _kill_pid_best_effort(pid: int, *, pgid: int | None = None) -> None:
    if pid <= 0:
        return
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            PROCESS_TERMINATE = 0x0001
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            h = kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
            if not h:
                return
            try:
                kernel32.TerminateProcess(h, 1)
            finally:
                kernel32.CloseHandle(h)
        except Exception:
            return
    else:
        # try killpg if we know it's a dedicated group leader
        if pgid and hasattr(os, "killpg"):
            with contextlib.suppress(Exception):
                os.killpg(int(pgid), signal.SIGKILL)
                return
        with contextlib.suppress(Exception):
            os.kill(int(pid), signal.SIGTERM)
        with contextlib.suppress(Exception):
            os.kill(int(pid), signal.SIGKILL)


def _cmd_local_dir_within_models_dir(cmd: List[str], models_dir: Path) -> bool:
    """
    Safety check to avoid killing unrelated processes:
    only allow cleanup when the recorded command line clearly targets our models_dir.
    """
    try:
        if not cmd:
            return False
        # Find "--local_dir <path>"
        local_dir = None
        for i, tok in enumerate(cmd):
            if str(tok) == "--local_dir" and i + 1 < len(cmd):
                local_dir = str(cmd[i + 1])
                break
        if not local_dir:
            return False
        abs_local = Path(local_dir).expanduser().resolve()
        abs_models = Path(models_dir).expanduser().resolve()
        # allow local_dir == models_dir or inside it
        abs_local.relative_to(abs_models)
        return True
    except Exception:
        return False


def cleanup_stale_download_cli_procs(*, models_dir: Path, service_id: str | None = None) -> int:
    """
    On service startup: kill leftover CLI subprocesses from previous crashed/forced-killed runs.
    Only targets processes recorded in our registry, optionally scoped by service_id.
    Returns number of stale PIDs we attempted to kill.
    """
    path = _registry_file(models_dir)
    reg = _load_registry(path)
    if not reg:
        return 0

    current_parent = str(os.getpid())
    sid = str(service_id or os.environ.get("VITOOM_SERVICE_ID") or "")
    killed = 0

    # We only consider entries under *other* parent pids as stale, to avoid killing a concurrently running service.
    for parent_pid, entries in list(reg.items()):
        if parent_pid == current_parent:
            continue
        if not isinstance(entries, list):
            continue
        new_entries: list = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            if sid and str(e.get("service_id") or "") != sid:
                new_entries.append(e)
                continue
            pid = int(e.get("pid") or 0)
            pgid = int(e.get("pgid") or 0) if e.get("pgid") is not None else None
            cmd = e.get("cmd") if isinstance(e.get("cmd"), list) else []
            # Extra safety: only kill when the recorded cmd targets our models_dir (e.g. "--local_dir ...")
            if not _cmd_local_dir_within_models_dir([str(x) for x in cmd], Path(models_dir)):
                new_entries.append(e)
                continue
            if _pid_alive(pid):
                _kill_pid_best_effort(pid, pgid=pgid)
                killed += 1
            # drop it from registry regardless (avoid repeated killing loops)
        if new_entries:
            reg[parent_pid] = new_entries
        else:
            reg.pop(parent_pid, None)

    _save_registry(path, reg)
    return killed


def best_effort_set_child_die_with_parent(proc: asyncio.subprocess.Process | None) -> None:
    """
    Best-effort cross-platform "parent dies => child dies" attachment.

    - Linux: handled by `wrap_cmd_with_pdeathsig()` (must be set in child before exec).
    - Windows: uses Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (best available for bare runs).

    This function must NOT raise (best effort).
    """
    if proc is None:
        return
    if proc.pid is None:
        return
    if sys.platform != "win32":
        return

    # Windows Job Object: kill all processes in the job when the job handle is closed.
    # Store handle on proc to keep it alive for the duration of the subprocess.
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # Constants
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", wintypes.ULARGE_INTEGER),
                ("WriteOperationCount", wintypes.ULARGE_INTEGER),
                ("OtherOperationCount", wintypes.ULARGE_INTEGER),
                ("ReadTransferCount", wintypes.ULARGE_INTEGER),
                ("WriteTransferCount", wintypes.ULARGE_INTEGER),
                ("OtherTransferCount", wintypes.ULARGE_INTEGER),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE

        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.INT,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL

        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE

        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        # Create job and set KILL_ON_JOB_CLOSE.
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(job)
            return

        # Open the child process and assign it to the job.
        ph = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, False, int(proc.pid))
        if not ph:
            kernel32.CloseHandle(job)
            return
        try:
            ok2 = kernel32.AssignProcessToJobObject(job, ph)
            if not ok2:
                # If parent is already in a job without breakaway, this may fail. Best-effort only.
                return
            setattr(proc, "_job_handle", job)
        finally:
            kernel32.CloseHandle(ph)
    except Exception:
        return


async def terminate_subprocess_tree(proc: asyncio.subprocess.Process | None) -> None:
    """
    Best-effort terminate a subprocess and its process group (when started with start_new_session=True).
    Prevents orphaned `modelscope download` / `hf download` processes when parent is cancelled (Ctrl+C).
    """
    if proc is None:
        return
    if proc.returncode is not None:
        return

    # Windows: if we attached a Job Object, kill the whole job.
    if sys.platform == "win32":
        try:
            job = getattr(proc, "_job_handle", None)
            if job:
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
                kernel32.TerminateJobObject.restype = wintypes.BOOL
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle.restype = wintypes.BOOL

                with contextlib.suppress(Exception):
                    kernel32.TerminateJobObject(job, 1)
                with contextlib.suppress(Exception):
                    kernel32.CloseHandle(job)
                setattr(proc, "_job_handle", None)

                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=5)
                return
        except Exception:
            pass

    # Try graceful termination first.
    # IMPORTANT: only killpg() when we are sure the subprocess is the leader of its own process group
    # (typically when created with start_new_session=True). Otherwise, killpg(proc.pid, ...) may
    # target an unrelated process group with the same id, which is dangerous.
    try:
        use_killpg = False
        if hasattr(os, "getpgid") and hasattr(os, "killpg"):
            try:
                pgid = os.getpgid(proc.pid)
                use_killpg = (pgid == proc.pid)
            except Exception:
                use_killpg = False
        if use_killpg:
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass

    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
        return
    except Exception:
        pass

    # Escalate to kill
    try:
        use_killpg = False
        if hasattr(os, "getpgid") and hasattr(os, "killpg"):
            try:
                pgid = os.getpgid(proc.pid)
                use_killpg = (pgid == proc.pid)
            except Exception:
                use_killpg = False
        if use_killpg:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass

    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except Exception:
        pass

