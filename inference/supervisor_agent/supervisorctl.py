from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Sequence

import psutil
import yaml

from .schemas import ProgramStatus


SUPERVISOR_SERVER_URL = os.getenv("VITOOM_SUPERVISOR_SOCKET", "unix:///tmp/supervisor.sock")
DEFAULT_TIMEOUT_SECONDS = 15
PROGRAM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_STATUS_LINE_PATTERN = re.compile(r"^(?P<name>\S+)\s+(?P<state>[A-Z]+)\s*(?P<description>.*)$")
_PID_PATTERN = re.compile(r"\bpid\s+(?P<pid>\d+)\b")
_UPTIME_PATTERN = re.compile(r"\buptime\s+(?P<uptime>\S+)")
_INFERENCE_MAIN_PATTERN = re.compile(r"(?:^|/)inference/(?:[^/]+/)*main\.py$")
_LOG_SUFFIXES = (".log", ".out", ".err", ".txt")


class SupervisorCtlError(RuntimeError):
    def __init__(self, message: str, *, returncode: int | None = None):
        super().__init__(message)
        self.returncode = returncode


@dataclass(frozen=True)
class SupervisorCtlResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def output(self) -> str:
        return (self.stdout or self.stderr).strip()


def validate_program_name(name: str) -> str:
    if not PROGRAM_NAME_PATTERN.fullmatch(name):
        raise ValueError("Program name may only contain letters, numbers, '.', '_' and '-'.")
    return name


def run_supervisorctl(
    args: Sequence[str],
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    check: bool = True,
) -> SupervisorCtlResult:
    command = ["supervisorctl", "-s", SUPERVISOR_SERVER_URL, *args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SupervisorCtlError(f"supervisorctl timed out after {timeout}s") from exc
    except OSError as exc:
        raise SupervisorCtlError(f"failed to execute supervisorctl: {exc}") from exc

    result = SupervisorCtlResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
    if check and completed.returncode != 0:
        message = result.output or f"supervisorctl exited with code {completed.returncode}"
        raise SupervisorCtlError(message, returncode=completed.returncode)
    return result


def parse_status_output(output: str) -> list[ProgramStatus]:
    programs: list[ProgramStatus] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _STATUS_LINE_PATTERN.match(line)
        if not match:
            continue

        description = match.group("description").strip()
        pid_match = _PID_PATTERN.search(description)
        uptime_match = _UPTIME_PATTERN.search(description)
        programs.append(
            ProgramStatus(
                name=match.group("name"),
                state=match.group("state"),
                description=description,
                pid=int(pid_match.group("pid")) if pid_match else None,
                uptime=uptime_match.group("uptime") if uptime_match else None,
            )
        )
    return programs


def _supervised_program_state(name: str) -> str | None:
    result = run_supervisorctl(["status"], check=False)
    if result.returncode != 0:
        raise SupervisorCtlError(
            result.output or f"supervisorctl status exited with code {result.returncode}",
            returncode=result.returncode,
        )

    for program in parse_status_output(result.output):
        if program.name == name:
            return program.state
    return None


def _format_uptime(create_time: float | None) -> str | None:
    if not create_time:
        return None
    seconds = max(0, int(datetime.now().timestamp() - create_time))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _configured_service_ids() -> set[str]:
    config_dir = Path(__file__).resolve().parents[1] / "config"
    service_ids: set[str] = set()
    for path in config_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        service_id = str(data.get("service_id") or "").strip()
        if service_id:
            service_ids.add(service_id)
    return service_ids


def _service_id_from_cmdline(cmdline: list[str], configured_ids: set[str]) -> str:
    for item in reversed(cmdline):
        value = str(item or "").strip()
        if value in configured_ids:
            return value
    for index, item in enumerate(cmdline):
        if _INFERENCE_MAIN_PATTERN.search(str(item or "")) and index + 1 < len(cmdline):
            candidate = str(cmdline[index + 1] or "").strip()
            if validate_candidate_program_name(candidate):
                return candidate
    return ""


def validate_candidate_program_name(name: str) -> bool:
    return bool(name and PROGRAM_NAME_PATTERN.fullmatch(name))


def _pid_to_service_id() -> dict[int, str]:
    configured_ids = _configured_service_ids()
    mapping: dict[int, str] = {}

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = [str(part) for part in (proc.info.get("cmdline") or [])]
            pid = int(proc.info.get("pid") or 0)
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
            continue
        if not cmdline or pid <= 0:
            continue
        if not any(_INFERENCE_MAIN_PATTERN.search(part) for part in cmdline):
            continue
        service_id = _service_id_from_cmdline(cmdline, configured_ids)
        if service_id:
            mapping[pid] = service_id
    return mapping


def _supervisor_conf_paths() -> list[Path]:
    candidates: list[Path] = []
    env_path = str(os.getenv("VITOOM_SUPERVISOR_CONF") or "").strip()
    if env_path:
        candidates.append(Path(env_path))
    service_group = str(os.getenv("VITOOM_SERVICE_GROUP") or "").strip()
    if service_group:
        candidates.append(
            Path(__file__).resolve().parents[2]
            / "docker"
            / "inference"
            / "services"
            / service_group
            / "supervisord.conf"
        )
    services_root = Path(__file__).resolve().parents[2] / "docker" / "inference" / "services"
    if services_root.is_dir():
        candidates.extend(sorted(services_root.glob("*/supervisord.conf")))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _supervisor_conf_path() -> Path | None:
    for path in _supervisor_conf_paths():
        if path.is_file():
            return path
    return None


def _program_service_ids_from_supervisor_conf() -> dict[str, str]:
    configured_ids = _configured_service_ids()
    mapping: dict[str, str] = {}

    for path in _supervisor_conf_paths():
        if not path.is_file():
            continue
        current_program = ""
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("[program:") and line.endswith("]"):
                current_program = line[len("[program:") : -1].strip()
                continue
            if not current_program or not line.startswith("command="):
                continue
            cmdline = line.split("=", 1)[1].strip().split()
            service_id = _service_id_from_cmdline(cmdline, configured_ids)
            if service_id:
                mapping[current_program] = service_id
    return mapping


def enrich_programs_with_service_ids(programs: list[ProgramStatus]) -> list[ProgramStatus]:
    configured_ids = _configured_service_ids()
    pid_map = _pid_to_service_id()
    conf_map = _program_service_ids_from_supervisor_conf()
    enriched: list[ProgramStatus] = []

    for program in programs:
        service_id = ""
        if program.name in configured_ids:
            service_id = program.name
        elif program.pid and program.pid in pid_map:
            service_id = pid_map[program.pid]
        elif program.name in conf_map:
            service_id = conf_map[program.name]

        enriched.append(
            program.model_copy(
                update={"service_id": service_id or None},
            )
        )
    return enriched


def resolve_supervisor_program_name(service_id: str) -> str:
    normalized = validate_program_name(service_id.strip())
    programs = list_programs()
    scanned: list[ProgramStatus] = []

    for program in programs:
        if program.source == "process_scan":
            scanned.append(program)
            continue
        if program.service_id == normalized or program.name == normalized:
            return program.name

    for program_name, mapped_service_id in _program_service_ids_from_supervisor_conf().items():
        if mapped_service_id == normalized:
            return program_name

    for program in scanned:
        if program.service_id == normalized or program.name == normalized:
            return program.name

    raise SupervisorCtlError(f"service_id {normalized!r} not found among supervised programs")


def control_service(service_id: str, action: str) -> SupervisorCtlResult:
    program_name = resolve_supervisor_program_name(service_id)
    return control_program(program_name, action)


def tail_service_logs(service_id: str, stream: str, tail: int) -> list[str]:
    program_name = resolve_supervisor_program_name(service_id)
    return tail_program_logs(program_name, stream, tail)


def scan_inference_processes() -> list[ProgramStatus]:
    configured_ids = _configured_service_ids()
    running: dict[str, ProgramStatus] = {}

    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            cmdline = [str(part) for part in (proc.info.get("cmdline") or [])]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not cmdline:
            continue
        if not any(_INFERENCE_MAIN_PATTERN.search(part) for part in cmdline):
            continue

        service_id = _service_id_from_cmdline(cmdline, configured_ids)
        if not service_id:
            continue

        command = " ".join(cmdline)
        running[service_id] = ProgramStatus(
            name=service_id,
            state="RUNNING",
            description=f"pid {proc.info.get('pid')}, process scan",
            pid=int(proc.info.get("pid") or 0) or None,
            uptime=_format_uptime(proc.info.get("create_time")),
            source="process_scan",
            command=command,
            service_id=service_id,
        )

    for service_id in configured_ids:
        running.setdefault(
            service_id,
            ProgramStatus(
                name=service_id,
                state="STOPPED",
                description="not found by process scan",
                source="process_scan",
                service_id=service_id,
            ),
        )

    return sorted(running.values(), key=lambda item: item.name)


def control_program(name: str, action: str) -> SupervisorCtlResult:
    try:
        return run_supervisorctl([action, name])
    except SupervisorCtlError as supervisor_exc:
        try:
            state = _supervised_program_state(name)
        except SupervisorCtlError:
            state = None
        else:
            if action == "stop" and state in {"STOPPED", "EXITED", "FATAL", "BACKOFF"}:
                return SupervisorCtlResult(
                    stdout=f"{name}: already stopped ({state})",
                    stderr="",
                    returncode=0,
                )
            if action == "start" and state == "RUNNING":
                return SupervisorCtlResult(
                    stdout=f"{name}: already running",
                    stderr="",
                    returncode=0,
                )
            if action == "start" and state in {"STOPPED", "EXITED", "FATAL", "BACKOFF"}:
                lowered = supervisor_exc.args[0].casefold()
                if "already started" in lowered:
                    return SupervisorCtlResult(
                        stdout=f"{name}: already running",
                        stderr="",
                        returncode=0,
                    )
            raise supervisor_exc

        raise supervisor_exc


def _app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _supervisor_logs_dir() -> Path:
    return _app_root() / "logs" / "supervisor"


def _tail_file(path: Path, line_count: int) -> list[str]:
    lines: deque[str] = deque(maxlen=max(1, line_count))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line.rstrip("\n"))
    return list(lines)


def _candidate_log_paths(name: str, stream: str) -> list[Path]:
    logs_root = _app_root() / "logs"
    supervisor_logs = _supervisor_logs_dir()
    names = [
        f"{name}.log",
        f"{name}.{stream}.log",
        f"{name}.out.log",
        f"{name}.err.log",
        f"{name}.out",
        f"{name}.err",
    ]
    if stream == "stderr":
        names = [
            f"{name}.stderr.log",
            f"{name}.err.log",
            f"{name}.err",
            f"{name}.log",
        ]

    candidates: list[Path] = []
    for base in (supervisor_logs, logs_root, logs_root / "supervisor", _app_root()):
        for filename in names:
            candidates.append(base / filename)

    if logs_root.exists():
        for path in logs_root.rglob(f"*{name}*"):
            if path.is_file() and path.suffix.lower() in _LOG_SUFFIXES:
                candidates.append(path)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def tail_program_logs(name: str, stream: str, tail: int) -> list[str]:
    try:
        result = run_supervisorctl(["tail", name, stream])
        lines = result.stdout.splitlines()
        if lines:
            return lines[-tail:]
    except SupervisorCtlError:
        pass

    for path in _candidate_log_paths(name, stream):
        if path.exists() and path.is_file():
            return _tail_file(path, tail)

    return [
        "No log file found for this program.",
        f"Expected logs under: {_supervisor_logs_dir()}",
        "If the service was just started, wait a few seconds and refresh.",
    ]


def list_programs() -> list[ProgramStatus]:
    try:
        result = run_supervisorctl(["status"])
        return enrich_programs_with_service_ids(parse_status_output(result.stdout))
    except SupervisorCtlError:
        return scan_inference_processes()

