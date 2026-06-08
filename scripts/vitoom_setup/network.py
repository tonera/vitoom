"""Host network helpers for setup."""

from __future__ import annotations

import platform
import re
import socket
import subprocess
from typing import Callable


def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def _parse_ipv4_from_text(text: str) -> list[str]:
    pattern = re.compile(
        r"\b("
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
        r")\b"
    )
    found: list[str] = []
    for match in pattern.finditer(text):
        ip = match.group(1)
        if ip not in found and not ip.endswith(".0") and not ip.endswith(".255"):
            found.append(ip)
    return found


def _run_command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "") + (result.stderr or "")


def _collect_ips_from_commands() -> list[str]:
    system = platform.system().lower()
    outputs: list[str] = []
    if system == "windows":
        outputs.append(_run_command_output(["ipconfig"]))
    elif system == "darwin":
        outputs.append(_run_command_output(["ifconfig"]))
    else:
        outputs.append(_run_command_output(["ip", "-4", "addr"]))
        if not outputs[-1].strip():
            outputs.append(_run_command_output(["ifconfig"]))
    ips: list[str] = []
    for text in outputs:
        for ip in _parse_ipv4_from_text(text):
            if ip not in ips:
                ips.append(ip)
    return ips


def _guess_primary_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def list_private_ipv4_addresses() -> list[str]:
    """Return private LAN IPs, preferring 192.168.x.x, then 10.x, then 172.16-31.x."""
    candidates = _collect_ips_from_commands()
    primary = _guess_primary_ip()
    if primary and primary not in candidates:
        candidates.insert(0, primary)

    def sort_key(ip: str) -> tuple[int, str]:
        if ip.startswith("192.168."):
            tier = 0
        elif ip.startswith("10."):
            tier = 1
        elif ip.startswith("172."):
            tier = 2
        else:
            tier = 3
        return (tier, ip)

    private = [ip for ip in candidates if sort_key(ip)[0] < 3]
    private.sort(key=sort_key)
    return private


def pick_ipv4(
    prompt_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
    *,
    allow_manual: bool = True,
    empty_message: str,
    select_message: str,
    manual_message: str,
) -> str:
    ips = list_private_ipv4_addresses()
    if not ips:
        if not allow_manual:
            raise SystemExit(empty_message)
        while True:
            manual = prompt_fn(manual_message).strip()
            if manual:
                return manual
            print_fn(empty_message)

    if len(ips) == 1:
        print_fn(f"{select_message}: {ips[0]}")
        return ips[0]

    print_fn(select_message)
    for index, ip in enumerate(ips, start=1):
        print_fn(f"  [{index}] {ip}")
    if allow_manual:
        print_fn(f"  [0] {manual_message}")
    while True:
        raw = prompt_fn("> ").strip()
        if not raw:
            return ips[0]
        if raw == "0" and allow_manual:
            manual = prompt_fn(manual_message).strip()
            if manual:
                return manual
            continue
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(ips):
                return ips[choice - 1]
        if raw in ips:
            return raw
        print_fn(empty_message)
