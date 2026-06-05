import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.supervisor_agent import app as agent_app
from inference.supervisor_agent.schemas import ProgramStatus
from inference.supervisor_agent import supervisorctl as supervisorctl_mod
from inference.supervisor_agent.supervisorctl import (
    SupervisorCtlError,
    SupervisorCtlResult,
    control_program,
    enrich_programs_with_service_ids,
    parse_status_output,
    resolve_supervisor_program_name,
    tail_program_logs,
)


client = TestClient(agent_app.app)


def set_agent_token(monkeypatch, token: str):
    monkeypatch.setattr(agent_app, "_configured_token", lambda: token)


def test_health_does_not_require_token(monkeypatch):
    set_agent_token(monkeypatch, "")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_v1_requires_configured_token(monkeypatch):
    set_agent_token(monkeypatch, "")

    response = client.get("/v1/programs")

    assert response.status_code == 503


def test_v1_rejects_invalid_token(monkeypatch):
    set_agent_token(monkeypatch, "secret")

    response = client.get("/v1/programs", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_list_programs_with_valid_token(monkeypatch):
    set_agent_token(monkeypatch, "secret")
    monkeypatch.setattr(
        agent_app,
        "list_programs",
        lambda: [
            ProgramStatus(
                name="image",
                state="RUNNING",
                description="pid 123, uptime 0:00:05",
                pid=123,
                uptime="0:00:05",
            )
        ],
    )

    response = client.get("/v1/programs", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json()["programs"][0]["name"] == "image"
    assert response.json()["programs"][0]["state"] == "RUNNING"


def test_control_rejects_supervisor_agent(monkeypatch):
    set_agent_token(monkeypatch, "secret")

    response = client.post(
        "/v1/programs/supervisor-agent/restart",
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 403


def test_control_rejects_invalid_program_name(monkeypatch):
    set_agent_token(monkeypatch, "secret")

    response = client.post(
        "/v1/programs/image%3Bshutdown/start",
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 400


def test_start_program_calls_supervisorctl(monkeypatch):
    calls = []
    set_agent_token(monkeypatch, "secret")

    def fake_run_supervisorctl(args):
        calls.append(args)
        return SupervisorCtlResult(stdout="image: started", stderr="", returncode=0)

    monkeypatch.setattr(supervisorctl_mod, "run_supervisorctl", fake_run_supervisorctl)

    response = client.post(
        "/v1/programs/image/start",
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert calls == [["start", "image"]]
    assert response.json()["output"] == "image: started"


def test_logs_are_limited_by_tail(monkeypatch):
    set_agent_token(monkeypatch, "secret")

    def fake_run_supervisorctl(args):
        assert args == ["tail", "image", "stdout"]
        return SupervisorCtlResult(stdout="one\ntwo\nthree\n", stderr="", returncode=0)

    monkeypatch.setattr(supervisorctl_mod, "run_supervisorctl", fake_run_supervisorctl)

    response = client.get(
        "/v1/programs/image/logs?tail=2",
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.json()["lines"] == ["two", "three"]


def test_get_global_config(monkeypatch, tmp_path):
    set_agent_token(monkeypatch, "secret")
    config_path = tmp_path / "inference.yaml"
    config_path.write_text("api_base_url: http://127.0.0.1:8888\n", encoding="utf-8")
    monkeypatch.setattr(agent_app, "read_global_config", lambda: {"api_base_url": "http://127.0.0.1:8888"})

    response = client.get("/v1/config/global", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json()["path"] == "inference.yaml"
    assert response.json()["config"]["api_base_url"] == "http://127.0.0.1:8888"


def test_put_service_config_merges(monkeypatch):
    set_agent_token(monkeypatch, "secret")
    writes = []

    def fake_write(service_id, patch):
        writes.append((service_id, patch))
        return ({"service_id": service_id, **patch}, f"{service_id}.yaml")

    monkeypatch.setattr(agent_app, "write_service_config", fake_write)

    response = client.put(
        "/v1/config/services/download",
        headers={"Authorization": "Bearer secret"},
        json={"config": {"config": {"civitai_token": "abc"}}},
    )

    assert response.status_code == 200
    assert writes == [("download", {"config": {"civitai_token": "abc"}})]


def test_parse_status_output_extracts_pid_and_uptime():
    output = "\n".join(
        [
            "image                            RUNNING   pid 123, uptime 0:01:02",
            "download                         STOPPED   May 26 10:00 AM",
        ]
    )

    programs = parse_status_output(output)

    assert programs[0].name == "image"
    assert programs[0].state == "RUNNING"
    assert programs[0].pid == 123
    assert programs[0].uptime == "0:01:02"
    assert programs[1].name == "download"
    assert programs[1].state == "STOPPED"


def test_control_stop_does_not_kill_supervised_process_on_command_error(monkeypatch):
    def fake_run_supervisorctl(args, *, timeout=15, check=True):
        if args == ["stop", "download"]:
            raise SupervisorCtlError("download: ERROR (no such process)", returncode=2)
        if args == ["status"]:
            return SupervisorCtlResult(
                stdout="download                         RUNNING   pid 123, uptime 0:01:02\n",
                stderr="",
                returncode=0,
            )
        raise AssertionError(f"unexpected supervisorctl args: {args}")

    monkeypatch.setattr(supervisorctl_mod, "run_supervisorctl", fake_run_supervisorctl)

    with pytest.raises(SupervisorCtlError):
        control_program("download", "stop")


def test_control_stop_treats_supervised_stopped_program_as_success(monkeypatch):
    def fake_run_supervisorctl(args, *, timeout=15, check=True):
        if args == ["stop", "download"]:
            raise SupervisorCtlError("download: ERROR (not running)", returncode=2)
        if args == ["status"]:
            return SupervisorCtlResult(
                stdout="download                         STOPPED   May 26 10:00 AM\n",
                stderr="",
                returncode=0,
            )
        raise AssertionError(f"unexpected supervisorctl args: {args}")

    monkeypatch.setattr(supervisorctl_mod, "run_supervisorctl", fake_run_supervisorctl)

    result = control_program("download", "stop")

    assert result.returncode == 0
    assert "already stopped" in result.output


def test_enrich_maps_supervisor_program_to_service_id_without_env(monkeypatch):
    for key in ("VITOOM_SUPERVISOR_CONF", "VITOOM_SERVICE_GROUP"):
        monkeypatch.delenv(key, raising=False)

    programs = enrich_programs_with_service_ids(
        [
            ProgramStatus(
                name="text",
                state="STOPPED",
                description="not running",
            )
        ]
    )

    assert programs[0].service_id == "text"


def test_resolve_supervisor_program_name_without_env(monkeypatch):
    for key in ("VITOOM_SUPERVISOR_CONF", "VITOOM_SERVICE_GROUP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        supervisorctl_mod,
        "list_programs",
        lambda: enrich_programs_with_service_ids(
            [
                ProgramStatus(
                    name="text",
                    state="STOPPED",
                    description="not running",
                )
            ]
        ),
    )

    assert resolve_supervisor_program_name("text") == "text"


def test_control_start_treats_running_program_as_success(monkeypatch):
    def fake_run_supervisorctl(args, *, timeout=15, check=True):
        if args == ["start", "text"]:
            raise SupervisorCtlError("text: ERROR (already started)", returncode=2)
        if args == ["status"]:
            return SupervisorCtlResult(
                stdout="text                             RUNNING   pid 123, uptime 0:01:02\n",
                stderr="",
                returncode=0,
            )
        raise AssertionError(f"unexpected supervisorctl args: {args}")

    monkeypatch.setattr(supervisorctl_mod, "run_supervisorctl", fake_run_supervisorctl)

    result = control_program("text", "start")

    assert result.returncode == 0
    assert "already running" in result.output


def test_enrich_maps_supervisor_program_to_service_id(monkeypatch, tmp_path):
    conf_path = tmp_path / "supervisord.conf"
    conf_path.write_text(
        "\n".join(
            [
                "[program:text]",
                "command=python inference/text/main.py text",
            ]
        ),
        encoding="utf-8",
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "text.yaml").write_text("service_id: text\n", encoding="utf-8")

    monkeypatch.setenv("VITOOM_SUPERVISOR_CONF", str(conf_path))
    monkeypatch.setattr(supervisorctl_mod, "_configured_service_ids", lambda: {"text"})

    programs = enrich_programs_with_service_ids(
        [
            ProgramStatus(
                name="text",
                state="STOPPED",
                description="not running",
            )
        ]
    )

    assert programs[0].service_id == "text"


def test_resolve_supervisor_program_name_from_process_scan_fallback(monkeypatch):
    for key in ("VITOOM_SUPERVISOR_CONF", "VITOOM_SERVICE_GROUP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        supervisorctl_mod,
        "list_programs",
        lambda: [
            ProgramStatus(
                name="text",
                state="STOPPED",
                description="not found by process scan",
                source="process_scan",
                service_id="text",
            )
        ],
    )

    assert resolve_supervisor_program_name("text") == "text"


def test_resolve_supervisor_program_name_by_service_id(monkeypatch):
    monkeypatch.setattr(
        supervisorctl_mod,
        "list_programs",
        lambda: [
            ProgramStatus(
                name="text",
                state="RUNNING",
                description="pid 1",
                pid=1,
                service_id="text",
            )
        ],
    )

    assert resolve_supervisor_program_name("text") == "text"


def test_start_service_uses_resolved_program_name(monkeypatch):
    calls = []
    set_agent_token(monkeypatch, "secret")
    monkeypatch.setattr(agent_app, "resolve_supervisor_program_name", lambda service_id: "text")

    def fake_control_program(name, action):
        calls.append((name, action))
        return SupervisorCtlResult(stdout="text: started", stderr="", returncode=0)

    monkeypatch.setattr(agent_app, "control_program", fake_control_program)

    response = client.post(
        "/v1/services/text/start",
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert calls == [("text", "start")]
    assert response.json()["name"] == "text"


def test_service_logs_use_resolved_program_name(monkeypatch):
    set_agent_token(monkeypatch, "secret")
    monkeypatch.setattr(agent_app, "resolve_supervisor_program_name", lambda service_id: "text")

    def fake_tail_program_logs(name, stream, tail):
        assert name == "text"
        assert stream == "stdout"
        assert tail == 3
        return ["a", "b", "c"]

    monkeypatch.setattr(agent_app, "tail_program_logs", fake_tail_program_logs)

    response = client.get(
        "/v1/services/text/logs?tail=3",
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "text"
    assert response.json()["lines"] == ["a", "b", "c"]


def test_tail_program_logs_reads_supervisor_log_file(monkeypatch, tmp_path):
    logs_dir = tmp_path / "logs" / "supervisor"
    logs_dir.mkdir(parents=True)
    (logs_dir / "text.log").write_text("line1\nline2\nline3\n", encoding="utf-8")

    def fake_run_supervisorctl(*args, **kwargs):
        raise SupervisorCtlError("supervisorctl tail failed")

    monkeypatch.setattr(supervisorctl_mod, "run_supervisorctl", fake_run_supervisorctl)
    monkeypatch.setattr(supervisorctl_mod, "_app_root", lambda: tmp_path)

    lines = tail_program_logs("text", "stdout", 2)

    assert lines == ["line2", "line3"]

