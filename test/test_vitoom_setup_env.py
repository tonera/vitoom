"""Vitoom setup .env helpers."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vitoom_setup.constants import DEFAULT_ADMIN_PASSWORD_LENGTH  # noqa: E402
from vitoom_setup.env_file import (  # noqa: E402
    generate_admin_password,
    is_placeholder_admin_password,
    is_valid_backend_url,
    resolve_admin_password,
    resolve_backend_url_from_input,
)


def test_resolve_backend_url_accepts_default_shortcuts():
    default = "http://192.168.31.163:8888"
    for raw in ("", "1", "y", "YES", "default"):
        assert resolve_backend_url_from_input(raw, default) == default


def test_resolve_backend_url_accepts_explicit_url():
    assert (
        resolve_backend_url_from_input("http://10.0.0.2:9000/", "http://192.168.1.1:8888")
        == "http://10.0.0.2:9000"
    )


def test_resolve_backend_url_rejects_invalid():
    assert resolve_backend_url_from_input("not-a-url", "http://192.168.1.1:8888") is None


def test_is_valid_backend_url_rejects_localhost():
    assert not is_valid_backend_url("http://127.0.0.1:8888")
    assert is_valid_backend_url("http://192.168.31.163:8888")


def test_generate_admin_password_length():
    password = generate_admin_password()
    assert len(password) == DEFAULT_ADMIN_PASSWORD_LENGTH
    assert password.isalnum()


def test_resolve_admin_password_preserves_custom():
    env = {"DEFAULT_ADMIN_PASSWORD": "MySecret99"}
    password, preserved = resolve_admin_password(env)
    assert password == "MySecret99"
    assert preserved is True


def test_resolve_admin_password_regenerates_placeholder():
    env = {"DEFAULT_ADMIN_PASSWORD": "admin123456"}
    password, preserved = resolve_admin_password(env)
    assert preserved is False
    assert len(password) == DEFAULT_ADMIN_PASSWORD_LENGTH
    assert password != "admin123456"


def test_is_placeholder_admin_password():
    assert is_placeholder_admin_password("")
    assert is_placeholder_admin_password("admin123456")
    assert not is_placeholder_admin_password("xK9m2pQ1aB")
