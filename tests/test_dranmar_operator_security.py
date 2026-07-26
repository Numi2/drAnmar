import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/dr_anmar_operator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dranmar_operator_security", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "[::1]", "localhost"])
def test_loopback_bind_needs_no_remote_override(monkeypatch, host):
    module = _load_module()
    for name in (
        "DR_ANMAR_ALLOW_REMOTE",
        "DR_ANMAR_ACCESS_TOKEN",
        "DR_ANMAR_TLS_TERMINATED",
        "DR_ANMAR_FIREWALL_CONFIRMED",
    ):
        monkeypatch.delenv(name, raising=False)
    module.validate_bind_security(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.20", "workstation"])
def test_nonloopback_bind_fails_closed(monkeypatch, host):
    module = _load_module()
    for name in (
        "DR_ANMAR_ALLOW_REMOTE",
        "DR_ANMAR_ACCESS_TOKEN",
        "DR_ANMAR_TLS_TERMINATED",
        "DR_ANMAR_FIREWALL_CONFIRMED",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="Refusing non-loopback bind"):
        module.validate_bind_security(host)


def test_nonloopback_bind_requires_every_control(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("DR_ANMAR_ALLOW_REMOTE", "1")
    monkeypatch.setenv("DR_ANMAR_ACCESS_TOKEN", "test-only-strong-token")
    monkeypatch.setenv("DR_ANMAR_TLS_TERMINATED", "1")
    monkeypatch.setenv("DR_ANMAR_FIREWALL_CONFIRMED", "1")
    module.validate_bind_security("0.0.0.0")


def test_access_token_comparison_stays_fail_closed_when_configured(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("DR_ANMAR_ACCESS_TOKEN", "configured-secret")
    assert not module.access_is_authorized(None)
    assert not module.access_is_authorized(None, "wrong")
    assert module.access_is_authorized(None, "configured-secret")
