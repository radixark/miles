from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import subprocess
import sys
from miles.rollout.agentic.session import resolve_session_url


def test_appends_v1_to_the_session_url(monkeypatch):
    monkeypatch.delenv("MILES_ROUTER_EXTERNAL_HOST", raising=False)
    assert resolve_session_url("http://10.0.0.1:30000/sessions/abc") == "http://10.0.0.1:30000/sessions/abc/v1"


def test_external_host_replaces_the_host_and_keeps_the_port(monkeypatch):
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "trainer.tailnet")
    assert resolve_session_url("http://10.0.0.1:30000/sessions/abc") == "http://trainer.tailnet:30000/sessions/abc/v1"


def test_external_host_without_a_port(monkeypatch):
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "trainer.tailnet")
    assert resolve_session_url("http://10.0.0.1/sessions/abc") == "http://trainer.tailnet/sessions/abc/v1"


def test_shared_package_is_torch_free():
    """Agent functions import it on CPU-only hosts and in offline tests (see nemogym_agent_function)."""
    code = (
        "import sys; import miles.rollout.agentic.session, miles.rollout.agentic.credentials; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
