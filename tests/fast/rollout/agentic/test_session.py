import subprocess
import sys

from miles.rollout.agentic.session import openai_session_url


def test_appends_v1_to_the_session_url():
    assert openai_session_url("http://10.0.0.1:30000/sessions/abc") == "http://10.0.0.1:30000/sessions/abc/v1"


def test_shared_package_is_torch_free():
    """Agent functions import it on CPU-only hosts and in offline tests (see nemogym_agent_function)."""
    code = (
        "import sys; import miles.rollout.agentic.session, miles.rollout.agentic.credentials; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
