import subprocess
import sys


def test_shared_package_is_torch_free():
    """Agent functions import it on CPU-only hosts and in offline tests (see nemogym_agent_function)."""
    code = (
        "import sys; import miles.rollout.agentic.session, miles.rollout.agentic.credentials; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
