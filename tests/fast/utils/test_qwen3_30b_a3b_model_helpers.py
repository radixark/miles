import shlex
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_rotary_base(script_relpath: str) -> str:
    script_path = REPO_ROOT / script_relpath
    command = (
        f"source {shlex.quote(str(script_path))}\n"
        'printf "%s\\n" "${MODEL_ARGS[@]}"\n'
    )
    result = subprocess.run(
        ["/bin/bash", "-c", command],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    args = result.stdout.splitlines()
    rotary_base_index = args.index("--rotary-base")
    return args[rotary_base_index + 1]


@pytest.mark.parametrize(
    ("script_relpath", "expected_rotary_base"),
    [
        ("scripts/models/qwen3-30B-A3B.sh", "1000000"),
        ("scripts/models/qwen3-30B-A3B-Instruct-2507.sh", "10000000"),
        ("scripts/models/qwen3-30B-A3B-Thinking-2507.sh", "10000000"),
    ],
)
def test_model_helpers_set_correct_rotary_base(script_relpath: str, expected_rotary_base: str):
    assert _resolve_rotary_base(script_relpath) == expected_rotary_base
