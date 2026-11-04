import shlex
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree


def verify_sdk_contract(
    *,
    execute: Callable[[str], str | None],
    venv: Path,
    repo_root: Path,
    report_path: Path,
    pythonpath: str,
) -> None:
    test_file = repo_root / "tests/fast/examples/experimental/verifiers/test_verifiers_rollout.py"
    execute(
        shlex.join(
            [
                "env",
                "CUDA_VISIBLE_DEVICES=",
                f"VIRTUAL_ENV={venv}",
                f"PYTHONPATH={pythonpath}",
                "uv",
                "run",
                "--no-project",
                "--active",
                "python",
                "-m",
                "pytest",
                f"{test_file}::test_canonical_tokenizer_selects_tool_renderer_for_ambiguous_local_checkpoint",
                f"{test_file}::test_verifiers_episode_owns_group_reward_computation",
                "-v",
                f"--junitxml={report_path}",
            ]
        )
    )
    _assert_contract_report(report_path)


def _assert_contract_report(report_path: Path) -> None:
    cases = ElementTree.parse(report_path).getroot().findall(".//testcase")
    assert len(cases) == 3, f"Expected three Verifiers SDK contract cases, found {len(cases)}"
    for case in cases:
        assert all(
            case.find(status) is None for status in ("skipped", "failure", "error")
        ), f"Verifiers SDK contract did not pass: {case.attrib}"
