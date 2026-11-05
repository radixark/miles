"""The skips the miles-ext-kernels GPU harness accepts (tests/fast-gpu/kernels/miles_ext_kernels_skips.py): the reasons
of pytest's ``-rs`` summary are read in full, and only a listed hardware limit of the skipped test's op passes."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "fast-gpu" / "kernels" / "miles_ext_kernels_skips.py"
_spec = importlib.util.spec_from_file_location("miles_ext_kernels_skips", _PATH)
allowlist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(allowlist)

KCP = "tests/cake/gdn/test_gdn_kcp_gpu.py"
CP4 = "context parallelism 4 needs 4 GPUs; 2 visible"
NO_FLA = "could not import 'fla.ops.gated_delta_rule': No module named 'fla'"
STDOUT = f"""\
..s.s.ss                                                                 [100%]
=========================== short test summary info ============================
SKIPPED [2] {KCP}:543: {CP4}
SKIPPED [1] tests/cake/gdn/test_fla_parity.py:32: {NO_FLA}
SKIPPED [1] tests/cake/gdn/test_gdn_det_gpu.py: the GDN kernels need SM100a / SM103a / SM107a
4 passed, 4 skipped in 3.40s
"""


def test_every_line_of_the_skip_summary_is_read_with_its_whole_reason():
    assert allowlist.skips(STDOUT) == [
        (2, KCP, CP4),
        (1, "tests/cake/gdn/test_fla_parity.py", NO_FLA),
        (1, "tests/cake/gdn/test_gdn_det_gpu.py", "the GDN kernels need SM100a / SM103a / SM107a"),
    ]
    assert allowlist.skips("4 passed in 3.40s\n") == []


def test_a_listed_reason_passes_with_any_formatted_values():
    assert allowlist.is_allowed(KCP, CP4)
    assert allowlist.is_allowed(KCP, "context parallelism 8 needs 8 GPUs; 4 visible")


def test_a_missing_dependency_or_an_unsupported_gpu_is_not_a_listed_limit():
    assert not allowlist.is_allowed("tests/cake/gdn/test_fla_parity.py", NO_FLA)
    assert not allowlist.is_allowed(
        "tests/cake/gdn/test_gdn_det_gpu.py", "the GDN kernels need SM100a / SM103a / SM107a"
    )
    assert not allowlist.is_allowed("tests/cake/gdn/test_gdn_det_gpu.py", "CUDA device required")


def test_a_reason_passes_only_whole_and_only_for_the_op_that_lists_it():
    assert not allowlist.is_allowed(KCP, f"{CP4}, and another reason")
    assert not allowlist.is_allowed(KCP, "context parallelism needs GPUs; visible")
    assert not allowlist.is_allowed("tests/cake/_runtime/test_workspace.py", CP4)
    assert not allowlist.is_allowed("tests/conftest.py", CP4)


def test_every_op_lists_its_reasons_as_strings():
    assert "_runtime" in allowlist.SKIP_ALLOWLIST
    for op, reasons in allowlist.SKIP_ALLOWLIST.items():
        assert isinstance(reasons, tuple) and all(isinstance(r, str) and r for r in reasons), op
