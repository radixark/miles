"""The tilelang sparse-MLA backward kernels must not take shape dims as jit arguments.

`@tilelang.jit` caches on the call arguments, so a batch or sequence-length argument makes
every distinct packed length a fresh nvcc compile. The forward and the indexer already
declare those dims symbolically; this keeps the backward from drifting back to literals.
"""

import ast
from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

REPO_ROOT = Path(__file__).resolve().parents[2]

BWD_KERNEL_FILES = [
    "miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_bwd.py",
]

# Which dims each jit-compiled entry point must resolve symbolically rather than take as an
# argument. preprocess reads Q/dO, postprocess only writes dKV, bwd touches both.
EXPECTED_SYMBOLIC_DIMS = {
    "preprocess": {"B", "S"},
    "postprocess": {"B", "S_kv"},
    "bwd": {"B", "S", "S_kv"},
}


def _symbolic_assignments(func: ast.FunctionDef) -> set[str]:
    """Names assigned directly from a `T.dynamic(...)` / `T.symbolic(...)` call."""
    names = set()
    for stmt in func.body:
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            continue
        callee = stmt.value.func
        if isinstance(callee, ast.Attribute) and callee.attr in ("dynamic", "symbolic"):
            names.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    return names


@pytest.mark.parametrize("rel_path", BWD_KERNEL_FILES, ids=lambda p: Path(p).parts[2])
def test_backward_kernels_resolve_shapes_symbolically(rel_path):
    path = REPO_ROOT / rel_path
    assert path.exists(), f"{rel_path} moved; update this test"
    functions = {
        node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)
    }

    missing = set(EXPECTED_SYMBOLIC_DIMS) - set(functions)
    assert not missing, f"{rel_path} no longer defines {sorted(missing)}; update this test"

    for name, dims in EXPECTED_SYMBOLIC_DIMS.items():
        func = functions[name]
        params = {arg.arg for arg in func.args.args}
        leaked = params & dims
        assert not leaked, (
            f"{rel_path}::{name} takes {sorted(leaked)} as jit argument(s); every distinct value "
            f"becomes a separate nvcc compile. Declare them with T.dynamic() instead."
        )
        assert _symbolic_assignments(func) >= dims, (
            f"{rel_path}::{name} must declare {sorted(dims)} via T.dynamic()/T.symbolic(), "
            f"found {sorted(_symbolic_assignments(func))}"
        )
