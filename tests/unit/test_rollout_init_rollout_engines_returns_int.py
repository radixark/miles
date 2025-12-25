import ast
from pathlib import Path

import miles
import pytest


@pytest.mark.unit
def test_init_rollout_engines_never_returns_tuple():
    rollout_py = Path(miles.__file__).resolve().parent / "ray" / "rollout.py"
    source = rollout_py.read_text(encoding="utf-8")
    module = ast.parse(source)

    init_fn = next(
        (node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "init_rollout_engines"),
        None,
    )
    assert init_fn is not None, "Expected init_rollout_engines() in miles/ray/rollout.py"

    tuple_return_lines = [
        node.lineno
        for node in ast.walk(init_fn)
        if isinstance(node, ast.Return) and isinstance(getattr(node, "value", None), ast.Tuple)
    ]
    assert tuple_return_lines == [], f"init_rollout_engines() returns tuple on lines: {tuple_return_lines}"

