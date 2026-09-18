import ast
from functools import cache

from tests.fast.launch_scripts.sh_harness import REPO_ROOT


@cache
def _module_attributes(module_name):
    """Read explicit module bindings without importing Ray or other runtime dependencies."""
    path = REPO_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert all(alias.name != "*" for alias in node.names), f"star import in {path}"
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id
                for assignment in node.targets
                for target in ast.walk(assignment)
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)
            )
        elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_script_utility_attributes_exist():
    """Check every U attribute, including references inside functions and decorators."""
    offenders = []
    checked = 0
    for path in sorted((REPO_ROOT / "scripts").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.asname == "U"
        }
        references = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "U"
        ]
        if not references:
            continue
        assert len(imports) == 1, f"expected one module imported as U in {path}"
        module_name = next(iter(imports))
        attributes = _module_attributes(module_name)
        for node in references:
            checked += 1
            if node.attr not in attributes:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {module_name}.{node.attr}")

    assert checked, "no script utility attributes found"
    assert not offenders, "\n".join(offenders)
