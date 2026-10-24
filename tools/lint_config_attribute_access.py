import ast
import io
import tokenize
from pathlib import Path


_EXEMPTION = "config-access-exempt:"


def main() -> None:
    violations = [
        message
        for root in (Path("miles"), Path("miles_plugins"))
        for path in root.rglob("*.py")
        for message in _violations(path)
    ]
    if violations:
        raise SystemExit("\n".join(violations))


def _violations(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source)
    exemptions = _exempted_lines(source)
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"getattr", "hasattr"} or not node.args:
            continue
        if node.lineno not in exemptions:
            violations.append(f"{path}:{node.lineno}: dynamic attribute access needs a specific inline exemption")
    return violations


def _exempted_lines(source: str) -> set[int]:
    exempted = set()
    statement_start = None
    has_exemption = False
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in {tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.COMMENT, tokenize.ENDMARKER}:
            if statement_start is None:
                statement_start = token.start[0]
        if token.type == tokenize.COMMENT and statement_start is not None:
            _, marker, reason = token.string.partition(_EXEMPTION)
            has_exemption |= bool(marker and reason.strip() and reason.strip() != "runtime reflection is required")
        if token.type == tokenize.NEWLINE:
            if has_exemption and statement_start is not None:
                exempted.update(range(statement_start, token.end[0] + 1))
            statement_start = None
            has_exemption = False
    return exempted


if __name__ == "__main__":
    main()
