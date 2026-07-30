import argparse
import ast
import re
from pathlib import Path

from miles.utils.arguments import get_miles_extra_args_provider

REPO_ROOT = Path(__file__).resolve().parents[3]

MILES_OWNED_FLAG_PREFIXES = ("--sglang-", "--miles-", "--use-miles-")

EXTERNAL_ROUTER_FLAGS = ("--sglang-router-ip", "--sglang-router-port")

_FLAG_PATTERN = re.compile(r"(?:(?<=\s)|^)(--[a-zA-Z0-9][a-zA-Z0-9-]*)(?=\s|$)")


def test_example_launchers_only_emit_registered_miles_flags() -> None:
    """Every miles-namespaced flag written into an examples/ launcher command string is still registered in the miles argument parser."""
    registered: set[str] = _collect_registered_flags()
    emitted: dict[str, set[Path]] = _collect_emitted_flags(root=REPO_ROOT / "examples")

    unregistered: dict[str, set[Path]] = {
        flag: sources
        for flag, sources in emitted.items()
        if flag.startswith(MILES_OWNED_FLAG_PREFIXES) and flag not in registered
    }

    assert not unregistered, "examples/ launchers pass flags that miles no longer registers: " + "; ".join(
        f"{flag} ({', '.join(sorted(str(p.relative_to(REPO_ROOT)) for p in sources))})"
        for flag, sources in sorted(unregistered.items())
    )


def test_example_launchers_do_not_request_an_external_router() -> None:
    """No examples/ launcher passes the external-router flags, which start_rollout_servers asserts are unset."""
    emitted: dict[str, set[Path]] = _collect_emitted_flags(root=REPO_ROOT / "examples")

    offenders: dict[str, set[Path]] = {flag: emitted[flag] for flag in EXTERNAL_ROUTER_FLAGS if flag in emitted}

    assert not offenders, "external router mode was removed, so examples/ launchers must not pass: " + "; ".join(
        f"{flag} ({', '.join(sorted(str(p.relative_to(REPO_ROOT)) for p in sources))})"
        for flag, sources in sorted(offenders.items())
    )


def _collect_registered_flags() -> set[str]:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    get_miles_extra_args_provider()(parser)
    return {option for action in parser._actions for option in action.option_strings}


def _collect_emitted_flags(root: Path) -> dict[str, set[Path]]:
    emitted: dict[str, set[Path]] = {}
    for path in sorted(root.rglob("*.py")):
        for text in _iter_string_constants(source=path.read_text()):
            for match in _FLAG_PATTERN.finditer(text):
                emitted.setdefault(match.group(1), set()).add(path)
    return emitted


def _iter_string_constants(source: str) -> list[str]:
    tree: ast.Module = ast.parse(source)
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
