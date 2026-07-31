import argparse
import ast
import re
from pathlib import Path

from miles.utils.arguments import get_miles_extra_args_provider

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"

MILES_OWNED_FLAG_PREFIXES: tuple[str, ...] = (
    "--sglang-",
    "--miles-",
    "--use-miles-",
    "--session-",
    "--use-session-",
    "--num-session-",
)

_FLAG_PATTERN = re.compile(r"(?<![\w-])--[a-z0-9][a-z0-9-]*")


def test_examples_only_emit_registered_miles_flags() -> None:
    """Every miles-owned --flag an examples/ launcher emits must be registered by the miles args provider."""
    registered: set[str] = _registered_option_strings()
    unregistered: dict[str, list[str]] = {}

    for flag, source_path in _iter_miles_owned_flags():
        if flag not in registered:
            unregistered.setdefault(flag, []).append(str(source_path.relative_to(REPO_ROOT)))

    assert not unregistered, f"examples/ emit miles-owned flags that miles does not register: {unregistered}"


def test_launcher_flag_scan_is_not_vacuous() -> None:
    """The scan must actually find miles-owned flags, so a broken scan cannot silently pass the check."""
    found: list[str] = [flag for flag, _ in _iter_miles_owned_flags()]

    assert len(found) > 20, f"scan of {EXAMPLES_ROOT} found suspiciously few miles-owned flags: {sorted(set(found))}"


def _registered_option_strings() -> set[str]:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    get_miles_extra_args_provider()(parser)
    return {option for action in parser._actions for option in action.option_strings}


def _iter_miles_owned_flags() -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    for source_path in sorted(EXAMPLES_ROOT.rglob("*.py")):
        tree = ast.parse(source=source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            for flag in _FLAG_PATTERN.findall(node.value):
                if flag.startswith(MILES_OWNED_FLAG_PREFIXES):
                    results.append((flag, source_path))
    return results
