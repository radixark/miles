import json
import os
from collections.abc import Sequence
from pathlib import Path

CI_FT_TEST_ACTIONS_FLAG: str = "--ci-ft-test-actions"


def compute_ft_test_actions_arg(actions: Sequence[dict]) -> str:
    return f"{CI_FT_TEST_ACTIONS_FLAG} '{render_ft_test_actions(actions)}' "


def render_ft_test_actions(actions: Sequence[dict]) -> str:
    return json.dumps(list(actions))


# ============ adhoc file delivery (revert after the args refactor) ============


CI_FT_TEST_ACTIONS_PATH_FLAG: str = "--ci-ft-test-actions-path"


# TODO ad hoc hack: revert after the args refactor
def read_declared_actions(args: object) -> str:
    inline: str | None = args.ci_ft_test_actions
    path: str | None = args.ci_ft_test_actions_path

    assert inline is None or path is None, (
        f"{CI_FT_TEST_ACTIONS_FLAG} and {CI_FT_TEST_ACTIONS_PATH_FLAG} both name the actions a run performs, and a "
        f"run given both silently follows one of them"
    )
    return read_ft_test_actions(Path(path)) if path is not None else (inline or "")


# TODO ad hoc hack: revert after the args refactor
def write_ft_test_actions(path: Path, actions: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    scratch = path.with_name(f"{path.name}.{os.getpid()}.partial")
    scratch.write_text(render_ft_test_actions(actions))
    scratch.replace(path)


# TODO ad hoc hack: revert after the args refactor
def read_ft_test_actions(path: Path) -> str:
    stamp = _stat_or_none(path)
    assert stamp is not None, (
        f"{CI_FT_TEST_ACTIONS_PATH_FLAG} names {path}, which does not exist; a run told to read its plan from a "
        f"file nothing wrote would quietly perform no action at all"
    )

    stamped_at = (stamp.st_mtime_ns, stamp.st_size)
    if (cached := _ACTIONS_OF_STAMP.get(path)) is not None and cached[0] == stamped_at:
        return cached[1]

    text = path.read_text()
    _ACTIONS_OF_STAMP[path] = (stamped_at, text)
    return text


# TODO ad hoc hack: revert after the args refactor
def _stat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


# TODO ad hoc hack: revert after the args refactor
_ACTIONS_OF_STAMP: dict[Path, tuple[tuple[int, int], str]] = {}
