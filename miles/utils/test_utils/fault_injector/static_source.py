import json
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.fault_injector.models import FaultHookRequest

CI_FAULT_HOOKS_FLAG: str = "--ci-fault-hooks"
CI_FAULT_HOOKS_PATH_FLAG: str = "--ci-fault-hooks-path"

_REQUEST_LIST_ADAPTER: TypeAdapter[list[FaultHookRequest]] = TypeAdapter(list[FaultHookRequest])


def compute_fault_hooks_arg(requests: Sequence[FaultHookRequest]) -> str:
    return f"{CI_FAULT_HOOKS_FLAG} '{render_fault_hooks(requests)}' "


def render_fault_hooks(requests: Sequence[FaultHookRequest]) -> str:
    return _REQUEST_LIST_ADAPTER.dump_json(list(requests)).decode()


def read_declared_fault_hooks(args: object) -> list[FaultHookRequest]:
    inline: str | None = args.ci_fault_hooks
    path: str | None = args.ci_fault_hooks_path
    assert inline is None or path is None, (
        f"{CI_FAULT_HOOKS_FLAG} and {CI_FAULT_HOOKS_PATH_FLAG} both name the hooks a run sets, and a run given both "
        f"silently follows one of them"
    )
    if path is not None:
        assert Path(path).is_file(), (
            f"{CI_FAULT_HOOKS_PATH_FLAG} names {path}, which does not exist; a run told to read its hooks from a file "
            f"nothing wrote would quietly set no hook at all"
        )
        return _REQUEST_LIST_ADAPTER.validate_json(Path(path).read_text())
    return _REQUEST_LIST_ADAPTER.validate_json(inline) if inline else []


def write_fault_hooks(path: Path, requests: Sequence[FaultHookRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, render_fault_hooks(requests))


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
