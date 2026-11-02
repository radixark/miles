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
