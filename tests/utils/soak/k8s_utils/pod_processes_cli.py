import sys

import typer
from tests.utils.soak.k8s_utils.pod_processes import (
    ProcessSignal,
    ProcessSignalReceipt,
    ProcessTarget,
    observe_processes,
    signal_observed_processes,
)

app = typer.Typer()


@app.command()
def observe(pod_uid: str, pattern: str) -> None:
    print(observe_processes(pod_uid=pod_uid, pattern=pattern).model_dump_json(), flush=True)


@app.command()
def kill(request_id: str) -> None:
    _print_receipt(request_id=request_id, operation=ProcessSignal.KILL)


@app.command()
def stop(request_id: str) -> None:
    _print_receipt(request_id=request_id, operation=ProcessSignal.STOP)


def _print_receipt(*, request_id: str, operation: ProcessSignal) -> None:
    target = ProcessTarget.model_validate_json(sys.stdin.read())
    receipt = ProcessSignalReceipt(
        request_id=request_id,
        target=target,
        operation=operation,
        signalled_pids=signal_observed_processes(target=target, operation=operation),
    )
    print(receipt.model_dump_json(), flush=True)


if __name__ == "__main__":
    app()
