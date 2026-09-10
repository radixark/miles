import logging
import os
import subprocess
import threading

from miles.utils.misc import NodeProbeMixin
from miles.utils.test_utils import fault_injector
from miles.utils.test_utils.fault_witness import publish_exit_receipt, publish_stop_receipt
from miles.utils.workers import process_utils

logger = logging.getLogger(__name__)


class CommandActor(NodeProbeMixin):
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._process_pidfd: int | None = None
        self._shutting_down = False
        self._fault_lock = threading.Lock()

    def run(self, cmd: str, envs: dict[str, str]) -> None:
        assert self._process is None, "CommandActor.run can only be called once"

        logger.info(f"CommandActor launches subprocess cmd={cmd!r} env_names={sorted(envs)}")
        self._process = process_utils.launch_bound_subprocess(["/bin/sh", "-c", cmd], envs=envs)
        try:
            self._process_pidfd = os.pidfd_open(self._process.pid)
        except Exception:
            logger.exception("Failed to pin subprocess identity")
            process_utils.terminate_process_tree(self._process)
            raise

        threading.Thread(target=self._babysit, args=(self._process,), daemon=True).start()

    def shutdown(self) -> None:
        if self._process is None:
            return

        with self._fault_lock:
            self._shutting_down = True
            try:
                if self._process_pidfd is not None:
                    process_utils.terminate_process_tree(self._process)
            finally:
                self._close_process_pidfd()

    def kill_subprocess(self) -> None:
        assert self._process is not None, "CommandActor has no subprocess to kill"
        process_utils.kill_process_tree(self._process)

    def inject_fault(self, mode: str, *, request_id: str | None = None, receipt_url: str | None = None) -> None:
        assert self._process is not None, "CommandActor has no subprocess to inject a fault into"
        assert (failure_mode := fault_injector.FailureMode(mode)) in {
            fault_injector.FailureMode.SIGKILL,
            fault_injector.FailureMode.SIGSTOP,
        }, (
            f"{failure_mode.value} is a fault a process inflicts on itself from the inside, and no signal reproduces "
            f"it from the outside, so only sigkill or sigstop can be injected into a subprocess"
        )

        logger.warning("CommandActor injects %s into subprocess tree pid=%s", mode, self._process.pid)
        with self._fault_lock:
            if self._process_pidfd is None:
                raise ProcessLookupError("CommandActor has no live pinned subprocess")
            if failure_mode is fault_injector.FailureMode.SIGSTOP:
                stopped_pids = process_utils.stop_process_tree_and_wait(self._process, root_pidfd=self._process_pidfd)
                if request_id is not None and receipt_url is not None:
                    publish_stop_receipt(receipt_url=receipt_url, request_id=request_id, stopped_pids=stopped_pids)
                return
            exited_pids = process_utils.kill_process_tree_and_wait(self._process, root_pidfd=self._process_pidfd)
            if request_id is not None:
                logger.info("Fault injection request_id=%s exited_pids=%s", request_id, exited_pids)
                if receipt_url is not None:
                    publish_exit_receipt(receipt_url=receipt_url, request_id=request_id, exited_pids=exited_pids)

    def _babysit(self, process: subprocess.Popen) -> None:
        returncode = process.wait()

        with self._fault_lock:
            self._close_process_pidfd()
            if self._shutting_down:
                logger.info(f"CommandActor subprocess exited with returncode={returncode} during shutdown")
                return

            logger.info(f"CommandActor exits since its subprocess exited with returncode={returncode}")
            os._exit(returncode if 0 <= returncode <= 255 else 1)

    def _close_process_pidfd(self) -> None:
        if self._process_pidfd is not None:
            os.close(self._process_pidfd)
            self._process_pidfd = None
