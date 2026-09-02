import logging
import os
import subprocess
import threading

from miles.utils.misc import NodeProbeMixin
from miles.utils.test_utils import fault_injector
from miles.utils.workers import process_utils

logger = logging.getLogger(__name__)


class CommandActor(NodeProbeMixin):
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._shutting_down = False
        self._debug_exit_when_subprocess_exits = True

    def run(self, cmd: str, envs: dict[str, str]) -> None:
        assert self._process is None, "CommandActor.run can only be called once"

        logger.info(f"CommandActor launches subprocess cmd={cmd!r} env_names={sorted(envs)}")
        self._process = process_utils.launch_bound_subprocess(["/bin/sh", "-c", cmd], envs=envs)

        threading.Thread(target=self._babysit, args=(self._process,), daemon=True).start()

    def shutdown(self) -> None:
        if self._process is None:
            return

        self._shutting_down = True
        process_utils.terminate_process_tree(self._process)

    def kill_subprocess(self) -> None:
        assert self._process is not None, "CommandActor has no subprocess to kill"
        process_utils.kill_process_tree(self._process)

    def inject_fault(self, mode: str, *, keep_actor_alive_until_ack: bool = False) -> None:
        assert self._process is not None, "CommandActor has no subprocess to inject a fault into"
        assert (failure_mode := fault_injector.FailureMode(mode)) is fault_injector.FailureMode.SIGKILL, (
            f"{failure_mode.value} is a fault a process inflicts on itself from the inside, and no signal reproduces "
            f"it from the outside, so only sigkill can be injected into a subprocess"
        )

        logger.warning(f"CommandActor kills its subprocess group pid={self._process.pid}")
        if keep_actor_alive_until_ack:
            self._debug_exit_when_subprocess_exits = False
        process_utils.kill_process_tree(self._process)

    def _babysit(self, process: subprocess.Popen) -> None:
        returncode = process.wait()

        if self._shutting_down:
            logger.info(f"CommandActor subprocess exited with returncode={returncode} during shutdown")
            return
        if not self._debug_exit_when_subprocess_exits:
            logger.info(f"CommandActor keeps running after acknowledged subprocess exit with returncode={returncode}")
            return

        logger.info(f"CommandActor exits since its subprocess exited with returncode={returncode}")
        os._exit(returncode if 0 <= returncode <= 255 else 1)
