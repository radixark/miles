from pathlib import Path

import miles.utils.external_utils.command_utils as command_utils
from miles.utils.external_utils import exec_command

_DISPATCHERS = ("exec_command_cpu", "exec_command_gpu", "exec_command_multi_node", "use_backend")


class TestExecCommandModule:
    def test_still_answers_to_the_import_path_the_launchers_use(self):
        """Scripts outside this repo import it by name, so the facade is part of the public surface."""
        for name in _DISPATCHERS:
            assert callable(getattr(exec_command, name))

    def test_is_a_module_rather_than_a_package(self):
        """The old exec_command/ package is gone, and a stray __init__.py would resurrect its submodules."""
        assert Path(exec_command.__file__).name == "exec_command.py"

    def test_dispatches_to_whatever_backend_is_active(self):
        """A launcher patches the backend, not the facade, so the facade must not hold its own binding."""
        assert exec_command.exec_command_cpu.__module__ == "miles.utils.external_utils.command_utils.base_backend"


class TestCommandUtilsPackage:
    def test_exports_the_dispatchers_the_launch_scripts_call(self):
        """Launch scripts reach the shell only through these, and losing one breaks them at import time."""
        for name in _DISPATCHERS:
            assert callable(getattr(command_utils, name))

    def test_exports_the_train_request_types(self):
        """Every launch script builds an ExecuteTrainConfig, and the request is what a backend receives."""
        assert command_utils.ExecuteTrainConfig().num_nodes >= 1
        assert set(command_utils.ExecuteTrainRequest.model_fields) >= {"train_args", "config"}
