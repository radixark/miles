from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import LAYOUT, engine

from miles.utils.external_utils.command_utils.helm_backend.launcher.values import placeholders
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values


class TestTheWorkerIndex:
    def test_replaces_only_the_node_rank_with_the_kubelet_placeholder(self):
        """Every pod of a group shares one command, so the rank must be the one part left to kubelet."""
        command = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert command[command.index("--node-rank") + 1] == placeholders._WORKER_INDEX_PLACEHOLDER
        assert command[command.index("--base-gpu-id") + 1] == "0"

    def test_allows_a_command_that_never_mentions_its_rank(self):
        """Some engines do not take one; what matters is that no sentinel survives into the command."""
        spec = engine().model_copy(update={"launch_command": lambda ctx: "python -m sglang.launch_server"})

        command = build_values([spec], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert str(placeholders.WORKER_INDEX_SENTINEL) not in " ".join(command)
