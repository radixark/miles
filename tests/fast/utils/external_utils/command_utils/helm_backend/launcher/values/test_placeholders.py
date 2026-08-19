import itertools

import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import LAYOUT, engine, trainer

from miles.utils.external_utils.command_utils.helm_backend.launcher.values import placeholders
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.worker_spec import BaseWorkerSpec


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

    def test_refuses_a_spec_that_builds_the_rank_into_a_larger_argument(self):
        """Kubelet substitutes whole arguments, so --node-rank=N would reach the engine unexpanded."""
        spec = engine().model_copy(
            update={
                "launch_command": lambda ctx: f"python -m sglang.launch_server --node-rank={ctx.worker_in_cell_index}"
            }
        )

        with pytest.raises(AssertionError, match="out of its pod index"):
            build_values([spec], LAYOUT).as_values()


COLOCATE_LAYOUT = LAYOUT.model_copy(update={"colocate": True})

_LARGEST_PLAUSIBLE_CARD_OR_RANK = 1_000_000


def _engine_command(specs: list[BaseWorkerSpec], plan: LaunchPlan) -> list[str]:
    return build_values(specs, plan).as_values()["run"]["inferenceEngines"][0]["command"]


def _base_gpu_id_argument(command: list[str]) -> str:
    return command[command.index("--base-gpu-id") + 1]


class TestBaseGpuIdOfASubNodeEngine:
    def test_leaves_a_whole_node_engine_to_be_handed_its_cards_from_zero(self):
        """A pod holding every card of its node is given them as devices 0..n-1, so nothing has to be told."""
        command = _engine_command(
            [engine(num_cells=2, gpus_per_engine=8), trainer(num_cells=2, gpus_per_cell=8)], COLOCATE_LAYOUT
        )

        assert _base_gpu_id_argument(command) == "0"

    def test_refuses_a_spec_that_builds_the_card_into_a_larger_argument(self):
        """Kubelet substitutes whole arguments, so a spelling like --gpus=N would reach sglang unexpanded."""
        spec = engine(num_cells=2, gpus_per_engine=4).model_copy(
            update={"launch_command": lambda ctx: f"python -m sglang.launch_server --base-gpu-id={ctx.gpu_ids[0]}"}
        )

        with pytest.raises(AssertionError, match="out of its base gpu id"):
            build_values([spec, trainer(num_cells=1, gpus_per_cell=8)], COLOCATE_LAYOUT).as_values()


class TestTheSubstitutionTableItself:
    def test_no_two_entries_share_a_sentinel(self):
        """Two entries on one sentinel would make which variable a command asks for depend on table order."""
        sentinels = [substitution.sentinel for substitution in placeholders._SUBSTITUTIONS]

        assert len(set(sentinels)) == len(sentinels)

    def test_no_two_entries_share_a_placeholder(self):
        """Two sentinels expanding to one variable means one of them is silently reading the other's value."""
        expansions = [substitution.placeholder for substitution in placeholders._SUBSTITUTIONS]

        assert len(set(expansions)) == len(expansions)

    def test_no_sentinel_is_a_number_a_card_or_a_rank_could_be(self):
        """A sentinel inside the real domain would make a genuine card number expand into a kubelet variable."""
        for substitution in placeholders._SUBSTITUTIONS:
            assert substitution.sentinel > _LARGEST_PLAUSIBLE_CARD_OR_RANK

    def test_no_sentinel_reads_as_part_of_another(self):
        """The guard tests substring containment, so one sentinel inside another would report the wrong spec."""
        for first, second in itertools.permutations(placeholders._SUBSTITUTIONS, 2):
            assert str(first.sentinel) not in str(second.sentinel)
