from types import SimpleNamespace

from miles.rollout.base_types import (
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    compute_kv_cache_namespace,
    stamp_kv_cache_namespace,
)
from miles.utils.types import Sample


def _args(*, partition: bool = True) -> SimpleNamespace:
    return SimpleNamespace(namespaced_radix_cache=partition)


class TestComputeKvCacheNamespace:
    def test_a_single_policy_train_call_names_its_rollout_id(self):
        """Without a policy the namespace still identifies the train call by its rollout id."""
        assert compute_kv_cache_namespace(_args(), RolloutFnTrainInput(rollout_id=7)) == "train:-:7"

    def test_consecutive_rollout_ids_get_different_namespaces(self):
        """Each train call opens its own namespace, so prefix KV never crosses a weight update."""
        assert compute_kv_cache_namespace(_args(), RolloutFnTrainInput(rollout_id=7)) != compute_kv_cache_namespace(
            _args(), RolloutFnTrainInput(rollout_id=8)
        )

    def test_two_policies_at_the_same_rollout_id_get_different_namespaces(self):
        """Multi-policy runs advance independently, so each policy keeps its own namespace."""
        assert compute_kv_cache_namespace(
            _args(), RolloutFnTrainInput(rollout_id=7, trainer_model_id="solver")
        ) != compute_kv_cache_namespace(_args(), RolloutFnTrainInput(rollout_id=7, trainer_model_id="verifier"))

    def test_eval_and_train_at_the_same_rollout_id_get_different_namespaces(self):
        """Eval generates under different weights than the train call of the same rollout id."""
        assert compute_kv_cache_namespace(_args(), RolloutFnEvalInput(rollout_id=7)) != compute_kv_cache_namespace(
            _args(), RolloutFnTrainInput(rollout_id=7)
        )

    def test_the_partition_being_off_names_no_namespace(self):
        """The gate sits at the single place a namespace is decided, so nothing downstream has to check it."""
        assert compute_kv_cache_namespace(_args(partition=False), RolloutFnTrainInput(rollout_id=7)) is None
        assert compute_kv_cache_namespace(_args(partition=False), RolloutFnEvalInput(rollout_id=7)) is None


class TestStampKvCacheNamespace:
    def test_an_unstamped_sample_takes_the_namespace(self):
        """A sample entering a call for the first time is bound to that call's namespace."""
        sample = Sample(prompt="p")

        stamp_kv_cache_namespace(sample, namespace="train:-:7")

        assert sample.kv_cache_namespace == "train:-:7"

    def test_a_started_sample_keeps_the_namespace_it_started_under(self):
        """A resumed multi-turn or partial sample must keep reusing the KV it already filled."""
        sample = Sample(prompt="p", kv_cache_namespace="train:-:7")

        stamp_kv_cache_namespace(sample, namespace="train:-:8")

        assert sample.kv_cache_namespace == "train:-:7"

    def test_no_namespace_stamps_nothing(self):
        """With the partition off every request must look byte for byte as it did before."""
        sample = Sample(prompt="p")

        stamp_kv_cache_namespace(sample, namespace=None)

        assert sample.kv_cache_namespace is None

    def test_every_sample_of_a_nested_group_is_stamped(self):
        """Groups arrive as lists of samples, and multi-agent groups nest one level deeper."""
        flat = [Sample(prompt="a"), Sample(prompt="b")]
        nested = [[Sample(prompt="c")], [Sample(prompt="d")]]

        stamp_kv_cache_namespace([flat, nested], namespace="train:-:7")

        assert {s.kv_cache_namespace for s in flat} == {"train:-:7"}
        assert {s.kv_cache_namespace for group in nested for s in group} == {"train:-:7"}
