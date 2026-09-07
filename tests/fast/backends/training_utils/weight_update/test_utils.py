from dataclasses import dataclass

import pytest
import torch

from miles.backends.training_utils.weight_update.utils import ModelParamStager


@dataclass
class _FakeMapping:
    sglang_name: str
    num_shards: int
    num_local_experts: int | None = None


class _FakeParameterMapper:
    def __init__(self, mappings: dict[str, _FakeMapping]):
        self.mappings = mappings

    def map(self, name: str) -> _FakeMapping:
        return self.mappings[name]


def _stager_with(mappings: dict[str, _FakeMapping], sglang_names: list[str]):
    return (
        ModelParamStager(),
        _FakeParameterMapper(mappings),
        {name: torch.zeros(1) for name in sglang_names},
    )


class TestGetTransferReadyParams:
    """Staging of HF-named shards until their mapped sglang parameter is complete."""

    def test_a_single_shard_parameter_is_ready_as_soon_as_it_arrives(self) -> None:
        """A parameter that maps one-to-one needs no accumulation and transfers immediately."""
        stager, mapper, params_dict = _stager_with({"hf.embed": _FakeMapping("embed", num_shards=1)}, ["embed"])
        tensor = torch.zeros(2)

        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [("hf.embed", tensor)], param_mapper=mapper, params_dict=params_dict
        )

        assert ready_params == ["embed"]
        assert ready_tensors == [("hf.embed", tensor)]
        stager.assert_all_done()

    def test_a_fused_parameter_waits_until_every_shard_has_been_staged(self) -> None:
        """q/k/v arrive separately, so nothing may transfer before the third shard lands."""
        mappings = {
            "hf.q": _FakeMapping("qkv_proj", num_shards=3),
            "hf.k": _FakeMapping("qkv_proj", num_shards=3),
            "hf.v": _FakeMapping("qkv_proj", num_shards=3),
        }
        stager, mapper, params_dict = _stager_with(mappings, ["qkv_proj"])
        tensors = {name: torch.zeros(2) for name in mappings}

        first = stager.get_transfer_ready_params(
            [("hf.q", tensors["hf.q"])], param_mapper=mapper, params_dict=params_dict
        )
        second = stager.get_transfer_ready_params(
            [("hf.k", tensors["hf.k"])], param_mapper=mapper, params_dict=params_dict
        )
        third = stager.get_transfer_ready_params(
            [("hf.v", tensors["hf.v"])], param_mapper=mapper, params_dict=params_dict
        )

        assert first == ([], [])
        assert second == ([], [])
        assert third == (
            ["qkv_proj"],
            [("hf.q", tensors["hf.q"]), ("hf.k", tensors["hf.k"]), ("hf.v", tensors["hf.v"])],
        )
        stager.assert_all_done()

    def test_all_shards_of_a_fused_parameter_inside_one_bucket_are_returned_together(self) -> None:
        """A bucket that already carries every shard completes the parameter in a single call."""
        mappings = {
            "hf.q": _FakeMapping("qkv_proj", num_shards=2),
            "hf.k": _FakeMapping("qkv_proj", num_shards=2),
        }
        stager, mapper, params_dict = _stager_with(mappings, ["qkv_proj"])
        tensors = {name: torch.zeros(2) for name in mappings}

        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [("hf.q", tensors["hf.q"]), ("hf.k", tensors["hf.k"])],
            param_mapper=mapper,
            params_dict=params_dict,
        )

        assert ready_params == ["qkv_proj"]
        assert ready_tensors == [("hf.q", tensors["hf.q"]), ("hf.k", tensors["hf.k"])]

    def test_an_expert_parameter_expects_one_shard_per_local_expert(self) -> None:
        """MoE weights are fused over experts too, so the expected count multiplies by the local expert count."""
        mappings = {
            f"hf.expert{i}.{part}": _FakeMapping("w13", num_shards=2, num_local_experts=2)
            for i in range(2)
            for part in ("gate", "up")
        }
        stager, mapper, params_dict = _stager_with(mappings, ["w13"])
        names = list(mappings)

        partial = stager.get_transfer_ready_params(
            [(name, torch.zeros(1)) for name in names[:3]], param_mapper=mapper, params_dict=params_dict
        )
        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [(names[3], torch.zeros(1))], param_mapper=mapper, params_dict=params_dict
        )

        assert partial == ([], [])
        assert ready_params == ["w13"]
        assert [name for name, _tensor in ready_tensors] == names

    def test_a_parameter_missing_from_the_replica_is_skipped_without_being_staged(self) -> None:
        """The shared replica holds only the target's shard, so unknown mapped names must not accumulate."""
        mappings = {
            "hf.unknown": _FakeMapping("not_in_replica", num_shards=2),
            "hf.embed": _FakeMapping("embed", num_shards=1),
        }
        stager, mapper, params_dict = _stager_with(mappings, ["embed"])

        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [("hf.unknown", torch.zeros(1)), ("hf.embed", torch.zeros(1))],
            param_mapper=mapper,
            params_dict=params_dict,
        )

        assert ready_params == ["embed"]
        assert [name for name, _tensor in ready_tensors] == ["hf.embed"]
        stager.assert_all_done()

    def test_two_fused_parameters_are_accumulated_independently(self) -> None:
        """Interleaved shards of different parameters must not be mixed into one transfer."""
        mappings = {
            "hf.q": _FakeMapping("qkv_proj", num_shards=2),
            "hf.k": _FakeMapping("qkv_proj", num_shards=2),
            "hf.gate": _FakeMapping("gate_up_proj", num_shards=2),
            "hf.up": _FakeMapping("gate_up_proj", num_shards=2),
        }
        stager, mapper, params_dict = _stager_with(mappings, ["qkv_proj", "gate_up_proj"])

        first = stager.get_transfer_ready_params(
            [("hf.q", torch.zeros(1)), ("hf.gate", torch.zeros(1))], param_mapper=mapper, params_dict=params_dict
        )
        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [("hf.up", torch.zeros(1))], param_mapper=mapper, params_dict=params_dict
        )

        assert first == ([], [])
        assert ready_params == ["gate_up_proj"]
        assert [name for name, _tensor in ready_tensors] == ["hf.gate", "hf.up"]
        with pytest.raises(AssertionError, match="qkv_proj"):
            stager.assert_all_done()

        ready_params, ready_tensors = stager.get_transfer_ready_params(
            [("hf.k", torch.zeros(1))], param_mapper=mapper, params_dict=params_dict
        )

        assert ready_params == ["qkv_proj"]
        assert [name for name, _tensor in ready_tensors] == ["hf.q", "hf.k"]
        stager.assert_all_done()


class TestStagerLifecycle:
    def test_empty_buckets_preserve_an_incomplete_parameter(self) -> None:
        """An empty bucket must neither discard pending shards nor make a partial parameter transferable."""
        stager, mapper, params = _stager_with({"q": _FakeMapping("qkv", 2), "k": _FakeMapping("qkv", 2)}, ["qkv"])
        q, k = torch.ones(2), torch.full((2,), 2)

        assert stager.get_transfer_ready_params([], param_mapper=mapper, params_dict=params) == ([], [])
        stager.get_transfer_ready_params([("q", q)], param_mapper=mapper, params_dict=params)
        assert stager.get_transfer_ready_params([], param_mapper=mapper, params_dict=params) == ([], [])
        with pytest.raises(AssertionError, match="qkv"):
            stager.assert_all_done()

        assert stager.get_transfer_ready_params([("k", k)], param_mapper=mapper, params_dict=params) == (
            ["qkv"],
            [("q", q), ("k", k)],
        )
        stager.assert_all_done()

    def test_a_completed_update_does_not_leak_tensors_into_the_next_update(self) -> None:
        """Reusing a stager must wait for fresh shards and return only the current update's tensors."""
        stager, mapper, params = _stager_with({"q": _FakeMapping("qkv", 2), "k": _FakeMapping("qkv", 2)}, ["qkv"])

        for version in range(2):
            q, k = torch.full((2,), version), torch.full((2,), version + 10)
            assert stager.get_transfer_ready_params([("q", q)], param_mapper=mapper, params_dict=params) == ([], [])
            ready, tensors = stager.get_transfer_ready_params([("k", k)], param_mapper=mapper, params_dict=params)

            assert ready == ["qkv"]
            assert tensors[0][0] == "q" and tensors[0][1] is q
            assert tensors[1][0] == "k" and tensors[1][1] is k
            assert len(tensors) == 2
            stager.assert_all_done()

    def test_independent_stagers_cannot_complete_each_others_parameter(self) -> None:
        """Shards staged for separate updaters must never combine into a complete parameter."""
        first, mapper, params = _stager_with({"q": _FakeMapping("qkv", 2), "k": _FakeMapping("qkv", 2)}, ["qkv"])
        second = ModelParamStager()

        assert first.get_transfer_ready_params([("q", torch.ones(1))], param_mapper=mapper, params_dict=params) == (
            [],
            [],
        )
        assert second.get_transfer_ready_params([("k", torch.ones(1))], param_mapper=mapper, params_dict=params) == (
            [],
            [],
        )
        for stager in (first, second):
            with pytest.raises(AssertionError, match="qkv"):
                stager.assert_all_done()

    @pytest.mark.parametrize("num_experts", [None, 0, 1])
    def test_non_expert_mappings_still_wait_for_all_fused_shards(self, num_experts: int | None) -> None:
        """Missing or trivial expert counts must preserve the ordinary fused-shard completion rule."""
        stager, mapper, params = _stager_with(
            {name: _FakeMapping("qkv", 2, num_experts) for name in ("q", "k")}, ["qkv"]
        )

        assert stager.get_transfer_ready_params([("q", torch.ones(1))], param_mapper=mapper, params_dict=params) == (
            [],
            [],
        )
        ready, tensors = stager.get_transfer_ready_params(
            [("k", torch.ones(1))], param_mapper=mapper, params_dict=params
        )

        assert ready == ["qkv"]
        assert [name for name, _ in tensors] == ["q", "k"]
        stager.assert_all_done()


class TestAssertAllDone:
    """The end-of-stream guard against silently dropped shards."""

    def test_a_freshly_created_stager_holds_nothing(self) -> None:
        """Nothing was staged yet, so the guard must stay quiet."""
        ModelParamStager().assert_all_done()

    def test_an_incomplete_fused_parameter_is_reported(self) -> None:
        """A missing shard would silently ship a half-written buffer, so the leftover must raise."""
        mappings = {"hf.q": _FakeMapping("qkv_proj", num_shards=3)}
        stager, mapper, params_dict = _stager_with(mappings, ["qkv_proj"])

        stager.get_transfer_ready_params([("hf.q", torch.zeros(1))], param_mapper=mapper, params_dict=params_dict)

        with pytest.raises(AssertionError, match="qkv_proj"):
            stager.assert_all_done()
