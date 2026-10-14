import torch

from miles.backends.training_utils.weight_update.protocols.utils.loader_probe import HfNameMapping
from miles.backends.training_utils.weight_update.protocols.utils.model_param_stager import ModelParamStager


def _stager(hf_names_by_param_name: dict[str, set[str]]) -> ModelParamStager:
    return ModelParamStager(
        HfNameMapping.from_hf_names_by_param_name(
            {param_name: frozenset(hf_names) for param_name, hf_names in hf_names_by_param_name.items()}
        )
    )


class TestStage:
    def test_params_chained_by_shared_hf_tensors_form_one_group(self) -> None:
        """One HF tensor can fill several params (a tied embedding) and fused params chain them further; loading part
        of such a group would hand the loader a param that is not bound."""
        stager = _stager({"a": {"hf.x"}, "b": {"hf.x", "hf.y"}, "c": {"hf.y"}})

        assert stager.param_groups == [("a", "b", "c")]
        assert stager.stage([("hf.x", torch.zeros(1))]) == {}
        assert list(stager.stage([("hf.y", torch.zeros(1))])) == [("a", "b", "c")]

    def test_groups_accumulate_independently_and_keep_their_own_tensors(self) -> None:
        stager = _stager({"qkv": {"hf.q", "hf.k"}, "gate_up": {"hf.gate", "hf.up"}})
        q, gate, up, k = (torch.full((1,), value) for value in range(4))

        assert stager.stage([("hf.q", q), ("hf.gate", gate)]) == {}
        assert stager.stage([("hf.up", up)]) == {("gate_up",): [("hf.gate", gate), ("hf.up", up)]}
        assert stager.stage([("hf.k", k)]) == {("qkv",): [("hf.q", q), ("hf.k", k)]}
