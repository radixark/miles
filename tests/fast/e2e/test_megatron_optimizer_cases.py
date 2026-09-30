import importlib
import shlex
from dataclasses import replace

import pytest

from miles.utils.external_utils import command_utils

_CASES = (
    ("test_qwen3_30B_A3B", "test_nospec_nor3_bf16_sgl_tp4_meg_tp2cp2"),
    ("test_qwen3_5_35B_A3B", "test_nospec_nor3_bf16_sgl_dpattn2x2_meg_tp2pp2"),
    ("test_glm47_flash", "test_nospec_r3_bf16_deepep_sgl_dpattn2x2_meg_tp2cp2"),
)
_ADAM_ONLY_FLAGS = {
    "--optimizer-cpu-offload",
    "--overlap-cpu-optimizer-d2h-h2d",
    "--use-precision-aware-optimizer",
    "--rematerialize-param-from-master-weight",
    "--exp-avg-dtype",
    "--exp-avg-sq-dtype",
    "--main-params-dtype",
}


@pytest.fixture(params=_CASES, ids=[family for family, _ in _CASES])
def case_modules(request, monkeypatch):
    family, name = request.param
    common = importlib.import_module(f"tests.e2e.megatron.{family}._common")
    case = importlib.import_module(f"tests.e2e.megatron.{family}.{name}").CASE
    monkeypatch.setattr(command_utils, "get_default_wandb_args", lambda _: "")
    return common, case


@pytest.mark.parametrize("tight_host_memory", [False, True])
def test_muon_case_renders_compatible_distributed_offload(case_modules, monkeypatch, tight_host_memory):
    common, case = case_modules
    if hasattr(common, "TIGHT_HOST_MEMORY"):
        monkeypatch.setattr(common, "TIGHT_HOST_MEMORY", tight_host_memory)

    tokens = shlex.split(common.build_train_args(case, wandb_file=__file__))

    assert tokens.count("--optimizer") == 1
    assert tokens[tokens.index("--optimizer") + 1] == "dist_muon"
    assert "--chunked-optimizer-state-offload" in tokens
    assert tokens[tokens.index("--optimizer-state-offload-fraction") + 1] == "1.0"
    assert tokens[tokens.index("--optimizer-state-offload-chunk-size-mb") + 1] == "1024"
    assert not _ADAM_ONLY_FLAGS.intersection(tokens)
    assert tokens[tokens.index("--num-rollout") + 1] == "2"


@pytest.mark.parametrize("tight_host_memory", [False, True])
def test_adam_retains_its_existing_offload_path(case_modules, monkeypatch, tight_host_memory):
    common, muon_case = case_modules
    if hasattr(common, "TIGHT_HOST_MEMORY"):
        monkeypatch.setattr(common, "TIGHT_HOST_MEMORY", tight_host_memory)

    case = replace(muon_case, optimizer="adam")
    tokens = shlex.split(common.build_train_args(case, wandb_file=__file__))

    assert tokens[tokens.index("--optimizer") + 1] == "adam"
    assert "--optimizer-cpu-offload" in tokens
    assert "--overlap-cpu-optimizer-d2h-h2d" in tokens
    assert "--use-precision-aware-optimizer" in tokens
    assert "--chunked-optimizer-state-offload" not in tokens
    if "qwen3" in common.__name__:
        assert "--rematerialize-param-from-master-weight" in tokens
    if hasattr(common, "TIGHT_HOST_MEMORY"):
        assert ("--main-params-dtype" in tokens) == tight_host_memory


def test_plain_muon_does_not_silently_select_the_adam_path(case_modules):
    _, case = case_modules
    with pytest.raises(ValueError, match="unsupported optimizer"):
        replace(case, optimizer="muon")
