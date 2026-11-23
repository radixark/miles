"""Qwen GDN TOP refuses, before launch, the trainer paths it has no program for."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1

from miles.backends.megatron_utils.arguments import admit_true_on_policy_trainer
from miles.true_on_policy.gdn import GDN_CHUNKED, GDN_MODE_ENV
from miles_plugins.top.registry import TrainerLayout, admit_trainer


def _args(**overrides):
    fields = dict(
        true_on_policy_mode=True,
        true_on_policy_contract=TRUE_ON_POLICY_V1,
        hf_checkpoint="/ckpt",
        use_session_server=False,
        tensor_model_parallel_size=1,
        allgather_cp=False,
        loss_type="policy_loss",
        entropy_coef=0,
        log_probs_backend="torch",
        use_sampling_support_replay=False,
        recompute_loss_function=False,
        megatron_to_hf_mode="raw",
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        (dict(recompute_loss_function=True), "--recompute-loss-function"),
        (dict(megatron_to_hf_mode="bridge"), "--megatron-to-hf-mode raw"),
    ],
)
def test_qwen_gdn_refuses_before_launch(overrides, refusal, monkeypatch):
    monkeypatch.setenv(GDN_MODE_ENV, GDN_CHUNKED)
    with patch("miles.true_on_policy.gdn.is_top_gdn", return_value=True):
        with pytest.raises(NotImplementedError, match=refusal):
            admit_true_on_policy_trainer(_args(**overrides))


def test_recompute_loss_function_passes_when_the_head_is_not_deferred():
    """With full logits (here: an entropy term) the checkpointed loss has a tensor input, so the flag is not refused."""
    from miles.backends.megatron_utils.arguments import _refuse_top_gdn_paths

    with patch("miles.true_on_policy.gdn.is_top_gdn", return_value=True):
        _refuse_top_gdn_paths(_args(recompute_loss_function=True, entropy_coef=0.01))
        _refuse_top_gdn_paths(_args())


def test_qwen_gdn_trainer_pipeline_stages():
    """The image gates train dense at PP2 and PP4 and MoE at PP2 x EP2, images included."""
    for model_type, ep, pp in (("qwen3_5", 1, 2), ("qwen3_5", 1, 4), ("qwen3_5_moe", 2, 2)):
        layout = TrainerLayout(
            tp=1, etp=1, ep=ep, cp=1, pp=pp, virtual_pipeline=False, sequence_parallel=False, allgather_cp=False
        )
        assert admit_trainer(model_type, layout, fp8=False, vision=True)
