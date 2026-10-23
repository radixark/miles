from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.diffusion_gemma.test_model import tiny_config

from miles.backends.fsdp_utils.adaptations.precision import PrecisionPolicy
from miles.backends.fsdp_utils.diffusion_gemma.config import validate_training_args
from miles.backends.fsdp_utils.diffusion_gemma.engine import _batch_seed, _optimizer_step, _prepare
from miles.backends.fsdp_utils.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion


def actor():
    config = tiny_config()
    config.canvas_length = 4
    config.eos_token_id = 2
    config.text_config.sliding_window = 3
    config._attn_implementation = "sdpa"
    model = DiffusionGemmaForBlockDiffusion(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return SimpleNamespace(
        hf_config=config,
        model=model,
        optimizer=optimizer,
        global_step=0,
        micro_step=0,
        precision_policy=PrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, keep_fp32_master=True),
        args=SimpleNamespace(
            seed=42,
            diffusion_noise_epsilon=0.001,
            diffusion_self_conditioning_probability=0.5,
            diffusion_encoder_loss_weight=0.7,
        ),
        _zero_grad=lambda: optimizer.zero_grad(set_to_none=True),
        _apply_step=lambda: optimizer.step(),
    )


def rows():
    return [
        {"tokens": [torch.tensor([3, 4, 5])], "response_lengths": [1], "loss_masks": [torch.ones(1)]},
        {"tokens": [torch.tensor([3, 5, 7, 4, 8])], "response_lengths": [3], "loss_masks": [torch.ones(3)]},
    ]


def test_optimizer_updates_and_restored_counters_reproduce_next_step(monkeypatch):
    # Single-process CPU orchestration; collectives are tested separately in the CUDA smoke test.
    monkeypatch.setattr("miles.backends.fsdp_utils.diffusion_gemma.engine.dist.all_reduce", lambda *a, **k: None)
    original = actor()
    before = original.model.model.decoder.layers[0].mlp.up_proj.weight.detach().clone()
    _, diff, ar = _optimizer_step(original, rows=rows(), dp_size=1, rank=0, group=None)
    assert diff > 0 and ar > 0
    assert original.global_step == 1 and original.micro_step == 2
    assert not torch.equal(before, original.model.model.decoder.layers[0].mlp.up_proj.weight)
    restored = actor()
    restored.model.load_state_dict(original.model.state_dict())
    restored.optimizer.load_state_dict(deepcopy(original.optimizer.state_dict()))
    restored.global_step, restored.micro_step = original.global_step, original.micro_step
    _optimizer_step(original, rows=rows(), dp_size=1, rank=0, group=None)
    _optimizer_step(restored, rows=rows(), dp_size=1, rank=0, group=None)
    for expected, actual in zip(original.model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_seed_changes_with_rank_step_microbatch_but_repeats_on_resume():
    baseline = _batch_seed(seed=42, step=7, microbatch=0, rank=0)
    assert baseline == _batch_seed(seed=42, step=7, microbatch=0, rank=0)
    assert (
        len(
            {
                baseline,
                _batch_seed(seed=42, step=8, microbatch=0, rank=0),
                _batch_seed(seed=42, step=7, microbatch=1, rank=0),
                _batch_seed(seed=42, step=7, microbatch=0, rank=1),
            }
        )
        == 4
    )


def test_terminal_fill_uses_tokenizer_eos_when_config_has_multiple_stop_ids():
    trainer = actor()
    trainer.tokenizer = SimpleNamespace(eos_token_id=9)
    trainer.hf_config.eos_token_id = [2, 3]
    batch = _prepare(trainer, row=rows()[0], microbatch=0, rank=0)
    assert batch.targets.tolist() == [[5, 9, 9, 9]]


def valid_args():
    return SimpleNamespace(
        loss_type="sft_loss",
        debug_train_only=True,
        rollout_function_path="miles.rollout.diffusion_gemma_sft.generate_rollout",
        n_samples_per_prompt=1,
        apply_chat_template=False,
        compute_advantages_and_returns=False,
        qkv_format="bshd",
        attn_implementation="sdpa",
        kernel_backend="native",
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        global_batch_size=4,
        micro_batch_size=1,
        rollout_batch_size=4,
        diffusion_noise_epsilon=0.001,
        diffusion_self_conditioning_probability=0.5,
        diffusion_encoder_loss_weight=1.0,
    )


def test_valid_offline_configuration():
    validate_training_args(valid_args())


@pytest.mark.parametrize(
    "field,value",
    [
        ("loss_type", "policy_loss"),
        ("debug_train_only", False),
        ("compute_advantages_and_returns", True),
        ("compute_advantages_and_returns", None),
        ("attn_implementation", "flash_attention_2"),
        ("lora_rank", 4),
        ("use_dynamic_batch_size", True),
        ("use_routing_replay", True),
        ("eval_num_gpus", 1),
        ("global_batch_size", 3),
        ("diffusion_encoder_loss_weight", float("nan")),
        ("diffusion_noise_epsilon", 0),
    ],
)
def test_rejects_unsupported_or_invalid_launches(field, value):
    args = valid_args()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_training_args(args)
