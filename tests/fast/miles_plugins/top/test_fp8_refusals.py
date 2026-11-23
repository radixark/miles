"""Trainer paths the TOP FP8 masters do not cover are refused at model build, before anything runs."""

from types import SimpleNamespace

import pytest

fp8_loader = pytest.importorskip("miles_plugins.top.fp8_loader")
glm_config = pytest.importorskip("miles_plugins.top.glm_config")


def _args(**overrides):
    values = dict(
        lora_rank=0,
        lora_adapter_path=None,
        save_hf=None,
        eval_uses_snapshots=False,
        eval_hf_dir=None,
    )
    backend = SimpleNamespace(overlap_param_gather=overrides.pop("overlap_param_gather", False))
    return SimpleNamespace(backend=backend, **(values | overrides))


def test_the_recipe_arguments_are_admitted():
    fp8_loader.refuse_unsupported(_args())
    # Snapshot eval without an HF export dir publishes nothing.
    fp8_loader.refuse_unsupported(_args(eval_uses_snapshots=True))


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"overlap_param_gather": True}, "--overlap-param-gather"),
        ({"lora_rank": 8}, "LoRA"),
        ({"lora_adapter_path": "/adapter"}, "LoRA"),
        ({"save_hf": "/out/{rollout_id}"}, "--save-hf"),
        ({"eval_uses_snapshots": True, "eval_hf_dir": "/dev/shm/eval"}, "snapshot eval"),
    ],
    ids=["overlap-param-gather", "lora-rank", "lora-adapter", "save-hf", "snapshot-eval"],
)
def test_paths_the_fp8_masters_do_not_cover_are_refused(overrides, refusal):
    with pytest.raises(NotImplementedError, match=rf"\[top\].*{refusal}"):
        fp8_loader.refuse_unsupported(_args(**overrides))


def test_glm_block_fp8_refuses_a_trainer_mtp_block(monkeypatch):
    """The FP8 install never fills an MTP block, so it would keep its random init (and train noise)."""
    from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1

    monkeypatch.setattr(glm_config, "checkpoint_is_block_fp8", lambda _: True)
    with pytest.raises(NotImplementedError, match=r"\[top\].*--mtp-num-layers 0"):
        glm_config.prepare_glm_config(
            SimpleNamespace(hf_checkpoint="/ckpt"), SimpleNamespace(mtp_num_layers=1), contract=TRUE_ON_POLICY_V1
        )
