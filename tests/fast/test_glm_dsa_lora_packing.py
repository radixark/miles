from __future__ import annotations

import pytest

from scripts import run_glm5_1_744b_a40b_lora, run_glm5_2_744b_a40b_lora

SCRIPTS = pytest.mark.parametrize(
    "script",
    [run_glm5_1_744b_a40b_lora, run_glm5_2_744b_a40b_lora],
    ids=["glm5.1", "glm5.2"],
)


def _config(script, backend: str, **kw) -> str:
    return script._get_parallel_config(script.ScriptArgs(dsa_attention_backend=backend, **kw))


@SCRIPTS
def test_tilelang_uses_thd_and_packs(script):
    """thd is the packed layout, so the fused kernels can take a dynamic batch."""
    cfg = _config(script, "tilelang")

    assert "--qkv-format thd" in cfg
    assert "--use-dynamic-batch-size" in cfg
    assert "--max-tokens-per-gpu" in cfg
    assert "--micro-batch-size" not in cfg


@SCRIPTS
def test_megatron_uses_bshd_and_does_not_pack(script):
    """bshd is the only layout that forbids dynamic batching, so it stays on a fixed micro-batch."""
    cfg = _config(script, "megatron")

    assert "--qkv-format bshd" in cfg
    assert "--micro-batch-size 1" in cfg
    assert "--use-dynamic-batch-size" not in cfg


@SCRIPTS
def test_max_tokens_per_gpu_is_honoured(script):
    assert "--max-tokens-per-gpu 24576" in _config(script, "tilelang", max_tokens_per_gpu=24576)


@SCRIPTS
def test_the_two_layouts_agree_on_everything_except_batching(script):
    """Parity check: the backend picks the query layout and the batching, nothing else."""

    def drop_batching(flags: list[str]) -> list[str]:
        out, skip = [], 0
        for f in flags:
            if skip:
                skip -= 1
                continue
            if f in ("--qkv-format", "--max-tokens-per-gpu", "--micro-batch-size"):
                skip = 1
                continue
            if f == "--use-dynamic-batch-size":
                continue
            out.append(f)
        return out

    assert drop_batching(_config(script, "tilelang").split()) == drop_batching(_config(script, "megatron").split())
