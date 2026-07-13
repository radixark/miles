"""Agent V2 launcher (Qwen3 / Qwen3-Coder): Miles <-> Harbor agent orchestration.

Thin wrapper over ``run.py`` — reuses its ``cleanup()`` / ``prepare()`` /
``execute()`` and only overrides the model identity and the SGLang/TITO
parsers, which are the sole differences from the GLM-4.7-Flash default.

Usage:
    # Qwen3-30B-A3B (general model)
    python run-qwen3-swe.py --prompt-data /root/swe_train.jsonl

    # Qwen3-Coder-30B-A3B-Instruct (coding-specialised; recommended base for
    # SWE-bench so rollouts produce non-zero reward variance)
    python run-qwen3-swe.py --coder --prompt-data /root/swe_train.jsonl

    # quick pipeline check (rollout only, no weight updates)
    python run-qwen3-swe.py --coder --mode debug_rollout_only
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

# run.py lives in the same directory (added to PYTHONPATH by execute()).
from run import ScriptArgs as BaseScriptArgs, cleanup, execute, prepare


@dataclass
class ScriptArgs(BaseScriptArgs):
    # Model identity (Qwen3-30B-A3B). ``--coder`` swaps these to the coder below.
    megatron_model_type: str = "qwen3-30B-A3B"
    model_name: str = "Qwen3-30B-A3B"
    hf_checkpoint: str = "Qwen/Qwen3-30B-A3B"
    ref_load: str = "/root/Qwen3-30B-A3B_torch_dist"
    save_dir: str = "/root/Qwen3-30B-A3B_agent_v2/"

    # SGLang / TITO parsers for Qwen3 (GLM defaults are glm47 / glm45 / glm47).
    sglang_tool_call_parser: str = "qwen25"
    sglang_reasoning_parser: str = "qwen3"
    tito_model: str = "qwen3"

    wandb_project: str = os.environ.get("WANDB_PROJECT", "qwen3-swe-agentic")
    wandb_run_name: str = "qwen3-swe-tito"
    prometheus_run_name: str = "qwen3-swe-tito"

    # Use Qwen3-Coder-30B-A3B-Instruct instead of the general model. The coder
    # is arch-identical (same converter/model script) but uses rope_theta=1e7
    # (vs 1e6), so we override MODEL_ARGS_ROTARY_BASE for the Megatron side;
    # SGLang picks up the correct value from the HF config automatically.
    coder: bool = False


@U.dataclass_cli
def main(args: ScriptArgs):
    if args.coder:
        args.model_name = "Qwen3-Coder-30B-A3B-Instruct"
        args.hf_checkpoint = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        args.ref_load = "/root/Qwen3-Coder-30B-A3B-Instruct_torch_dist"
        args.save_dir = "/root/Qwen3-Coder-30B-A3B-Instruct_agent_v2/"
        # Read by scripts/models/qwen3-30B-A3B.sh (${MODEL_ARGS_ROTARY_BASE:-1000000})
        # when the model script is sourced for both conversion and training.
        os.environ["MODEL_ARGS_ROTARY_BASE"] = "10000000"

    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
