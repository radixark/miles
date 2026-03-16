"""``run`` and ``show-model-args`` CLI commands."""

import dataclasses
import json
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs
from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig
from miles.utils.debug_utils.run_megatron.cli.path_utils import resolve_megatron_path, resolve_model_script
from miles.utils.debug_utils.run_megatron.cli.prompt_utils import (
    PromptConfig,
    generate_token_ids,
    write_token_ids_to_tmpfile,
)
from miles.utils.debug_utils.run_megatron.cli.worker_executor import (
    build_dumper_env,
    build_torchrun_cmd,
    build_worker_args,
)
from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.misc import exec_command
from miles.utils.typer_utils import dataclass_cli


def register(app: typer.Typer) -> None:
    """Register ``run`` and ``show-model-args`` commands on *app*."""
    app.command()(run)
    app.command(name="show-model-args")(show_model_args)


@dataclasses.dataclass
class _RolloutSample:
    token_ids: list[int]
    sglang_logprobs: list[float]
    prompt_length: int


@dataclasses.dataclass
class _RolloutData:
    samples: list[_RolloutSample]
    routed_experts: list | None  # list of np arrays, shape [seq_len, num_layers, topk] per sample


def _load_rollout_data(rollout_path: Path) -> _RolloutData:
    """Load all samples from a rollout .pt file.

    Returns samples with token_ids, SGLang logprobs, prompt length,
    and optionally routed_experts for routing replay.
    """
    import torch

    data = torch.load(rollout_path, weights_only=False)
    samples = data["samples"]
    assert len(samples) > 0, f"No samples in {rollout_path}"

    results = []
    routed_experts_list = []
    has_routing = False
    for i, sample in enumerate(samples):
        if sample.get("rollout_log_probs") is None:
            continue
        token_ids = sample["tokens"]
        response_length = sample["response_length"]
        prompt_length = len(token_ids) - response_length
        sglang_logprobs = sample["rollout_log_probs"]
        assert (
            len(sglang_logprobs) == response_length
        ), f"sample {i}: rollout_log_probs length {len(sglang_logprobs)} != response_length {response_length}"
        results.append(
            _RolloutSample(
                token_ids=token_ids,
                sglang_logprobs=sglang_logprobs,
                prompt_length=prompt_length,
            )
        )
        re = sample.get("rollout_routed_experts")
        if re is not None:
            has_routing = True
        routed_experts_list.append(re)

    print(
        f"[cli] Loaded {len(results)}/{len(samples)} samples (routing_replay={'yes' if has_routing else 'no'})",
        flush=True,
    )
    return _RolloutData(
        samples=results,
        routed_experts=routed_experts_list if has_routing else None,
    )


def _pad_to_same_length(all_token_ids: list[list[int]], pad_multiple: int = 128) -> list[list[int]]:
    """Pad all sequences to the same length (max, rounded up to pad_multiple)."""
    max_len = max(len(t) for t in all_token_ids)
    if max_len % pad_multiple != 0:
        max_len = max_len + pad_multiple - (max_len % pad_multiple)
    return [t + [0] * (max_len - len(t)) for t in all_token_ids]


def _save_sglang_logprobs_as_baseline(
    rollout_samples: list[_RolloutSample],
    padded_seq_length: int,
    output_dir: Path,
) -> None:
    """Save all samples' SGLang logprobs in the comparator JSON format.

    Each sample becomes a separate batch entry. SGLang logprobs are for response tokens
    only. Megatron logprobs are next-token predictions at each position.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    for sample in rollout_samples:
        entries = []
        for i, lp in enumerate(sample.sglang_logprobs):
            global_position = sample.prompt_length + i - 1
            token_id = sample.token_ids[sample.prompt_length + i]
            entries.append(
                {
                    "global_position": global_position,
                    "token_id": token_id,
                    "logprob": float(lp),
                    "is_valid": True,
                }
            )
        all_entries.append(entries)

    payload = {
        "rank": 0,
        "tp_size": 1,
        "cp_size": 1,
        "pp_size": 1,
        "logprob_entries": all_entries,
    }

    output_path = output_dir / "rank_0.json"
    output_path.write_text(json.dumps(payload, indent=2))
    total_positions = sum(len(e) for e in all_entries)
    print(f"[cli] SGLang logprobs saved: {len(all_entries)} samples, {total_positions} total positions", flush=True)


def _save_routing_replay_from_rollout(
    routed_experts: list,
    padded_seq_length: int,
    output_dir: Path,
) -> Path:
    """Convert rollout routed_experts to run_megatron replay file format.

    Rollout format: per-sample np array [seq_len, num_layers, topk]
    Replay format: list[list[Tensor]] — outer=per MoE layer, inner=per sequence,
                   each tensor [seq_len, topk]

    For run_megatron, ALL layers are assumed MoE (moe_layer_freq=1 for V4).
    """
    import numpy as np
    import torch

    # Get num_layers from first valid sample
    first_valid = next(re for re in routed_experts if re is not None)
    num_layers = first_valid.shape[1]
    topk = first_valid.shape[2]

    # Build per-layer replay: list[list[Tensor]]
    # Outer: num_layers, Inner: num_samples, each [padded_seq_len, topk]
    per_layer: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

    for re in routed_experts:
        if re is None:
            # Pad with -1 (no routing)
            for layer_idx in range(num_layers):
                per_layer[layer_idx].append(torch.full((padded_seq_length, topk), -1, dtype=torch.int32))
            continue

        seq_len = re.shape[0]  # original seq_len (before padding)
        re_tensor = torch.from_numpy(np.asarray(re)).to(torch.int32)

        for layer_idx in range(num_layers):
            layer_data = re_tensor[:, layer_idx, :]  # [seq_len, topk]
            # Pad to padded_seq_length
            if seq_len < padded_seq_length:
                pad = torch.full((padded_seq_length - seq_len, topk), -1, dtype=torch.int32)
                layer_data = torch.cat([layer_data, pad], dim=0)
            per_layer[layer_idx].append(layer_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    from miles.utils.replay_base import routing_replay_manager

    save_path = output_dir / f"rank0_{routing_replay_manager.filename}"
    torch.save(per_layer, save_path)
    print(
        f"[cli] Routing replay saved: {num_layers} layers, {len(routed_experts)} sequences → {save_path}", flush=True
    )
    return output_dir


def run_impl(args: RunArgs) -> None:
    """Core run logic, called by both ``run`` command and ``run_and_compare``."""
    parallel: ParallelConfig = ParallelConfig.from_run_args(args)

    if args.routing_replay_dump_path is not None and parallel.nproc != 1:
        raise ValueError(f"Routing replay dump requires single-rank run (nproc=1), got {parallel}")

    resolved_megatron: Path = resolve_megatron_path(args.megatron_path)

    # --- Load rollout data if provided ---
    sglang_logprob_dir: Path | None = None
    if args.rollout_data is not None:
        rollout_data = _load_rollout_data(args.rollout_data)
        assert rollout_data.samples, "No valid samples with logprobs found"

        all_token_ids = [s.token_ids for s in rollout_data.samples]
        padded = _pad_to_same_length(all_token_ids)
        seq_length = len(padded[0])
        args.seq_length = seq_length

        # Write multi-sequence token_ids (list of lists)
        token_ids_file = write_token_ids_to_tmpfile(padded)  # type: ignore[arg-type]
        print(f"[cli] Token IDs: {len(padded)} sequences × {seq_length} tokens", flush=True)

        # Save SGLang logprobs for comparison after Megatron run
        sglang_logprob_dir = args.output_dir / "sglang_logprobs"
        _save_sglang_logprobs_as_baseline(rollout_data.samples, seq_length, sglang_logprob_dir)

        # Enable Megatron logprob output
        if args.logprob_output is None:
            args.logprob_output = args.output_dir / "megatron_logprobs"

        # Convert routing replay data from rollout format
        if rollout_data.routed_experts is not None and args.routing_replay_load_path is None:
            replay_dir = _save_routing_replay_from_rollout(
                rollout_data.routed_experts, seq_length, args.output_dir / "routing_replay"
            )
            args.routing_replay_load_path = replay_dir
    else:
        prompt: PromptConfig = PromptConfig(
            mode=args.prompt_mode,  # type: ignore[arg-type]
            text=args.prompt_text,
            file=args.prompt_file,
            seq_length=args.seq_length,
            apply_chat_template=args.apply_chat_template,
        )
        token_ids: list[int] = generate_token_ids(prompt=prompt, tokenizer_path=args.hf_checkpoint)
        token_ids_file = write_token_ids_to_tmpfile(token_ids)
        print(f"[cli] Token IDs written to {token_ids_file} ({len(token_ids)} tokens)", flush=True)

    script_args: WorkerScriptArgs = WorkerScriptArgs(
        hf_checkpoint=args.hf_checkpoint,
        token_ids_file=token_ids_file,
        role=args.role,
        ref_load=args.ref_load,
        run_backward=args.run_backward,
        source_patcher_config=args.source_patcher_config,
        routing_replay_dump_path=args.routing_replay_dump_path,
        routing_replay_load_path=args.routing_replay_load_path,
        top_k=args.top_k,
        logprob_output=args.logprob_output,
        allgather_cp=args.allgather_cp,
    )
    worker_args_str: str = build_worker_args(
        parallel=parallel,
        sp=args.sp,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        script_args=script_args,
        extra_args=args.extra_args,
    )

    dumper_env: dict[str, str] = build_dumper_env(
        output_dir=args.output_dir,
        run_backward=args.run_backward,
        dumper_filter=args.dumper_filter,
    )
    env_exports: str = " && ".join(f"export {k}='{v}'" for k, v in dumper_env.items())

    cmd: str = build_torchrun_cmd(
        model_type=args.model_type,
        megatron_path=resolved_megatron,
        nproc=parallel.nproc,
        worker_args=worker_args_str,
    )
    exec_command(f"{env_exports} && {cmd}")
    print(f"[cli] Run completed. Output: {args.output_dir}", flush=True)

    # --- Compare SGLang vs Megatron logprobs ---
    if sglang_logprob_dir is not None and args.logprob_output is not None:
        from miles.utils.debug_utils.run_megatron.logprob_comparator import compare_logprobs

        print("\n[cli] Comparing SGLang vs Megatron logprobs...", flush=True)
        compare_logprobs(
            baseline_dir=sglang_logprob_dir,
            target_dir=args.logprob_output,
            threshold=1e10,  # no pass/fail, just report
        )


@dataclass_cli(env_var_prefix="")
def run(args: RunArgs) -> None:
    """Launch torchrun to run Megatron standalone forward (or forward+backward)."""
    run_impl(args)


def show_model_args(
    model_type: Annotated[str, typer.Option(help="Model type matching scripts/models/{model_type}.sh")],
) -> None:
    """Show the MODEL_ARGS for a given model type (debug helper)."""
    output: str | None = exec_command(
        f'source "{resolve_model_script(model_type)}" && echo "${{MODEL_ARGS[@]}}"',
        capture_output=True,
    )
    if output:
        print(output.strip())
