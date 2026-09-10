# Standalone Tinker multi-LoRA pressure test

This PR adds example tools only, on top of #2846 at
`17b7ee73d684d9860fd6ec2b9103b7753ba9d735`. It does not change Miles runtime,
`serve_tinker.py`, the original multi-LoRA example, or CI/unit-test files.
The scripts are submitted for review **before running the N-user experiment**.
The standalone GPU probe and N-user E2E are not yet validated on hardware.

## Scripts to review

- [Single Tinker user](tinker_e2e_user.py): `run_user()` / `_train_step()`.
- [N concurrent users](pressure_client.py): starts N tasks running that exact user flow.
- [Capacity estimate](slot_capacity.py): `_probe()` / `_report()`; inspired by #3170.
- [Probe worker](slot_probe_actor.py): uses the existing `worker_class` extension;
  inherits the real training implementation and only adds memory observation.
- [Probe/serve launcher](run_pressure.py): selects the independent probe program
  or the **unchanged** `serve_tinker.py` with an integer slot count.
- [DAPO preparation and checks](pressure_dapo.py), [step timing](pressure_timing.py),
  and [optional telemetry uploader](pressure_telemetry.py).

The user flow follows the official
[Tinker cookbook RL loop](https://github.com/thinking-machines-lab/tinker-cookbook/blob/1f962eda3a2cec8de284725f2adc9978e93dfcd3/tinker_cookbook/recipes/rl_loop.py).
Its SDK sequence is adapted to DAPO/PPO, the DAPO math dataset and 8K context.
It does not call trainer methods directly; only the separate capacity probe does.

## Workload and pass criteria

Each user owns a distinct SDK session, tenant key, LoRA model and Adam stream:

1. Create the training client; publish its current weights and get a sampler.
2. Sample eight responses per prompt. Prompt plus response is at most **8192**
   tokens. Retain four groups with mixed correct/incorrect responses, with at
   most 64 prompt attempts per step. Use group-normalized advantages, response
   token normalization, DAPO's overlength penalty, and PPO bounds 0.8 / 1.28.
3. Submit `forward_backward_async(..., loss_fn="ppo")` and `optim_step_async()`;
   await both and check finite loss/logprobs and a positive, finite gradient norm.
4. Repeat for three optimizer steps by default. Finally publish the last update
   and sample again, so the final optimizer update also reaches inference.

The concurrent launcher uses **N independent SDK users in one process**, sharing
only tokenized input data. A one-time start gate ensures all N models exist before
rollout; an end gate keeps their sessions alive until everyone finishes. There
are **no cross-user barriers between training steps or before forward/backward**.
Each user trains as soon as its own batch is ready.

A trial passes only when every distinct model completes every requested step and
final sampling. An exception or deadline fails the trial and cancels the other
users. Filtering exhaustion, timeouts and RPC failures are not classified as OOM.
A passing N is a tested count, **not a confirmed maximum**. To find the memory
boundary later, restart with larger explicit N and repeat; retain GPU logs showing
CUDA OOM at the failing count. Engine request concurrency can still be smaller
than N without changing the number of resident adapters.

## Capacity estimate

The separate probe starts one trainer slot, warms up a maximum-token
forward/backward plus zero-learning-rate Adam step, then measures a second pass.
Every DP replica receives a full row. It launches no rollout engines or SDK users.
It writes raw per-rank measurements, stops its own workers and exits.

For rank `r`, with measured slot storage `S`, warmed free memory `F`, transient
activation peak `A` and margin `M`:

```text
N_measured[r] = floor(max(0, F[r] + S[r] - A[r] - M) / S[r])
N_gpu         = min_r N_measured[r]
N_host        = floor(per_engine_host_budget / (keep_K * full_adapter_bytes))
N             = min(N_gpu, N_host)   # host constraint optional; keep_K defaults to 2
```

The closed-form cross-check accounts for LayerWise optimizer ownership:
`P_local * (weight_bytes + gradient_bytes) + P_optimizer_owned * (master_bytes + moment_bytes)`.
It is separately reported as `n_theoretical_trainer`; it does not replace the
storage measurement. `n_slots` is the measured estimate after the optional host
constraint. The report identifies the limiting rank/constraint.

This is a **capacity estimate**, not a proven worst-case bound: allocator padding,
MoE routing, batching, and engine GPU LoRA/KV memory can change the actual limit.
The synthetic CE probe is not the DAPO E2E. `engine_gpu_capacity_checked` is false
and `e2e_max_n` remains null until an actual sweep provides evidence. An optional
host budget is **per engine**, so divide node RAM appropriately when multiple
engines share a node. The server uses N GPU LoRA buffers per engine and up to 2N
host adapter versions, through #2846's existing arguments.

## Run after script review

Use the Miles GPU environment (including `tinker`, `transformers` and math reward
dependencies); the clients need Python 3.11+. Start from this repository's root.
Have an existing four-node H200 Ray cluster and the full BF16
`Qwen3-30B-A3B` checkpoint on shared storage. The default placement is 16 trainer
GPUs (TP2/EP8) plus 16 rollout GPUs (eight TP2/EP2 engines), rank16/alpha32.
The launcher uses the repository's standard `execute_train` process preamble;
run it on the dedicated experiment cluster with the previous trial stopped.
It does not acquire or release devbox leases.

Pass the same checkpoint, topology, precision, recompute and token settings to
both jobs. If using separate source checkouts, pass `--sglang-pythonpath`, and
provide the cluster's Ray/network environment as with other Miles launchers.
`MILES_SCRIPT_EXTERNAL_RAY=1` preserves the already joined Ray cluster.

```bash
# 1. Trainer-only probe; exits after writing /shared/probe/slot-capacity.json.
MILES_SCRIPT_EXTERNAL_RAY=1 python examples/multi_lora/run_pressure.py \
  --mode probe --hf-checkpoint /models/Qwen3-30B-A3B --output-dir /shared/probe

# 2. Start the unmodified gateway with N from that report; no users start here.
MILES_SCRIPT_EXTERNAL_RAY=1 python examples/multi_lora/run_pressure.py \
  --mode serve --hf-checkpoint /models/Qwen3-30B-A3B --output-dir /shared/server \
  --capacity-report /shared/probe/slot-capacity.json

# 3. Single-user E2E, after the gateway is ready (use a fresh output directory).
python -m examples.multi_lora.tinker_e2e_user \
  --model /models/Qwen3-30B-A3B --dataset /datasets/dapo-math-17k.jsonl \
  --steps 3 --output-dir /shared/one-user

# 4. On a fresh gateway with the same N slots, run the N-user workload.
python -m examples.multi_lora.pressure_client \
  --capacity-report /shared/probe/slot-capacity.json \
  --model /models/Qwen3-30B-A3B --dataset /datasets/dapo-math-17k.jsonl \
  --steps 3 --output-dir /shared/n-users
```

Restart the serving job between trials to remove the preceding users/models.
For a later sweep, replace the server's `--capacity-report` with `--n-adapters N`
and the client's with `--clients N`; they must agree. No automatic search or
background client supervisor runs from these scripts. For sustained training,
use `--steps 0` and an appropriate `--timeout-seconds`; interruption is not a pass.
Sampler exports accumulate in the server checkpoint directory, so a long run
also needs sufficient disk space and a separately reviewed retention policy.

## Results and timing

`result.json` records aggregate pass/fail and completed users. Each LoRA has its
own phase, progress, error (on failure), result, raw step-time and timing-summary
files, plus an aggregate `timing-comparison.json` after a passing finite trial.
Summaries include mean, P50/P90/P95 and standard deviation. Raw timings
split publication, rollout/batch preparation, SDK submission, forward/backward
wait and optimizer wait. These waits include server queueing and execution;
they do not separately measure scheduler wait and GPU computation.

**W&B is off by default**, regardless of whether an API key exists in the
environment. `--wandb` opts into a separate uploader, with one run per LoRA;
`WANDB_ENTITY` and `WANDB_PROJECT` select the destination. Local journals remain
the primary results and are written without network access. Credentials are
never added to launcher arguments or committed scripts.
