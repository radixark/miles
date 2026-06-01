from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import enum
import logging
import shutil
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from sglang.srt.debug_utils.dumper import DumperConfig, _get_rank, dumper

from miles.backends.training_utils.parallel import ParallelState, get_parallel_state
from miles.utils.process_group_utils import GeneralPGUtil

logger = logging.getLogger(__name__)


class DumperPhase(enum.Enum):
    INFERENCE = "inference"
    FWD_ONLY = "fwd_only"
    FWD_BWD = "fwd_bwd"


# ------------------------------- SGLang -------------------------------------


def get_sglang_env(args: Namespace) -> dict[str, str]:
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return {}

    env: dict[str, str] = {"DUMPER_SERVER_PORT": "reuse"}
    overrides = _get_phase_override_configs(args, DumperPhase.INFERENCE)

    # SGLang registers non-intrusive hooks while loading the model. Configs that
    # affect hook registration must be present in the actor environment; the
    # later HTTP configure call only controls active dumping/output location.
    if non_intrusive_mode := overrides.get("non_intrusive_mode"):
        env["DUMPER_NON_INTRUSIVE_MODE"] = str(non_intrusive_mode)

    if source_patcher_config := args.dumper_source_patcher_config_inference:
        env["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config
    elif source_patcher_config := overrides.get("source_patcher_config"):
        env["DUMPER_SOURCE_PATCHER_CONFIG"] = str(source_patcher_config)

    return env


async def configure_sglang(args: Namespace) -> None:
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return

    from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
    from miles.utils.http_utils import post

    worker_urls = await get_worker_urls(args)
    overrides = _get_phase_override_configs(args, DumperPhase.INFERENCE)

    engines_dir: Path = _get_dir(args) / "engines"
    _cleanup_dump_dir(engines_dir)

    coros = []
    for i, url in enumerate(worker_urls):
        body = {
            "enable": True,
            "dir": str(_get_dir(args)),
            "exp_name": f"engines/engine_{i}",
            **overrides,
        }
        coros.append(post(f"{url}/dumper/configure", body))

    await asyncio.gather(*coros)
    logger.info("Configured dumper on %d SGLang engines", len(worker_urls))


# ------------------------------- Megatron -------------------------------------


class DumperMegatronUtil:
    def __init__(self, args: Namespace, model: Sequence[torch.nn.Module], phase: DumperPhase) -> None:
        self.args = args
        self.phase = phase
        self.overrides = _get_phase_override_configs(args, phase)
        self.enabled = self._configure(args, phase, self.overrides)
        if self.enabled:
            dumper.register_non_intrusive_dumper(self._extract_model(model))

    def wrap_forward_step(self, forward_step_func: Callable) -> Callable:
        if not self.enabled:
            return forward_step_func

        return _wrap_forward_step_with_stepping(forward_step_func)

    def finalize(self, model: Sequence[torch.nn.Module]) -> None:
        if not self.enabled:
            return

        extracted_model = self._extract_model(model)
        if self.phase is DumperPhase.FWD_BWD and self.overrides.get("enable_model_grad"):
            _log_model_grad_coverage(extracted_model)
            # The raw per-rank ``main_grad`` is not topology-invariant: a non-FT
            # baseline dumps a single intra-DP rank's local (un-reduced) grad,
            # while an indep-DP (fault-tolerant) target dumps the cross-replica
            # SUM. Reduce each side to the same global gradient so dumps are
            # directly comparable across different DP topologies.
            with _topology_invariant_grad_for_dump(extracted_model, self.args):
                dumper.dump_model(extracted_model)
        else:
            dumper.dump_model(extracted_model)
        dumper.step()
        dumper.configure(enable=False)

    @staticmethod
    def _extract_model(model: Sequence[torch.nn.Module]) -> torch.nn.Module:
        assert (
            len(model) == 1
        ), f"Dumper does not yet support virtual pipeline parallelism (got {len(model)} model chunks)"
        return model[0]

    @staticmethod
    def _configure(args: Namespace, phase: DumperPhase, overrides: dict[str, Any] | None = None) -> bool:
        if overrides is None:
            overrides = _get_phase_override_configs(args, phase)
        if not overrides.get("enable"):
            return False

        merged = {
            "dir": str(_get_dir(args)),
            "exp_name": phase.value,
            "enable_output_console": False,
            **overrides,
        }

        # Only write dump files on effective DP rank 0 (covers both intra-DP
        # and indep-DP). Other DP ranks still participate in dumper collectives
        # (barrier, broadcast, allgather) but don't produce output files.
        # TODO: optimize — non-DP-rank-0 ranks currently run full dumper logic
        # (forward hooks, model iteration) without producing output.
        if get_parallel_state().effective_dp.rank != 0:
            merged["enable_output_file"] = False
            merged["enable_output_console"] = False

        full_config = DumperConfig(**merged)
        dumper.reset()
        _cleanup_dump_dir(Path(merged["dir"]) / merged["exp_name"])
        dumper.configure(**dataclasses.asdict(full_config))
        return True


def _loss_parallel_size(parallel_state: ParallelState, args: Namespace) -> int:
    # Mirror the loss scaling applied in
    # miles.backends.training_utils.loss.loss_function: the loss (hence the
    # accumulated grad) is multiplied by exactly this factor, so dividing it out
    # cancels the loss scaling symmetrically on both compared runs.
    if args.true_on_policy_mode and parallel_state.is_ulysses_cp:
        return parallel_state.intra_dp.size
    return parallel_state.intra_dp_cp.size


@contextlib.contextmanager
def _topology_invariant_grad_for_dump(model: torch.nn.Module, args: Namespace) -> Iterator[None]:
    """Temporarily expose the global, loss-scale-normalized gradient to the dumper.

    The dumper reads ``param.grad``, falling back to ``param.main_grad``. Under
    Megatron's fp32 grad accumulation ``param.grad`` is ``None`` and the gradient
    lives in the fp32 ``main_grad`` buffer, so we rebind ``main_grad`` to the
    reduced tensor for the duration of the dump (rebinding ``param.grad`` would
    hit PyTorch's grad-dtype check, since ``main_grad`` is fp32 while the param is
    bf16). We compute ``sum_over_intra_dp(main_grad) / loss_parallel_size``:

    - Non-FT baseline: ``intra_dp`` spans all DP ranks, so the all-reduce
      reconstructs the full gradient from per-rank local accumulators.
    - Indep-DP (FT) target: ``intra_dp`` is trivial (size 1); the cross-replica
      SUM already happened upstream, so the all-reduce is a no-op.

    Dividing by ``loss_parallel_size`` removes the Megatron loss scaling that
    differs between the two topologies. The remaining per-microbatch / batch-size
    factors are identical across runs and cancel in the comparison.

    All intra-DP ranks run the dumper (only effective-DP rank 0 writes files), so
    the all-reduce is collective and symmetric across the group. Rebinding the
    ``main_grad`` attribute does not affect the optimizer, which holds its own
    references into the gradient buffer.
    """
    parallel_state = get_parallel_state()
    intra_dp = parallel_state.intra_dp
    loss_parallel_size = _loss_parallel_size(parallel_state, args)

    saved_main_grads: list[tuple[torch.nn.Parameter, torch.Tensor | None]] = []
    try:
        for _name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            main_grad = getattr(param, "main_grad", None)
            if main_grad is None or param.grad is not None:
                # main_grad accumulation is the only path the dumper would read
                # here; skip anything that does not match it rather than risk a
                # wrong-source or dtype-mismatched override.
                continue

            reduced = main_grad.detach().float().clone()
            if intra_dp.size > 1 and intra_dp.group is not None:
                GeneralPGUtil.create(intra_dp.group).all_reduce(reduced, intra_dp.group, op=dist.ReduceOp.SUM)
            reduced /= loss_parallel_size

            saved_main_grads.append((param, param.main_grad))
            param.main_grad = reduced

        yield
    finally:
        for param, original_main_grad in saved_main_grads:
            param.main_grad = original_main_grad


def _log_model_grad_coverage(model: torch.nn.Module) -> None:
    missing: list[str] = []
    with_grad = 0
    total = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        total += 1
        grad = param.grad if param.grad is not None else getattr(param, "main_grad", None)
        if grad is None:
            missing.append(name)
        else:
            with_grad += 1

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else _get_rank()
    logger.info(
        "Dumper fwd_bwd model grad coverage rank=%s with_grad=%d total=%d missing=%d missing_names=%s",
        rank,
        with_grad,
        total,
        len(missing),
        missing[:20],
    )


def _wrap_forward_step_with_stepping(forward_step_func: Callable) -> Callable:
    is_first_call = True

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal is_first_call
        if not is_first_call:
            dumper.step()
        is_first_call = False
        return forward_step_func(*args, **kwargs)

    return _wrapped


# ------------------------------- Common -------------------------------------


def _cleanup_dump_dir(dump_dir: Path) -> None:
    # Only cell 0's rank 0 deletes — avoids race when multiple cells' rank 0
    # all see _get_rank()==0 and try to rmtree the same directory.
    # Best-effort: stale handles from a peer that crashed (NFS .nfsXXXX stubs)
    # can make rmtree fail with "Directory not empty"; we don't want that to
    # propagate up and mark the (healthy) cell as errored.
    indep_dp = get_parallel_state().indep_dp
    if (_get_rank() == 0) and (indep_dp.rank == 0) and dump_dir.is_dir():
        try:
            shutil.rmtree(dump_dir)
        except OSError:
            logger.warning("dump dir cleanup failed; continuing", exc_info=True)
    if dist.is_initialized():
        dist.barrier()
    if indep_dp.group is not None:
        indep_dp.group.barrier()


def _get_phase_override_configs(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    raw = getattr(args, f"dumper_{phase.value}")
    return {"enable": args.dumper_enable, **DumperConfig._kv_pairs_to_dict(raw)}


def _is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    return _get_phase_override_configs(args, phase).get("enable", False)


def _get_dir(args: Namespace) -> Path:
    return Path(args.dumper_dir)
