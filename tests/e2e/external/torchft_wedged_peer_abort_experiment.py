"""Reproduce the FT abort hang with a GPU-wedged peer (NOT a sleep-wedged one).

NOT a CI test (needs a Ray GPU cluster). Mirrors the causal chain proven by
py-spy --native on the real pp2 with_failure run (see agent-context
2026-06-07-abort-hang-bisection-report):

    D (W's cell sibling) crashes -> W wedges inside a REAL native NCCL
    collective (its kernel spins on the GPU forever; this is cell1_rank1's
    state, stuck in cudaStreamSynchronize) -> the survivor S has an in-flight
    cross-cell allreduce to W -> S calls errored() (bug A: full-device cuda
    synchronize, indep_dp.py:115) or abort() (bug B: c10d abort ->
    waitForFutureOrTimeout, indep_dp.py:77) BEFORE the 120s userspace timer.

Every earlier attempt used a sleep-wedged peer: process stuck but GPU idle and
comms drained, so abort returned in <1s. The differentiator under test is the
peer's GPU state.

Scenarios (fresh actors each; env mirrors the real run:
CUDA_DEVICE_MAX_CONNECTIONS=1, NCCL_NVLS_ENABLE=1, NCCL_ALGO=Ring):
    matched_abort   Fully faithful shape: W's spinning cell kernel occupies its
                    single hw connection, W THEN posts the MATCHING cross
                    allreduce (queued behind, never launches), S's cross kernel
                    engages mid-collective. S aborts. (bug B candidate)
    matched_errored Same but S calls errored() (bug A on the matched shape).
    wedged_abort    No-show shape: W never posts the cross allreduce. Round-1
                    result: abort returns 0.52s (NOT the hang).
    wedged_errored  Round-1 result: REPRODUCED bug A -- errored() blocked the
                    full 120s until the userspace timer aborted S's comm.
    sleep_control   Old (false-negative) shape: W sleep-wedged, GPU idle.
                    Round-1 result: abort 0.52s, as always.

Usage:
    python tests/e2e/external/torchft_wedged_peer_abort_experiment.py
    python tests/e2e/external/torchft_wedged_peer_abort_experiment.py --only wedged_abort
"""

import logging
import os
import time
from datetime import timedelta
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)

_VERDICT_S = 75.0  # < cross timeout (120s) so the userspace timer cannot self-heal first
_SETTLE_S = 3.0
_REAL_RUN_ENV = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_ALGO": "Ring",
    "NCCL_DEBUG": "WARN",
}


@ray.remote(num_gpus=1, max_concurrency=4)
class _Worker:
    """One rank. max_concurrency=4 so probes run while another call blocks forever."""

    def init(self, *, name: str) -> dict:
        import torch

        self._name = name
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        return {"name": name, "device": str(self._device)}

    def build_cross_pg(self, *, store_addr: str, rank: int, timeout_s: float) -> dict:
        """torchft cross-cell NCCL PG (S<->W), warmed up. Mirrors create_indep_dp_group."""
        import torch
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupNCCL

        self._cross = ProcessGroupNCCL(timeout=timedelta(seconds=timeout_s))
        self._cross.configure(store_addr=store_addr, replica_id=self._name, rank=rank, world_size=2, quorum_id=0)

        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        t = torch.ones(8, device=self._device)
        assert self._cross.allreduce([t], opts).wait(), f"{self._name}: cross warmup failed"
        return {"name": self._name, "use_abort": self._cross._use_abort}

    def build_native_cell_pg(
        self, *, store_addr: str, rank: int, prefix: str = "wedged_peer_cell", long_timeout_s: float = 3600.0
    ) -> dict:
        """RAW c10d NCCL PG for a 2-rank 'cell' (no torchft timer): the wedge medium."""
        import torch
        from torch.distributed import PrefixStore
        from torch.distributed import ProcessGroupNCCL as BaseProcessGroupNCCL
        from torch.distributed import TCPStore

        host, port = store_addr.split(":")
        store = TCPStore(host_name=host, port=int(port), is_master=False)
        prefixed = PrefixStore(prefix, store)

        opts = BaseProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=long_timeout_s)
        self._cell = BaseProcessGroupNCCL(prefixed, rank, 2, opts)

        t = torch.ones(8, device=self._device)
        self._cell.allreduce([t]).wait()
        return {"name": self._name, "cell_warmup": float(t[0].item())}

    def die(self) -> None:
        os._exit(1)

    def wedge_sleep(self) -> None:
        time.sleep(1_000_000)

    def wedge_native_collective(self) -> None:
        """Block forever inside a real cell collective whose peer (D) is dead:
        a spinning NCCL kernel on this GPU -- cell1_rank1's wedge state."""
        import torch

        t = torch.ones(1 << 20, device=self._device)
        self._cell.allreduce([t]).wait()  # D never joins; raw c10d, huge timeout

    def enqueue_native_inflight(self) -> dict:
        """Post the cell collective WITHOUT waiting: the spinning kernel occupies
        this GPU's single hardware connection (CUDA_DEVICE_MAX_CONNECTIONS=1), so
        anything W enqueues afterwards (its matching cross allreduce) queues
        behind it and never launches -- the exact real-run shape where W's
        Megatron cell-internal comm wedges first, then W posts the cross grad
        allreduce."""
        import torch

        t = torch.ones(1 << 20, device=self._device)
        self._native_inflight = self._cell.allreduce([t])
        self._native_inflight_tensor = t
        return {"name": self._name, "native_inflight": True}

    def enqueue_inflight_cross(self, *, numel: int) -> dict:
        """S posts the cross allreduce W never joins (the in-flight grad allreduce
        at crash time). wait() is non-blocking for the nonblocking comm; do NOT
        touch errored() here -- that is bug A, measured separately."""
        import torch
        import torch.distributed as dist

        t = torch.ones(numel, device=self._device)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        start = time.monotonic()
        work = self._cross.allreduce([t], opts)
        ok = work.wait()
        self._inflight_tensor = t
        return {"name": self._name, "wait_ok": bool(ok), "wait_s": round(time.monotonic() - start, 3)}

    def abort_cross(self) -> dict:
        """Bug B shape: reconfigure_indep_dp_group's g.abort(errored=False)."""
        start = time.monotonic()
        self._cross.abort(errored=False)
        return {"name": self._name, "abort_s": round(time.monotonic() - start, 2)}

    def errored_cross(self) -> dict:
        """Bug A shape: _allreduce_grads_across_replicas' pg.errored() (internally
        a full torch.cuda.synchronize())."""
        start = time.monotonic()
        e = self._cross.errored()
        return {"name": self._name, "errored_s": round(time.monotonic() - start, 2), "errored": str(e)[:120]}

    def cuda_probe(self) -> dict:
        """Second-thread torch.cuda.synchronize() timing: GPU-wedged vs idle."""
        import torch

        start = time.monotonic()
        torch.cuda.synchronize()
        return {"name": self._name, "synchronize_s": round(time.monotonic() - start, 3)}

    def ping(self) -> str:
        return f"{self._name} alive at {time.monotonic():.1f}"


def _spawn(name: str) -> object:
    worker = _Worker.options(runtime_env={"env_vars": dict(_REAL_RUN_ENV)}).remote()
    ray.get(worker.init.remote(name=name), timeout=60)
    return worker


def _confirm_gpu_wedged(worker: object, *, who: str) -> str:
    try:
        probe = ray.get(worker.cuda_probe.remote(), timeout=10)
        return f"{who} GPU NOT wedged (synchronize returned: {probe})"
    except ray.exceptions.GetTimeoutError:
        return f"{who} GPU WEDGED (synchronize blocked >10s) -- faithful to cell1_rank1"


def _kill_then_await(ref: object, *, victim: object, victim_name: str, what: str) -> str:
    """The dead-peer law check: kill the named victim, see if the blocked call returns."""
    ray.kill(victim, no_restart=True)
    start = time.monotonic()
    try:
        out = ray.get(ref, timeout=120)
        return f"UNBLOCKED {round(time.monotonic() - start, 1)}s after killing {victim_name}: {out}"
    except ray.exceptions.GetTimeoutError:
        return f"STILL STUCK 120s after killing {victim_name} ({what})"
    except Exception as e:  # noqa: BLE001 - blocked call may surface the abort as an error
        return f"UNBLOCKED-with-exception {round(time.monotonic() - start, 1)}s after killing {victim_name}: {type(e).__name__}: {str(e)[:200]}"


def _run(exp: str, *, store_base: str, cell_store_addr: str, timeout_s: float, numel: int) -> str:
    print(f"== experiment {exp} ==")
    surv = _spawn("S")
    wedged = _spawn("W")
    sibling = _spawn("D")
    workers = [surv, wedged, sibling]

    try:
        # Cross pair (S rank0, W rank1) + W<->D native cell PG, all warmed up.
        cross_store = f"{store_base}/{exp}/cross"
        ray.get(
            [
                surv.build_cross_pg.remote(store_addr=cross_store, rank=0, timeout_s=timeout_s),
                wedged.build_cross_pg.remote(store_addr=cross_store, rank=1, timeout_s=timeout_s),
            ],
            timeout=120,
        )
        ray.get(
            [
                wedged.build_native_cell_pg.remote(store_addr=cell_store_addr, rank=0),
                sibling.build_native_cell_pg.remote(store_addr=cell_store_addr, rank=1),
            ],
            timeout=120,
        )
        print("  cross pair + native cell PG built and warmed")

        if exp.startswith("self_occluded"):
            # 503903's shape: the SURVIVOR's own orphaned cell collective spins on
            # its single hw connection, so S's cross kernel (and the teardown work
            # commDestroySync waits on) can never launch/retire. The far peer W is
            # alive and idle -- irrelevant by construction.
            s2 = _spawn("S2")
            workers.append(s2)
            ray.get(
                [
                    surv.build_native_cell_pg.remote(store_addr=cell_store_addr, rank=0, prefix="surv_cell"),
                    s2.build_native_cell_pg.remote(store_addr=cell_store_addr, rank=1, prefix="surv_cell"),
                ],
                timeout=120,
            )
            native = ray.get(surv.enqueue_native_inflight.remote(), timeout=30)  # S2 alive, never joins
            print(f"  S native orphaned in-flight (S2 alive, never posts): {native}")
            time.sleep(_SETTLE_S)
            print(f"  {_confirm_gpu_wedged(surv, who='S')} (S itself must be occluded)")

            inflight = ray.get(surv.enqueue_inflight_cross.remote(numel=numel), timeout=30)
            print(f"  S cross allreduce queued behind its own spinning cell kernel: {inflight}")

            what = "errored()" if exp.endswith("errored") else "abort()"
            ref = (surv.errored_cross if exp.endswith("errored") else surv.abort_cross).remote()
            try:
                out = ray.get(ref, timeout=_VERDICT_S)
                return f"{what} returned in <{_VERDICT_S}s: {out} (NOT reproduced)"
            except ray.exceptions.GetTimeoutError:
                print(f"  *** {what} BLOCKED >{_VERDICT_S}s (REPRODUCED bug B shape) *** killing S2 (cell sibling)")
                return f"*** {what} BLOCKED >{_VERDICT_S}s (REPRODUCED) *** then: {_kill_then_await(ref, victim=s2, victim_name='S2 (occluder peer)', what=what)}"

        # The real failure shape: D dies, then W wedges in a real cell collective.
        sibling.die.remote()
        time.sleep(_SETTLE_S)
        if exp == "sleep_control":
            wedged.wedge_sleep.remote()
            time.sleep(_SETTLE_S)
            print(f"  {_confirm_gpu_wedged(wedged, who='W')} (control expects NOT wedged)")
        elif exp.startswith("matched_"):
            # The fully faithful shape: W's spinning cell kernel occupies its single
            # hardware connection (CUDA_DEVICE_MAX_CONNECTIONS=1), THEN W posts the
            # MATCHING cross allreduce, which queues behind it and never launches.
            # S's cross kernel therefore engages mid-collective with a peer whose
            # kernel will never arrive -- unlike the no-show shape above.
            native = ray.get(wedged.enqueue_native_inflight.remote(), timeout=30)
            print(f"  W native in-flight: {native}")
            w_cross = ray.get(wedged.enqueue_inflight_cross.remote(numel=numel), timeout=30)
            print(f"  W matched cross posted (queued behind spinning kernel): {w_cross}")
            time.sleep(_SETTLE_S)
            print(f"  {_confirm_gpu_wedged(wedged, who='W')}")
        else:
            wedged.wedge_native_collective.remote()
            time.sleep(_SETTLE_S)
            print(f"  {_confirm_gpu_wedged(wedged, who='W')}")

        inflight = ray.get(surv.enqueue_inflight_cross.remote(numel=numel), timeout=30)
        print(f"  S in-flight cross allreduce (numel={numel}): {inflight}")

        target = surv.errored_cross if exp.endswith("errored") else surv.abort_cross
        what = "errored()" if exp.endswith("errored") else "abort()"
        ref = target.remote()
        try:
            out = ray.get(ref, timeout=_VERDICT_S)
            return f"{what} returned in <{_VERDICT_S}s: {out} (NOT reproduced)"
        except ray.exceptions.GetTimeoutError:
            print(f"  *** {what} BLOCKED >{_VERDICT_S}s (REPRODUCED the hang) *** now testing the dead-peer law")
            return f"*** {what} BLOCKED >{_VERDICT_S}s (REPRODUCED) *** then: {_kill_then_await(ref, victim=wedged, victim_name='W', what=what)}"
    finally:
        for w in [surv, wedged, sibling]:
            try:
                ray.kill(w, no_restart=True)
            except Exception:  # noqa: BLE001 - idempotent cleanup
                pass


def main(
    timeout_s: Annotated[float, typer.Option(help="torchft cross PG timeout (miles uses 120)")] = 120.0,
    numel: Annotated[int, typer.Option(help="in-flight cross allreduce size (grad bucket scale)")] = 1 << 22,
    only: Annotated[str | None, typer.Option(help="run a single experiment")] = None,
) -> None:
    """Run the wedged-peer abort reproduction matrix; one RESULT line per experiment."""
    ray.init(ignore_reinit_error=True)

    from torch.distributed import TCPStore

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    store_base = f"localhost:{store.port}/wedged_peer"

    experiments = ["matched_abort", "matched_errored", "wedged_abort", "wedged_errored", "sleep_control"]
    if only is not None:
        experiments = [only]

    results: dict[str, str] = {}
    for i, exp in enumerate(experiments):
        cell_store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
        results[exp] = _run(
            exp,
            store_base=store_base,
            cell_store_addr=f"localhost:{cell_store.port}",
            timeout_s=timeout_s,
            numel=numel,
        )
        print(f"RESULT {exp}: {results[exp]}\n")
        del cell_store
        time.sleep(3)

    print("==== SUMMARY ====")
    for exp, res in results.items():
        print(f"RESULT {exp}: {res}")
    del store


if __name__ == "__main__":
    typer.run(main)
