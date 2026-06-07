"""Reproduce the FATAL FT hang: two concurrent abort() deadlock on a host mutex.

NOT a CI test (needs a Ray GPU cluster). This targets the root cause proven by
gdb 'thread apply all bt' on the real pp2 hang (see agent-context
2026-06-07-hang-understanding-v4-ROOT-CAUSE): the healthy cell's fatal ~600s
hang is NOT GPU-blocked. It is two concurrent ``ProcessGroupNCCL::abort()``
calls on the SAME PG (torchft's 120s _WorkAcceleratorTimeout timer abort +
miles ``reconfigure_indep_dp_group``'s explicit ``g.abort(errored=False)``)
deadlocking in c10d ``abortComms``: one abort holds the PG mutex stuck in NCCL
``commFree``, the other blocks at ``pthread_mutex_lock`` at abortComms entry.

Every earlier UT issued a SINGLE abort and never hung (wedged_abort: 0.52s).
The differentiator is concurrency.

Scenarios (env mirrors the real run: CUDA_DEVICE_MAX_CONNECTIONS=1,
NCCL_NVLS_ENABLE=1, NCCL_ALGO=Ring):
    single_abort   one abort() on the wedged-peer cross PG. EXPECT: returns fast.
    double_abort   TWO threads call abort() on the SAME cross PG at a barrier.
                   EXPECT (claim): host mutex deadlock -> blocks.
    double_abort_clean  two concurrent aborts on a HEALTHY (non-wedged) cross PG,
                   to see if concurrency alone deadlocks or it needs the
                   errored/teardown work.

Usage:
    python tests/e2e/external/torchft_concurrent_abort_experiment.py
    python tests/e2e/external/torchft_concurrent_abort_experiment.py --only double_abort
"""

import logging
import os
import threading
import time
from datetime import timedelta
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)

_VERDICT_S = 75.0
_SETTLE_S = 3.0
_REAL_RUN_ENV = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_ALGO": "Ring",
    "NCCL_DEBUG": "WARN",
}


@ray.remote(num_gpus=1, max_concurrency=4)
class _Worker:
    def init(self, *, name: str) -> dict:
        import torch

        self._name = name
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        return {"name": name, "device": str(self._device)}

    def build_cross_pg(self, *, store_addr: str, rank: int, timeout_s: float) -> dict:
        import torch
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupNCCL

        self._cross = ProcessGroupNCCL(timeout=timedelta(seconds=timeout_s))
        self._cross.configure(store_addr=store_addr, replica_id=self._name, rank=rank, world_size=2, quorum_id=0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        t = torch.ones(8, device=self._device)
        assert self._cross.allreduce([t], opts).wait(), f"{self._name}: warmup failed"
        return {"name": self._name, "use_abort": self._cross._use_abort}

    def build_many_pgs(self, *, store_base: str, rank: int, timeout_s: float, k: int) -> dict:
        """Build K torchft NCCL PGs in one process (mimics the survivor holding many
        co-resident Megatron comms), each warmed up."""
        import torch
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupNCCL

        self._pgs = []
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        for i in range(k):
            pg = ProcessGroupNCCL(timeout=timedelta(seconds=timeout_s))
            pg.configure(store_addr=f"{store_base}/pg{i}", replica_id=self._name, rank=rank, world_size=2, quorum_id=0)
            t = torch.ones(8, device=self._device)
            assert pg.allreduce([t], opts).wait(), f"{self._name}: pg{i} warmup failed"
            self._pgs.append(pg)
        return {"name": self._name, "k": len(self._pgs)}

    def enqueue_inflight_many(self, *, numel: int) -> dict:
        import torch
        import torch.distributed as dist

        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        self._inflight = []
        for pg in self._pgs:
            t = torch.ones(numel, device=self._device)
            pg.allreduce([t], opts).wait()
            self._inflight.append(t)
        return {"name": self._name, "posted": len(self._inflight)}

    def abort_concurrent_raw(self, *, n: int = 2) -> dict:
        """Call the UNDERLYING c10d ProcessGroup.abort() from N threads at a barrier.

        torchft's wrapper.abort() does `pg=self._pg; pg.abort(); self._pg=None`, so
        when abort is fast the 2nd caller sees _pg=None and no-ops (why the earlier
        double_abort didn't deadlock). Here we grab `self._cross._pg` (the c10d PG)
        ONCE and call its abort() directly from N threads -> N genuinely concurrent
        c10d ProcessGroupNCCL::abort() (std::async abortComms) on the same comm,
        exactly the gdb-observed real shape."""
        raw = self._cross._pg  # the c10d BaseProcessGroup behind the torchft wrapper
        assert raw is not None and hasattr(raw, "abort"), f"no raw c10d abort: {type(raw)}"
        barrier = threading.Barrier(n)
        results: list[float] = [0.0] * n
        errs: list[str] = [""] * n

        def _do(i: int) -> None:
            barrier.wait()
            t0 = time.monotonic()
            try:
                raw.abort()
            except Exception as e:  # noqa: BLE001
                errs[i] = f"{type(e).__name__}: {str(e)[:120]}"
            results[i] = round(time.monotonic() - t0, 2)

        threads = [threading.Thread(target=_do, args=(i,)) for i in range(n)]
        start = time.monotonic()
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=_VERDICT_S)
        alive = [i for i, th in enumerate(threads) if th.is_alive()]
        return {
            "name": self._name,
            "raw_type": type(raw).__name__,
            "n": n,
            "total_s": round(time.monotonic() - start, 2),
            "per_thread_s": results,
            "threads_still_alive": alive,
            "errs": [e for e in errs if e],
        }

    def abort_many_concurrent(self) -> dict:
        """Fire one abort() per PG, all started together at a barrier: many
        concurrent abortComms contending NCCL/global teardown locks."""
        n = len(self._pgs)
        barrier = threading.Barrier(n)
        results: list[float] = [0.0] * n
        errs: list[str] = [""] * n

        def _do(i: int) -> None:
            barrier.wait()
            t0 = time.monotonic()
            try:
                self._pgs[i].abort(errored=False)
            except Exception as e:  # noqa: BLE001
                errs[i] = f"{type(e).__name__}: {str(e)[:80]}"
            results[i] = round(time.monotonic() - t0, 2)

        threads = [threading.Thread(target=_do, args=(i,)) for i in range(n)]
        start = time.monotonic()
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=_VERDICT_S)
        alive = [i for i, th in enumerate(threads) if th.is_alive()]
        return {
            "name": self._name,
            "k": n,
            "total_s": round(time.monotonic() - start, 2),
            "threads_still_alive": alive,
            "max_thread_s": max(results),
            "errs": [e for e in errs if e][:3],
        }

    def die(self) -> None:
        os._exit(1)

    def wedge_sleep(self) -> None:
        time.sleep(1_000_000)

    def enqueue_inflight_cross(self, *, numel: int) -> dict:
        import torch
        import torch.distributed as dist

        t = torch.ones(numel, device=self._device)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        work = self._cross.allreduce([t], opts)
        ok = work.wait()
        self._inflight_tensor = t
        return {"name": self._name, "wait_ok": bool(ok)}

    def abort_once(self) -> dict:
        start = time.monotonic()
        self._cross.abort(errored=False)
        return {"name": self._name, "abort_s": round(time.monotonic() - start, 2)}

    def abort_concurrent(self, *, n: int = 2) -> dict:
        """Fire N abort() calls on the SAME PG from N threads, started together at
        a barrier -- the timer-abort + reconfigure-abort race the real run hit."""
        barrier = threading.Barrier(n)
        results: list[float] = [0.0] * n
        errs: list[str] = [""] * n

        def _do(i: int) -> None:
            barrier.wait()
            t0 = time.monotonic()
            try:
                self._cross.abort(errored=False)
            except Exception as e:  # noqa: BLE001 - record, do not mask the hang
                errs[i] = f"{type(e).__name__}: {str(e)[:120]}"
            results[i] = round(time.monotonic() - t0, 2)

        threads = [threading.Thread(target=_do, args=(i,)) for i in range(n)]
        start = time.monotonic()
        for th in threads:
            th.start()
        # Join with a budget; if it deadlocks the joins never complete (ray.get times out).
        for th in threads:
            th.join(timeout=_VERDICT_S)
        alive = [i for i, th in enumerate(threads) if th.is_alive()]
        return {
            "name": self._name,
            "total_s": round(time.monotonic() - start, 2),
            "per_thread_s": results,
            "threads_still_alive": alive,
            "errs": [e for e in errs if e],
        }

    def ping(self) -> str:
        return f"{self._name} alive at {time.monotonic():.1f}"


def _try_ping(worker: object) -> str:
    try:
        return str(ray.get(worker.ping.remote(), timeout=10))
    except Exception as e:  # noqa: BLE001
        return f"ping failed: {type(e).__name__}"


def _spawn(name: str) -> object:
    w = _Worker.options(runtime_env={"env_vars": dict(_REAL_RUN_ENV)}).remote()
    ray.get(w.init.remote(name=name), timeout=60)
    return w


def _run(exp: str, *, store_base: str, timeout_s: float, numel: int, k: int) -> str:
    print(f"== experiment {exp} ==")
    surv = _spawn("S")
    peer = _spawn("W")
    try:
        if exp == "many_comm":
            base = f"{store_base}/{exp}"
            ray.get(
                [
                    surv.build_many_pgs.remote(store_base=base, rank=0, timeout_s=timeout_s, k=k),
                    peer.build_many_pgs.remote(store_base=base, rank=1, timeout_s=timeout_s, k=k),
                ],
                timeout=240,
            )
            print(f"  {k} PGs built + warmed on each side")
            peer.wedge_sleep.remote()
            time.sleep(_SETTLE_S)
            ray.get(surv.enqueue_inflight_many.remote(numel=numel), timeout=60)
            print(f"  peer wedged + survivor posted in-flight on all {k} PGs")
            ref = surv.abort_many_concurrent.remote()
            try:
                out = ray.get(ref, timeout=_VERDICT_S + 30)
                stuck = out.get("threads_still_alive")
                return f"{'DEADLOCK (REPRODUCED)' if stuck else 'no deadlock'}: {out}"
            except ray.exceptions.GetTimeoutError:
                alive = _try_ping(surv)
                return f"*** many_comm abort DEADLOCK >{_VERDICT_S + 30}s (REPRODUCED) *** ping={alive}"

        cross_store = f"{store_base}/{exp}/cross"
        ray.get(
            [
                surv.build_cross_pg.remote(store_addr=cross_store, rank=0, timeout_s=timeout_s),
                peer.build_cross_pg.remote(store_addr=cross_store, rank=1, timeout_s=timeout_s),
            ],
            timeout=120,
        )
        print("  cross pair built + warmed")

        if exp != "double_abort_clean":
            # Make the comm non-trivial to tear down: wedge peer + in-flight collective.
            peer.wedge_sleep.remote()
            time.sleep(_SETTLE_S)
            ray.get(surv.enqueue_inflight_cross.remote(numel=numel), timeout=30)
            print("  peer wedged + survivor in-flight cross allreduce posted")

        if exp == "single_abort":
            ref = surv.abort_once.remote()
        elif exp == "raw_double_abort":
            ref = surv.abort_concurrent_raw.remote(n=2)
        elif exp == "raw_quad_abort":
            ref = surv.abort_concurrent_raw.remote(n=4)
        else:
            ref = surv.abort_concurrent.remote(n=2)

        try:
            out = ray.get(ref, timeout=_VERDICT_S + 30)
            stuck = out.get("threads_still_alive") if isinstance(out, dict) else None
            verdict = "DEADLOCK (REPRODUCED)" if stuck else "no deadlock"
            return f"{verdict}: {out}"
        except ray.exceptions.GetTimeoutError:
            alive = None
            try:
                alive = ray.get(surv.ping.remote(), timeout=10)
            except Exception as e:  # noqa: BLE001
                alive = f"ping failed: {type(e).__name__}"
            return f"*** abort DEADLOCK >{_VERDICT_S + 30}s (REPRODUCED) *** survivor_ping={alive}"
    finally:
        for w in [surv, peer]:
            try:
                ray.kill(w, no_restart=True)
            except Exception:  # noqa: BLE001 - idempotent cleanup
                pass


def main(
    timeout_s: Annotated[float, typer.Option(help="torchft cross PG timeout")] = 120.0,
    numel: Annotated[int, typer.Option(help="in-flight cross allreduce size")] = 1 << 22,
    k: Annotated[int, typer.Option(help="number of co-resident PGs for many_comm")] = 16,
    only: Annotated[str | None, typer.Option(help="run a single experiment")] = None,
) -> None:
    """Run the concurrent-abort deadlock reproduction matrix."""
    ray.init(ignore_reinit_error=True)
    from torch.distributed import TCPStore

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    store_base = f"localhost:{store.port}/concurrent_abort"

    experiments = ["single_abort", "raw_double_abort", "raw_quad_abort", "double_abort", "many_comm"]
    if only is not None:
        experiments = [only]

    results: dict[str, str] = {}
    for exp in experiments:
        results[exp] = _run(exp, store_base=store_base, timeout_s=timeout_s, numel=numel, k=k)
        print(f"RESULT {exp}: {results[exp]}\n")
        time.sleep(3)

    print("==== SUMMARY ====")
    for exp, res in results.items():
        print(f"RESULT {exp}: {res}")
    del store


if __name__ == "__main__":
    typer.run(main)
