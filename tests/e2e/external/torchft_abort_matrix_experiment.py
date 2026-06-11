"""torchft PG abort() behavior matrix: under which peer states does abort hang?

NOT a CI test. Minimal reproduction of the r2 reconfigure hang (see
agent-context 2026-06-07-indep-dp-pg-graveyard-explainer) plus boundary
conditions, to answer: why do torchft/torchtitan deployments not hit the hang
miles hit, and which teardown path is actually safe.

Topology (4 GPUs, mirrors two miles cells of 2 ranks each):
    cell A = {a0, a1}, cell B = {b0, b1}
    per-cell torchft NCCL PG:    (a0,a1) and (b0,b1)
    cross-cell torchft NCCL+Gloo pair PGs: (a0,b0) and (a1,b1)
All PGs are warmed up with one allreduce, then left idle.

Experiments (each on a fresh actor set):
    torchft_form  kill a0 AND a1 (the whole peer cell dies, mirroring the
                  replica-granularity failures torchft deployments produce)
                  -> b1 aborts its cross pair. Expect: fast.
    r2_form       kill a0 only; a1 stays alive but wedged (sleeping, comms
                  idle) -> b1 aborts its cross pair. Expect (r2): hang; then
                  kill a1 and expect the abort to unblock.
    locate_nccl   r2_form but b1 aborts ONLY the cross NCCL PG.
    locate_gloo   r2_form but b1 aborts ONLY the cross Gloo PG.
    gpu_spin      r2_form but a1 spins a GPU kernel while wedged (closer to
                  the real wedged-in-collective state).

Usage:
    python tests/e2e/external/torchft_abort_matrix_experiment.py
    python tests/e2e/external/torchft_abort_matrix_experiment.py --only r2_form
"""

import logging
import os
import time
from datetime import timedelta
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)

_HANG_VERDICT_TIMEOUT_S = 45.0
_SETTLE_S = 3.0


@ray.remote(num_gpus=1, max_concurrency=2)
class _MatrixWorker:
    """One rank: holds a cell-internal PG and a cross-cell pair PG (NCCL + Gloo).

    max_concurrency=2 so a liveness ``ping`` can run while another call blocks
    (wedge methods still wedge: they simply never return)."""

    def setup(self, *, store_base: str, name: str, timeout_s: float) -> dict:
        import torch
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL

        self._name = name
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")

        cell = name[0]  # "a" / "b"
        cell_rank = int(name[1])  # 0 / 1
        pair_idx = cell_rank  # pair0 = (a0,b0), pair1 = (a1,b1)
        pair_rank = 0 if cell == "a" else 1

        def _mk(pg_cls: type, store_suffix: str, rank: int) -> object:
            pg = pg_cls(timeout=timedelta(seconds=timeout_s))
            pg.configure(
                store_addr=f"{store_base}/{store_suffix}",
                replica_id=name,
                rank=rank,
                world_size=2,
                quorum_id=0,
            )
            return pg

        self._cell_pg = _mk(ProcessGroupNCCL, f"cell_{cell}", cell_rank)
        self._cross_pg = _mk(ProcessGroupNCCL, f"pair{pair_idx}_nccl", pair_rank)
        self._cross_gloo = _mk(ProcessGroupGloo, f"pair{pair_idx}_gloo", pair_rank)

        # Warm every comm up with one allreduce, then leave it idle (mirrors a
        # quorum that worked in the previous step and is torn down later).
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        for pg, dev in [(self._cell_pg, self._device), (self._cross_pg, self._device), (self._cross_gloo, "cpu")]:
            t = torch.ones(8, device=dev)
            assert pg.allreduce([t], opts).wait(), f"{name}: warmup failed"

        return {"name": name, "nccl": torch.cuda.nccl.version()}

    def die(self) -> None:
        os._exit(1)

    def wedge_sleep(self) -> None:
        """Stay alive but never respond (idle comms, blocked main thread)."""
        time.sleep(1_000_000)

    def wedge_gpu_spin(self) -> None:
        """Stay alive with a busy GPU (closer to wedged-in-collective)."""
        import torch

        x = torch.randn(4096, 4096, device=self._device)
        while True:
            x = x @ x
            x = x / x.norm()

    def build_native_cell_pg(self, *, store_addr: str, prefix: str, long_timeout_s: float = 600.0) -> dict:
        """Build a RAW c10d NCCL PG for the cell pair (no torchft wrapper, no
        userspace timeout) — mirrors Megatron's cell-internal comms in r2."""
        import torch
        import torch.distributed as dist
        from torch.distributed import ProcessGroupNCCL as BaseProcessGroupNCCL
        from torch.distributed import TCPStore

        cell_rank = int(self._name[1])
        host, port = store_addr.split(":")
        store = TCPStore(host_name=host, port=int(port), is_master=False)
        # Prefix must be unique per experiment: reusing one collides with the
        # rendezvous records of a previous run's dead ranks (connection refused).
        prefixed = dist.PrefixStore(f"{prefix}/native_cell_{self._name[0]}", store)

        opts = BaseProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=long_timeout_s)
        self._native_cell = BaseProcessGroupNCCL(prefixed, cell_rank, 2, opts)

        t = torch.ones(8, device=self._device)
        self._native_cell.allreduce([t]).wait()
        return {"name": self._name, "native_warmup": float(t[0].item())}

    def wedge_native_collective(self) -> None:
        """Block forever inside a real NCCL collective whose peer never joins
        (the peer is dead): a true wedged-in-collective state with a spinning
        NCCL kernel, like r2's cell0 survivors."""
        import torch

        t = torch.ones(1024, device=self._device)
        work = self._native_cell.allreduce([t])
        work.wait()  # peer never participates; raw c10d with long timeout -> wedged

    def start_inflight_cross_allreduce(self) -> dict:
        """Enqueue a cross-pair allreduce WITHOUT waiting: the NCCL kernel sits
        in-flight waiting for a peer that never joins. No wait() means torchft's
        userspace timeout is never armed."""
        import torch
        import torch.distributed as dist

        t = torch.ones(1 << 20, device=self._device)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        self._inflight_work = self._cross_pg.allreduce([t], opts)
        self._inflight_tensor = t
        return {"name": self._name, "inflight": True}

    def wait_inflight_until_timeout(self) -> dict:
        """wait() the in-flight work so torchft's userspace timeout fires and
        aborts/errors the comm (the idiomatic detection path)."""
        start = time.monotonic()
        ok = None
        err = None
        try:
            ok = self._inflight_work.wait()
        except Exception as e:  # noqa: BLE001 - the expected timeout/abort error
            err = f"{type(e).__name__}: {e}"
        return {
            "name": self._name,
            "ok": ok,
            "err": (err or "")[:200],
            "elapsed_s": round(time.monotonic() - start, 2),
            "errored": str(self._cross_pg.errored())[:120],
        }

    def blocked_wait_min1(self) -> dict:
        """Exact r19 shape: enqueue a 1-element MIN allreduce on the cross pair
        (collective_bool_and's exact op) whose peer never joins, then BLOCK in
        wait(). torchft's _WorkAcceleratorTimeout fires abort() from its timer
        thread while this thread is parked inside wait() — the r19 logs show
        that abort's backing future timing out after 600s."""
        import torch
        import torch.distributed as dist

        t = torch.tensor([1.0], dtype=torch.float32, device=self._device)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.MIN

        errored_before = str(self._cross_pg.errored())[:80]
        start = time.monotonic()
        work = self._cross_pg.allreduce([t], opts)
        enqueue_s = round(time.monotonic() - start, 3)
        ok = None
        err = None
        try:
            ok = work.wait()
        except Exception as e:  # noqa: BLE001 - the expected abort/timeout error
            err = f"{type(e).__name__}: {str(e)[:200]}"
        return {
            "name": self._name,
            "work_type": type(work).__name__,
            "errored_before": errored_before,
            "enqueue_s": enqueue_s,
            "wait_s": round(time.monotonic() - start, 2),
            "ok": ok,
            "err": err,
            "errored_after": str(self._cross_pg.errored())[:80],
        }

    def blocked_item_min1(self) -> dict:
        """Exact collective_bool_and shape: 1-element MIN allreduce + tensor.item().

        torchft's wait() is non-blocking (stream-ordered); the D2H sync in
        .item() is where the MAIN THREAD parks inside cudaStreamSynchronize
        waiting for the never-completing kernel. The userspace timer then calls
        pg.abort() CONCURRENTLY with that synchronize — the suspected r19
        deadlock pair."""
        import torch
        import torch.distributed as dist

        t = torch.tensor([1.0], dtype=torch.float32, device=self._device)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.MIN

        start = time.monotonic()
        work = self._cross_pg.allreduce([t], opts)
        wait_ok = work.wait()
        wait_s = round(time.monotonic() - start, 3)
        value = None
        err = None
        try:
            value = t.item()  # blocks in D2H sync until kernel completes or abort
        except Exception as e:  # noqa: BLE001 - the expected abort error
            err = f"{type(e).__name__}: {str(e)[:200]}"
        item_s = round(time.monotonic() - start, 2)

        errored_start = time.monotonic()
        errored_after = str(self._cross_pg.errored())[:80]
        return {
            "name": self._name,
            "wait_ok": wait_ok,
            "wait_s": wait_s,
            "item_s": item_s,
            "value": value,
            "err": err,
            "errored_after": errored_after,
            "errored_probe_s": round(time.monotonic() - errored_start, 2),
        }

    def ping(self) -> str:
        """Liveness probe; runs on a second actor thread while another call blocks."""
        return f"{self._name} alive at {time.monotonic():.1f}"

    def cuda_probe(self) -> dict:
        """GPU-side probe on a second actor thread: time torch.cuda.synchronize().

        Separates host-blocked from GPU-blocked: if the device/stream is occupied
        by a never-completing kernel, this blocks too; if only a host thread is
        stuck (lock/GIL/socket), this returns immediately."""
        import torch

        start = time.monotonic()
        torch.cuda.synchronize()
        return {"name": self._name, "synchronize_s": round(time.monotonic() - start, 3)}

    def abort_cross(self, *, target: str) -> dict:
        """Abort this rank's cross-cell PG(s) like reconfigure_indep_dp_group does; time it."""
        groups = {
            "nccl": [self._cross_pg],
            "gloo": [self._cross_gloo],
            "both": [self._cross_pg, self._cross_gloo],
        }[target]

        timings: list[tuple[str, float]] = []
        start_all = time.monotonic()
        for pg in groups:
            start = time.monotonic()
            pg.abort(errored=False)
            timings.append((type(pg).__name__, round(time.monotonic() - start, 3)))
        return {
            "name": self._name,
            "target": target,
            "total_s": round(time.monotonic() - start_all, 3),
            "per_pg_s": timings,
        }


def _spawn_cells(*, store_base: str, timeout_s: float, env_vars: dict | None = None) -> dict:
    actor_cls = _MatrixWorker.options(runtime_env={"env_vars": env_vars}) if env_vars else _MatrixWorker
    workers = {name: actor_cls.remote() for name in ("a0", "a1", "b0", "b1")}
    infos = ray.get(
        [w.setup.remote(store_base=store_base, name=name, timeout_s=timeout_s) for name, w in workers.items()],
        timeout=120,
    )
    print(f"  setup: {infos}")
    return workers


def _fire_and_forget(ref: object) -> None:
    del ref


def _kill_all(workers: dict) -> None:
    for w in workers.values():
        try:
            ray.kill(w, no_restart=True)
        except Exception:  # noqa: BLE001 - idempotent cleanup
            pass


def _await_abort(ref: object, *, workers: dict, expect: str) -> str:
    """Wait for the abort call; on hang, kill the wedged peer and wait again."""
    try:
        out = ray.get(ref, timeout=_HANG_VERDICT_TIMEOUT_S)
        return f"returned in {out['total_s']}s per_pg={out['per_pg_s']} (expect={expect})"
    except ray.exceptions.GetTimeoutError:
        pass

    print(f"  abort STUCK for >{_HANG_VERDICT_TIMEOUT_S}s -> killing wedged peer a1 to test unblock")
    ray.kill(workers["a1"], no_restart=True)
    try:
        out = ray.get(ref, timeout=_HANG_VERDICT_TIMEOUT_S)
        return f"HUNG then UNBLOCKED by peer kill: {out['total_s']}s total (expect={expect})"
    except ray.exceptions.GetTimeoutError:
        return f"HUNG and STILL STUCK {_HANG_VERDICT_TIMEOUT_S}s after peer kill (expect={expect})"


def _run_experiment(exp: str, *, store_base: str, store_hostport: str, timeout_s: float) -> str:
    print(f"== experiment {exp} ==")
    env_vars = {"CUDA_DEVICE_MAX_CONNECTIONS": "1"} if exp.endswith("_conn1") else None
    workers = _spawn_cells(store_base=f"{store_base}/{exp}", timeout_s=timeout_s, env_vars=env_vars)

    try:
        if exp == "torchft_form":
            _fire_and_forget(workers["a0"].die.remote())
            _fire_and_forget(workers["a1"].die.remote())
            time.sleep(_SETTLE_S)
            ref = workers["b1"].abort_cross.remote(target="both")
            return _await_abort(ref, workers=workers, expect="fast")

        if exp.startswith("blocked_item"):
            # collective_bool_and shape: main thread parks in .item()'s D2H sync
            # while the userspace timer aborts concurrently.
            if "native" in exp:
                ray.get(
                    [
                        workers[n].build_native_cell_pg.remote(store_addr=store_hostport, prefix=exp)
                        for n in ("a0", "a1")
                    ],
                    timeout=60,
                )
                _fire_and_forget(workers["a1"].wedge_native_collective.remote())
            else:
                _fire_and_forget(workers["a1"].wedge_sleep.remote())
            time.sleep(_SETTLE_S)
            _fire_and_forget(workers["a0"].die.remote())
            time.sleep(_SETTLE_S)

            ref = workers["b1"].blocked_item_min1.remote()
            time.sleep(5)  # while .item() is parked, probe the GPU from the second thread
            probe_ref = workers["b1"].cuda_probe.remote()
            try:
                out = ray.get(ref, timeout=timeout_s + _HANG_VERDICT_TIMEOUT_S)
                probe = ray.get(probe_ref, timeout=_HANG_VERDICT_TIMEOUT_S)
                return (
                    f"item returned: {out}; gpu probe during block: {probe} "
                    f"(expect: both unblocked ~{timeout_s}s by timer abort, or HANG like r19)"
                )
            except ray.exceptions.GetTimeoutError:
                probe = ray.get(workers["b1"].ping.remote(), timeout=10)
                print(f"  blocked_item STUCK; probe: {probe}; killing wedged a1")
                ray.kill(workers["a1"], no_restart=True)
                try:
                    out = ray.get(ref, timeout=_HANG_VERDICT_TIMEOUT_S)
                    return f"HUNG then UNBLOCKED by peer kill: {out}"
                except ray.exceptions.GetTimeoutError:
                    return f"HUNG and STILL STUCK {_HANG_VERDICT_TIMEOUT_S}s after peer kill"

        if exp.startswith("blocked_wait"):
            # Exact r19 shape: peer wedged, then b1 BLOCKS in wait() on a
            # 1-element MIN allreduce; torchft's timer thread aborts concurrently.
            if "native" in exp:
                ray.get(
                    [
                        workers[n].build_native_cell_pg.remote(store_addr=store_hostport, prefix=exp)
                        for n in ("a0", "a1")
                    ],
                    timeout=60,
                )
                _fire_and_forget(workers["a1"].wedge_native_collective.remote())
            else:
                _fire_and_forget(workers["a1"].wedge_sleep.remote())
            time.sleep(_SETTLE_S)
            _fire_and_forget(workers["a0"].die.remote())
            time.sleep(_SETTLE_S)

            ref = workers["b1"].blocked_wait_min1.remote()
            try:
                out = ray.get(ref, timeout=timeout_s + _HANG_VERDICT_TIMEOUT_S)
                return f"wait returned: {out} (expect: unblocked ~{timeout_s}s by timer abort)"
            except ray.exceptions.GetTimeoutError:
                probe = ray.get(workers["b1"].ping.remote(), timeout=10)
                print(f"  blocked_wait STUCK; probe: {probe}; killing wedged a1")
                ray.kill(workers["a1"], no_restart=True)
                try:
                    out = ray.get(ref, timeout=_HANG_VERDICT_TIMEOUT_S)
                    return f"HUNG then UNBLOCKED by peer kill: {out}"
                except ray.exceptions.GetTimeoutError:
                    return f"HUNG and STILL STUCK {_HANG_VERDICT_TIMEOUT_S}s after peer kill"

        if exp.startswith("native_wedge"):
            # Cell A gets a RAW c10d NCCL PG; a1 wedges inside a real collective.
            ray.get(
                [workers[n].build_native_cell_pg.remote(store_addr=store_hostport, prefix=exp) for n in ("a0", "a1")],
                timeout=60,
            )
            _fire_and_forget(workers["a1"].wedge_native_collective.remote())
            time.sleep(_SETTLE_S)
            _fire_and_forget(workers["a0"].die.remote())
            time.sleep(_SETTLE_S)
            if "inflight" in exp:
                # b1's cross allreduce kernel sits in-flight waiting for wedged a1.
                _fire_and_forget(workers["b1"].start_inflight_cross_allreduce.remote())
                time.sleep(_SETTLE_S)
            ref = workers["b1"].abort_cross.remote(target="both")
            return _await_abort(ref, workers=workers, expect="hang? (closest to r2)")

        if exp.startswith("inflight"):
            _fire_and_forget(workers["a1"].wedge_sleep.remote())
            time.sleep(_SETTLE_S)
            _fire_and_forget(workers["a0"].die.remote())
            time.sleep(_SETTLE_S)
            _fire_and_forget(workers["b1"].start_inflight_cross_allreduce.remote())
            time.sleep(_SETTLE_S)
            if exp == "inflight_post_timeout":
                detect = ray.get(workers["b1"].wait_inflight_until_timeout.remote(), timeout=timeout_s + 60)
                print(f"  post-timeout detect: {detect}")
            ref = workers["b1"].abort_cross.remote(target="both")
            return _await_abort(ref, workers=workers, expect="hang? (in-flight work variable)")

        wedge_method = "wedge_gpu_spin" if exp == "gpu_spin" else "wedge_sleep"
        target = {"locate_nccl": "nccl", "locate_gloo": "gloo"}.get(exp, "both")

        _fire_and_forget(getattr(workers["a1"], wedge_method).remote())
        time.sleep(_SETTLE_S)  # a1 occupies its single actor thread = wedged
        _fire_and_forget(workers["a0"].die.remote())
        time.sleep(_SETTLE_S)
        ref = workers["b1"].abort_cross.remote(target=target)
        return _await_abort(ref, workers=workers, expect="hang? (r2 says yes for some target)")
    finally:
        _kill_all(workers)


def main(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout")] = 20.0,
    only: Annotated[str | None, typer.Option(help="run a single experiment by name")] = None,
) -> None:
    """Run the abort-behavior matrix; prints one RESULT line per experiment."""
    ray.init(ignore_reinit_error=True)

    from torch.distributed import TCPStore

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    store_hostport = f"localhost:{store.port}"
    store_base = f"{store_hostport}/abort_matrix"

    experiments = [
        "blocked_item",
        "blocked_item_conn1",
        "blocked_item_native",
        "blocked_item_native_conn1",
    ]
    if only is not None:
        experiments = [only]

    results: dict[str, str] = {}
    for exp in experiments:
        results[exp] = _run_experiment(exp, store_base=store_base, store_hostport=store_hostport, timeout_s=timeout_s)
        print(f"RESULT {exp}: {results[exp]}\n")
        time.sleep(2)

    print("==== SUMMARY ====")
    for exp, res in results.items():
        print(f"RESULT {exp}: {res}")

    del store


if __name__ == "__main__":
    typer.run(main)
