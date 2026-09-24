"""Portable LoRA weight transfer: stage the adapter, then tell engines to load it.

Miles has two transports that declare ``supports_lora``: ``broadcast`` (an NCCL
collective spanning trainer ranks and engines) and ``cuda_ipc`` (same host only).
Both are bound to one vendor, and cuda_ipc to one host, so neither can carry an
adapter from a trainer on NVIDIA GPUs to engines on AMD GPUs -- NCCL and RCCL do
not interoperate. The transports that DO work across hosts, ``p2p`` and
``disk-delta``, both ``assert lora_rank <= 0``.

Two engine routes carry an adapter over plain HTTP, and this transport speaks both
(``--http-lora-ship``):

* ``path`` (default): rank 0 writes the adapter as a versioned PEFT directory under
  --update-weight-disk-dir and POSTs the path to ``/load_lora_adapter``. The
  directory must be visible to the engine hosts (a shared filesystem), or
  --custom-update-weight-post-write-path copies it there, as with disk-delta.
  Versioned files on disk are what make a served adapter auditable afterwards.
* ``tensors``: rank 0 POSTs the weights themselves to
  ``/load_lora_adapter_from_tensors``, so the engines need nothing but an HTTP
  port. The payload is a plain pickle of ``{name: cpu tensor}`` -- NOT SGLang's
  ``MultiprocessingSerializer``, which ships shared-memory handles that only
  resolve on the same host -- base64-encoded once per TP rank, as the request
  type expects. About 1.3x the adapter size per rank.

Either way only HTTP crosses the boundary, so the trainer and the engines need
not share a vendor, an interconnect, or a host. Only sane for LoRA: a
full-weight sync this way would move the whole model every step.

``use_weight_update_session = False`` because this protocol owns registration and
reload itself; it does not need the session frame's pause/resume around a
collective that is not happening.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import os
import pickle
import shutil
import time
from argparse import Namespace
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.session import pause_engines, resume_engines, set_weight_version

logger = logging.getLogger(__name__)

# Versioned staging dirs are written every sync; keep a few so an engine can still
# be reading the previous one, and no more.
KEEP_VERSIONS = 3
# A load is a file read on the engine host. 1800s meant one wedged engine stalled
# training for half an hour before anyone noticed.
LOAD_TIMEOUT_S = 300.0


def _pickle_with_torch_globals(obj) -> bytes:
    """``pickle.dumps`` that names torch's own storage loader in the stream.

    A tensor pickles its storage as ``(torch.storage._load_from_bytes, (bytes,))``,
    looked up in ``torch.storage`` at pickling time. Megatron replaces that module
    attribute with ``megatron.core.safe_globals.safe_load_from_bytes``, so anything
    pickled inside a trainer process references a Megatron function -- and the
    engine's allowlisting unpickler (CVE-2025-10164) refuses it, fatally, in the
    scheduler. The engine runs stock torch, so emit the stock name: put a function
    with torch's module/qualname in the slot for the duration of the dump.
    """
    import torch.storage as ts

    current = ts._load_from_bytes
    if getattr(current, "__module__", "") == "torch.storage":
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_from_bytes(b):  # only the name is used here; the engine runs its own
        return torch.load(io.BytesIO(b))

    _load_from_bytes.__module__, _load_from_bytes.__qualname__ = "torch.storage", "_load_from_bytes"
    ts._load_from_bytes = _load_from_bytes
    try:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        ts._load_from_bytes = current


class UpdateWeightHttpLora(WeightTransferProtocol):
    """Stage the adapter to a directory, then POST its path to every engine."""

    # gather_pp so one rank holds the whole adapter; tp/ep are always gathered.
    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True
    # Engines already hold the base; a resync request here is a misconfiguration,
    # not something to silently do at 70 GB per step.
    needs_base_resync_for_lora = False
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._buf: dict[str, torch.Tensor] = {}
        self._ship: str = getattr(args, "http_lora_ship", "path")
        # 'path' is validated to have a staging dir; 'tensors' keeps versioned
        # copies on disk only if one was given.
        self._stage_root: str | None = getattr(args, "update_weight_disk_dir", None) or None
        assert (
            self._ship == "tensors" or self._stage_root
        ), "http-lora --http-lora-ship path needs --update-weight-disk-dir"
        self._load_route: str = "/load_lora_adapter" if self._ship == "path" else "/load_lora_adapter_from_tensors"
        # The tensor route wants one serialized copy per TP rank of the engine.
        self._engine_tp: int = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
        # Per-slot rank/alpha are not visible to a transport (send_bucket carries
        # only "{lora_name}:{hf_key}"), so a multi-LoRA run would be published with
        # the wrong config. Refuse rather than mislabel it.
        from miles.utils.multi_lora import is_multi_lora_enabled

        assert not is_multi_lora_enabled(args), (
            "http-lora does not support multi-LoRA yet: per-adapter rank/alpha are not "
            "available to a weight-transfer protocol, so adapters would be published "
            "with the global --lora-rank/--lora-alpha."
        )
        # Probed conclusively on the first sync and remembered; it decides whether
        # the swap may run under a pause. None = not yet known.
        self._upsert_supported: bool | None = None
        self._keep_versions: int = KEEP_VERSIONS
        self._load_timeout_s: float = LOAD_TIMEOUT_S
        self._post_write_hook: Callable | None = None
        if getattr(args, "custom_update_weight_post_write_path", None):
            from miles.utils.function_registry import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    # ------------------------------------------------------------------ setup

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        self.rollout_engines = rollout_engines
        assert placement.gather_pp, "http-lora needs the full adapter on one rank"
        # One sender: the adapter is fully gathered, so fanning the upload across
        # ranks would write the same bytes N times.
        self.is_sender = dist.get_rank() == 0 if dist.is_initialized() else True
        self.group_name = "miles-http-lora"
        self._buf.clear()

    # ------------------------------------------------------------------ stream

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        """Buffer adapter tensors. Names are ``{lora_name}:{hf_key}``; anything
        else is a base weight and is dropped loudly rather than written into an
        adapter file where it would be nonsense."""
        for name, tensor in bucket:
            if ":" not in name:
                logger.warning("http-lora: dropping non-adapter tensor %r", name)
                continue
            self._buf[name] = tensor.detach().to("cpu", copy=True)

    # --------------------------------------------------------------- finalize

    def finalize(self, weight_version: int) -> None:
        if not self.is_sender:
            return
        if not self._buf:
            logger.warning("http-lora: nothing buffered at finalize; skipping publish")
            return

        by_adapter: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in self._buf.items():
            lora_name, hf_key = name.split(":", 1)
            by_adapter.setdefault(lora_name, {})[hf_key] = tensor
        self._buf.clear()

        # Pause only when nothing in the swap can wait on in-flight requests.
        # With upsert the engine replaces weights in place -- no unload, nothing to
        # wait for -- so pausing is safe, and it is what restores the cache flush
        # (the radix cache is keyed by adapter id, which upsert keeps) and labels
        # every sample with the version it was actually generated under. The
        # first sync is also safe: the name is not registered yet, so there is
        # nothing to unload either way. The only case that must NOT pause is the
        # unload fallback, which waits for running requests to finish; a paused
        # engine under retract/in_place never finishes them.
        engines = list(self.rollout_engines or [])
        first_sync = self._upsert_supported is None
        paused = False
        try:
            if first_sync or bool(self._upsert_supported):
                # Flag first, then pause: if the pause or its cache flush fails
                # part-way (stock SGLang refuses to flush while retracted requests
                # are queued), the fleet may already be paused on some engines and
                # must still be resumed below. Nothing here can wait on in-flight
                # requests, so pausing is safe in both branches.
                paused = True
                pause_engines(self.args, engines)
            for lora_name, tensors in by_adapter.items():
                t0 = time.time()
                local_dir = self._write_adapter(lora_name, weight_version, tensors) if self._stage_root else None
                nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
                t_write = time.time() - t0

                t0 = time.time()
                if self._ship == "path":
                    # Same escape hatch disk-delta uses: when the staging dir is not
                    # genuinely shared with the engine hosts, the operator supplies
                    # --custom-update-weight-post-write-path to replicate it.
                    if self._post_write_hook is not None:
                        self._post_write_hook(self.args, local_dir, engines)
                    load_body = {"lora_name": lora_name, "lora_path": local_dir, "pinned": False}
                else:
                    load_body = self._tensor_body(lora_name, tensors)
                t_ship = time.time() - t0

                t0 = time.time()
                self._load_everywhere(lora_name, load_body)
                t_load = time.time() - t0

                if self._stage_root:
                    self._prune_old_versions(lora_name, weight_version)

                # Namespaced per adapter: a shared key set would leave only the last
                # adapter's numbers on the step log.
                self.update_weight_metrics.update(
                    {
                        f"http_lora/{lora_name}/bytes": float(nbytes),
                        f"http_lora/{lora_name}/write_s": t_write,
                        f"http_lora/{lora_name}/ship_s": t_ship,
                        f"http_lora/{lora_name}/load_s": t_load,
                    }
                )
                logger.info(
                    "http-lora: %s v%d  %.0f MB  write %.1fs  ship %.1fs  load %.1fs  -> %d engine(s)",
                    lora_name,
                    weight_version,
                    nbytes / 1e6,
                    t_write,
                    t_ship,
                    t_load,
                    len(engines),
                )

            # Without this the engines report a stale version forever and any staleness
            # or off-policy-lag accounting downstream reads the wrong number.
            set_weight_version(engines, weight_version)
        finally:
            if paused:
                resume_engines(engines)

    # ---------------------------------------------------------------- helpers

    def _write_adapter(self, lora_name: str, version: int, tensors: dict[str, torch.Tensor]) -> str:
        from safetensors.torch import save_file

        # Versioned dir: an engine may still be reading the previous one, and
        # overwriting in place is how a half-written adapter gets loaded.
        d = os.path.join(self._stage_root, f"{lora_name}_v{version}")
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, ".adapter_model.safetensors.tmp")
        save_file(tensors, tmp, metadata={"format": "pt"})
        os.replace(tmp, os.path.join(d, "adapter_model.safetensors"))  # atomic publish

        cfg = self._adapter_config(tensors)
        cfg_tmp = os.path.join(d, ".adapter_config.json.tmp")
        with open(cfg_tmp, "w") as fh:
            json.dump(cfg, fh, indent=2)
        os.replace(cfg_tmp, os.path.join(d, "adapter_config.json"))  # atomic, like the weights

        # Consumed by a different process and often a different user (trainer as
        # root in a container, shipper not). safetensors writes 0600, so without
        # this the reader gets "Permission denied" on a file that is plainly there.
        os.chmod(d, 0o755)
        for f in os.listdir(d):
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(d, f), 0o644)
        return d

    def _tensor_body(self, lora_name: str, tensors: dict[str, torch.Tensor]) -> dict:
        """Request body for ``/load_lora_adapter_from_tensors``.

        A plain pickle carries the bytes; SGLang's own ``MultiprocessingSerializer``
        would carry a shared-memory handle no other host can open. The server
        unpickles a ``{name: tensor}`` dict and each TP rank reads entry
        ``[tp_rank]`` of the list, so every rank gets an identical copy.
        """
        blob = base64.b64encode(_pickle_with_torch_globals({k: v.contiguous() for k, v in tensors.items()})).decode(
            "ascii"
        )
        return {
            "lora_name": lora_name,
            "config_dict": self._adapter_config(tensors),
            "serialized_named_tensors": [blob] * self._engine_tp,
            "pinned": False,
        }

    @staticmethod
    def _infer_rank(tensors: dict[str, torch.Tensor]) -> int | None:
        """lora_A is [r, in], so the rank is readable off the weights."""
        for k, v in tensors.items():
            if k.endswith(".lora_A.weight") and v.dim() >= 2:
                return int(v.shape[-2])
        return None

    def _adapter_config(self, tensors: dict[str, torch.Tensor]) -> dict:
        """PEFT-shaped config. Target modules come from the tensors we actually
        hold, not the training args: claiming a module whose weights are absent is
        how engines apply uninitialised slots."""
        args = self.args
        targets = sorted({k.split(".lora_")[0].rsplit(".", 1)[-1] for k in tensors if ".lora_" in k})
        # Short names match by name, so "down_proj"/"up_proj" also select the
        # fused MoE experts. Without this exclusion an adapter carrying no expert
        # weights is rejected for the missing tensors, reporting only expert
        # paths -- which reads as stale data and is not.
        has_expert_weights = any(".mlp.experts." in k for k in tensors)
        exclude = None if has_expert_weights else r".*mlp\.experts.*"
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            # Derived from the tensors actually present, for the same reason
            # target_modules is: the published config should describe the file,
            # not the run's flags.
            "r": self._infer_rank(tensors) or int(getattr(args, "lora_rank", 0)),
            "lora_alpha": float(getattr(args, "lora_alpha", 0)),
            "lora_dropout": float(getattr(args, "lora_dropout", 0.0) or 0.0),
            "bias": "none",
            "inference_mode": True,
            "target_modules": targets,
            "exclude_modules": exclude,
        }

    @staticmethod
    def _is_name_conflict(resp) -> bool:
        """The engine already holds this adapter name and did not replace it."""
        body = (resp.text or "").lower()
        return "already exists" in body or "already loaded" in body

    @staticmethod
    def _is_absent(resp) -> bool:
        """Nothing registered under this name (SGLang answers 400 'does not exist')."""
        return "does not exist" in (resp.text or "").lower()

    def _probe_upsert(self, client, base: str, lora_name: str, load_body: dict) -> bool:
        """First sync: install the adapter, then find out whether this engine can
        replace it in place.

        A server without ``upsert`` does not reject the field -- the request type
        is a dataclass and pydantic ignores unknown keys -- so the only conclusive
        test is a second load against the name we just registered: 200 means it
        replaced in place, "already exists" means it cannot.
        """
        plain = load_body
        r = client.post(base + self._load_route, json=plain)
        if r.status_code != 200 and not self._is_name_conflict(r):
            raise RuntimeError(f"http-lora: load failed on {base}: HTTP {r.status_code} {r.text[:200]}")
        stale = r.status_code != 200  # a leftover from a previous run holds the name

        r = client.post(base + self._load_route, json={**plain, "upsert": True})
        if r.status_code == 200:
            return True
        if not self._is_name_conflict(r):
            raise RuntimeError(f"http-lora: upsert probe failed on {base}: HTTP {r.status_code} {r.text[:200]}")
        if stale:
            # We are paused (first sync) and cannot unload without deadlocking.
            raise RuntimeError(
                f"http-lora: {lora_name!r} is already registered on {base} from an earlier run "
                "and this engine cannot replace it in place. Restart the engine, or use one "
                f"with upsert on {self._load_route}."
            )
        return False

    def _load_one(self, base: str, lora_name: str, load_body: dict) -> bool | None:
        """Install one version on one engine. Returns the probe result on the
        first sync, None afterwards.

        The updater hands a transport a STABLE adapter name every sync
        (``LORA_ADAPTER_NAME``, or ``slot_lora_name(slot)``), so every sync after
        the first names an adapter the engine already holds, and a plain load is
        rejected with "LoRA with name X already exists". Treating that as success
        is how a run silently serves the step-1 adapter forever.

        Upsert replaces in place: one call, no window, safe under a pause. No
        released SGLang exposes it on either HTTP load route yet; the probe finds out.

        Fallback for a stock engine: unload, then load. Only safe UNPAUSED, and
        only in synchronous RL. ``unload_lora_adapter`` waits for the adapter's
        usage counter to reach zero; the counter is released when a request
        FINISHES. Paused under retract/in_place they never finish (deadlock), and
        unpaused under --fully-async they may take longer than the timeout while
        the name is unregistered and every new request fails. In sync RL the
        window is empty: generation drained before the sync began.
        """
        import httpx

        plain = load_body
        with httpx.Client(timeout=self._load_timeout_s) as client:
            if self._upsert_supported is None:
                return self._probe_upsert(client, base, lora_name, load_body)

            if self._upsert_supported:
                r = client.post(base + self._load_route, json={**plain, "upsert": True})
                if r.status_code != 200:
                    raise RuntimeError(f"http-lora: upsert load failed on {base}: HTTP {r.status_code} {r.text[:200]}")
                return None

            u = client.post(base + "/unload_lora_adapter", json={"lora_name": lora_name})
            if u.status_code != 200 and not self._is_absent(u):
                raise RuntimeError(
                    f"http-lora: {lora_name!r} is registered on {base} and unload failed: "
                    f"HTTP {u.status_code} {u.text[:200]}"
                )
            r = client.post(base + self._load_route, json=plain)
            if r.status_code != 200:
                raise RuntimeError(
                    f"http-lora: reload after unload failed on {base}: HTTP {r.status_code} {r.text[:200]}"
                )
            return None

    def _load_everywhere(self, lora_name: str, load_body: dict) -> None:
        """Install the staged adapter on every engine, in parallel.

        On the first sync this also settles whether the fleet can replace in
        place, and refuses to continue under --fully-async if it cannot: the
        fallback there would leave the only adapter unregistered while it waits
        on rollouts that can outlast the timeout.
        """
        from concurrent.futures import ThreadPoolExecutor

        bases = [eng.server_url.rstrip("/") for eng in self.rollout_engines or []]
        if not bases:
            return
        first_sync = self._upsert_supported is None
        with ThreadPoolExecutor(max_workers=len(bases)) as pool:
            results = list(pool.map(lambda b: self._load_one(b, lora_name, load_body), bases))
        if not first_sync:
            return

        # Conservative across a mixed fleet: one engine that cannot upsert means
        # the pause is unsafe for the fleet, so treat the fleet as unable.
        self._upsert_supported = all(bool(x) for x in results)
        if self._upsert_supported:
            return
        logger.warning(
            "http-lora: engines cannot replace an adapter in place (no upsert on %s); "
            "later syncs will unload+reload unpaused.",
            self._load_route,
        )
        if getattr(self.args, "fully_async", False):
            raise RuntimeError(
                f"http-lora under --fully-async needs engines that support `upsert` on {self._load_route}. "
                "Without it the only safe replacement is unload+reload "
                "while unpaused, which leaves the adapter unregistered for as long as the "
                "slowest in-flight rollout takes to finish -- every request in that window "
                "fails, and the wait can exceed the load timeout. Run synchronous RL, or use "
                "an engine with upsert on that route."
            )

    def _prune_old_versions(self, lora_name: str, current: int) -> None:
        """Keep the few newest version dirs. Each sync writes a full adapter, so
        an unbounded run would otherwise fill the staging volume."""
        if self._keep_versions <= 0:
            return
        prefix = f"{lora_name}_v"
        try:
            versions = sorted(
                int(d[len(prefix) :])
                for d in os.listdir(self._stage_root)
                if d.startswith(prefix) and d[len(prefix) :].isdigit()
            )
        except OSError:
            return
        for v in versions[: max(0, len(versions) - self._keep_versions)]:
            if v == current:
                continue
            with contextlib.suppress(OSError):
                shutil.rmtree(os.path.join(self._stage_root, f"{prefix}{v}"))
