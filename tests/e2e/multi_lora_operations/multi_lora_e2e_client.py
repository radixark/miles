#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

import ray

API = "http://127.0.0.1:8068"
ROUTER = "http://127.0.0.1:20080"  # rebound to the head-node IP at startup
NAME = "e2e_a"
SAVE_ROOT = "/personal/tinker_e2e/save"  # rebound from --save-root

PASS: list[str] = []
FAIL: list[str] = []


def report(phase: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    (PASS if ok else FAIL).append(phase)
    print(f"[{tag}] {phase}: {detail}", flush=True)
    if not ok:
        print("--- aborting on first failure ---", flush=True)
        sys.exit(1)


def http(method: str, path: str, body: dict | None = None, base: str = API) -> dict:
    req = urllib.request.Request(
        base + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def wait_state(name: str, want: str, timeout_s: float = 300) -> str:
    deadline = time.monotonic() + timeout_s
    state = None
    while time.monotonic() < deadline:
        state = http("GET", f"/adapter_runs/state?names={name}")["states"].get(name)
        if state == want:
            return state
        time.sleep(2)
    raise TimeoutError(f"adapter '{name}' state {state!r}, wanted {want!r} within {timeout_s}s")


class Ops:
    """Operation plane over the controller Ray actor."""

    def __init__(self):
        self.controller = ray.get_actor("miles_tinker_controller", namespace="miles")
        # Ordinals are consecutive from 1 PER REGISTRATION; a re-registered
        # name is a new tenant and restarts at 1 (reset_ordinals).
        self.ordinals: dict[str, int] = {}

    def enqueue(self, kind: str, payload: dict | None = None, name: str = NAME) -> str:
        ordinal = self.ordinals.get(name, 0) + 1
        self.ordinals[name] = ordinal
        op_id = f"op-{name}-{ordinal}-{kind}-{uuid.uuid4().hex[:8]}"
        view = ray.get(self.controller.enqueue_operation.remote(name, op_id, ordinal, kind, payload))
        assert view["state"] == "QUEUED", view
        return op_id

    def wait(self, op_id: str, timeout_s: float = 600) -> dict:
        deadline = time.monotonic() + timeout_s
        view = None
        while time.monotonic() < deadline:
            view = ray.get(self.controller.get_operation.remote(op_id))
            if view is not None and view["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return view
            time.sleep(1)
        raise TimeoutError(f"operation {op_id} not terminal within {timeout_s}s: {view}")

    def ack(self, op_id: str) -> None:
        ray.get(self.controller.ack_operation.remote(op_id))

    def run(self, kind: str, payload: dict | None = None, name: str = NAME, timeout_s: float = 600) -> dict:
        op_id = self.enqueue(kind, payload, name=name)
        view = self.wait(op_id, timeout_s)
        self.ack(op_id)
        return view

    def snapshot(self) -> dict:
        return ray.get(self.controller.snapshot.remote())

    def step_of(self, name: str) -> int:
        return ray.get(self.controller.adapter_step.remote(name))

    def reset_ordinals(self, name: str) -> None:
        self.ordinals.pop(name, None)


def fb_payload(sample_lens: list[tuple[int, int]], base_token: int = 2000) -> dict:
    """CE forward_backward payload: (total_len, response_len) per sample."""
    samples = []
    for i, (total, resp) in enumerate(sample_lens):
        tokens = [base_token + i * 100 + j for j in range(total)]
        samples.append(
            dict(
                tokens=tokens,
                response_length=resp,
                loss_mask=[1] * resp,
                loss_weights=[1.0 / resp] * resp,
            )
        )
    return dict(samples=samples, loss=dict(loss_fn="cross_entropy"))


def check_fb_result(view: dict, sample_lens: list[tuple[int, int]], phase: str) -> list[list[float]]:
    ok = view["state"] == "SUCCEEDED"
    detail = f"state={view['state']}"
    logprobs, loss = None, None
    if ok:
        result = view["result"] or {}
        logprobs = result.get("logprobs")
        metrics = result.get("metrics") or {}
        loss = metrics.get("loss:sum")
        shapes_ok = (
            isinstance(logprobs, list)
            and len(logprobs) == len(sample_lens)
            and all(len(lp) == resp for lp, (_, resp) in zip(logprobs, sample_lens, strict=True))
        )
        loss_ok = isinstance(loss, float) and math.isfinite(loss)
        ok = shapes_ok and loss_ok
        detail = (
            f"state=SUCCEEDED shapes={[len(lp) for lp in logprobs] if isinstance(logprobs, list) else None} "
            f"want={[r for _, r in sample_lens]} loss:sum={loss} "
            f"unmasked_tokens:sum={metrics.get('unmasked_tokens:sum')}"
        )
    else:
        detail += f" error={view.get('error')}"
    report(phase, ok, detail)
    return logprobs


def check_optim(view: dict, phase: str, expect_zero: bool = False) -> float | None:
    result = view.get("result") or {}
    grad_norm = result.get("grad_norm")
    finite = isinstance(grad_norm, float) and math.isfinite(grad_norm)
    ok = view["state"] == "SUCCEEDED" and finite and (grad_norm == 0.0 if expect_zero else grad_norm > 0)
    report(
        phase,
        ok,
        f"state={view['state']} grad_norm={grad_norm} lr={result.get('learning_rate')} error={view.get('error')}",
    )
    return grad_norm


def max_logprob_delta(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(x - y) for row_a, row_b in zip(a, b, strict=True) for x, y in zip(row_a, row_b, strict=True))


def register(ops: Ops, name: str, rank: int = 8) -> dict:
    """Register and wait READY; returns {slot, registration_id}. A fresh
    registration is a new tenant: its operation ordinals restart at 1."""
    ops.reset_ordinals(name)
    reg = http("POST", "/adapter_runs", {"name": name, "config": {"rank": rank}})
    wait_state(name, "READY", timeout_s=600)
    info = http("GET", f"/adapter_runs/{name}")
    return {"slot": reg.get("slot"), "registration_id": info["registration_id"]}


def deregister(ops: Ops, name: str, timeout_s: float = 300) -> str:
    http("DELETE", f"/adapter_runs/{name}")
    deadline = time.monotonic() + timeout_s
    state = None
    while time.monotonic() < deadline:
        state = http("GET", f"/adapter_runs/state?names={name}")["states"].get(name)
        if state == "COMPLETED":
            return state
        time.sleep(2)
    raise TimeoutError(f"adapter '{name}' not COMPLETED within {timeout_s}s (state={state})")


def sidecar_manifest(name: str) -> str:
    return f"{SAVE_ROOT}/adapters/{name}/slot_state/manifest.pt"


# Adapter lifecycle at DP=2.


def phase_a(ops: Ops) -> None:
    from miles.ray.multi_lora.identity import serving_lora_name  # noqa: PLC0415

    reg = http("POST", "/adapter_runs", {"name": NAME, "config": {"rank": 8}})
    slot_bound = reg.get("slot") is not None
    state = wait_state(NAME, "READY", timeout_s=600)
    info = http("GET", f"/adapter_runs/{NAME}")
    registration_id = info["registration_id"]
    report(
        "phase1-register",
        slot_bound and state == "READY",
        f"slot={reg.get('slot')} state={state} rid={registration_id[:8]}",
    )

    fb_shapes = [
        [(24, 16), (20, 12), (28, 16)],  # 3 samples: count not divisible by DP=2
        [(16, 8), (32, 24)],
        [(24, 16), (24, 16), (20, 10), (30, 20)],
    ]
    fb3_payload = fb_payload(fb_shapes[2], base_token=5000)
    payloads = [fb_payload(fb_shapes[0]), fb_payload(fb_shapes[1], base_token=3500), fb3_payload]
    fb3_logprobs = None
    for i, (shapes, payload) in enumerate(zip(fb_shapes, payloads, strict=True), start=1):
        view = ops.run("forward_backward", payload)
        lp = check_fb_result(view, shapes, f"phase2-fb{i}")
        if i == 3:
            fb3_logprobs = lp

    # One-sample fb: at DP=2 one whole rank runs only the zero-weight padding
    # row; the result plane must carry exactly the client's single row.
    view = ops.run("forward_backward", fb_payload([(22, 14)], base_token=7000))
    check_fb_result(view, [(22, 14)], "phase2-fb-odd1")

    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)))
    check_optim(view, "phase3-optim_step")

    view = ops.run("save_weights_for_sampler", {})
    result = view.get("result") or {}
    serving_version = result.get("serving_version")
    serving_name = result.get("serving_name")
    expected_name = serving_lora_name(NAME, registration_id)
    ok = view["state"] == "SUCCEEDED" and serving_version == 1 and serving_name == expected_name
    report(
        "phase4-save_weights_for_sampler",
        ok,
        f"state={view['state']} serving_version={serving_version} serving_name={serving_name} error={view.get('error')}",
    )

    sample_body = dict(
        text="The capital of France is",
        sampling_params=dict(max_new_tokens=8, temperature=0.0),
        lora_path=serving_name,
    )
    try:
        gen = http("POST", "/generate", sample_body, base=ROUTER)
        text = gen.get("text")
        report("phase4-sample", isinstance(text, str) and len(text) > 0, f"text={text!r}")
    except urllib.error.HTTPError as e:
        report("phase4-sample", False, f"HTTP {e.code}: {e.read().decode()[:500]}")

    view = ops.run("save_state", dict(tag="e2e-t0"))
    result = view.get("result") or {}
    state_path = result.get("path")
    manifest_ok = bool(state_path) and os.path.exists(os.path.join(state_path, "manifest.pt"))
    ok = view["state"] == "SUCCEEDED" and manifest_ok and result.get("step") == 1
    report(
        "phase5-save_state",
        ok,
        f"state={view['state']} path={state_path} manifest={manifest_ok} step={result.get('step')} error={view.get('error')}",
    )

    view = ops.run("load_state", dict(path=state_path))
    result = view.get("result") or {}
    ok = view["state"] == "SUCCEEDED" and result.get("step") == 1
    report("phase6-load_state", ok, f"state={view['state']} step={result.get('step')} error={view.get('error')}")

    view = ops.run("forward_backward", fb3_payload)
    fb4_logprobs = check_fb_result(view, fb_shapes[2], "phase6-fb-post-restore")

    # weights actually moved: identical payload, logprobs differ pre/post optim
    max_delta = max_logprob_delta(fb3_logprobs, fb4_logprobs)
    report("phase6-weights-moved", max_delta > 1e-9, f"max |dlogprob| fb3 vs post-optim fb = {max_delta:.6g}")

    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)))
    check_optim(view, "phase6-optim-post-restore")

    http("DELETE", f"/adapter_runs/{NAME}")
    deadline = time.monotonic() + 300
    final_state, snapshot = None, None
    while time.monotonic() < deadline:
        snapshot = ops.snapshot()
        final_state = http("GET", f"/adapter_runs/state?names={NAME}")["states"].get(NAME)
        if final_state == "COMPLETED":
            break
        time.sleep(2)
    slot_free = (
        NAME not in {**snapshot["pending"], **snapshot["ready"], **snapshot["retiring"]}
        and NAME not in snapshot["cleanup"]
    )

    sidecar_ok = os.path.exists(sidecar_manifest(NAME))

    rejected = False
    reject_detail = "enqueue unexpectedly accepted"
    try:
        ops.enqueue("forward_backward", fb_payload([(16, 8)]))
    except Exception as e:  # noqa: BLE001
        rejected = "not accepting operations" in str(e) or "fenced" in str(e)
        reject_detail = str(e).splitlines()[-1][:200]
    ok = final_state == "COMPLETED" and slot_free and sidecar_ok and rejected
    report(
        "phase7-deregister",
        ok,
        f"final_state={final_state} slot_free={slot_free} sidecar={sidecar_ok} post-dereg-enqueue-rejected={rejected} ({reject_detail})",
    )

    reg_b = http("POST", "/adapter_runs", {"name": "e2e_b", "config": {"rank": 8}})
    state_b = wait_state("e2e_b", "READY", timeout_s=600)
    http("DELETE", "/adapter_runs/e2e_b")
    report(
        "phase7-second-adapter",
        reg_b.get("slot") is not None and state_b == "READY",
        f"slot={reg_b.get('slot')} state={state_b}",
    )
    wait_state("e2e_b", "COMPLETED", timeout_s=300)


# Forward-only operations and empty optimizer steps.


def phase_b(ops: Ops) -> None:
    name = "e2e_f"
    reg = register(ops, name)
    report("phaseB-register", reg["slot"] is not None, f"slot={reg['slot']} rid={reg['registration_id'][:8]}")

    shapes = [(24, 16), (20, 12)]
    payload = fb_payload(shapes, base_token=9000)

    view = ops.run("forward", dict(samples=payload["samples"]), name=name)
    result = view.get("result") or {}
    fwd_logprobs = result.get("logprobs")
    shapes_ok = (
        view["state"] == "SUCCEEDED"
        and isinstance(fwd_logprobs, list)
        and len(fwd_logprobs) == len(shapes)
        and all(len(lp) == resp for lp, (_, resp) in zip(fwd_logprobs, shapes, strict=True))
    )
    report(
        "phaseB-forward",
        shapes_ok,
        f"state={view['state']} shapes={[len(lp) for lp in fwd_logprobs] if isinstance(fwd_logprobs, list) else None} "
        f"metrics={result.get('metrics')} error={view.get('error')}",
    )

    # no dirty pin: a save_state right after the forward must not be rejected
    # by the unstepped-gradients gate
    view = ops.run("save_state", dict(tag="b-nodirty"), name=name)
    dirty_gated = "unstepped gradients" in (view.get("error") or "")
    report(
        "phaseB-no-dirty-pin",
        view["state"] == "SUCCEEDED" and not dirty_gated,
        f"state={view['state']} error={view.get('error')}",
    )

    # optim_step with nothing accumulated: the backend contract is an empty
    # step — SUCCEEDED with grad_norm == 0.0 (fresh Adam moments: weights
    # cannot move), never a user-side rejection
    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)), name=name)
    check_optim(view, "phaseB-optim-after-forward-only", expect_zero=True)
    step = ops.step_of(name)
    report("phaseB-empty-step-clock", step == 1, f"step={step} (empty optim_step advances the clock)")

    # identical payload through forward_backward: same weights (the empty step
    # moved nothing), so the logprob planes must agree
    view = ops.run("forward_backward", payload, name=name)
    fb_logprobs = check_fb_result(view, shapes, "phaseB-fb-same-payload")
    delta = max_logprob_delta(fwd_logprobs, fb_logprobs)
    report("phaseB-forward-vs-fb-logprobs", delta <= 1e-4, f"max |dlogprob| forward vs fb = {delta:.6g}")

    # the fb DID pin dirty (contrast with the forward): its optim_step has real gradients
    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)), name=name)
    check_optim(view, "phaseB-optim-after-fb")

    deregister(ops, name)
    report("phaseB-deregister", True, "COMPLETED")


# Slot-state ownership at DP=2.


def _rank_swapped_copy(state_path: str, dest: str) -> str:
    """A byte-identical copy of a two-rank state with the rank shards swapped:
    same save generation, same shapes, but each rank now reads a shard whose
    recorded per-rank ownership signature is the OTHER rank's — exactly the
    'sharded with a different per-rank parameter ownership' condition."""
    import shutil  # noqa: PLC0415

    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(state_path, dest)
    r0, r1 = os.path.join(dest, "shard_rank00000.pt"), os.path.join(dest, "shard_rank00001.pt")
    tmp = os.path.join(dest, "shard_rank_tmp.pt")
    os.rename(r0, tmp)
    os.rename(r1, r0)
    os.rename(tmp, r1)
    return dest


def phase_c(ops: Ops) -> None:
    import torch  # noqa: PLC0415

    reg = register(ops, "e2e_c")
    report("phaseC-register-slot0", reg["slot"] == 0, f"slot={reg['slot']}")
    ops.run("forward_backward", fb_payload([(24, 16), (20, 12)], base_token=11000), name="e2e_c")
    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)), name="e2e_c")
    check_optim(view, "phaseC-seed-optim")
    view = ops.run("save_state", dict(tag="c0"), name="e2e_c")
    state_path = (view.get("result") or {}).get("path")
    report(
        "phaseC-save-slot0-state",
        view["state"] == "SUCCEEDED" and (view.get("result") or {}).get("step") == 1,
        f"state={view['state']} path={state_path} step={(view.get('result') or {}).get('step')}",
    )

    # LayerWise DP sharding is real: the two rank shards carry disjoint,
    # non-trivial ownership signatures
    sig = [
        torch.load(os.path.join(state_path, f"shard_rank{r:05d}.pt"), map_location="cpu", weights_only=True)[
            "optimizer_param_names"
        ]
        for r in (0, 1)
    ]
    flat0 = {name for child in sig[0] for name in child}
    flat1 = {name for child in sig[1] for name in child}
    report(
        "phaseC-dp-sharding-real",
        sig[0] != sig[1] and flat0 and flat1 and not (flat0 & flat1),
        f"rank0 owns {len(flat0)} params, rank1 owns {len(flat1)}, overlap {len(flat0 & flat1)}",
    )
    deregister(ops, "e2e_c")

    reg1 = register(ops, "e2e_c1")
    reg2 = register(ops, "e2e_c2")
    report("phaseC-slot-arrangement", reg1["slot"] == 0 and reg2["slot"] == 1, f"c1={reg1['slot']} c2={reg2['slot']}")

    # Cross-slot restore under MATCHING signatures: on this deployment every
    # numel-class block is a multiple of 4, so slot 0 and slot 1 get identical
    # per-rank ownership in LayerWise's DP-2 ping-pong — the fence must allow
    # the restore (the contract is signature equality, not same-slot).
    view = ops.run("load_state", dict(path=state_path), name="e2e_c2")
    restored = view["state"] == "SUCCEEDED" and (view.get("result") or {}).get("step") == 1
    report(
        "phaseC-cross-slot-matching-sig-restore",
        restored and ops.step_of("e2e_c2") == 1,
        f"state={view['state']} step={(view.get('result') or {}).get('step')} error={view.get('error')}",
    )

    # ... and bitwise-correctly: a state saved back out of slot 1 carries the
    # same weights and optimizer tensors (only the slot tag differs)
    view = ops.run("save_state", dict(tag="c2snap"), name="e2e_c2")
    snap_path = (view.get("result") or {}).get("path")
    mismatch = None
    for r in (0, 1):
        shard = f"shard_rank{r:05d}.pt"
        before = torch.load(os.path.join(state_path, shard), map_location="cpu", weights_only=True)
        after = torch.load(os.path.join(snap_path, shard), map_location="cpu", weights_only=True)
        for key in ("weights", "optimizer_state", "optimizer_param_names"):
            mismatch = mismatch or _payload_tensors_equal(before[key], after[key], f"{shard}:{key}")
    report(
        "phaseC-cross-slot-restore-correct",
        mismatch is None,
        "slot0 state == slot1 re-save (bitwise)" if mismatch is None else mismatch,
    )

    # A state with a genuinely DIFFERENT per-rank ownership (the same save
    # with its rank shards swapped) must be refused by the ownership fence:
    # clean user-category failure, unanimous across ranks, nothing mutated.
    swapped = _rank_swapped_copy(state_path, os.path.join(os.path.dirname(state_path), "c0-rankswap"))
    view = ops.run("load_state", dict(path=swapped), name="e2e_c2")
    fence_msg = view.get("error") or ""
    refused = (
        view["state"] == "FAILED"
        and view.get("error_category") == "user"
        and "different per-rank parameter ownership" in fence_msg
    )
    report(
        "phaseC-ownership-fence-refused",
        refused,
        f"state={view['state']} category={view.get('error_category')} error={fence_msg[:220]}",
    )

    # trainer stayed healthy: the refused tenant keeps training
    ops.run("forward_backward", fb_payload([(18, 10)], base_token=12000), name="e2e_c2")
    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)), name="e2e_c2")
    check_optim(view, "phaseC-post-refusal-train")

    view = ops.run("load_state", dict(path=state_path), name="e2e_c1")
    restored = view["state"] == "SUCCEEDED" and (view.get("result") or {}).get("step") == 1
    step = ops.step_of("e2e_c1")
    report(
        "phaseC-same-slot-restore",
        restored and step == 1,
        f"state={view['state']} result_step={(view.get('result') or {}).get('step')} registry_step={step} "
        f"error={view.get('error')}",
    )
    ops.run("forward_backward", fb_payload([(24, 16)], base_token=13000), name="e2e_c1")
    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)), name="e2e_c1")
    check_optim(view, "phaseC-post-restore-train")

    deregister(ops, "e2e_c1")
    deregister(ops, "e2e_c2")

    # sidecar variant of the fence: swap the retired tenant's sidecar shards
    # so its recorded ownership is foreign on every rank; re-registration must
    # fall back to a fresh init (no crash, step 0) instead of resuming it
    sidecar_base = os.path.dirname(sidecar_manifest("e2e_c2"))
    r0, r1 = os.path.join(sidecar_base, "shard_rank00000.pt"), os.path.join(sidecar_base, "shard_rank00001.pt")
    tmp = os.path.join(sidecar_base, "shard_rank_tmp.pt")
    os.rename(r0, tmp)
    os.rename(r1, r0)
    os.rename(tmp, r1)
    reg2b = register(ops, "e2e_c2")
    step = ops.step_of("e2e_c2")
    report(
        "phaseC-foreign-sidecar-fresh-init",
        reg2b["slot"] == 0 and step == 0,
        f"slot={reg2b['slot']} step={step} (rank-swapped sidecar refused by the fence; reconcile fresh-inits)",
    )
    ops.run("forward_backward", fb_payload([(18, 10)], base_token=14000), name="e2e_c2")
    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)), name="e2e_c2")
    check_optim(view, "phaseC-fresh-init-train")
    deregister(ops, "e2e_c2")


# Sidecar resume preserves step, weights, and FP32 masters.


def _payload_tensors_equal(a, b, where: str = "") -> str | None:
    """First mismatch path between two saved payload subtrees, or None."""
    import torch  # noqa: PLC0415

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return f"{where}: tensor vs {type(b).__name__}"
        return None if torch.equal(a, b) else f"{where}: tensors differ (max|d|={(a - b).abs().max().item():.3g})"
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{where}: keys {sorted(a)} != {sorted(b)}"
        for key in a:
            if key == "miles_multi_lora_slot":  # the destination slot's tag: differs across slots by design
                continue
            if (m := _payload_tensors_equal(a[key], b[key], f"{where}.{key}")) is not None:
                return m
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{where}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if (m := _payload_tensors_equal(x, y, f"{where}[{i}]")) is not None:
                return m
        return None
    return None if a == b else f"{where}: {a!r} != {b!r}"


def phase_d(ops: Ops) -> None:
    import torch  # noqa: PLC0415

    name = "e2e_d"
    probe = dict(samples=fb_payload([(26, 18), (20, 12)], base_token=15000)["samples"])

    reg = register(ops, name)
    report("phaseD-register", reg["slot"] is not None and ops.step_of(name) == 0, f"slot={reg['slot']} step=0")

    # two real steps so the resume has a non-trivial clock and Adam state
    for i, base in enumerate((16000, 17000), start=1):
        ops.run("forward_backward", fb_payload([(24, 16), (28, 18)], base_token=base), name=name)
        view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)), name=name)
        check_optim(view, f"phaseD-optim{i}")

    view = ops.run("save_weights_for_sampler", {}, name=name)
    serving_version = (view.get("result") or {}).get("serving_version")
    report(
        "phaseD-publish",
        view["state"] == "SUCCEEDED" and serving_version == 1,
        f"state={view['state']} serving_version={serving_version}",
    )

    view = ops.run("forward", probe, name=name)
    probe_before = (view.get("result") or {}).get("logprobs")
    report("phaseD-probe-before", view["state"] == "SUCCEEDED" and probe_before is not None, "captured L1")

    deregister(ops, name)
    sidecar_ok = os.path.exists(sidecar_manifest(name))
    report("phaseD-final-sidecar", sidecar_ok, sidecar_manifest(name))

    # re-register the SAME name: reconcile must auto-resume from the sidecar
    reg2 = register(ops, name)
    step = ops.step_of(name)
    report("phaseD-resume-step", step == 2, f"slot={reg2['slot']} restored step={step} (want 2)")

    view = ops.run("forward", probe, name=name)
    probe_after = (view.get("result") or {}).get("logprobs")
    delta = max_logprob_delta(probe_before, probe_after)
    report("phaseD-resume-logprobs", delta <= 1e-6, f"max |dlogprob| pre-dereg vs post-resume = {delta:.6g}")

    # the resumed masters are the checkpoint's fp32 masters, NOT re-quantized
    # through bf16: a state saved now must carry bitwise-identical weights and
    # optimizer state (fp32 masters + Adam moments) to the retirement sidecar
    view = ops.run("save_state", dict(tag="d-resumed"), name=name)
    resumed_path = (view.get("result") or {}).get("path")
    report("phaseD-save-resumed", view["state"] == "SUCCEEDED" and resumed_path is not None, f"path={resumed_path}")

    sidecar_base = os.path.dirname(sidecar_manifest(name))
    shards = sorted(f for f in os.listdir(sidecar_base) if f.startswith("shard_rank"))
    mismatch, compared = None, 0
    for shard in shards:
        before = torch.load(os.path.join(sidecar_base, shard), map_location="cpu", weights_only=True)
        after = torch.load(os.path.join(resumed_path, shard), map_location="cpu", weights_only=True)
        for key in ("weights", "optimizer_state", "optimizer_param_names"):
            mismatch = mismatch or _payload_tensors_equal(before[key], after[key], f"{shard}:{key}")
        compared += 1
    report(
        "phaseD-fp32-masters-preserved",
        compared > 0 and mismatch is None,
        f"{compared} rank shards bitwise-compared (weights + optimizer fp32 masters/moments): "
        + ("identical" if mismatch is None else mismatch),
    )

    # and training continues from the restored state
    ops.run("forward_backward", fb_payload([(24, 16), (20, 12)], base_token=18000), name=name)
    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)), name=name)
    check_optim(view, "phaseD-post-resume-train")
    step = ops.step_of(name)
    report("phaseD-post-resume-step", step == 3, f"step={step} (want 3)")

    deregister(ops, name)
    report("phaseD-deregister", True, "COMPLETED")


PHASES = {"a": phase_a, "b": phase_b, "c": phase_c, "d": phase_d}


def main() -> None:
    global SAVE_ROOT, ROUTER
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--phases", default="a,b,c,d", help="comma-separated subset of a,b,c,d")
    parser.add_argument("--save-root", default=SAVE_ROOT, help="the service's --save dir (sidecar/state paths)")
    args = parser.parse_args()
    ray.init(address=args.ray_address, namespace="miles", ignore_reinit_error=True, log_to_driver=False)

    SAVE_ROOT = args.save_root.rstrip("/")

    ops = Ops()
    # The sglang router binds the node IP (the control API advertises its own
    # loopback bind host, which never reaches the router's socket).
    from miles.utils.misc import get_current_node_ip  # noqa: PLC0415

    ROUTER = f"http://{get_current_node_ip()}:20080"
    print(f"router: {ROUTER}", flush=True)

    for phase in args.phases.split(","):
        print(f"\n=== phase {phase.upper()} ===", flush=True)
        PHASES[phase.strip().lower()](ops)

    print(f"\n=== E2E SUMMARY: {len(PASS)} passed, {len(FAIL)} failed ===", flush=True)


if __name__ == "__main__":
    main()
