#!/usr/bin/env python3
"""7-phase GPU E2E client for the tinker-compatible backend.

Drives one adapter ("e2e_a") through the full operation lifecycle against a
live service: register -> forward_backward x3 -> optim_step ->
save_weights_for_sampler (+ router sampling) -> save_state -> load_state ->
post-restore fb/optim -> deregister (+ post-deregister rejection).

Registration goes over the controller HTTP API; operations go through the
controller Ray actor (operation enqueue/get/ack are not HTTP-exposed yet).
Run on the head node: PYTHONPATH must include /personal/miles.
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
import uuid

import ray

API = "http://127.0.0.1:8068"
ROUTER = "http://127.0.0.1:20080"  # rebound to the head-node IP at startup
NAME = "e2e_a"

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
        self.ordinal = 0

    def enqueue(self, kind: str, payload: dict | None = None, name: str = NAME) -> str:
        self.ordinal += 1
        op_id = f"op-{self.ordinal}-{kind}-{uuid.uuid4().hex[:8]}"
        view = ray.get(self.controller.enqueue_operation.remote(name, op_id, self.ordinal, kind, payload))
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

    def run(self, kind: str, payload: dict | None = None, timeout_s: float = 600) -> dict:
        op_id = self.enqueue(kind, payload)
        view = self.wait(op_id, timeout_s)
        self.ack(op_id)
        return view

    def snapshot(self) -> dict:
        return ray.get(self.controller.snapshot.remote())


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    args = parser.parse_args()
    ray.init(address=args.ray_address, namespace="miles", ignore_reinit_error=True, log_to_driver=False)
    from miles.utils.tinker_backend import serving_lora_name  # noqa: PLC0415

    ops = Ops()
    global ROUTER
    router_host = ray.get(ops.controller.http_host.remote())
    ROUTER = f"http://{router_host}:20080"
    print(f"router: {ROUTER}", flush=True)

    # ---------------- phase 1: register ----------------
    reg = http("POST", "/adapter_runs", {"name": NAME, "config": {"rank": 8}})
    slot_bound = reg.get("slot") is not None
    state = wait_state(NAME, "READY", timeout_s=600)
    info = http("GET", f"/adapter_runs/{NAME}")
    registration_id = info["registration_id"]
    report("phase1-register", slot_bound and state == "READY", f"slot={reg.get('slot')} state={state} rid={registration_id[:8]}")

    # ---------------- phase 2: forward_backward x3 ----------------
    fb_shapes = [
        [(24, 16), (20, 12), (28, 16)],  # 3 samples: count not divisible by 2
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

    # ---------------- phase 3: optim_step ----------------
    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)))
    result = view.get("result") or {}
    grad_norm = result.get("grad_norm")
    ok = (
        view["state"] == "SUCCEEDED"
        and isinstance(grad_norm, float)
        and math.isfinite(grad_norm)
        and grad_norm > 0
    )
    report("phase3-optim_step", ok, f"state={view['state']} grad_norm={grad_norm} lr={result.get('learning_rate')} error={view.get('error')}")

    # ---------------- phase 4: save_weights_for_sampler + sample ----------------
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

    # ---------------- phase 5: save_state ----------------
    view = ops.run("save_state", dict(tag="e2e-t0"))
    result = view.get("result") or {}
    state_path = result.get("path")
    manifest_ok = False
    if state_path:
        import os  # noqa: PLC0415

        manifest_ok = os.path.exists(os.path.join(state_path, "manifest.pt"))
    ok = view["state"] == "SUCCEEDED" and manifest_ok and result.get("step") == 1
    report("phase5-save_state", ok, f"state={view['state']} path={state_path} manifest={manifest_ok} step={result.get('step')} error={view.get('error')}")

    # ---------------- phase 6: load_state + fb/optim still work ----------------
    view = ops.run("load_state", dict(path=state_path))
    result = view.get("result") or {}
    ok = view["state"] == "SUCCEEDED" and result.get("step") == 1
    report("phase6-load_state", ok, f"state={view['state']} step={result.get('step')} error={view.get('error')}")

    view = ops.run("forward_backward", fb3_payload)
    fb4_logprobs = check_fb_result(view, fb_shapes[2], "phase6-fb-post-restore")

    # weights actually moved: identical payload, logprobs differ pre/post optim
    max_delta = max(
        abs(a - b) for row_a, row_b in zip(fb3_logprobs, fb4_logprobs, strict=True) for a, b in zip(row_a, row_b, strict=True)
    )
    report("phase6-weights-moved", max_delta > 1e-9, f"max |dlogprob| fb3 vs post-optim fb = {max_delta:.6g}")

    view = ops.run("optim_step", dict(adam_params=dict(learning_rate=1e-4)))
    result = view.get("result") or {}
    grad_norm2 = result.get("grad_norm")
    ok = view["state"] == "SUCCEEDED" and isinstance(grad_norm2, float) and math.isfinite(grad_norm2) and grad_norm2 > 0
    report("phase6-optim-post-restore", ok, f"state={view['state']} grad_norm={grad_norm2} error={view.get('error')}")

    # ---------------- phase 7: deregister ----------------
    http("DELETE", f"/adapter_runs/{NAME}")
    deadline = time.monotonic() + 300
    final_state, snapshot = None, None
    while time.monotonic() < deadline:
        snapshot = ops.snapshot()
        final_state = http("GET", f"/adapter_runs/state?names={NAME}")["states"].get(NAME)
        if final_state == "COMPLETED":
            break
        time.sleep(2)
    slot_free = NAME not in {**snapshot["pending"], **snapshot["ready"], **snapshot["retiring"]} and NAME not in snapshot["cleanup"]

    import os  # noqa: PLC0415

    sidecar = f"/personal/tinker_e2e/save/adapters/{NAME}/slot_state/manifest.pt"
    sidecar_ok = os.path.exists(sidecar)

    # a fb enqueued AFTER deregister must be rejected as a user error
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

    # second adapter registers cleanly into the freed pool
    reg_b = http("POST", "/adapter_runs", {"name": "e2e_b", "config": {"rank": 8}})
    state_b = wait_state("e2e_b", "READY", timeout_s=600)
    http("DELETE", "/adapter_runs/e2e_b")
    report("phase7-second-adapter", reg_b.get("slot") is not None and state_b == "READY", f"slot={reg_b.get('slot')} state={state_b}")

    print(f"\n=== E2E SUMMARY: {len(PASS)} passed, {len(FAIL)} failed ===", flush=True)


if __name__ == "__main__":
    main()
