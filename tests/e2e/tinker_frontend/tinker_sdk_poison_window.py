#!/usr/bin/env python3
"""Poison-window GPU acceptance: a FAILED forward_backward chunk poisons the
registration's gradient window (#2258 §5) — the window's ``optim_step`` is
rejected ("gradient window ... discarded") and the trainer executes the
discard (``zero_adapter_slot_grads``) on EVERY rank instead of stepping.

The CPU contract tests prove the control flow; this client proves the
collective semantics on a live DP>1 deployment through the UNMODIFIED
``tinker==0.24.1`` SDK:

  1. baseline   rank-8 client, 3x good fb+optim: finite losses/grad_norms,
                step clock exactly 3, publish bumps serving_version to 1
  2. poison     capture probe logprobs L0 on a fixed payload (forward: no
                gradients, no dirty pin) and a clean-window reference
                grad_norm for the SAME batch; then good fb (EXECUTES into
                the window) + channel-mismatch fb (typed reject) + optim.
                The fb error is typed; the optim FAILS with the poison
                message; step clock and serving version hold; the probe
                re-reads EXACTLY L0 — the good chunk's gradients were
                discarded on both ranks, no half-applied update
  3. recovery   the same batch again: optim SUCCEEDS and its grad_norm
                matches the clean reference exactly — the discard left no
                residue on any rank (residue would double the norm). A
                real step then MOVES the probe (sensitivity control)
  4. isolation  a second adapter runs the poison sequence CONCURRENTLY
                while the first trains normally: the victim's poison never
                perturbs the neighbor's losses or step clock
  5. late chunk a 1030-datum fb whose LATE chunk carries the bad datum: the
                SDK splits at 1024 and posts the first chunk last; the
                1024-datum chunk lands (real gradients on both ranks)
                before the poison is seen — same discard assertions

Step/serving clocks come from the operator plane (``GET /adapter_runs`` on
the same uvicorn; loopback-only), so run this on the head node from a venv
with ``tinker==0.24.1``:
  python tests/e2e/tinker_frontend/tinker_sdk_poison_window.py --out-dir <dir>

Numeric tolerances: dense deployments are bitwise-deterministic per forward,
so every probe comparison defaults to EXACT (0.0) and the grad-norm reference
comparison to rel_tol=1e-6. MoE deployments (e.g. GPT-OSS grouped-GEMM/Triton
kernels) have inherent run-to-run forward nondeterminism at the BASE model
(measured 0.09-0.21 max |dlogprob| on 4xH200 GPT-OSS 20B, pre-existing before
any multi-LoRA change), which fails the probe-stability precondition before
any mechanism is tested. For those deployments pass ``--probe-tolerance`` (and
``--grad-norm-rtol``) calibrated to the measured noise; the client then also
REQUIRES the real-step sensitivity to clear that tolerance (reporting the
margin), so a discard check can never hide a real update inside the noise
band — measured on 4xH200 GPT-OSS 20B at LR=1e-4: noise 0.130, real-step
movement 0.406 (3.1x the noise, 1.6x a 2x-noise tolerance). Every MECHANISM
assertion — typed fb/optim failures, step/serving clocks held, discard
executed, neighbor isolation, no-hang — stays exact regardless of tolerance.
"""

import argparse
import json
import math
import os
import threading
import time
import urllib.request

import tinker
from tinker import types

CORPUS = [
    "The old lighthouse keeper climbed the spiral stairs every evening at dusk.",
    "He lit the great lamp so that ships could find their way home through the fog.",
    "One autumn night a fierce storm rolled in from the north and shook the tower.",
    "The keeper held his lantern steady and watched the waves crash on the rocks.",
    "By morning the sea was calm again and a small fishing boat waved its thanks.",
    "The keeper smiled, poured his tea, and wrote the night's story in his logbook.",
    "Years later his granddaughter found the logbook and read every page aloud.",
    "She decided then that she too would keep the light burning for the ships.",
]

LR = 1e-4

# Deployment noise tolerances; overridden from --probe-tolerance /
# --grad-norm-rtol in main(). 0.0 / 1e-6 = the exact dense-deployment contract.
PROBE_TOLERANCE = 0.0
GRAD_NORM_RTOL = 1e-6


def assert_probe_still(delta: float, what: str) -> None:
    """A probe that must NOT have moved (discard/isolation checks): exact on
    dense deployments, within the deployment's measured forward-noise band on
    nondeterministic (MoE) ones."""
    assert delta <= PROBE_TOLERANCE, f"{what}: max|dlogprob|={delta} > tolerance {PROBE_TOLERANCE}"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{threading.current_thread().name}] {msg}", flush=True)


def ce_datum(tokens: list[int]) -> types.Datum:
    inputs, targets = tokens[:-1], tokens[1:]
    return types.Datum(
        model_input=types.ModelInput.from_ints(inputs),
        loss_fn_inputs={"target_tokens": targets, "weights": [1.0] * len(targets)},
    )


def channel_mismatch_datum(tokens: list[int]) -> types.Datum:
    """importance_sampling requires 'logprobs'; deliberately missing -> the
    frontend rejects the chunk typed, consuming (and poisoning) its ordinal."""
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "advantages": [1.0] * (len(tokens) - 1)},
    )


def bad_target_datum(tokens: list[int]) -> types.Datum:
    """cross_entropy datum whose active target is not the next input token."""
    inputs, targets = tokens[:-1], list(tokens[1:])
    targets[0] += 7  # non-next-token target with non-zero weight -> typed reject
    return types.Datum(
        model_input=types.ModelInput.from_ints(inputs),
        loss_fn_inputs={"target_tokens": targets, "weights": [1.0] * len(targets)},
    )


# ---------------- operator plane (loopback, same uvicorn) ----------------


def adapter_record(base_url: str, api_key: str, session_id: str) -> dict:
    req = urllib.request.Request(f"{base_url}/adapter_runs", headers={"X-API-Key": api_key})
    with urllib.request.urlopen(req, timeout=30) as resp:
        adapters = json.load(resp)["adapters"]
    for status in adapters:
        if (status.get("metadata") or {}).get("session_id") == session_id:
            return status
    raise AssertionError(f"no adapter registered for session {session_id}")


def session_of(client) -> str:
    return client.model_id.split(":")[0]


def clocks(args, client) -> tuple[int, int, int]:
    record = adapter_record(args.base_url, args.api_key, session_of(client))
    return record["step"], record["version"], record["slot"]


def wait_version(args, client, version: int, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if clocks(args, client)[1] == version:
            return
        time.sleep(2)
    raise AssertionError(f"serving version never reached {version}")


# ---------------- probes and typed-failure helpers ----------------


def probe_rows(client, probe_data) -> list[list[float]]:
    forward = client.forward(probe_data, "cross_entropy").result()
    return [out["logprobs"].tolist() for out in forward.loss_fn_outputs]


def max_abs_delta(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(x - y) for ra, rb in zip(a, b, strict=True) for x, y in zip(ra, rb, strict=True))


def expect_typed_failure(future, needle: str, what: str) -> str:
    try:
        future.result()
    except tinker.RequestFailedError as exc:
        message = str(exc)
        assert needle in message, f"{what}: expected {needle!r} in: {message}"
        return message
    raise AssertionError(f"{what}: expected a typed RequestFailedError, got success")


def poison_round(args, client, data, bad_datum, bad_loss_fn, bad_needle) -> tuple[str, str]:
    """Submit good fb + bad fb + optim in one window (cookbook style: all
    posted before any await). Returns (fb_error, optim_error); asserts the
    step clock and serving version held still."""
    step_pre, version_pre, _ = clocks(args, client)
    good_future = client.forward_backward(data, "cross_entropy")
    bad_future = client.forward_backward([bad_datum], bad_loss_fn)
    optim_future = client.optim_step(types.AdamParams(learning_rate=LR))
    good = good_future.result()  # the good chunk EXECUTED: gradients are live on every rank
    assert len(good.loss_fn_outputs) == len(data)
    t0 = time.time()
    fb_error = expect_typed_failure(bad_future, bad_needle, "bad fb chunk")
    optim_error = expect_typed_failure(optim_future, "gradient window", "poisoned optim_step")
    assert "discarded" in optim_error, optim_error
    log(f"typed fb reject + poisoned optim discard in {time.time() - t0:.1f}s (no hang)")
    step_post, version_post, _ = clocks(args, client)
    assert (step_post, version_post) == (step_pre, version_pre), (
        f"clocks moved across a poisoned window: step {step_pre}->{step_post}, "
        f"version {version_pre}->{version_post}"
    )
    return fb_error, optim_error


def train_round(client, data, lr: float = LR) -> tuple[float, float]:
    fb_future = client.forward_backward(data, "cross_entropy")
    optim_future = client.optim_step(types.AdamParams(learning_rate=lr))
    fb = fb_future.result()
    optim = optim_future.result()
    loss = fb.metrics["loss:sum"]
    grad_norm = optim.metrics["grad_norm"]
    assert math.isfinite(loss) and math.isfinite(grad_norm) and grad_norm > 0, (loss, grad_norm)
    return loss, grad_norm


def assert_close(observed: float, reference: float, what: str) -> None:
    assert math.isclose(
        observed, reference, rel_tol=GRAD_NORM_RTOL
    ), f"{what}: {observed} != {reference} (rel_tol {GRAD_NORM_RTOL})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8068")
    parser.add_argument("--api-key", default=os.environ.get("MILES_TINKER_API_KEY", "tml-miles-gpu-acceptance"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--large-fb-datums", type=int, default=1030, help=">1024 forces multi-chunk posting")
    parser.add_argument(
        "--probe-tolerance",
        type=float,
        default=0.0,
        help="allowed max |dlogprob| for probe comparisons; 0.0 (exact) for dense deployments, "
        "the measured base-model forward-noise band for nondeterministic MoE kernels",
    )
    parser.add_argument(
        "--grad-norm-rtol",
        type=float,
        default=1e-6,
        help="rel_tol for grad-norm reference comparisons (recovery/residue checks)",
    )
    args = parser.parse_args()
    global PROBE_TOLERANCE, GRAD_NORM_RTOL
    PROBE_TOLERANCE = args.probe_tolerance
    GRAD_NORM_RTOL = args.grad_norm_rtol
    os.makedirs(args.out_dir, exist_ok=True)
    summary: dict = {}

    service_a = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    capabilities = service_a.get_server_capabilities()
    [base_model] = [m.model_name for m in capabilities.supported_models]
    summary["base_model"] = base_model

    # ================= phase 1: baseline sanity =================
    client_a = service_a.create_lora_training_client(base_model=base_model, rank=8)
    assert client_a.get_info().lora_rank == 8
    tokenizer = client_a.get_tokenizer()
    data = [ce_datum(tokenizer.encode(text)) for text in CORPUS]
    probe_data = [ce_datum(tokenizer.encode(text)) for text in CORPUS[:4]]
    log(f"adapter A ready: model_id={client_a.model_id} rank=8")

    baseline = [train_round(client_a, data) for _ in range(3)]
    step, version, slot_a = clocks(args, client_a)
    assert step == 3, f"baseline step clock: {step} != 3"
    sampling = client_a.save_weights_and_get_sampling_client()
    assert sampling.get_base_model() == base_model
    wait_version(args, client_a, 1)
    summary["phase1_baseline"] = {"rounds": baseline, "step": step, "serving_version": 1, "slot": slot_a}
    log(f"baseline: 3 rounds, losses {[round(loss, 3) for loss, _ in baseline]}, step=3, published version=1")

    # ================= phase 2: poison the window =================
    l0 = probe_rows(client_a, probe_data)
    _, grad_norm_ref = train_round(client_a, data, lr=0.0)  # clean-window reference, weights unchanged
    l0_control = probe_rows(client_a, probe_data)
    control_delta = max_abs_delta(l0_control, l0)
    # Precondition: the deployment's inherent forward noise must sit inside
    # the configured tolerance, or every later stillness check is meaningless.
    assert_probe_still(control_delta, "probe not stable across an lr=0 round")
    step_pre, version_pre, _ = clocks(args, client_a)
    assert (step_pre, version_pre) == (4, 1)

    fb_error, optim_error = poison_round(
        args, client_a, data, channel_mismatch_datum(tokenizer.encode(CORPUS[0])), "importance_sampling", "logprobs"
    )
    l1 = probe_rows(client_a, probe_data)
    poison_delta = max_abs_delta(l1, l0)
    assert_probe_still(poison_delta, "weights moved across a poisoned window")
    summary["phase2_poison"] = {
        "grad_norm_ref": grad_norm_ref,
        "control_probe_delta": control_delta,
        "fb_error": fb_error[:200],
        "optim_error": optim_error[:200],
        "step_held": step_pre,
        "version_held": version_pre,
        "probe_delta_after_discard": poison_delta,
    }
    log(f"poison: optim rejected, step/version held at {step_pre}/{version_pre}, probe delta {poison_delta}")

    # ================= phase 3: recovery, no residue =================
    loss_rec, grad_norm_rec = train_round(client_a, data)  # same batch, same weights
    assert_close(grad_norm_rec, grad_norm_ref, "recovery grad_norm vs clean reference (residue would double it)")
    step, version, _ = clocks(args, client_a)
    assert step == step_pre + 1, f"recovery step clock: {step} != {step_pre + 1}"
    l2 = probe_rows(client_a, probe_data)
    sensitivity = max_abs_delta(l2, l0)
    # The minimum meaningful bar: a real update must be distinguishable from
    # the configured noise band, or the stillness checks above prove nothing.
    assert sensitivity > PROBE_TOLERANCE, (
        f"probe blind: a real optim step moved the logprobs by {sensitivity}, "
        f"inside the noise tolerance {PROBE_TOLERANCE}"
    )
    if PROBE_TOLERANCE > 0.0:
        log(f"sensitivity margin: real step moved {sensitivity:.4f} = {sensitivity / PROBE_TOLERANCE:.2f}x tolerance")
    summary["phase3_recovery"] = {
        "loss": loss_rec,
        "grad_norm": grad_norm_rec,
        "grad_norm_ref": grad_norm_ref,
        "step": step,
        "probe_moved_by_real_step": sensitivity,
    }
    log(f"recovery: grad_norm {grad_norm_rec} == ref {grad_norm_ref}, step->{step}, probe moved {sensitivity:.4f}")

    # ================= phase 4: concurrent isolation =================
    service_b = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    client_b = service_b.create_lora_training_client(base_model=base_model, rank=8)
    _, _, slot_b = clocks(args, client_b)
    assert slot_b != slot_a, (slot_a, slot_b)
    lb0 = probe_rows(client_b, probe_data)
    _, grad_norm_ref_b = train_round(client_b, data, lr=0.0)  # quiet reference for B
    assert_probe_still(max_abs_delta(probe_rows(client_b, probe_data), lb0), "B probe not stable")
    step_a_pre = clocks(args, client_a)[0]

    barrier = threading.Barrier(2)
    neighbor_rounds: list[tuple[float, float]] = []
    victim_errors: list[str] = []
    failures: list[BaseException] = []

    def neighbor() -> None:
        try:
            barrier.wait(timeout=60)
            for _ in range(4):
                neighbor_rounds.append(train_round(client_a, data))
        except BaseException as exc:  # noqa: BLE001 - surfaced after join
            failures.append(exc)

    def victim() -> None:
        try:
            barrier.wait(timeout=60)
            errors = poison_round(
                args,
                client_b,
                data,
                channel_mismatch_datum(tokenizer.encode(CORPUS[1])),
                "importance_sampling",
                "logprobs",
            )
            victim_errors.extend(errors)
        except BaseException as exc:  # noqa: BLE001 - surfaced after join
            failures.append(exc)

    threads = [
        threading.Thread(target=neighbor, name="neighbor-A"),
        threading.Thread(target=victim, name="victim-B"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=600)
        assert not thread.is_alive(), f"{thread.name} hung"
    assert not failures, failures

    step_a_post = clocks(args, client_a)[0]
    assert step_a_post == step_a_pre + 4, f"neighbor step clock: {step_a_pre}->{step_a_post}, expected +4"
    neighbor_losses = [loss for loss, _ in neighbor_rounds]
    assert neighbor_losses[-1] < neighbor_losses[0], f"neighbor loss did not decrease: {neighbor_losses}"
    assert all(b <= a * 1.02 for a, b in zip(neighbor_losses, neighbor_losses[1:], strict=False)), neighbor_losses
    lb1 = probe_rows(client_b, probe_data)
    victim_delta = max_abs_delta(lb1, lb0)
    assert_probe_still(victim_delta, "victim weights moved")
    _, grad_norm_rec_b = train_round(client_b, data)  # victim recovery (quiet)
    assert_close(grad_norm_rec_b, grad_norm_ref_b, "victim recovery grad_norm vs quiet reference")
    step_b, version_b, _ = clocks(args, client_b)
    assert (step_b, version_b) == (2, 0), (step_b, version_b)
    summary["phase4_isolation"] = {
        "neighbor_losses": neighbor_losses,
        "neighbor_grad_norms": [grad_norm for _, grad_norm in neighbor_rounds],
        "neighbor_steps": [step_a_pre, step_a_post],
        "victim_errors": [error[:200] for error in victim_errors],
        "victim_probe_delta": victim_delta,
        "victim_recovery_grad_norm": grad_norm_rec_b,
        "victim_grad_norm_ref": grad_norm_ref_b,
    }
    log(
        f"isolation: neighbor stepped {step_a_pre}->{step_a_post} losses {[round(loss, 2) for loss in neighbor_losses]}; "
        f"victim poisoned+discarded (delta {victim_delta}), recovered grad_norm {grad_norm_rec_b}"
    )

    # ================= phase 5: late chunk fails after early chunk landed =================
    lb2 = probe_rows(client_b, probe_data)
    _, grad_norm_ref_late = train_round(client_b, data, lr=0.0)
    step_pre_b, version_pre_b, _ = clocks(args, client_b)
    short = tokenizer.encode("The sea was calm.")
    big = [ce_datum(short) for _ in range(args.large_fb_datums - 1)] + [bad_target_datum(short)]
    fb_future = client_b.forward_backward(big, "cross_entropy")  # 2 chunks; the bad datum rides the late one
    optim_future = client_b.optim_step(types.AdamParams(learning_rate=LR))
    t0 = time.time()
    fb_error = expect_typed_failure(fb_future, "next input", "late bad chunk")
    optim_error = expect_typed_failure(optim_future, "gradient window", "poisoned optim after landed chunk")
    log(f"late-chunk poison surfaced in {time.time() - t0:.1f}s")
    step_post_b, version_post_b, _ = clocks(args, client_b)
    assert (step_post_b, version_post_b) == (step_pre_b, version_pre_b)
    lb3 = probe_rows(client_b, probe_data)
    late_delta = max_abs_delta(lb3, lb2)
    assert_probe_still(late_delta, "1024 landed datums leaked into the weights")
    loss_late, grad_norm_late = train_round(client_b, data)  # residue of 1024 datums would explode this
    assert_close(grad_norm_late, grad_norm_ref_late, "post-late-chunk recovery grad_norm vs quiet reference")
    summary["phase5_late_chunk"] = {
        "datums": args.large_fb_datums,
        "fb_error": fb_error[:200],
        "optim_error": optim_error[:200],
        "step_held": step_pre_b,
        "probe_delta": late_delta,
        "recovery_grad_norm": grad_norm_late,
        "grad_norm_ref": grad_norm_ref_late,
        "recovery_loss": loss_late,
    }
    log(
        f"late chunk: discard held (delta {late_delta}), recovery grad_norm {grad_norm_late} == ref {grad_norm_ref_late}"
    )

    summary["ok"] = True
    with open(os.path.join(args.out_dir, "poison_window_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("=== POISON-WINDOW ACCEPTANCE (DP=2): PASS ===")


if __name__ == "__main__":
    main()
