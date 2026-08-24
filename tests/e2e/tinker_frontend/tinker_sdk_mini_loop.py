#!/usr/bin/env python3
"""Golden-acceptance mini-loop: the UNMODIFIED official ``tinker==0.24.1`` SDK
drives the miles tinker frontend end to end, cookbook style.

    ServiceClient(base_url, api_key)
      -> get_server_capabilities (the deployment's one base model)
      -> create_lora_training_client(rank=16)
      -> ~10x [forward_backward(cross_entropy, teacher-forced prompt-masked
               SFT datums: prompt weight 0, completion weight 1)
               + optim_step(AdamParams(lr=1e-4))]      loss:sum must decrease
      -> save_weights_and_get_sampling_client -> sample  coherent continuation
      -> save_state -> load_state_with_optimizer -> one more fb/optim
      -> out-of-order large fb (>MAX_CHUNK_LEN datums: the SDK splits chunks
         and posts the first one LAST; the backend ledger reorders)
      -> a deliberate channel-mismatch datum surfacing as a typed SDK error;
         it poisons its gradient window (#2258 §5) so the window's optim_step
         fails as a discard, and the next round steps normally

The SFT per-token loss divides by ``loss_weight:sum`` (Σ weight·mask), NOT by
``unmasked_tokens:sum`` — the latter counts the weight-0 prompt positions too
(codex-0817-sft-fix §7). The prompt masking here keeps the two metrics
distinct, so this loop regression-tests the denominator on real GPUs: with
the old all-ones weights they were equal and the bug was invisible.

Run on the head node from a venv with ``tinker==0.24.1`` installed:
  python tests/e2e/tinker_frontend/tinker_sdk_mini_loop.py --out-dir <dir>
"""

import argparse
import json
import os
import time

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

SAMPLE_PROMPT = "The old lighthouse keeper climbed"


def ce_datum(tokens: list[int]) -> types.Datum:
    """Plain LM datum: model_input = tokens[:-1], next-token targets, weight 1."""
    inputs, targets = tokens[:-1], tokens[1:]
    return types.Datum(
        model_input=types.ModelInput.from_ints(inputs),
        loss_fn_inputs={"target_tokens": targets, "weights": [1.0] * len(targets)},
    )


def sft_datum(prompt_tokens: list[int], completion_tokens: list[int]) -> tuple[types.Datum, float, int]:
    """Teacher-forced SFT datum (the correct shape, codex-0817-sft-fix §2):
    position i predicts tokens[i+1], so the prompt-internal next-token
    positions get weight 0 and the completion positions weight 1. Returns the
    datum plus its CE weight sum and its total target-position count."""
    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights},
    )
    return datum, sum(weights), len(weights)


def split_prompt_completion(text: str) -> tuple[str, str]:
    """First half of the words is the prompt (weight 0), the rest completion."""
    words = text.split()
    split = max(1, len(words) // 2)
    return " ".join(words[:split]), " " + " ".join(words[split:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8068")
    parser.add_argument("--api-key", default=os.environ.get("MILES_TINKER_API_KEY", "tml-miles-gpu-acceptance"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--large-fb-datums", type=int, default=1030, help=">1024 forces multi-chunk posting")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary: dict = {}

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)

    # ---- capabilities: the deployment serves exactly one base model ----
    capabilities = service.get_server_capabilities()
    base_models = [m.model_name for m in capabilities.supported_models]
    log(f"server capabilities: supported_models={base_models}")
    assert len(base_models) == 1 and base_models[0], base_models
    base_model = base_models[0]
    summary["base_model"] = base_model

    client = service.create_lora_training_client(base_model=base_model, rank=16)
    info = client.get_info()
    assert info.lora_rank == 16, info
    log(f"training client ready: model_id={client.model_id} rank={info.lora_rank}")

    tokenizer = client.get_tokenizer()
    pairs = [split_prompt_completion(text) for text in CORPUS]
    built = [sft_datum(tokenizer.encode(prompt), tokenizer.encode(completion)) for prompt, completion in pairs]
    data = [datum for datum, _, _ in built]
    expected_weight_sum = sum(weight_sum for _, weight_sum, _ in built)
    expected_positions = sum(positions for _, _, positions in built)
    assert expected_positions > expected_weight_sum > 0, (expected_positions, expected_weight_sum)
    n_tokens = sum(len(d.model_input.to_ints()) for d in data)
    log(
        f"corpus: {len(data)} prompt-masked SFT datums, {n_tokens} input tokens, "
        f"{expected_weight_sum:.0f} completion positions of {expected_positions} targets"
    )

    # ---- supervised mini-loop: loss must decrease ----
    losses: list[float] = []
    t0 = time.time()
    for iteration in range(1, args.iterations + 1):
        fb_future = client.forward_backward(data, "cross_entropy")
        optim_future = client.optim_step(types.AdamParams(learning_rate=args.lr))
        fb = fb_future.result()
        optim = optim_future.result()
        loss_sum = fb.metrics["loss:sum"]
        # The SFT denominator is the CE weight sum (completion positions),
        # not unmasked_tokens:sum, which also counts the weight-0 prompt
        # (codex-0817-sft-fix §7). Guarded: weights are arbitrary floats.
        weight_sum = fb.metrics["loss_weight:sum"]
        unmasked = fb.metrics["unmasked_tokens:sum"]
        assert abs(weight_sum - expected_weight_sum) < 1e-6, (weight_sum, expected_weight_sum)
        assert abs(unmasked - expected_positions) < 1e-6, (unmasked, expected_positions)
        assert unmasked > weight_sum, "prompt masking must keep the two denominators distinct"
        per_token = loss_sum / weight_sum if weight_sum > 0 else None
        losses.append(loss_sum)
        log(
            f"iter {iteration:2d}/{args.iterations}: loss:sum={loss_sum:.3f} "
            f"per_token={per_token:.4f} grad_norm={optim.metrics.get('grad_norm')}"
        )
    train_dt = time.time() - t0
    summary["losses"] = losses
    summary["loss_weight_sum"] = expected_weight_sum
    summary["unmasked_tokens"] = expected_positions
    summary["train_seconds"] = round(train_dt, 1)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert all(b <= a * 1.02 for a, b in zip(losses, losses[1:], strict=False)), f"loss not (near-)monotone: {losses}"
    log(f"loss decreased {losses[0]:.3f} -> {losses[-1]:.3f} over {args.iterations} iterations ({train_dt:.0f}s)")

    # ---- publish + sample: the tuned adapter must speak ----
    sampling = client.save_weights_and_get_sampling_client()
    assert sampling.get_base_model() == base_model
    prompt_ids = tokenizer.encode(SAMPLE_PROMPT)
    response = sampling.sample(
        prompt=types.ModelInput.from_ints(prompt_ids),
        num_samples=2,
        sampling_params=types.SamplingParams(max_tokens=24, temperature=0.0),
    ).result()
    continuations = [tokenizer.decode(seq.tokens) for seq in response.sequences]
    for i, (seq, text) in enumerate(zip(response.sequences, continuations, strict=True)):
        log(f"sample[{i}] stop={seq.stop_reason} logprobs[:3]={[round(p, 3) for p in (seq.logprobs or [])[:3]]}")
        log(f"sample[{i}] text: {SAMPLE_PROMPT}{text!s}")
        assert seq.tokens and seq.logprobs and len(seq.logprobs) == len(seq.tokens)
    summary["sample_prompt"] = SAMPLE_PROMPT
    summary["sample_continuations"] = continuations

    # ---- save_state -> load_state_with_optimizer -> training continues ----
    path = client.save_state("mini-loop-golden").result().path
    log(f"save_state -> {path}")
    assert path.startswith("tinker://")
    client.load_state_with_optimizer(path).result()
    fb = client.forward_backward(data, "cross_entropy").result()
    client.optim_step(types.AdamParams(learning_rate=args.lr)).result()
    resumed_loss = fb.metrics["loss:sum"]
    summary["checkpoint_path"] = path
    summary["loss_after_restore"] = resumed_loss
    # The restored state is the post-loop state: its loss must match the
    # trained trajectory, not the untrained start.
    assert resumed_loss < losses[0], (resumed_loss, losses[0])
    log(f"restored from checkpoint; fb/optim after load works (loss:sum={resumed_loss:.3f})")

    # ---- large out-of-order fb: SDK chunks >MAX_CHUNK_LEN and posts the ----
    # ---- first chunk last; the backend gap-buffers and reassembles.     ----
    short = tokenizer.encode("The sea was calm.")
    big = [ce_datum(short) for _ in range(args.large_fb_datums)]
    t1 = time.time()
    result = client.forward_backward(big, "cross_entropy").result()
    client.optim_step(types.AdamParams(learning_rate=0.0)).result()  # release the dirty-grad pin
    assert len(result.loss_fn_outputs) == args.large_fb_datums, len(result.loss_fn_outputs)
    row = result.loss_fn_outputs[0]["logprobs"].tolist()
    assert len(row) == len(short) - 1
    summary["large_fb"] = {"datums": args.large_fb_datums, "seconds": round(time.time() - t1, 1)}
    log(
        f"large fb: {args.large_fb_datums} datums (multi-chunk, out-of-order) -> "
        f"{len(result.loss_fn_outputs)} rows in {summary['large_fb']['seconds']}s"
    )

    # ---- deliberate user error: channel mismatch -> typed SDK error, no hang ----
    bad = types.Datum(
        model_input=types.ModelInput.from_ints(short[:-1]),
        loss_fn_inputs={"target_tokens": short[1:], "advantages": [1.0] * (len(short) - 1)},
        # importance_sampling requires 'logprobs'; it is deliberately missing.
    )
    t2 = time.time()
    try:
        client.forward_backward([bad], "importance_sampling").result()
        raise AssertionError("channel-mismatch datum was accepted")
    except tinker.RequestFailedError as exc:
        err_dt = time.time() - t2
        summary["typed_user_error"] = {"error": str(exc)[:200], "seconds": round(err_dt, 1)}
        log(f"typed user error in {err_dt:.1f}s (no hang): {str(exc)[:120]}")
    # The rejected submission consumed its ordinal AND poisoned its gradient
    # window (#2258 §5): the window's optim_step must discard, not step.
    good = client.forward_backward(data[:2], "cross_entropy").result()
    assert len(good.loss_fn_outputs) == 2
    try:
        client.optim_step(types.AdamParams(learning_rate=args.lr)).result()
        raise AssertionError("optim_step on a poisoned window succeeded")
    except tinker.RequestFailedError as exc:
        assert "gradient window" in str(exc), exc
        summary["poisoned_optim_error"] = str(exc)[:200]
        log(f"poisoned-window optim_step failed typed: {str(exc)[:120]}")
    # The discard reset the window: the next round steps normally.
    good = client.forward_backward(data[:2], "cross_entropy").result()
    client.optim_step(types.AdamParams(learning_rate=0.0)).result()
    assert len(good.loss_fn_outputs) == 2
    log("post-error round stepped: the discard left no residue and no gap")

    summary["ok"] = True
    with open(os.path.join(args.out_dir, "mini_loop_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("=== MINI-LOOP GOLDEN ACCEPTANCE: PASS ===")


if __name__ == "__main__":
    main()
