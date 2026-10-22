"""Manual GEO3K RL validation through the Tinker SDK and miles gateway.

From the repository root, with four GPUs and the runtime dependencies installed:
    python -m examples.multi_lora.validate_geo3k
Or use the pinned Modal environment:
    modal run examples/multi_lora/modal_geo3k.py --output-dir ./geo3k-results

Uses 64 fixed training problems, 32 held-out validation problems, groups of four,
and up to 48 batches to obtain 16 updates with nonconstant rewards. Metrics and
rollouts are saved to MILES_GEO3K_OUTPUT_DIR. Accuracy improvement is measured,
not asserted: this small run validates the RL path, not statistical convergence.
"""

import io
import json
import math
import os
import random
import shlex
import signal
import subprocess
import tempfile
import time
import urllib.request
from contextlib import contextmanager, suppress
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor

import tinker
from miles.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from miles.utils.http_utils import is_port_available

DATASET = "hiyouga/geometry3k"
DATASET_REVISION = "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7"
SEED = 2026
GROUP_SIZE = 4
BATCH_SIZE = 4
UPDATES = 16
MAX_TOKENS = 1536


BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"


def _image_prompt(processor, images, text):
    rendered = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [*({"type": "image", "image": image} for image in images), {"type": "text", "text": text}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    parts = rendered.split(processor.image_token)
    assert len(parts) == len(images) + 1
    chunks = []
    for prefix, image in zip(parts[:-1], images, strict=True):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_ids = processor(text=processor.image_token, images=[image], add_special_tokens=False)["input_ids"][0]
        chunks.extend(
            [
                tinker.types.EncodedTextChunk(tokens=processor.tokenizer.encode(prefix, add_special_tokens=False)),
                tinker.types.ImageChunk(data=buffer.getvalue(), format="png", expected_tokens=len(image_ids)),
            ]
        )
    chunks.append(
        tinker.types.EncodedTextChunk(tokens=processor.tokenizer.encode(parts[-1], add_special_tokens=False))
    )
    return tinker.ModelInput(chunks=chunks)


def _qwen3_vl_serve_args(checkpoint):
    megatron_path = os.environ.get("MILES_MEGATRON_PATH", "/root/Megatron-LM")
    return (
        f"--hf-checkpoint {shlex.quote(checkpoint)} --model-type qwen3-vl-30B-A3B "
        f"--megatron-path {shlex.quote(megatron_path)} "
        "--num-gpus-per-node 4 --actor-num-gpus 2 --rollout-num-gpus 2 "
        "--tp 2 --ep 2 --n-adapters 1 --target-modules attn "
        f"--extra-args '--tinker-base-model {BASE_MODEL} "
        "--sglang-context-length 4096 --sglang-cuda-graph-backend-decode disabled'"
    )


GATEWAY_PORT = 10613
SERVE_TIMEOUT_S = 1200


def _wait_for_gateway(server: subprocess.Popen) -> None:
    deadline = time.time() + SERVE_TIMEOUT_S
    url = f"http://127.0.0.1:{GATEWAY_PORT}/api/v1/healthz"
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"gateway exited during startup with code {server.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"gateway not serving after {SERVE_TIMEOUT_S}s")


@contextmanager
def _running_gateway(checkpoint):
    if not is_port_available(GATEWAY_PORT):
        raise RuntimeError(f"port {GATEWAY_PORT} already has a listener; refusing to reuse a gateway not started here")
    serve_cmd = (
        "python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve "
        f"--lora-rank 8 --lora-alpha 16 {_qwen3_vl_serve_args(checkpoint)}"
    )
    server = subprocess.Popen(["bash", "-c", f"exec {serve_cmd}"], start_new_session=True)
    try:
        _wait_for_gateway(server)
        yield f"http://127.0.0.1:{GATEWAY_PORT}"
    finally:
        try:
            with suppress(ProcessLookupError):
                server.terminate()
            returncode = server.wait(timeout=180)
            if returncode not in (0, 128 + signal.SIGTERM):
                raise RuntimeError(f"gateway launcher failed during shutdown with code {returncode}")
        except BaseException:
            # A stuck launcher cannot finish its own Ray cleanup.
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=30)
            subprocess.run(["ray", "stop", "--force"], check=True, timeout=120)
            raise


def _record(output_dir, event):
    with (output_dir / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    if event["kind"] != "rollout":
        print("GEO3K " + json.dumps(event), flush=True)


def _load_examples(split, count):
    dataset = load_dataset(DATASET, revision=DATASET_REVISION, split=split)
    indices = random.Random(SEED).sample(range(len(dataset)), count)
    return [{**dataset[index], "id": f"{split}:{index}"} for index in indices]


def _prompt(processor, example, *, blank=False):
    images = [image.convert("RGB") for image in example["images"]]
    if blank:
        images = [Image.new("RGB", image.size, "white") for image in images]
    question = example["problem"].replace("<image>", "").strip()
    return _image_prompt(processor, images, question + "\nReason briefly and put your final answer in \\boxed{}.")


def _reward(answer, target):
    extracted = extract_answer(answer)
    return int(
        extracted is not None and (grade_answer_mathd(extracted, target) or grade_answer_sympy(extracted, target))
    )


def _sample(sampler, processor, examples, prompts, *, count, temperature, seed, phase, output_dir):
    futures = [
        sampler.sample(
            prompt,
            num_samples=count,
            sampling_params=tinker.SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                seed=seed + index * count,
            ),
        )
        for index, prompt in enumerate(prompts)
    ]
    groups = []
    for example, prompt, future in zip(examples, prompts, futures, strict=True):
        group = []
        for sequence in future.result(timeout=600).sequences:
            assert sequence.tokens and sequence.logprobs is not None
            assert len(sequence.tokens) == len(sequence.logprobs)
            assert all(math.isfinite(x) for x in sequence.logprobs)
            answer = processor.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            reward = _reward(answer, example["answer"])
            _record(
                output_dir,
                {
                    "kind": "rollout",
                    "phase": phase,
                    "id": example["id"],
                    "problem": example["problem"],
                    "target": example["answer"],
                    "answer": answer,
                    "reward": reward,
                    "stop_reason": sequence.stop_reason,
                    "tokens": sequence.tokens,
                    "logprobs": sequence.logprobs,
                },
            )
            group.append({"prompt": prompt, "sequence": sequence, "reward": reward})
        groups.append(group)
    rewards = [item["reward"] for group in groups for item in group]
    sequences = [item["sequence"] for group in groups for item in group]
    _record(
        output_dir,
        {
            "kind": "sampling",
            "phase": phase,
            "reward": float(np.mean(rewards)),
            "samples": len(rewards),
            "truncated": sum(sequence.stop_reason == "length" for sequence in sequences),
            "mean_completion_tokens": float(np.mean([len(sequence.tokens) for sequence in sequences])),
        },
    )
    return groups


def _rl_datums(groups):
    selected = []
    for group in groups:
        mean = sum(item["reward"] for item in group) / len(group)
        selected.extend((item, item["reward"] - mean) for item in group if item["reward"] != mean)
    # Tinker accumulates sums. Normalize by the number of active completion tokens.
    normalizer = sum(len(item["sequence"].tokens) for item, _ in selected)
    datums = []
    for item, advantage in selected:
        prompt, sequence = item["prompt"], item["sequence"]
        prefix = prompt.length - 1
        datums.append(
            tinker.Datum(
                model_input=prompt.append(tinker.types.EncodedTextChunk(tokens=sequence.tokens[:-1])),
                loss_fn_inputs={
                    "target_tokens": [0] * prefix + sequence.tokens,
                    "logprobs": [0.0] * prefix + sequence.logprobs,
                    "advantages": [0.0] * prefix + [advantage / normalizer] * len(sequence.tokens),
                },
            )
        )
    return datums


def _logprob_check(result, datums):
    differences = []
    assert len(result.loss_fn_outputs) == len(datums)
    for output, datum in zip(result.loss_fn_outputs, datums, strict=True):
        actual = np.asarray(output["logprobs"].data)
        expected = np.asarray(datum.loss_fn_inputs["logprobs"].data)
        active = np.asarray(datum.loss_fn_inputs["advantages"].data) != 0
        assert len(actual) == datum.model_input.length
        assert np.isfinite(actual).all()
        differences.extend((actual[active] - expected[active]).tolist())
    differences = np.asarray(differences)
    return {
        "logprob_mae": float(np.abs(differences).mean()),
        "logprob_p99_abs": float(np.quantile(np.abs(differences), 0.99)),
        "ratio_mean": float(np.exp(differences).mean()),
    }


def _publish(service, trainer, step, output_dir):
    path = trainer.save_weights_for_sampler(name=f"geo3k-{step:03d}").result(timeout=300).path
    _record(output_dir, {"kind": "checkpoint", "step": step, "path": path})
    return service.create_sampling_client(model_path=path)


def _evaluate(sampler, processor, examples, prompts, phase, output_dir):
    groups = _sample(
        sampler,
        processor,
        examples,
        prompts,
        count=1,
        temperature=0.0,
        seed=SEED,
        phase=phase,
        output_dir=output_dir,
    )
    return sum(group[0]["reward"] for group in groups) / len(groups)


def _train(service, trainer, sampler, processor, examples, prompts, output_dir):
    updates, history = 0, []
    for batch_index in range(48):
        start = (batch_index * BATCH_SIZE) % len(examples)
        group = _sample(
            sampler,
            processor,
            examples[start : start + BATCH_SIZE],
            prompts[start : start + BATCH_SIZE],
            count=GROUP_SIZE,
            temperature=1.0,
            seed=SEED + batch_index * 100,
            phase=f"train-{batch_index:03d}",
            output_dir=output_dir,
        )
        datums = _rl_datums(group)
        if not datums:
            _record(output_dir, {"kind": "skipped_constant_rewards", "batch": batch_index})
            continue
        before = trainer.forward(datums, loss_fn="importance_sampling").result(timeout=600)
        metrics = _logprob_check(before, datums)
        _record(output_dir, {"kind": "parity", "step": updates, **metrics})
        # BF16 kernels differ between Megatron and SGLang; large systematic drift
        # would invalidate on-policy RL even if forward/backward did not crash.
        assert metrics["logprob_mae"] < 0.15, metrics
        backward = trainer.forward_backward(datums, loss_fn="importance_sampling").result(timeout=600)
        assert math.isfinite(backward.metrics["loss:sum"])
        _logprob_check(backward, datums)
        optim = trainer.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=600)
        assert math.isfinite(optim.metrics["grad_norm"]) and optim.metrics["grad_norm"] > 0, optim.metrics
        updates += 1
        record = {
            "kind": "update",
            "step": updates,
            "batch": batch_index,
            "datums": len(datums),
            "reward": float(np.mean([item["reward"] for samples in group for item in samples])),
            "loss": backward.metrics["loss:sum"],
            "grad_norm": optim.metrics["grad_norm"],
            **metrics,
        }
        history.append(record)
        _record(output_dir, record)
        sampler = _publish(service, trainer, updates, output_dir)
        if updates == UPDATES:
            break
    assert updates == UPDATES, f"only {updates} nonconstant-reward updates in 48 batches"
    return sampler, history


def main():
    output_dir = Path(os.environ.get("MILES_GEO3K_OUTPUT_DIR") or tempfile.mkdtemp(prefix="miles-geo3k-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = os.environ["MILES_MULTIMODAL_CHECKPOINT"]
    processor = AutoProcessor.from_pretrained(checkpoint)
    train, validation = _load_examples("train", 64), _load_examples("validation", 32)
    train_prompts = [_prompt(processor, row) for row in train]
    eval_prompts = [_prompt(processor, row) for row in validation]
    blank_prompts = [_prompt(processor, row, blank=True) for row in validation]
    metadata = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "model": BASE_MODEL,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "train_ids": [x["id"] for x in train],
        "validation_ids": [x["id"] for x in validation],
        "group_size": GROUP_SIZE,
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        "learning_rate": 1e-4,
    }
    _record(output_dir, {"kind": "config", **metadata})
    with _running_gateway(checkpoint) as base_url:
        service = tinker.ServiceClient(base_url=base_url, api_key="tml-geo3k-validation")
        trainer = service.create_lora_training_client(
            base_model=BASE_MODEL, rank=8, train_mlp=False, train_unembed=False
        )
        sampler = _publish(service, trainer, 0, output_dir)
        before = _evaluate(sampler, processor, validation, eval_prompts, "eval-before", output_dir)
        blank_before = _evaluate(sampler, processor, validation, blank_prompts, "blank-before", output_dir)
        sampler, updates = _train(service, trainer, sampler, processor, train, train_prompts, output_dir)
        after = _evaluate(sampler, processor, validation, eval_prompts, "eval-after", output_dir)
        blank_after = _evaluate(sampler, processor, validation, blank_prompts, "blank-after", output_dir)
        summary = {
            **metadata,
            "updates": updates,
            "accuracy_before": before,
            "accuracy_after": after,
            "blank_accuracy_before": blank_before,
            "blank_accuracy_after": blank_after,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        _record(
            output_dir,
            {
                "kind": "complete",
                "updates": len(updates),
                "accuracy_before": before,
                "accuracy_after": after,
                "blank_accuracy_before": blank_before,
                "blank_accuracy_after": blank_after,
            },
        )


if __name__ == "__main__":
    if "MILES_MULTIMODAL_CHECKPOINT" not in os.environ:
        os.environ["MILES_MULTIMODAL_CHECKPOINT"] = snapshot_download(BASE_MODEL, revision=MODEL_REVISION)
    main()
