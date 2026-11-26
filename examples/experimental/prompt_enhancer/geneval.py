"""GenEval images for a prompt enhancer: rewrite GenEval's prompts with a policy checkpoint, render every image through
SGLang Diffusion engines, and write GenEval's folder layout for its own `evaluation/evaluate_images.py` to score.

The rewrite is RePrompt's GenEval call (`evaluation/reprompt_infer_geneval.py`): its system prompt, the user's prompt and
the assistant prefix, rendered with `add_generation_prompt=True`, greedy decoding of up to 512 new tokens with the
weights in bf16, and the first `<answer>` taken (or "a photo"). `--raw` sends GenEval's prompt unchanged. Image `k` of
prompt `i` uses seed `i * 1000 + k + --seed-offset` in every arm, so images pair across arms by (prompt, seed).

Engines are plain `sglang serve` processes, one per GPU; space their ports, which SGLang derives others from (README.md):
    CUDA_VISIBLE_DEVICES=0 sglang serve --model-path /root/models/Z-Image-Turbo --port 30000 \\
        --master-port 31000 --scheduler-port 32000

    python -m examples.experimental.prompt_enhancer.geneval --policy /root/shared_data/<run>/hf/iter_0000099 \\
        --metadata geneval/prompts/evaluation_metadata.jsonl --image-urls http://127.0.0.1:30000 --out out/trained

Output: `<out>/inputs.json`, `<out>/rewrites.jsonl`, and `<out>/images/<index:05>/{metadata.jsonl,samples/<k:05>.png}`.
A rerun with the same inputs keeps the files that exist, so a stopped run resumes; a rerun with any other input (policy,
prompts, seeds, image count or sampling) is refused, so every image in a directory belongs to one cohort.
"""

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

from examples.experimental.prompt_enhancer.make_data import messages
from miles.rollout.generate_utils.image_generation import ImageGenerationClient, ImageRequest

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SEEDS_PER_PROMPT = 1000


def final_prompt(response: str) -> str:
    """RePrompt's GenEval extraction, which (unlike the training reward) spans lines and strips."""
    match = _ANSWER.search(response)
    return match.group(1).strip() if match else "a photo"


def image_seed(prompt_index: int, k: int, offset: int) -> int:
    return prompt_index * _SEEDS_PER_PROMPT + k + offset


def rewrite(policy: str, prompts: list[str], *, batch_size: int, max_new_tokens: int = 512) -> list[str]:
    # Imported here: rendering a --raw arm or an existing rewrites.jsonl needs neither torch nor a GPU.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(policy, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(policy, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    texts = [tokenizer.apply_chat_template(messages(p), tokenize=False, add_generation_prompt=True) for p in prompts]
    responses = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start : start + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        responses += tokenizer.batch_decode(out[:, batch.input_ids.shape[1] :], skip_special_tokens=True)
        print(f"rewrote {len(responses)}/{len(texts)}", flush=True)
    return responses


def bind_inputs(out: Path, inputs: dict) -> None:
    """Record the cohort's inputs in a new directory, or refuse a directory made from other inputs."""
    path = out / "inputs.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if recorded != inputs:
            raise SystemExit(f"{out} holds images made from {recorded}; give a fresh --out for {inputs}")
        return
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty and records no inputs; give a fresh --out")
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inputs, indent=2))


async def render_all(jobs: list[tuple[ImageRequest, Path]], urls: list[str], *, concurrency: int) -> None:
    clients = [ImageGenerationClient(url, timeout_secs=600) for url in urls]
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def one(i: int, request: ImageRequest, path: Path) -> None:
        nonlocal done
        async with semaphore:
            png = await clients[i % len(clients)].generate_image(request)
        path.write_bytes(png)
        done += 1
        if done % 500 == 0:
            print(f"rendered {done}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(i, request, path) for i, (request, path) in enumerate(jobs)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", help="HF checkpoint of the rewriter; omit with --raw.")
    parser.add_argument("--raw", action="store_true", help="Render GenEval's prompts unchanged.")
    parser.add_argument("--metadata", type=Path, required=True, help="GenEval's evaluation_metadata.jsonl.")
    parser.add_argument("--image-urls", required=True, help="Comma-separated diffusion engine base URLs.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images-per-prompt", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=9)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64, help="Prompts per rewrite batch.")
    parser.add_argument("--concurrency", type=int, default=64, help="Images in flight across all engines.")
    args = parser.parse_args()
    assert args.raw != bool(args.policy), "give exactly one of --policy and --raw"
    assert args.images_per_prompt <= _SEEDS_PER_PROMPT

    metadata_text = args.metadata.read_text()
    metadatas = [json.loads(line) for line in metadata_text.splitlines() if line.strip()]
    prompts = [m["prompt"] for m in metadatas]
    policy = str(Path(args.policy).resolve()) if args.policy and Path(args.policy).exists() else args.policy
    bind_inputs(
        args.out,
        dict(
            policy=None if args.raw else policy,
            metadata_sha256=hashlib.sha256(metadata_text.encode()).hexdigest(),
            images_per_prompt=args.images_per_prompt,
            seed_offset=args.seed_offset,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ),
    )
    rewrites_path = args.out / "rewrites.jsonl"
    if not rewrites_path.exists():
        responses = [""] * len(prompts) if args.raw else rewrite(args.policy, prompts, batch_size=args.batch_size)
        finals = prompts if args.raw else [final_prompt(r) for r in responses]
        rows = [{"response": r, "final_prompt": f} for r, f in zip(responses, finals, strict=True)]
        rewrites_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rows = [json.loads(line) for line in rewrites_path.read_text().splitlines()]
    assert len(rows) == len(metadatas), f"{rewrites_path} holds {len(rows)} rows for {len(metadatas)} prompts"

    jobs = []
    for i, (metadata, row) in enumerate(zip(metadatas, rows, strict=True)):
        folder = args.out / "images" / f"{i:05}"
        (folder / "samples").mkdir(parents=True, exist_ok=True)
        (folder / "metadata.jsonl").write_text(json.dumps(metadata | row))
        for k in range(args.images_per_prompt):
            path = folder / "samples" / f"{k:05}.png"
            if not path.exists():
                request = ImageRequest(
                    prompt=row["final_prompt"],
                    seed=image_seed(i, k, args.seed_offset),
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                )
                jobs.append((request, path))
    asyncio.run(render_all(jobs, args.image_urls.split(","), concurrency=args.concurrency))
    print(f"GENEVAL_IMAGES_DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
