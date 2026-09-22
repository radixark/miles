"""Prepare and submit a reproducible Qwen FP8 reasoning smoke test."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from datasets import load_dataset
from ray.job_submission import JobSubmissionClient
from tap import Tap


class Args(Tap):
    phase: str
    root: str = "/scratch/260922-de0b2498"


def main() -> None:
    args = Args().parse_args()
    root = Path(args.root)
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    for name in ("checkpoints", "wandb", "traces"):
        (root / name).mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train", cache_dir=str(root / "hf-cache"))
        rows = []
        seen = set()
        for row in dataset:
            key = row["extra_info"]["index"]
            if key in seen:
                continue
            seen.add(key)
            rows.append({"prompt": row["prompt"], "label": row["reward_model"]["ground_truth"], "metadata": {"source": "DAPO-Math-17k", "id": key}})
            if len(rows) == 512:
                break
        output = root / "prompts.jsonl"
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = {"source": "BytedTsinghua-SIA/DAPO-Math-17k", "rows": len(rows), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "dataset_fingerprint": dataset._fingerprint}
        (root / "data-manifest.json").write_text(json.dumps(manifest, indent=2))
        print(manifest)
    elif args.phase == "submit":
        repos = {"miles": root / "miles", "sglang": Path("/sgl-workspace/sglang"), "megatron-lm": Path("/root/Megatron-LM")}
        sources = {}
        for name, repo in repos.items():
            assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip(), f"Dirty {repo}"
            sources[name] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        (root / "source-pins.json").write_text(json.dumps(sources, indent=2))
        assert (root / "snapshots-confirmed.json").is_file()
        assert json.loads((root / "snapshots-confirmed.json").read_text()) == sources
        client = JobSubmissionClient("http://127.0.0.1:8265")
        job = client.submit_job(submission_id="fp8-" + config["run_id"], entrypoint=shlex.join(config["command"]), runtime_env=config["runtime_env"])
        print("SUBMITTED", job)
    else:
        raise ValueError(args.phase)


if __name__ == "__main__":
    main()
