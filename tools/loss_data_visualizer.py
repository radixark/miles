"""
Loss data visualization tool for analyzing loss data (.pt files) saved during miles-sunrise RL training.

CLI subcommands:

1. info - View data file information
   uv run python tools/loss_data_visualizer.py info <data_path> --rollout-id 0 --step-id 0 --rank 0

2. distribution - Visualize loss distribution (histograms in linear + log scale)
   uv run python tools/loss_data_visualizer.py distribution <data_path> --rollout-id 0 --step-id 0 --save-path metrics_distribution.png

3. train_infer_diff - Compare training vs inference log prob differences
   uv run python tools/loss_data_visualizer.py train_infer_diff <data_path> --rollout-id 0 --step-id 0 --save-path train_infer_diff.png

Data file format:
- Filename: {rollout_id}_{step_id}_{rank}.pt
- Structure: {"microbatches": [{"microbatch_offset": int, "loss_data": {...}, "batch": {...}}, ...]}
"""
import glob
import math
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

app = typer.Typer()


def _find_files(data_path: Path, rollout_id: int, step_id: int):
    pattern = f"{rollout_id}_{step_id}_*.pt"
    files = glob.glob(str(data_path / pattern))
    ranks = []
    for f in files:
        match = re.search(rf"{rollout_id}_{step_id}_(\d+)\.pt$", f)
        if match:
            ranks.append(int(match.group(1)))
    return sorted(ranks)


def _get_rank(rank: Optional[int], data_path: Path, rollout_id: int, step_id: int) -> int:
    ranks = _find_files(data_path, rollout_id, step_id)
    print(f"Available ranks: {ranks}")
    if not ranks:
        raise typer.Exit(1)
    if rank is None:
        rank = min(ranks)
        print(f"Auto-selected rank: {rank}")
    return rank


def _load_loss_data(data_path: Path, rollout_id: int, step_id: int, rank: int) -> dict:
    file_path = data_path / f"{rollout_id}_{step_id}_{rank}.pt"
    return torch.load(file_path, map_location="cpu")


def _aggregate_microbatches(data: dict, microbatch_id: Optional[int] = None) -> dict:
    microbatches = data["microbatches"]
    if microbatch_id is not None:
        microbatches = [mb for mb in microbatches if mb["microbatch_offset"] == microbatch_id]

    merged = {}
    for key in microbatches[0]["loss_data"].keys():
        merged[key] = torch.cat([mb["loss_data"][key].flatten() for mb in microbatches])
    return merged


def _format_field(v) -> str:
    if isinstance(v, torch.Tensor):
        return f"{list(v.shape)} {v.dtype}"
    if isinstance(v, list):
        return f"list[{len(v)}]" + (f" of {_format_field(v[0])}" if v and isinstance(v[0], torch.Tensor) else "")
    return type(v).__name__


@app.command()
def info(
    data_path: str,
    rollout_id: int = 0,
    step_id: int = 0,
    rank: Optional[int] = None,
):
    path = Path(data_path)
    rank = _get_rank(rank, path, rollout_id, step_id)
    data = _load_loss_data(path, rollout_id, step_id, rank)

    print(f"\nFile: {rollout_id}_{step_id}_{rank}.pt")
    print(f"Number of microbatches: {len(data['microbatches'])}")
    print(f"Microbatch offsets: {[mb['microbatch_offset'] for mb in data['microbatches']]}")

    mb = data["microbatches"][0]
    print(f"\nloss_data:")
    for k, v in mb["loss_data"].items():
        print(f"  {k}: {_format_field(v)}")
    print(f"\nbatch:")
    for k, v in mb["batch"].items():
        print(f"  {k}: {_format_field(v)}")


def _visualize_distribution(loss_data: dict, save_path: Optional[str] = None):
    metrics = {k: v.flatten().float().numpy() for k, v in loss_data.items()}
    n_metrics = len(metrics)
    cols = 3
    rows = math.ceil(n_metrics / cols)

    fig, axes = plt.subplots(rows * 2, cols, figsize=(5 * cols, 4 * rows * 2))
    axes = axes.flatten()

    for i, (name, data) in enumerate(metrics.items()):
        title = f"{name}\nn={len(data)}, min={data.min():.2e}, max={data.max():.2e}, mean={data.mean():.2e}"

        axes[i].hist(data, bins=80, edgecolor='black', alpha=0.7)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Token Count")
        axes[i].set_title(title)

        ax_log = axes[i + rows * cols]
        positive_data = data[data > 0]
        if len(positive_data) > 0:
            log_bins = np.logspace(np.log10(positive_data.min()), np.log10(positive_data.max()), 80)
            ax_log.hist(positive_data, bins=log_bins, edgecolor='black', alpha=0.7)
            ax_log.set_xscale('log')
        ax_log.set_xlabel("Value (log)")
        ax_log.set_ylabel("Token Count")
        ax_log.set_title(f"{title} [log]")

    for i in range(n_metrics, rows * cols):
        axes[i].axis('off')
        axes[i + rows * cols].axis('off')

    plt.tight_layout(w_pad=10)
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


@app.command()
def distribution(
    data_path: str,
    rollout_id: int = 0,
    step_id: int = 0,
    microbatch_id: Optional[int] = None,
    rank: Optional[int] = None,
    save_path: Optional[str] = "metrics_distribution.png",
):
    path = Path(data_path)
    rank = _get_rank(rank, path, rollout_id, step_id)
    data = _load_loss_data(path, rollout_id, step_id, rank)
    loss_data = _aggregate_microbatches(data, microbatch_id)
    _visualize_distribution(loss_data, save_path)


def _extract_samples(data: dict) -> list[dict]:
    """Extract per-sample data from all microbatches."""
    samples = []
    for mb in data["microbatches"]:
        batch = mb["batch"]
        n = len(batch["unconcat_tokens"])
        for i in range(n):
            prompt_len = batch["total_lengths"][i] - batch["response_lengths"][i]
            samples.append({
                "prompt": tuple(batch["unconcat_tokens"][i][:prompt_len].tolist()),
                "log_probs": batch["log_probs"][i].float().numpy(),
                "rollout_log_probs": batch["rollout_log_probs"][i].float().numpy(),
            })
    return samples


def _group_by_prompt(samples: list[dict]) -> dict[tuple, list[dict]]:
    """Group samples by prompt tokens."""
    from collections import defaultdict
    groups = defaultdict(list)
    for s in samples:
        groups[s["prompt"]].append(s)
    return dict(groups)


@app.command()
def train_infer_diff(
    data_path: str,
    rollout_id: int = 0,
    step_id: int = 0,
    rank: Optional[int] = None,
    save_path: Optional[str] = "train_infer_diff.png",
    max_groups: int = 6,
):
    path = Path(data_path)
    rank = _get_rank(rank, path, rollout_id, step_id)
    data = _load_loss_data(path, rollout_id, step_id, rank)

    samples = _extract_samples(data)
    groups = _group_by_prompt(samples)
    print(f"Found {len(samples)} samples in {len(groups)} prompt groups")

    group_items = list(groups.items())[:max_groups]
    n_groups = len(group_items)
    cols = min(3, n_groups)
    rows = math.ceil(n_groups / cols)

    fig = plt.figure(figsize=(6 * cols, 4 * rows + 14))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [3.5])

    for idx, (prompt, group) in enumerate(group_items):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        for i, s in enumerate(group):
            diff = np.abs(s["log_probs"] - s["rollout_log_probs"])
            ax.plot(diff, alpha=0.7, label=f"sample {i}")
        ax.set_xlabel("Token Index")
        ax.set_ylabel("|train_logp - infer_logp|")
        ax.set_title(f"Group {idx} ({len(group)} samples)")
        ax.legend(fontsize=8)

    all_train_logp = np.concatenate([s["log_probs"] for s in samples])
    all_rollout_logp = np.concatenate([s["rollout_log_probs"] for s in samples])

    ax_scatter = fig.add_subplot(gs[rows, :])
    ax_scatter.scatter(all_train_logp, all_rollout_logp, alpha=0.3, s=1)
    min_val = min(all_train_logp.min(), all_rollout_logp.min())
    max_val = max(all_train_logp.max(), all_rollout_logp.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='y=x')
    ax_scatter.set_xlabel("Train log prob")
    ax_scatter.set_ylabel("Rollout log prob")
    ax_scatter.set_title(f"Train vs Rollout log prob (n={len(all_train_logp)})")
    ax_scatter.legend()
    ax_scatter.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    app()
