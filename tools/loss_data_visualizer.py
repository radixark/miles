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
        print("No data files found.")
        raise typer.Exit(1)
    if rank is None:
        rank = min(ranks)
        print(f"Auto-selected rank: {rank}")
    else:
        assert rank in ranks, f"Rank {rank} not in available ranks: {ranks}"
    return rank


def _load_loss_data(data_path: Path, rollout_id: int, step_id: int, rank: int) -> dict:
    file_path = data_path / f"{rollout_id}_{step_id}_{rank}.pt"
    return torch.load(file_path, map_location="cuda" if torch.cuda.is_available() else "cpu")


def _aggregate_microbatches(data: dict, microbatch_id: Optional[int] = None) -> dict:
    microbatches = data["microbatches"]
    if microbatch_id is not None:
        microbatches = [mb for mb in microbatches if mb["microbatch_offset"] == microbatch_id]
        if not microbatches:
            print(f"Microbatch {microbatch_id} not found.")
            raise typer.Exit(1)

    all_loss_data = [mb["loss_data"] for mb in microbatches]
    merged = {}
    for key in all_loss_data[0].keys():
        tensors = [d[key] for d in all_loss_data if isinstance(d.get(key), torch.Tensor)]
        if tensors:
            merged[key] = torch.cat([t.flatten() for t in tensors])
    return merged


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

    if data["microbatches"]:
        print(f"\nFields in loss_data: {list(data['microbatches'][0]['loss_data'].keys())}")
        print(f"Fields in batch: {list(data['microbatches'][0]['batch'].keys())}")


def _visualize_distribution(loss_data: dict, save_path: Optional[str] = None):
    metrics = {k: v.flatten().float().cpu().numpy() for k, v in loss_data.items() if isinstance(v, torch.Tensor)}
    n_metrics = len(metrics)
    cols = 3
    rows = math.ceil(n_metrics / cols)

    fig, axes = plt.subplots(rows * 2, cols, figsize=(5 * cols, 4 * rows * 2))
    axes = axes.flatten()

    for i, (name, data) in enumerate(metrics.items()):
        total = len(data)
        vmin, vmax, vmean = data.min(), data.max(), data.mean()
        title = f"{name}\nn={total}, min={vmin:.2e}, max={vmax:.2e}, mean={vmean:.2e}"

        ax = axes[i]
        ax.hist(data, bins=80, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Value")
        ax.set_ylabel("Token Count")
        ax.set_title(title)

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


if __name__ == "__main__":
    app()
