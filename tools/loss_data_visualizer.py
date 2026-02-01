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


def _find_files(data_path: Path, rollout_id: int, step_id: int, microbatch_id: Optional[int] = None):
    pattern = f"{rollout_id}_{step_id}_*.pt" if microbatch_id is None else f"{rollout_id}_{step_id}_{microbatch_id}_*.pt"
    files = glob.glob(str(data_path / pattern))
    result = {}
    for f in files:
        match = re.search(rf"{rollout_id}_{step_id}_(\d+)_(\d+)\.pt$", f)
        if match:
            mb, rk = int(match.group(1)), int(match.group(2))
            if microbatch_id is None or mb == microbatch_id:
                result.setdefault(mb, []).append(rk)
    return result


def _get_rank(rank: int, data_path: Path, rollout_id: int, step_id: int, microbatch_id: int) -> int:
    files = _find_files(data_path, rollout_id, step_id, microbatch_id)
    ranks = files.get(microbatch_id, [])
    print(f"Available ranks: {sorted(ranks)}")
    if not ranks:
        print("No data files found.")
        raise typer.Exit(1)
    if rank is None:
        rank = min(ranks)
        print(f"Auto-selected rank: {rank}")
    else:
        assert rank in ranks, f"Rank {rank} not in available ranks: {ranks}"
    return rank


def _load_loss_data(data_path: Path, rollout_id: int, step_id: int, microbatch_id: int, rank: int) -> dict:
    file_path = data_path / f"{rollout_id}_{step_id}_{microbatch_id}_{rank}.pt"
    return torch.load(file_path, map_location="cuda" if torch.cuda.is_available() else "cpu")


@app.command()
def info(
    data_path: str,
    rollout_id: int = 0,
    step_id: int = 0,
    microbatch_id: int = 0,
    rank: Optional[int] = None,
):
    path = Path(data_path)
    rank = _get_rank(rank, path, rollout_id, step_id, microbatch_id)
    data = _load_loss_data(path, rollout_id, step_id, microbatch_id, rank)
    
    print(f"\nFields in loss_data:")
    print(data['loss_data'].keys())


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
        
        # Linear scale
        ax = axes[i]
        ax.hist(data, bins=80, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Value")
        ax.set_ylabel("Token Count")
        ax.set_title(title)
        
        # Log scale
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


def _aggregate_loss_data(data_path: Path, rollout_id: int, step_id: int) -> dict:
    files = _find_files(data_path, rollout_id, step_id)
    print(f"Found microbatches: {sorted(files.keys())}")
    all_data = []
    for mb in sorted(files.keys()):
        rk = min(files[mb])
        all_data.append(_load_loss_data(data_path, rollout_id, step_id, mb, rk)['loss_data'])
    merged = {}
    for key in all_data[0].keys():
        tensors = [d[key] for d in all_data if isinstance(d.get(key), torch.Tensor)]
        if tensors:
            merged[key] = torch.cat([t.flatten() for t in tensors])
    return merged


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
    if microbatch_id is None:
        loss_data = _aggregate_loss_data(path, rollout_id, step_id)
    else:
        rank = _get_rank(rank, path, rollout_id, step_id, microbatch_id)
        loss_data = _load_loss_data(path, rollout_id, step_id, microbatch_id, rank)['loss_data']
    _visualize_distribution(loss_data, save_path)


if __name__ == "__main__":
    app()

