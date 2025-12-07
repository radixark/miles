"""
Plotting utilities for performance visualization
"""
import os
import json
import glob
from typing import List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def scan_best_metadata_files(
    run_path: str,
    save_cache: bool = True,
    cache_filename: str = "performance_data.json"
) -> List[Tuple[int, float]]:
    """
    Scan best_metadata_step_*.json files in a run directory

    Args:
        run_path: Path to run directory (e.g., output_dir or {save_path}/{run_name})
        save_cache: Whether to save extracted data to cache file
        cache_filename: Name of cache file to save

    Returns:
        List of (training_step, score) tuples sorted by training_step
    """
    pattern = os.path.join(run_path, "best_program", "best_metadata_step_*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        return []

    data_points = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            metadata = json.load(f)

        training_step = metadata.get("training_step")
        score = metadata.get("metrics", {}).get("combined_score")

        if training_step is not None and score is not None:
            data_points.append((training_step, score))

    data_points.sort(key=lambda x: x[0])

    # Save cache file for later processing
    if save_cache and data_points:
        cache_dir = os.path.join(run_path, "visualizations")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_filename)

        cache_data = {
            "data_points": [[step, score] for step, score in data_points],
            "num_points": len(data_points),
            "min_step": min(step for step, _ in data_points),
            "max_step": max(step for step, _ in data_points),
            "min_score": min(score for _, score in data_points),
            "max_score": max(score for _, score in data_points)
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

    return data_points


def load_performance_data_from_cache(
    run_path: str,
    cache_filename: str = "performance_data.json"
) -> Optional[List[Tuple[int, float]]]:
    """
    Load performance data from cache file

    Args:
        run_path: Path to run directory
        cache_filename: Name of cache file

    Returns:
        List of (training_step, score) tuples, or None if cache not found
    """
    cache_path = os.path.join(run_path, "visualizations", cache_filename)
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, 'r') as f:
        cache_data = json.load(f)

    data_points = [tuple(point) for point in cache_data["data_points"]]
    return data_points


def plot_single_run_curve(
    data_points: List[Tuple[int, float]],
    output_path: str,
    title: str = "Performance Progression",
    xlabel: str = "Training Step",
    ylabel: str = "Best Combined Score",
    figsize: Tuple[int, int] = (12, 6),
    color: str = "#2E86AB",
    annotate_peak: bool = True
):
    """
    Plot performance curve for a single run

    Args:
        data_points: List of (training_step, score) tuples
        output_path: Path to save the figure (should end with .jpg or .png)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        color: Line color
        annotate_peak: Whether to annotate the peak score
    """
    if not data_points:
        raise ValueError("No data points provided for plotting")

    steps, scores = zip(*data_points)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot line
    ax.plot(steps, scores, marker='o', linestyle='-', linewidth=2, markersize=4, color=color)

    # Annotate max score
    if annotate_peak:
        max_score = max(scores)
        max_step = steps[scores.index(max_score)]
        ax.annotate(
            f'Peak: {max_score:.4f}',
            xy=(max_step, max_score),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            color='#A23B72',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#A23B72', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5)
        )

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
