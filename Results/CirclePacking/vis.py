import numpy as np
import json
import argparse
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load circle packing data from JSON"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    assert len(data) > 0, "Loaded data is empty."
    return data


def _convert_to_circles_format(centers: List[List[float]], radii: List[float]) -> List[List[float]]:
    """Convert centers and radii to circles format [[x,y,r], ...]"""
    assert len(centers) == len(radii), f"Mismatch: {len(centers)} centers vs {len(radii)} radii"
    return [[float(c[0]), float(c[1]), float(r)] for c, r in zip(centers, radii)]


def _normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert entry to circles format if needed. Auto-detects and converts from [centers, radii] format."""
    name = entry.get('name', 'unknown')
    data = entry.get('list', [])

    if not data:
        return {'name': name, 'list': []}

    # Already in circles format: [[x,y,r], [x,y,r], ...]
    if isinstance(data[0], list) and len(data[0]) == 3:
        return entry

    # Convert from [centers, radii] format: [[x,y],...], [r,...]
    if len(data) == 2 and isinstance(data[0], list) and isinstance(data[1], list):
        centers, radii = data
        if len(centers) > 0 and isinstance(centers[0], list) and len(centers[0]) == 2:
            circles = _convert_to_circles_format(centers, radii)
            return {'name': name, 'list': circles}

    # If we get here, data is already in correct format or empty
    return entry




def verify_circles(circles: np.ndarray, tolerance: float = 0) -> tuple[bool, str]:
    """Checks that the circles are disjoint and lie inside a unit square.

        Args:
        circles: A numpy array of shape (num_circles, 3), where each row is
            of the form (x, y, radius), specifying a circle.
        tolerance: Small value to account for floating point precision errors.

        Returns:
        (is_valid, error_message): Boolean indicating validity, and error message if invalid.
    """
    # Check pairwise disjointness with tolerance for floating point errors.
    for circle1, circle2 in itertools.combinations(circles, 2):
        center_distance = np.sqrt((circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2)
        radii_sum = circle1[2] + circle2[2]
        if center_distance < radii_sum - tolerance:
            return False, f"Circles NOT disjoint: distance={center_distance:.10f}, radii_sum={radii_sum:.10f}"

    # Check all circles lie inside the unit square [0,1]x[0,1].
    for circle in circles:
        if not (0 <= min(circle[0], circle[1]) - circle[2] + tolerance and max(circle[0],circle[1]) + circle[2] - tolerance <= 1):
            return False, f"Circle NOT inside unit square: {circle}"

    return True, ""


def plot_circles(circles: np.ndarray, title: str = "", save_path: str = None):
    """Plots the circles."""
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')  # Make axes scaled equally.

    # Draw unit square boundary.
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Draw the circles.
    for idx, circle in enumerate(circles):
        circ = patches.Circle((circle[0], circle[1]), circle[2], edgecolor='blue', facecolor='skyblue', alpha=0.5)
        ax.add_patch(circ)
        # Add index number in the center of each circle
        ax.text(circle[0], circle[1], str(idx), ha='center', va='center', fontsize=14, fontweight='bold')

    if title:
        plt.title(title)
    else:
        plt.title(f'Circle Packing: {len(circles)} circles, sum_radii = {np.sum(circles[:, 2]):.10f}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + ".png", dpi=100, bbox_inches='tight')
        # plt.savefig(save_path + ".pdf", bbox_inches='tight')
        print(f"    Saved: {save_path}.png/pdf")
    else:
        plt.show()

    plt.close()


def process_entry(entry: Dict[str, Any], save_dir: str, tolerance: float = 0):
    """Process single entry: verify and visualize"""
    # Auto-detect and convert format if needed
    entry = _normalize_entry(entry)

    name = entry.get('name', 'unknown')
    data = entry.get('list', [])

    if not data:
        print(f"  - {name}: empty data")
        return

    circles = np.array(data, dtype=float)

    # Verify circles (but still plot regardless)
    is_valid, error_msg = verify_circles(circles, tolerance=0)
    if not is_valid:
        # test with tolerance again
        print(f"  \t{name}: verification failed with tolerance=0, retrying with tolerance={tolerance}")
        is_valid, error_msg = verify_circles(circles, tolerance=tolerance)
        t = tolerance
    else:
        t = 0


    sum_radii = np.sum(circles[:, 2])
    status = f"[Success with t={t}]" if is_valid else f"[Failed with t={t}]"
    print(f"  {status} {name}: {len(circles)} circles, sum_radii = {sum_radii:.10f}", end="")

    if not is_valid:
        print(f" | {error_msg}")
    else:
        print()

    # Always plot, regardless of validation status
    try:
        title = f"Circle Packing: {name}, score={sum_radii:.10f}"
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, name)
        plot_circles(circles, title=title, save_path=save_path)
    except Exception as e:
        print(f"    Error plotting: {e}")

    
    print('-'*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and visualize circle packing solutions")
    parser.add_argument("--data_file", type=str, default="data.json", help="Input JSON file")
    parser.add_argument("--save_dir", type=str, default="figs", help="Output directory for plots")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Floating point tolerance for validation")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nLoading from {args.data_file}")
    data = load_data(args.data_file)

    print("Processing entries:")
    for entry in data:
      process_entry(entry, args.save_dir, tolerance=args.tolerance)

    print()
