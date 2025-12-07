import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import json

# load data
def load_data(file_path: str):
    # load json data
    with open(file_path, 'r') as f:
        data = json.load(f)

    assert len(data) > 0, "Loaded data is empty."
    print(f"Loaded data from {file_path}, total entries: {len(data)}, keys in first entry: {list(data[0].keys())}")
    # data would be a list contain {"name":..., "list":...}
    return data

# Verification
def verify(heights_sequence_2: list[float]):
    #@title Verification
    convolution_2 = np.convolve(heights_sequence_2, heights_sequence_2)
    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points) # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i+1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)

    print(f"This step function shows that C2 >= {C_lower_bound}")
    print(f"len(heights_sequence_2) = {len(heights_sequence_2)}, len(convolution_2) = {len(convolution_2)}")

    return convolution_2, C_lower_bound


def parse_name(name: str) -> tuple:
    """Parse name: '8B-w_RL@65' -> ('8B', 'w_RL', 65), 'Init' -> ('Init', None, None)
    Returns None if name format is not recognized."""
    if name == "Init":
        return ("Init", None, None)
    try:
        if '-' not in name or '@' not in name:
            return None
        model, rl_info = name.split('-', 1)
        rl_type, step = rl_info.split('@')
        return (model, rl_type, int(step))
    except (ValueError, IndexError):
        return None


def plot_step_function(step_heights_input: list[float], title="", save_dir=".", save_name=None):
    """Plots a step function with equally-spaced intervals on [-1/4,1/4]."""
    num_steps = len(step_heights_input)

    # increase font size
    plt.rcParams.update({'font.size': 14})
    # Generate x values for plotting (need to plot steps properly).
    step_edges_plot = np.linspace(-0.25, 0.25, num_steps + 1)
    x_plot = np.array([])
    y_plot = np.array([])

    for i in range(num_steps):
        x_start = step_edges_plot[i]
        x_end = step_edges_plot[i+1]
        x_step_vals = np.linspace(x_start, x_end, 100)  # Points within each step.
        y_step_vals = np.full_like(x_step_vals, step_heights_input[i])
        x_plot = np.concatenate((x_plot, x_step_vals))
        y_plot = np.concatenate((y_plot, y_step_vals))

    # Plot the step function.
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.xlim([-0.3, 0.3])  # Adjust x-axis limits if needed.
    plt.ylim([-1, max(step_heights_input) * 1.2])  # Adjust y-axis limits.
    plt.grid(True)
    plt.step(step_edges_plot[:-1], step_heights_input, where='post', color='green', linewidth=2)  # Overlay with plt.step for clarity.
    # compact
    plt.tight_layout()

    assert save_name is not None, "Please provide a save_name to save the plot. e.g., 'step_function.pdf'"
    assert not save_name.endswith(".pdf") and not save_name.endswith(".png"), "Please provide save_name without file extension."
    save_name_pdf = save_name + ".pdf"
    save_name_png = save_name + ".png"
    # save_path_pdf = os.path.join(save_dir, save_name_pdf)
    # plt.savefig(save_path_pdf)
    save_path_png = os.path.join(save_dir, save_name_png)
    plt.savefig(save_path_png)
    print(f"Plot saved to {save_path_png}")
  


def _plot_combined(model_groups: dict, plot_type: str, save_dir: str, if_normalize: bool, suffix_tags: str = ""):
    """Helper to plot combined figures (step or autoconv)."""
    model_names = {"8B": "DeepSeek-R1-0528-Qwen3-8B", "1.5B": "ProRL-1.5B-v2"}
    colors = {"Init": "#1f77b4", "w_RL": "#ff7f0e", "wo_RL": "#2ca02c"}
    is_autoconv = plot_type == "autoconv"

    for model, entries in model_groups.items():
        entries_sorted = sorted(entries, key=lambda x: (x[2] != "Init", -(x[3] or 0)))
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(10, 6))

        for name, heights, rl_type, step in entries_sorted:
            convolution_2, score = verify(heights)
            data = convolution_2 if is_autoconv else np.array(heights)
            if if_normalize:
                norm_inf = np.max(np.abs(convolution_2))
                norm_factor = norm_inf
                norm = norm_factor if is_autoconv else np.sqrt(norm_factor)
                data = data / norm
            num_steps = len(data)
            step_edges = np.linspace(-0.25, 0.25, num_steps + 1)

            label = f"Initial @ 0, score={score:.4f}" if rl_type is None else \
                    f"{'w/' if 'w_RL' in rl_type else 'w/o'} RL @ Step {step}, score={score:.4f}"
            plt.step(step_edges[:-1], data, where='post',
                    color=colors.get(rl_type, "#d62728"), linewidth=2, label=label)

        title_prefix = "Autoconvolution" if is_autoconv else "Step function"
        title_prefix += f" (Normalized)" if if_normalize else ""
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"{title_prefix}: {model_names.get(model, model)}")
        plt.xlim([-0.3, 0.3])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_suffix = "autoconv_f" if is_autoconv else "step_f"
        save_name = f"{model}_combined_{plot_suffix}{suffix_tags}"
        save_path_pdf = os.path.join(save_dir, save_name + ".pdf")
        save_path_png = os.path.join(save_dir, save_name + ".png")
        # plt.savefig(save_path_pdf)
        plt.savefig(save_path_png)
        print(f"Combined plot saved to {save_path_png}")
        plt.close()


def plot_combined_step_functions(data: list[dict], save_dir=".", if_combine_figures=True, if_normalize=True, if_add_init=False):
    """Plot combined step functions and autoconvolution by model."""
    if not if_combine_figures:
        return

    model_groups = {}
    init_entry = None
    for entry in data:
        name, heights = entry["name"], entry["list"]
        if name == "Init":
            init_entry = (name, heights, None, None)
        else:
            parsed = parse_name(name)
            if parsed is None:
                continue  # Skip names that can't be parsed
            model, rl_type, step = parsed
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append((name, heights, rl_type, step))

    if if_add_init and init_entry:
        for model in model_groups:
            model_groups[model].insert(0, init_entry)

    suffix_tags = ""
    if if_add_init:
        suffix_tags += "_ai_1"

    _plot_combined(model_groups, "step", save_dir, if_normalize, suffix_tags)
    _plot_combined(model_groups, "autoconv", save_dir, if_normalize, suffix_tags)


def verification_and_plots(heights_sequence_2: list[float], save_dir: str, save_name: str):
    print(f"================= Processing {save_name} =================")
    convolution_2, C_lower_bound = verify(heights_sequence_2)
    plot_step_function(heights_sequence_2, title=f"Step function: {save_name}, score={C_lower_bound:.4f}", save_dir=save_dir, save_name=save_name + '_step_f')
    plot_step_function(convolution_2, title=f"Autoconvolution: {save_name}, score={C_lower_bound:.4f}", save_dir=save_dir, save_name=save_name + '_autoconv_f')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_file", type=str, default="data.json", help="Path to the JSON data file.")
    args.add_argument("--save_dir", type=str, default="figs", help="Directory to save the plots.")
    args.add_argument("--if_combine_figures", type=int, default=0, help="Combine figures by model if True (0 or 1).")
    args.add_argument("--if_normalize", type=int, default=1, help="Normalize step/conv for drawing if True (0 or 1).")
    args.add_argument("--if_add_init", type=int, default=0, help="Add init to combined figures if True (0 or 1).")
    parsed_args = args.parse_args()

    os.makedirs(parsed_args.save_dir, exist_ok=True)

    data = load_data(parsed_args.data_file)
    for entry in data:
        name = entry["name"]
        heights_sequence_2 = entry["list"]
        verification_and_plots(heights_sequence_2, save_dir=parsed_args.save_dir, save_name=name)

    # Plot combined figures
    plot_combined_step_functions(data, save_dir=parsed_args.save_dir, if_combine_figures=bool(parsed_args.if_combine_figures), if_normalize=bool(parsed_args.if_normalize), if_add_init=bool(parsed_args.if_add_init))

