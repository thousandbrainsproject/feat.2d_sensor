"""Script to visualize and compare results between N experiments.

Reads eval_stats.csv from experiment directories and creates comparison plots
for accuracy, monty_steps, and rotation_error. Supports equivalence-class
reclassification for cross-object evaluation (e.g., disk-trained model evaluated
on cylinder/sphere/cross objects).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_experiment_data(csv_path):
    """Load eval_stats.csv from an experiment directory.

    Args:
        csv_path: Path to eval_stats.csv file

    Returns:
        pandas DataFrame with experiment data
    """
    df = pd.read_csv(csv_path)
    return df


def _parse_array(s):
    """Parse a numpy-style array string like '[ 1.0  2.0  3.0]' into a list."""
    return [float(x) for x in s.strip().replace("[", "").replace("]", "").split()]


def _compute_rotation_error_deg(most_likely_rotation_str, target_quat_str):
    """Compute geodesic rotation error in degrees from CSV string columns.

    Args:
        most_likely_rotation_str: Euler XYZ angles in degrees (string).
        target_quat_str: Target quaternion in [w,x,y,z] order (string).

    Returns:
        Rotation error in degrees, or NaN if parsing fails.
    """
    try:
        mlr = _parse_array(most_likely_rotation_str)
        tgt_q = _parse_array(target_quat_str)
        # CSV stores [w,x,y,z]; scipy expects [x,y,z,w]
        tgt_q_scipy = tgt_q[1:] + tgt_q[:1]
        pred = Rotation.from_euler("xyz", mlr, degrees=True)
        tgt = Rotation.from_quat(tgt_q_scipy)
        return np.round(np.degrees((pred * tgt.inv()).magnitude()), 4)
    except (ValueError, IndexError):
        return np.nan


def reclassify_performance(df, equivalence_classes):
    """Reclassify confused predictions using equivalence classes.

    When a model trained on one object shape (e.g., disk) is evaluated on another
    (e.g., cylinder), the framework marks predictions as "confused" because the
    object names differ. This function reclassifies those as "correct" when both
    objects belong to the same equivalence class.

    Also computes rotation_error for reclassified rows using most_likely_rotation
    and primary_target_rotation_quat, since the framework only computes
    rotation_error for originally-correct predictions.

    Args:
        df: DataFrame with primary_performance, primary_target_object,
            and most_likely_object columns.
        equivalence_classes: Mapping from class label to list of object names,
            e.g. {"tbp": ["007_disk_tbp_horz", "012_cylinder_tbp_horz"]}.

    Returns:
        Modified DataFrame (also modifies in-place).
    """
    reverse_lookup = {}
    for class_label, objects in equivalence_classes.items():
        for obj in objects:
            reverse_lookup[obj] = class_label

    for perf_val, replacement in [
        ("confused", "correct"),
        ("confused_mlh", "correct_mlh"),
    ]:
        mask = df["primary_performance"] == perf_val
        if not mask.any():
            continue

        target_classes = df.loc[mask, "primary_target_object"].map(reverse_lookup)
        predicted_classes = df.loc[mask, "most_likely_object"].map(reverse_lookup)

        same_class = (
            target_classes.notna()
            & predicted_classes.notna()
            & (target_classes == predicted_classes)
        )
        reclassified_mask = mask & same_class.reindex(df.index, fill_value=False)
        reclassified = reclassified_mask.sum()
        if reclassified > 0:
            df.loc[reclassified_mask, "primary_performance"] = replacement

            # Fill in rotation_error for rows that had it missing
            missing_rot = reclassified_mask & df["rotation_error"].isna()
            if missing_rot.any():
                df.loc[missing_rot, "rotation_error"] = df.loc[missing_rot].apply(
                    lambda r: _compute_rotation_error_deg(
                        r["most_likely_rotation"],
                        r["primary_target_rotation_quat"],
                    ),
                    axis=1,
                )
            print(f"  Reclassified {reclassified} rows: {perf_val} -> {replacement}")

    return df


def calculate_accuracy(df):
    """Calculate accuracy as percentage of correct or correct_mlh results.

    Args:
        df: DataFrame with primary_performance column

    Returns:
        Accuracy percentage (0-100)
    """
    if "primary_performance" not in df.columns:
        raise ValueError("DataFrame must contain 'primary_performance' column")

    correct_mask = df["primary_performance"].isin(["correct", "correct_mlh"])
    accuracy = (correct_mask.sum() / len(df)) * 100
    return accuracy


def calculate_accuracy_breakdown(df):
    """Calculate breakdown of correct and correct_mlh as percentages.

    Args:
        df: DataFrame with primary_performance column

    Returns:
        Tuple of (correct_percentage, correct_mlh_percentage) both 0-100
    """
    if "primary_performance" not in df.columns:
        raise ValueError("DataFrame must contain 'primary_performance' column")

    total = len(df)
    correct_count = (df["primary_performance"] == "correct").sum()
    correct_mlh_count = (df["primary_performance"] == "correct_mlh").sum()

    correct_pct = (correct_count / total) * 100
    correct_mlh_pct = (correct_mlh_count / total) * 100

    return correct_pct, correct_mlh_pct


def get_monty_steps_data(df):
    """Get raw monty_steps data.

    Args:
        df: DataFrame with monty_steps column

    Returns:
        Array of monty_steps values (with NaN removed)
    """
    if "monty_steps" not in df.columns:
        raise ValueError("DataFrame must contain 'monty_steps' column")

    monty_steps = df["monty_steps"].dropna().values
    return monty_steps


def get_rotation_error_data(df):
    """Get raw rotation_error data in degrees.

    Args:
        df: DataFrame with rotation_error column (in degrees)

    Returns:
        Array of rotation_error values in degrees (with NaN removed)
    """
    if "rotation_error" not in df.columns:
        raise ValueError("DataFrame must contain 'rotation_error' column")

    # Drop NaN values (rotation_error is already in degrees)
    rotation_error_deg = df["rotation_error"].dropna().values
    return rotation_error_deg


def _get_colors(n):
    """Return n (base_color, light_color) pairs from tab10 colormap."""
    cmap = plt.cm.tab10
    pairs = []
    for i in range(n):
        base = cmap(i)
        light = tuple(min(1.0, c * 1.3) for c in base[:3]) + (base[3],)
        pairs.append((base, light))
    return pairs


def _plot_accuracy_bars(ax, experiments):
    """Plot stacked accuracy bars comparing N experiments on given axes.

    Args:
        ax: Matplotlib axes.
        experiments: List of (name, DataFrame) tuples.
    """
    if all(len(df) == 0 for _, df in experiments):
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        return

    color_pairs = _get_colors(len(experiments))
    labels, correct_vals, mlh_vals, totals = [], [], [], []
    for name, df in experiments:
        n = len(df)
        labels.append(f"{name}\n(n={n})")
        if n > 0:
            c, m = calculate_accuracy_breakdown(df)
        else:
            c, m = 0.0, 0.0
        correct_vals.append(c)
        mlh_vals.append(m)
        totals.append(c + m)

    x = np.arange(len(experiments))
    ax.bar(
        x,
        correct_vals,
        color=[cp[0] for cp in color_pairs],
        alpha=0.8,
        label="correct",
    )
    ax.bar(
        x,
        mlh_vals,
        bottom=correct_vals,
        color=[cp[1] for cp in color_pairs],
        alpha=0.8,
        label="correct_mlh",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=8)

    for i, acc in enumerate(totals):
        ax.text(
            i,
            acc,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def _plot_violin(ax, experiments, column, ylabel, unit=""):
    """Plot violin comparison for a numeric column on given axes.

    Args:
        ax: Matplotlib axes.
        experiments: List of (name, DataFrame) tuples.
        column: Column name to plot.
        ylabel: Y-axis label.
        unit: Unit suffix for stat labels (e.g., "deg").
    """
    n_exp = len(experiments)
    color_pairs = _get_colors(n_exp)
    colors = [cp[0] for cp in color_pairs]
    labels, datasets = [], []
    for name, df in experiments:
        n = len(df)
        labels.append(f"{name}\n(n={n})")
        datasets.append(df[column].dropna().values if n > 0 else np.array([]))

    plottable = [len(d) >= 2 for d in datasets]
    if any(plottable):
        plot_data = [d for d, ok in zip(datasets, plottable) if ok]
        plot_positions = [i for i, ok in enumerate(plottable) if ok]
        parts = ax.violinplot(
            plot_data,
            positions=plot_positions,
            showmeans=True,
            showmedians=True,
            widths=0.6,
        )
        for pc, pos in zip(parts["bodies"], plot_positions):
            pc.set_facecolor(colors[pos])
            pc.set_alpha(0.7)
        for key in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1.5)

    for pos, (data, color) in enumerate(zip(datasets, colors)):
        if len(data) > 0:
            jitter = np.random.normal(0, 0.03, size=len(data))
            ax.scatter(
                pos + jitter,
                data,
                color=color,
                alpha=1.0,
                s=20,
                zorder=3,
                edgecolors="black",
                linewidths=0.5,
            )

    ax.set_xticks(range(n_exp))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    non_empty = [d for d in datasets if len(d) > 0]
    if non_empty:
        y_max = np.max(np.concatenate(non_empty))
    else:
        y_max = 1
    ax.set_ylim(top=y_max * 1.35)

    for pos, data in enumerate(datasets):
        if len(data) > 0:
            ax.text(
                pos,
                y_max * 1.22,
                f"Mean: {np.mean(data):.1f}{unit}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            ax.text(
                pos,
                y_max * 1.12,
                f"Median: {np.median(data):.1f}{unit}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )


def create_comparison_plots(experiments, output_dir):
    """Create comparison plots for N experiments.

    Args:
        experiments: List of (name, DataFrame) tuples.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_exp = len(experiments)
    fig_width = max(18, 6 * n_exp)

    # Combined 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 5))

    axes[0].set_title("Accuracy Comparison", fontsize=14, fontweight="bold")
    _plot_accuracy_bars(axes[0], experiments)

    axes[1].set_title("Monty Steps Distribution", fontsize=14, fontweight="bold")
    _plot_violin(axes[1], experiments, "monty_steps", "Monty Steps")

    axes[2].set_title("Rotation Error Distribution", fontsize=14, fontweight="bold")
    _plot_violin(
        axes[2],
        experiments,
        "rotation_error",
        "Rotation Error (degrees)",
        unit="\u00b0",
    )

    plt.tight_layout(pad=3.0)
    output_path = output_dir / "experiment_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined plot: {output_path}")

    # Individual plots
    plot_configs = [
        (
            "experiment_comparison_accuracy.png",
            "Accuracy Comparison",
            lambda ax: _plot_accuracy_bars(ax, experiments),
        ),
        (
            "experiment_comparison_monty_steps.png",
            "Monty Steps Distribution",
            lambda ax: _plot_violin(ax, experiments, "monty_steps", "Monty Steps"),
        ),
        (
            "experiment_comparison_rotation_error.png",
            "Rotation Error Distribution",
            lambda ax: _plot_violin(
                ax,
                experiments,
                "rotation_error",
                "Rotation Error (degrees)",
                unit="\u00b0",
            ),
        ),
    ]
    for filename, title, plot_fn in plot_configs:
        fig_i, ax_i = plt.subplots(figsize=(max(6, 3 * n_exp), 5))
        ax_i.set_title(title, fontsize=14, fontweight="bold")
        plot_fn(ax_i)
        plt.tight_layout(pad=2.0)
        path_i = output_dir / filename
        plt.savefig(path_i, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {title.lower()} plot: {path_i}")


def main():
    """Main function to compare N experiments with optional reclassification."""
    # -- Configuration ----------------------------------------------------------
    experiment_configs = [
        {
            "name": "2D-on-2D",
            "path": "/Users/hlee/tbp/results/monty/projects/2d_sensor_inference/disk_v0_2d_20pos_delta_thresholds/eval_stats.csv",
        },
        {
            "name": "3D-on-3D",
            "path": "/Users/hlee/tbp/results/monty/projects/2d_sensor_inference/disk_v0_3d_20pos_delta_thresholds/eval_stats.csv",
        },
        # {
        #     "name": "45 degree",
        #     "path": "/Users/hlee/tbp/results/monty/projects/2d_sensor_inference/disk_inference_2d_45deg/eval_stats.csv",
        # },
        # {
        #     "name": "90 degree",
        #     "path": "/Users/hlee/tbp/results/monty/projects/2d_sensor_inference/disk_inference_2d_90deg/eval_stats.csv",
        # },
        # Add more experiments here as needed:
        # {
        #     "name": "disk_on_cylinder",
        #     "path": "/Users/hlee/tbp/results/.../eval_stats.csv",
        # },
    ]

    # Objects that should be treated as equivalent across shapes.
    # Set to None or {} to skip reclassification.
    equivalence_classes = {
        "tbp": [
            "007_disk_tbp_horz",
            "012_cylinder_tbp_horz",
            "017_sphere_tbp_horz",
            "024_mug_tbp_horz",
        ],
        "numenta": [
            "009_disk_numenta_horz",
            "014_cylinder_numenta_horz",
            "019_sphere_numenta_horz",
            "026_mug_numenta_horz",
        ],
    }

    output_dir = Path("/Users/hlee/tbp/feat.2d_sensor/results")
    # --------------------------------------------------------------------------

    experiments = []
    for cfg in experiment_configs:
        path = Path(cfg["path"])
        print(f"Loading {cfg['name']}: {path}")
        df = load_experiment_data(path)
        print(f"  Loaded {len(df)} rows")

        if equivalence_classes:
            reclassify_performance(df, equivalence_classes)

        experiments.append((cfg["name"], df))

    print("\nCreating comparison plots...")
    create_comparison_plots(experiments, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
