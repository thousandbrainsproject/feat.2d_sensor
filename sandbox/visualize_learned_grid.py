"""Visualize learned edge detections against ground truth grid lines.

Loads a pretrained model checkpoint and renders edge tangent lines
(scaled by coherence * edge_strength) alongside the known ground truth
grid pattern for visual comparison. Includes an interactive slider to
filter points by edge strength.
"""

import argparse
from pathlib import Path

import math

import numpy as np
import torch
from vedo import Line, Plotter, Points

from model_loading_utils import load_object_model


def _normalize_rows(V, eps=1e-12):
    V = np.asarray(V, float)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(n, eps)


def add_ground_truth_grid(
    plotter,
    depth_value,
    ax0=0,
    ax1=1,
    depth_ax=2,
    half_extent=0.025,
    spacing=0.01,
    color="blue",
    lw=3,
    alpha=0.5,
):
    """Add ground truth grid lines centered at the origin.

    Grid lines are placed at multiples of ``spacing`` from 0 on both
    ``ax0`` and ``ax1``, drawn at ``depth_value`` on the depth axis.

    Returns:
        List of Line objects added to the plotter.
    """
    n = int(half_extent / spacing)
    offsets = np.arange(-n, n + 1) * spacing

    lines = []
    for t in offsets:
        # Line along ax0 at position t on ax1
        p_start = np.zeros(3)
        p_end = np.zeros(3)
        p_start[depth_ax] = p_end[depth_ax] = depth_value
        p_start[ax0] = -half_extent
        p_end[ax0] = half_extent
        p_start[ax1] = p_end[ax1] = t
        lines.append(Line(p_start, p_end, c=color, lw=lw, alpha=alpha))

        # Line along ax1 at position t on ax0
        p_start = np.zeros(3)
        p_end = np.zeros(3)
        p_start[depth_ax] = p_end[depth_ax] = depth_value
        p_start[ax1] = -half_extent
        p_end[ax1] = half_extent
        p_start[ax0] = p_end[ax0] = t
        lines.append(Line(p_start, p_end, c=color, lw=lw, alpha=alpha))

    if lines:
        plotter.add(*lines)
    return lines


def _detect_grid_axes(points):
    """Return (ax0, ax1, depth_ax) — grid plane axes and depth axis.

    ax0 and ax1 are the two axes with the largest spread; depth_ax is the
    smallest.
    """
    spreads = points.max(axis=0) - points.min(axis=0)
    order = np.argsort(spreads)  # ascending
    return int(order[2]), int(order[1]), int(order[0])


def visualize_learned_edges(
    model_data,
    title=None,
    show_ground_truth=True,
    arrow_scale=0.0025,
    tangent_color="black",
    tangent_lw=3,
    gt_half_extent=None,
    gt_spacing=0.01,
):
    """Visualize learned edge tangent lines with interactive strength slider.

    Only points with pose_from_edge=True are shown. Tangent lines are scaled
    by (coherence * edge_strength) / 4 when those features are available.
    A slider lets the user set a minimum edge strength threshold to declutter.
    """
    points = np.asarray(model_data["points"], float)
    features = model_data["features"]

    # Identify edge points
    if "pose_from_edge" not in features:
        print("[edge_viz] No pose_from_edge feature; showing all points.")
        edge_mask = np.ones(len(points), dtype=bool)
    else:
        edge_mask = np.asarray(features["pose_from_edge"], bool).reshape(-1)

    edge_idx = np.where(edge_mask[: len(points)])[0]
    if len(edge_idx) == 0:
        print("[edge_viz] No edge points found.")
        return

    edge_points = points[edge_idx]
    print(f"[edge_viz] {len(edge_idx)} edge points (of {len(points)} total)")

    # Extract tangent vectors for edge points
    tangents_at_edges = None
    if "pose_vectors" in features:
        pv = np.asarray(features["pose_vectors"], float)
        if pv.ndim == 2 and pv.shape[1] == 9:
            pv = pv.reshape(-1, 3, 3)
        all_tangents = _normalize_rows(pv[:, 1, :])
        tangents_at_edges = all_tangents[edge_idx]

    # Per-edge-point scale factors: coherence only (range [0, 1])
    edge_scales = np.ones(len(edge_idx))
    if "coherence" in features:
        coherence = np.asarray(features["coherence"], float).reshape(-1)
        for i, idx in enumerate(edge_idx):
            if idx < len(coherence):
                edge_scales[i] = coherence[idx]
        edge_scales = np.clip(edge_scales, 0, 1)
        print(f"[edge_viz] Coherence range: [{edge_scales.min():.3f}, {edge_scales.max():.3f}]")

    # Detect grid plane axes
    ax0, ax1, depth_ax = _detect_grid_axes(points)
    center = points.mean(axis=0)
    print(f"[edge_viz] Grid axes: {ax0}, {ax1}; depth axis: {depth_ax}")
    print(f"[edge_viz] Center: {center}")

    # Build visualization
    plotter = Plotter(size=(1400, 1000), title=title or "Learned Edges")

    # Mutable state for slider-driven redraw
    viz_state = {
        "pc": None,
        "edge_lines": [],
        "threshold": 0.0,
    }

    def redraw(threshold):
        """Rebuild points and tangent lines for the given threshold."""
        # Remove previous objects
        if viz_state["pc"] is not None:
            plotter.remove(viz_state["pc"])
        if viz_state["edge_lines"]:
            plotter.remove(*viz_state["edge_lines"])
            viz_state["edge_lines"] = []

        # Filter: keep edge points with scale >= threshold
        keep = edge_scales >= threshold
        kept_points = edge_points[keep]
        kept_scales = edge_scales[keep]

        if len(kept_points) == 0:
            viz_state["pc"] = None
            plotter.render()
            return

        pc = Points(kept_points, r=8).color("green4")

        plotter.add(pc)
        viz_state["pc"] = pc

        # Tangent lines
        if tangents_at_edges is not None:
            new_lines = []
            kept_tangents = tangents_at_edges[keep]
            for i in range(len(kept_points)):
                t_vec = kept_tangents[i]
                if not np.isfinite(t_vec).all() or np.linalg.norm(t_vec) < 1e-9:
                    continue
                half = (arrow_scale * kept_scales[i]) / 2
                p = kept_points[i]
                new_lines.append(
                    Line(p - half * t_vec, p + half * t_vec,
                         c=tangent_color, lw=tangent_lw)
                )
            if new_lines:
                plotter.add(*new_lines)
                viz_state["edge_lines"] = new_lines

        plotter.render()

    # Initial draw
    redraw(0.0)

    # Edge strength slider
    def slider_callback(widget, _event):
        val = widget.value
        viz_state["threshold"] = val
        redraw(val)

    plotter.add_slider(
        slider_callback,
        xmin=0.0,
        xmax=1.0,
        value=0.0,
        pos=[(0.15, 0.05), (0.85, 0.05)],
        title="Coherence",
    )

    # Ground truth grid overlay (centered at origin, at the point cloud depth)
    if show_ground_truth:
        depth_value = center[depth_ax]
        if gt_half_extent is None:
            max_extent = max(
                np.abs(points[:, ax0]).max(),
                np.abs(points[:, ax1]).max(),
            )
            step = 0.005  # 0.5 cm
            gt_half_extent = math.ceil(max_extent / step) * step
        add_ground_truth_grid(
            plotter, depth_value,
            ax0=ax0, ax1=ax1, depth_ax=depth_ax,
            half_extent=gt_half_extent,
            spacing=gt_spacing,
        )
        print(
            f"[edge_viz] Ground truth grid at origin, depth={depth_value:.4f},"
            f" half_extent={gt_half_extent * 100:.1f} cm"
        )

    # Camera along depth axis looking at centroid
    max_range = (points.max(axis=0) - points.min(axis=0)).max()
    camera_distance = max_range * 1.5
    camera_pos = center.copy()
    camera_pos[depth_ax] += camera_distance

    viewup = [0, 0, 0]
    viewup[ax1] = 1

    plotter.show(
        axes=dict(xtitle="X", ytitle="Y", ztitle="Z"),
        viewup=viewup,
        camera=dict(
            pos=tuple(camera_pos),
            focal_point=tuple(center),
            view_angle=45,
        ),
        interactive=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize learned edge detections vs ground truth grid."
    )
    parser.add_argument(
        "--run_name", required=True,
        help="Folder name under base_path (e.g. cylinder_grid_learning_2d)",
    )
    parser.add_argument(
        "--show_ground_truth", action="store_true", default=True,
        dest="show_ground_truth",
        help="Overlay ground truth grid (default: on)",
    )
    parser.add_argument(
        "--no_ground_truth", action="store_false", dest="show_ground_truth",
        help="Disable ground truth grid overlay",
    )
    parser.add_argument(
        "--base_path",
        default="~/tbp/results/monty/pretrained_models/2d_sensor/",
        help="Base directory for model checkpoints",
    )
    args = parser.parse_args()

    model_path = (
        Path(args.base_path).expanduser() / args.run_name / "pretrained" / "model.pt"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")

    lm_id = 0
    graph_memory = state_dict["lm_dict"][lm_id]["graph_memory"]
    available_objects = list(graph_memory.keys())
    print(f"Available objects: {available_objects}")

    for object_name in available_objects:
        print(f"\n--- {object_name} ---")
        try:
            model_data = load_object_model(model_path, object_name, lm_id=lm_id)
            print(f"  Points: {model_data['points'].shape}")
            print(f"  Features: {list(model_data['features'].keys())}")

            visualize_learned_edges(
                model_data,
                title=f"Edges: {object_name}",
                show_ground_truth=args.show_ground_truth,
            )
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
