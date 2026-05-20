"""Visualize H-LLM model: LM1 points colored by LM0 object_id.

Loads a hierarchical Monty model where LM0 recognizes sub-objects and LM1
learns composite objects. Points in LM1's "learning_module_0" channel are
colored by which sub-object LM0 identified at each location.
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from vedo import LegendBox, Plotter, Points


# ---------------------------------------------------------------------------
# Object ID encoding (mirrors EvidenceGraphLM._object_id_to_features)
# ---------------------------------------------------------------------------


def compute_object_id(name):
    """Compute the integer object_id for a graph name.

    Mirrors learning_module.py:1135-1137:
        sum(ord(i) for i in object_id)
    """
    return sum(ord(c) for c in name)


def build_id_to_name_mapping(state_dict, lm_id=0):
    """Build {int_id: object_name} from LM0's graph_memory keys."""
    graph_memory = state_dict["lm_dict"][lm_id]["graph_memory"]
    mapping = {}
    for name in graph_memory:
        int_id = compute_object_id(name)
        mapping[int_id] = name
    return mapping


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _to_numpy(x):
    """Convert Tensor or array-like to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_lm_channel_data(
    state_dict, object_name, lm_id=1, input_channel="learning_module_0"
):
    """Extract positions and object_id values from an LM channel.

    Returns:
        (positions, object_ids) -- both as numpy arrays, shapes (N,3) and (N,).
    """
    graph_memory = state_dict["lm_dict"][lm_id]["graph_memory"]
    model = graph_memory[object_name][input_channel]
    graph = model._graph

    pos = _to_numpy(graph.pos).astype(float)
    x = _to_numpy(graph.x).astype(float)
    fm = graph.feature_mapping

    start, end = fm["object_id"]
    object_ids = x[:, start:end].squeeze()

    return pos, object_ids


# ---------------------------------------------------------------------------
# Grid-averaging artifact correction
# ---------------------------------------------------------------------------


def snap_to_nearest_id(raw_ids, known_ids):
    """Snap each raw object_id to the nearest known integer ID.

    Grid averaging can produce non-integer values; this maps each to the
    closest legitimate ID.
    """
    known = np.array(sorted(known_ids))
    raw = np.asarray(raw_ids, dtype=float)
    # Vectorized: find index of nearest known ID for each raw value
    idx = np.abs(raw[:, None] - known[None, :]).argmin(axis=1)
    return known[idx]


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

SUBOBJECT_COLORS = [
    ("cube", (0, 160, 223)),  # numenta_blue #00a0df
    ("disk", (247, 55, 189)),  # pink #f737bd
    ("tbp", (93, 17, 191)),  # purple #5d11bf
    ("numenta", (0, 142, 67)),  # green #008e43
]
FALLBACK_COLOR = (160, 160, 160)  # gray


def color_for_name(name):
    """Return an RGB tuple for a sub-object name using substring matching."""
    lower = name.lower()
    for substring, color in SUBOBJECT_COLORS:
        if substring in lower:
            return color
    return FALLBACK_COLOR


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def print_statistics(composite_data, id_to_name):
    """Print per-object stats: point count, bounds, sub-object distribution."""
    print("\n" + "=" * 60)
    print("H-LLM Model Statistics")
    print("=" * 60)

    total_points = 0
    for obj_name, (pos, snapped_ids) in sorted(composite_data.items()):
        n = len(pos)
        total_points += n
        print(f"\n--- {obj_name} ({n} points) ---")
        print(f"  Bounds X: [{pos[:, 0].min():.4f}, {pos[:, 0].max():.4f}]")
        print(f"  Bounds Y: [{pos[:, 1].min():.4f}, {pos[:, 1].max():.4f}]")
        print(f"  Bounds Z: [{pos[:, 2].min():.4f}, {pos[:, 2].max():.4f}]")
        print(f"  Center:   {pos.mean(axis=0)}")

        counts = Counter(snapped_ids)
        print("  Sub-object distribution:")
        for sid, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            name = id_to_name.get(int(sid), f"unknown({int(sid)})")
            pct = 100 * count / n
            print(f"    {name:30s}  {count:5d} pts  ({pct:5.1f}%)")

    print(f"\nTotal points across all objects: {total_points}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_combined(composite_data, id_to_name, title=None):
    """Vedo plot of all composite objects, points colored by sub-object ID."""
    all_points = []
    all_colors = []

    for _obj_name, (pos, snapped_ids) in sorted(composite_data.items()):
        colors = np.zeros((len(pos), 3), dtype=np.uint8)
        for i, sid in enumerate(snapped_ids):
            name = id_to_name.get(int(sid), "unknown")
            colors[i] = color_for_name(name)
        all_points.append(pos)
        all_colors.append(colors)

    pts = np.concatenate(all_points, axis=0)
    cols = np.concatenate(all_colors, axis=0)

    print(f"Visualizing {len(pts)} total points")

    window_title = title or "H-LLM Model: LM1 colored by LM0 object_id"
    plotter = Plotter(size=(1400, 1000), title=window_title)
    point_cloud = Points(pts, r=8)
    point_cloud.pointcolors = cols.tolist()
    plotter.add(point_cloud)

    # Legend: create one colored marker per sub-object
    legend_entries = []
    seen = set()
    for _, (_, snapped_ids) in sorted(composite_data.items()):
        for sid in snapped_ids:
            key = int(sid)
            if key not in seen:
                seen.add(key)
                name = id_to_name.get(key, f"unknown({key})")
                color = color_for_name(name)
                marker = Points([[0, 0, 0]], r=12).c(color)
                marker.legend(name)
                legend_entries.append((name, marker))

    legend_entries.sort(key=lambda pair: pair[0])
    lb = LegendBox([m for _, m in legend_entries], width=0.2, height=0.15, padding=2)
    plotter.add(lb)

    # Camera: look straight at XY plane from +Z
    center = pts.mean(axis=0)
    max_range = np.ptp(pts, axis=0).max()
    cam_dist = max_range * 3
    camera_pos = (
        center[0],
        center[1],
        center[2] + cam_dist,
    )
    plotter.show(
        axes=dict(xtitle="X", ytitle="Y", ztitle="Z"),
        viewup="y",
        camera=dict(
            pos=camera_pos,
            focal_point=center.tolist(),
        ),
        interactive=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    default_path = os.path.join(
        os.environ.get("MONTY_MODELS", ""),
        "pretrained_ycb_v12",
        "supervised_pre_training_objects_with_logos_lvl1_comp_models_burst_sampling",
        "pretrained",
        "model.pt",
    )

    parser = argparse.ArgumentParser(
        description="Visualize H-LLM model: LM1 points colored by LM0 object_id"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_path,
        help="Path to the pretrained model.pt file",
    )
    parser.add_argument(
        "--lm0_id",
        type=int,
        default=0,
        help="LM index for the lower-level module (default: 0)",
    )
    parser.add_argument(
        "--lm1_id",
        type=int,
        default=1,
        help="LM index for the higher-level module (default: 1)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="learning_module_0",
        help="Input channel on LM1 that carries LM0's object_id",
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=["007_disk_tbp_horz"],
        help="Composite object name(s) to visualize (default: 007_disk_tbp_horz)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")

    # Build reverse mapping from integer ID -> sub-object name
    id_to_name = build_id_to_name_mapping(state_dict, lm_id=args.lm0_id)
    known_ids = set(id_to_name.keys())
    print(f"LM{args.lm0_id} sub-objects: {id_to_name}")

    # Extract data from each composite object in LM1
    graph_memory = state_dict["lm_dict"][args.lm1_id]["graph_memory"]
    composite_objects = list(graph_memory.keys())
    print(f"LM{args.lm1_id} composite objects: {composite_objects}")

    # Filter to requested objects
    selected = args.objects
    missing = [o for o in selected if o not in composite_objects]
    if missing:
        print(f"WARNING: objects not found in model: {missing}")
        print(f"  Available: {composite_objects}")
    selected = [o for o in selected if o in composite_objects]

    composite_data = {}
    for obj_name in selected:
        try:
            pos, raw_ids = extract_lm_channel_data(
                state_dict,
                obj_name,
                lm_id=args.lm1_id,
                input_channel=args.channel,
            )
            snapped_ids = snap_to_nearest_id(raw_ids, known_ids)
            composite_data[obj_name] = (pos, snapped_ids)
            print(f"  {obj_name}: {len(pos)} points")
        except Exception as e:
            print(f"  Skipping {obj_name}: {e}")

    if not composite_data:
        print("No composite objects found. Check --objects and model path.")
        return

    print_statistics(composite_data, id_to_name)
    title = ", ".join(sorted(composite_data.keys()))
    visualize_combined(composite_data, id_to_name, title=title)


if __name__ == "__main__":
    main()
