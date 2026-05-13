"""Surface normal error heatmap: TLS-estimated vs analytical cylinder normals.

Loads training logs (detailed_run_stats.json) alongside a 2D pretrained model,
raycasts gaze directions onto the cylinder mesh for undistorted 3D positions,
then computes the L2 error between TLS-estimated and analytical surface normals.
Results are visualized as a colored point cloud (blue=low error, red=high)
overlaid on a semi-transparent cylinder mesh.

Usage:
    python sandbox/visualize_normal_error.py
    python sandbox/visualize_normal_error.py --object 042_grid_cylinder_6cm
    python sandbox/visualize_normal_error.py --run_name cylinder_grid_learning_5000steps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from vedo import Mesh, Plotter, Points, Text2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_rows(V, eps=1e-12):
    V = np.asarray(V, float)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(n, eps)


def scalar_to_rgb(values, vmin=None, vmax=None):
    """Map scalars to RGB via HSV hue rotation (blue=low -> red=high)."""
    values = np.asarray(values, float)
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    span = vmax - vmin
    if span < 1e-12:
        t = np.zeros_like(values)
    else:
        t = np.clip((values - vmin) / span, 0, 1)

    h = 0.66 * (1.0 - t)
    s = np.ones_like(t)
    v = np.ones_like(t)

    hi = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    tt = v * (1 - (1 - f) * s)

    conditions = [hi == k for k in range(6)]
    choices_r = [v, q, p, p, tt, v]
    choices_g = [tt, v, v, q, p, p]
    choices_b = [p, p, tt, v, v, q]
    rgb = np.stack([
        np.select(conditions, choices_r),
        np.select(conditions, choices_g),
        np.select(conditions, choices_b),
    ], axis=1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_cylinder_mesh(object_name, mesh_dir, object_rotation):
    """Load a cylinder .glb mesh with rotation applied.

    Returns:
        (vedo.Mesh, trimesh.Trimesh) — mesh for display and trimesh for
        raycasting, both with rotation applied.
    """
    glb_path = Path(mesh_dir).expanduser() / object_name / "textured.glb"
    if not glb_path.exists():
        raise FileNotFoundError(f"Mesh not found: {glb_path}")

    scene = trimesh.load(str(glb_path))
    if hasattr(scene, "geometry") and scene.geometry:
        geom = next(iter(scene.geometry.values()))
    elif hasattr(scene, "vertices"):
        geom = scene
    else:
        raise RuntimeError(f"No geometry in {glb_path}")

    verts = np.asarray(geom.vertices, float)
    faces = np.asarray(geom.faces, int)

    rot = Rotation.from_quat(object_rotation)  # xyzw
    verts = rot.apply(verts)

    tri_geom = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    vmesh = Mesh([verts, faces])
    has_texture = (
        hasattr(geom.visual, "uv")
        and geom.visual.uv is not None
        and hasattr(geom.visual, "material")
        and getattr(geom.visual.material, "baseColorTexture", None) is not None
    )
    if has_texture:
        vmesh.texture(
            tname=np.array(geom.visual.material.baseColorTexture),
            tcoords=geom.visual.uv,
        )
    else:
        vmesh.color("lightgray")

    print(f"[mesh] {object_name}: {len(verts)} verts, {len(faces)} faces")
    return vmesh, tri_geom


# ---------------------------------------------------------------------------
# Gaze raycasting (from visualize_training_steps.py)
# ---------------------------------------------------------------------------

def _compute_gaze_directions(action_sequence, n_steps):
    """Compute (n_steps, 3) gaze directions from action_sequence rotations.

    LM step i uses the state from action_seq[max(0, i-1)]. Quaternions in
    action_sequence are [w, x, y, z]; scipy uses [x, y, z, w].
    """
    forward = np.array([0.0, 0.0, -1.0])
    directions = np.zeros((n_steps, 3))

    for i in range(n_steps):
        seq_idx = max(0, i - 1)
        if seq_idx >= len(action_sequence):
            directions[i] = forward
            continue

        _action, state = action_sequence[seq_idx]
        agent_id = next(iter(state))
        agent_state = state[agent_id]

        aq = agent_state["rotation"]
        agent_rot = Rotation.from_quat([aq[1], aq[2], aq[3], aq[0]])

        sq = agent_state["sensors"]["patch.depth"]["rotation"]
        sensor_rot = Rotation.from_quat([sq[1], sq[2], sq[3], sq[0]])

        combined = agent_rot * sensor_rot
        directions[i] = combined.apply(forward)

    return directions


def _raycast_closest(origins, directions, vertices, faces):
    """Closest ray-mesh intersections via Moller-Trumbore."""
    EPS = 1e-8
    N = len(origins)
    best_t = np.full(N, np.inf)
    hit_points = np.full((N, 3), np.nan)

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        e1 = v1 - v0
        e2 = v2 - v0

        h = np.cross(directions, e2)
        a = np.dot(h, e1)

        valid = np.abs(a) > EPS
        if not valid.any():
            continue

        f_inv = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)
        s = origins - v0
        u = f_inv * np.einsum("ij,ij->i", s, h)

        ok = valid & (u >= 0.0) & (u <= 1.0)
        if not ok.any():
            continue

        q = np.cross(s, e1)
        v = f_inv * np.einsum("ij,ij->i", directions, q)

        ok &= (v >= 0.0) & (u + v <= 1.0)
        if not ok.any():
            continue

        t = f_inv * np.dot(q, e2)
        ok &= (t > EPS) & (t < best_t)

        if ok.any():
            best_t[ok] = t[ok]
            hit_points[ok] = origins[ok] + t[ok, np.newaxis] * directions[ok]

    hit_mask = np.isfinite(best_t) & (best_t < np.inf)
    return hit_points, hit_mask


# ---------------------------------------------------------------------------
# Episode parsing and error computation
# ---------------------------------------------------------------------------

def parse_episodes(json_path):
    """Parse all episodes from detailed_run_stats.json (JSONL).

    Returns:
        dict mapping object_name -> episode dict with keys:
            object_name, object_position, object_rotation,
            n_steps, pose_vectors, action_sequence
    """
    episodes = {}
    json_path = Path(json_path).expanduser()

    with json_path.open() as f:
        for i, line in enumerate(f):
            raw = json.loads(line)
            ep = raw[next(iter(raw))]
            lm = ep["LM_0"]
            target = lm.get("target", {})
            obj_name = target.get("object", f"unknown_{i}")

            pv = np.asarray(lm["patch"]["pose_vectors"], float)
            n_steps = len(lm["locations"]["patch"])
            pv = pv.reshape(n_steps, 3, 3)

            object_position = np.asarray(target.get("position", [0, 0, 0]), float)
            object_rotation = np.asarray(
                target.get("quat_rotation", [0, 0, 0, 1]), float
            )

            action_seq = ep.get("motor_system", {}).get("action_sequence", [])

            # Resolve agent position from action_sequence
            sensor_position = None
            if action_seq:
                state = action_seq[0][1]
                agent_id = next(iter(state))
                sensor_position = np.array(
                    state[agent_id]["position"], float
                )

            episodes[obj_name] = {
                "object_name": obj_name,
                "object_position": object_position,
                "object_rotation": object_rotation,
                "n_steps": n_steps,
                "pose_vectors": pv,
                "action_sequence": action_seq,
                "sensor_position": sensor_position,
            }
            print(f"[episode {i}] {obj_name}: {n_steps} steps")

    return episodes


def compute_object_errors(episode, mesh_dir):
    """Raycast training steps onto mesh, compute analytical vs estimated errors.

    Returns:
        dict with positions_3d, estimated_normals, analytical_normals, errors,
        or None if raycasting fails.
    """
    obj_name = episode["object_name"]
    obj_pos = episode["object_position"]
    obj_rot = episode["object_rotation"]
    n_steps = episode["n_steps"]
    action_seq = episode["action_sequence"]
    sensor_pos = episode["sensor_position"]

    if sensor_pos is None or not action_seq:
        print(f"[{obj_name}] No sensor data for raycasting, skipping")
        return None

    vmesh, tri_geom = load_cylinder_mesh(obj_name, mesh_dir, obj_rot)

    # Gaze raycasting in object-local frame
    gaze_dirs = _compute_gaze_directions(action_seq, n_steps)
    agent_local = sensor_pos - obj_pos
    origins = np.tile(agent_local, (n_steps, 1))

    hit_points, hit_mask = _raycast_closest(
        origins, gaze_dirs,
        np.asarray(tri_geom.vertices, float),
        np.asarray(tri_geom.faces, int),
    )
    n_hits = int(hit_mask.sum())
    print(f"[{obj_name}] Gaze raycast: {n_hits}/{n_steps} hits")

    if n_hits == 0:
        return None

    positions_3d = hit_points[hit_mask]

    # TLS-estimated normals (first row of pose_vectors)
    all_normals = episode["pose_vectors"][:, 0, :]
    estimated = _normalize_rows(all_normals[hit_mask])

    # Analytical cylinder normal: radial outward in XZ, zero Y
    radial = positions_3d.copy()
    radial[:, 1] = 0.0
    analytical = _normalize_rows(radial)

    errors = np.linalg.norm(estimated - analytical, axis=1)

    return {
        "object_name": obj_name,
        "vmesh": vmesh,
        "positions_3d": positions_3d,
        "estimated_normals": estimated,
        "analytical_normals": analytical,
        "errors": errors,
        "n_hits": n_hits,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Colormap scaling strategies
# ---------------------------------------------------------------------------

CMAP_MODES = ["linear", "pctile95", "log", "sqrt", "quantile"]


def _apply_cmap_scaling(errors, mode):
    """Transform raw errors and return (mapped_values, vmin, vmax, legend_str).

    All modes produce values in [0, 1] suitable for scalar_to_rgb(v, 0, 1),
    plus a legend string describing the colorbar endpoints.
    """
    if mode == "linear":
        vmax = max(errors.max(), 0.01)
        return errors, 0.0, vmax, f"  blue = 0.0\n  red  = {vmax:.4f}"

    elif mode == "pctile95":
        vmax = max(np.percentile(errors, 95), 0.01)
        clamped = np.clip(errors, 0, vmax)
        return clamped, 0.0, vmax, f"  blue = 0.0\n  red  = {vmax:.4f} (p95)"

    elif mode == "log":
        # log10(1 + error / eps) with eps chosen so median maps to ~0.5
        eps = max(np.median(errors), 1e-6)
        transformed = np.log10(1.0 + errors / eps)
        vmax = transformed.max()
        # Compute the raw error values at colorbar ticks
        mid_raw = eps * (10 ** (vmax * 0.5) - 1)
        return (
            transformed, 0.0, max(vmax, 1e-12),
            f"  blue = 0.0\n"
            f"  mid  ~ {mid_raw:.4f}\n"
            f"  red  ~ {errors.max():.4f}",
        )

    elif mode == "sqrt":
        transformed = np.sqrt(errors)
        vmax = max(transformed.max(), 1e-6)
        mid_raw = (vmax * 0.5) ** 2
        return (
            transformed, 0.0, vmax,
            f"  blue = 0.0\n"
            f"  mid  ~ {mid_raw:.4f}\n"
            f"  red  ~ {errors.max():.4f}",
        )

    elif mode == "quantile":
        # Rank-based: each point's color reflects its percentile
        ranks = np.argsort(np.argsort(errors)).astype(float)
        n = len(errors)
        quantiles = ranks / max(n - 1, 1)
        p25 = np.percentile(errors, 25)
        p75 = np.percentile(errors, 75)
        return (
            quantiles, 0.0, 1.0,
            f"  blue = p0  ({errors.min():.4f})\n"
            f"  25%  = {p25:.4f}\n"
            f"  75%  = {p75:.4f}\n"
            f"  red  = p100 ({errors.max():.4f})",
        )

    raise ValueError(f"Unknown cmap mode: {mode}")


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class NormalErrorVisualizer:
    """Interactive vedo visualization of normal estimation error on cylinders."""

    def __init__(self, object_data_list, mesh_dir):
        self.object_data_list = object_data_list
        self.mesh_dir = Path(mesh_dir).expanduser()
        self.obj_idx = 0
        self._cmap_mode_idx = 1  # start with pctile95 (most useful default)

        self.plotter = Plotter(size=(1600, 1000), title="Normal Error Heatmap")
        self._static_actors = []
        self._dynamic_actors = []

    @property
    def _cmap_mode(self):
        return CMAP_MODES[self._cmap_mode_idx]

    def _clear_dynamic(self):
        if self._dynamic_actors:
            self.plotter.remove(*self._dynamic_actors)
        self._dynamic_actors = []

    def _clear_static(self):
        if self._static_actors:
            self.plotter.remove(*self._static_actors)
        self._static_actors = []

    def _build_scene(self, data):
        """Build all scene actors for a given object."""
        self._clear_dynamic()
        self._clear_static()

        positions = data["positions_3d"]
        errors = data["errors"]
        n = len(positions)

        # Semi-transparent mesh for shape reference
        vmesh = data["vmesh"]
        vmesh.alpha(0.15)
        self._static_actors.append(vmesh)

        # Apply colormap scaling
        mapped, vmin, vmax, legend_str = _apply_cmap_scaling(errors, self._cmap_mode)
        colors = scalar_to_rgb(mapped, vmin=vmin, vmax=vmax)
        pc = Points(positions, r=5)
        pc.pointcolors = colors
        self._dynamic_actors.append(pc)

        # Stats overlay
        stats_text = (
            f"Object: {data['object_name']}\n"
            f"Points: {n} / {data['n_steps']} steps\n"
            f"\n"
            f"L2 Normal Error\n"
            f"  mean:   {errors.mean():.4f}\n"
            f"  median: {np.median(errors):.4f}\n"
            f"  max:    {errors.max():.4f}\n"
            f"  std:    {errors.std():.4f}"
        )
        info = Text2D(stats_text, pos="top-left", s=0.65, font="Courier", c="black")
        self._dynamic_actors.append(info)

        # Colorbar text
        cbar_text = f"Scale: {self._cmap_mode}\n{legend_str}"
        cbar = Text2D(cbar_text, pos="top-right", s=0.6, font="Courier", c="black")
        self._dynamic_actors.append(cbar)

        self.plotter.add(*self._static_actors)
        self.plotter.add(*self._dynamic_actors)

    def _on_cycle_object(self, _obj, _event):
        self.obj_idx = (self.obj_idx + 1) % len(self.object_data_list)
        data = self.object_data_list[self.obj_idx]
        print(f"[viz] Switching to {data['object_name']}...")
        self._current_data = data
        self._build_scene(data)
        self.plotter.render()

    def _on_cycle_cmap(self, _obj, _event):
        self._cmap_mode_idx = (self._cmap_mode_idx + 1) % len(CMAP_MODES)
        print(f"[viz] Colormap: {self._cmap_mode}")
        self._build_scene(self._current_data)
        self.plotter.render()

    def show(self, initial_object=None):
        """Load initial object and launch the interactive viewer."""
        if initial_object:
            for i, d in enumerate(self.object_data_list):
                if d["object_name"] == initial_object:
                    self.obj_idx = i
                    break

        self._current_data = self.object_data_list[self.obj_idx]
        self._build_scene(self._current_data)

        # Buttons
        btn_y = 0.06

        if len(self.object_data_list) > 1:
            labels = [f" {d['object_name']} " for d in self.object_data_list]
            self.plotter.add_button(
                self._on_cycle_object,
                pos=(0.85, btn_y),
                states=labels,
                size=16,
                font="Calco",
            )
            btn_y += 0.07

        cmap_labels = [f" Scale: {m} " for m in CMAP_MODES]
        self.plotter.add_button(
            self._on_cycle_cmap,
            pos=(0.85, btn_y),
            states=cmap_labels,
            size=16,
            font="Calco",
        )

        # Camera: look along +Z toward cylinder front face
        pts = self._current_data["positions_3d"]
        center = pts.mean(axis=0)
        span = (pts.max(axis=0) - pts.min(axis=0)).max()
        cam_pos = (center[0], center[1], center[2] + span * 2.0)

        self.plotter.show(
            axes=dict(xtitle="X", ytitle="Y", ztitle="Z"),
            viewup="y",
            camera=dict(pos=cam_pos, focal_point=tuple(center), view_angle=45),
            interactive=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_BASE_PATH = "~/tbp/results/monty/pretrained_models/2d_sensor"
DEFAULT_MESH_DIR = "~/tbp/data/compositional_objects/meshes"


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TLS normal error as heatmap on cylinder mesh."
    )
    parser.add_argument(
        "--run_name", default="cylinder_grid_learning_5000steps",
        help="Pretraining run name (default: cylinder_grid_learning_5000steps)",
    )
    parser.add_argument(
        "--base_path", default=DEFAULT_BASE_PATH,
        help=f"Base path for pretrained models (default: {DEFAULT_BASE_PATH})",
    )
    parser.add_argument(
        "--mesh_dir", default=DEFAULT_MESH_DIR,
        help=f"Mesh directory (default: {DEFAULT_MESH_DIR})",
    )
    parser.add_argument(
        "--object", default=None,
        help="Initial object to display (default: first available)",
    )
    args = parser.parse_args()

    pretrained_dir = (
        Path(args.base_path).expanduser() / args.run_name / "pretrained"
    )
    json_path = pretrained_dir / "detailed_run_stats.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Training log not found: {json_path}")

    episodes = parse_episodes(json_path)

    # Compute errors for each object via raycasting
    object_data_list = []
    for obj_name in sorted(episodes.keys()):
        result = compute_object_errors(episodes[obj_name], args.mesh_dir)
        if result is not None:
            object_data_list.append(result)

    if not object_data_list:
        raise RuntimeError("No objects could be processed (raycasting failed)")

    print(f"\nObjects ready: {[d['object_name'] for d in object_data_list]}")

    viz = NormalErrorVisualizer(object_data_list, args.mesh_dir)
    viz.show(initial_object=args.object)


if __name__ == "__main__":
    main()
