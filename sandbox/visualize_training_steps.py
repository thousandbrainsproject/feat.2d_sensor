"""Interactive training step visualizer for 2D pretraining experiments.

Replays training steps from detailed_run_stats.json, showing how the learned
model grows point-by-point, what features are extracted at each step, and where
edges are detected. Uses LM buffer data and SM_0 sensor positions.

Optionally overlays the object mesh (semi-transparent .glb), marks the sensor
position per step, and draws a gaze line from sensor to the current surface point.
Sensor positions are resolved in priority order: SM_0.sm_properties (per-step),
motor_system.action_sequence (constant agent position), or --agent_pos override.

Usage:
    python sandbox/visualize_training_steps.py <path_to_detailed_run_stats.json> \
        [--episode N] [--mesh_dir DIR] [--agent_pos X Y Z]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from vedo import Line, Mesh, Plotter, Points, Sphere, Text2D


# ---------------------------------------------------------------------------
# Helpers (standalone copies from visualize_learned_edges.py)
# ---------------------------------------------------------------------------

def _normalize_rows(V, eps=1e-12):
    V = np.asarray(V, float)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(n, eps)


def _detect_grid_axes(points):
    """Return (ax0, ax1, depth_ax) by spread — grid plane vs depth."""
    spreads = points.max(axis=0) - points.min(axis=0)
    order = np.argsort(spreads)
    return int(order[2]), int(order[1]), int(order[0])


def scalar_to_rgb(values, vmin=None, vmax=None):
    """Map scalars to RGB via HSV hue rotation (blue->red). No matplotlib."""
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

    # Hue: 0.66 (blue) -> 0.0 (red) as t goes 0 -> 1
    h = 0.66 * (1.0 - t)
    s = np.ones_like(t)
    v = np.ones_like(t)

    # HSV -> RGB (vectorized)
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


def binary_to_rgb(values):
    """Map boolean/binary values to green (True) / red (False)."""
    colors = np.zeros((len(values), 3), dtype=np.uint8)
    mask = np.asarray(values, bool)
    colors[mask] = [0, 200, 0]
    colors[~mask] = [200, 0, 0]
    return colors


def _raycast_closest(origins, directions, vertices, faces):
    """Closest ray-mesh intersections via Moller-Trumbore.

    Iterates over faces (small F for typical meshes), vectorized across N rays
    per face. For each ray, keeps the closest positive-t intersection.

    Args:
        origins: (N, 3) ray origin points.
        directions: (N, 3) ray direction vectors (need not be unit length).
        vertices: (V, 3) mesh vertex positions.
        faces: (F, 3) triangle face indices into vertices.

    Returns:
        hit_points: (N, 3) intersection points (NaN for misses).
        hit_mask: (N,) boolean array, True where a hit was found.
    """
    EPS = 1e-8
    N = len(origins)
    best_t = np.full(N, np.inf)
    hit_points = np.full((N, 3), np.nan)

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        e1 = v1 - v0  # (3,)
        e2 = v2 - v0  # (3,)

        h = np.cross(directions, e2)  # (N, 3)
        a = np.dot(h, e1)             # (N,)

        valid = np.abs(a) > EPS
        if not valid.any():
            continue

        f_inv = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)
        s = origins - v0  # (N, 3)
        u = f_inv * np.einsum("ij,ij->i", s, h)

        ok = valid & (u >= 0.0) & (u <= 1.0)
        if not ok.any():
            continue

        q = np.cross(s, e1)  # (N, 3)
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


def _compute_gaze_directions(action_sequence, n_steps):
    """Compute (n_steps, 3) gaze directions from action_sequence rotations.

    LM step i uses the state from action_seq[max(0, i-1)] (the action that
    produced step i's observation). Each entry combines agent rotation (yaw)
    and sensor rotation (pitch).

    Quaternions in action_sequence are stored as [w, x, y, z]; scipy uses
    [x, y, z, w].
    """
    from scipy.spatial.transform import Rotation

    forward = np.array([0.0, 0.0, -1.0])  # Habitat camera forward
    directions = np.zeros((n_steps, 3))

    for i in range(n_steps):
        # Step 0 has no preceding action; reuse action_seq[0] (same as step 1)
        seq_idx = max(0, i - 1)
        if seq_idx >= len(action_sequence):
            directions[i] = forward
            continue

        _action, state = action_sequence[seq_idx]
        agent_id = next(iter(state))
        agent_state = state[agent_id]

        # Agent rotation (yaw) — [w,x,y,z] -> [x,y,z,w]
        aq = agent_state["rotation"]
        agent_rot = Rotation.from_quat([aq[1], aq[2], aq[3], aq[0]])

        # Sensor rotation (pitch) — nested under sensors.patch.depth
        sq = agent_state["sensors"]["patch.depth"]["rotation"]
        sensor_rot = Rotation.from_quat([sq[1], sq[2], sq[3], sq[0]])

        combined = agent_rot * sensor_rot
        directions[i] = combined.apply(forward)

    return directions


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class EpisodeData:
    """Parsed LM buffer data for a single episode."""

    object_name: str
    locations: np.ndarray  # (N, 3)
    features: dict  # name -> np.ndarray
    stepwise_targets: list
    n_steps: int = 0
    available_color_modes: list = field(default_factory=list)
    object_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    object_rotation: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    sensor_positions: np.ndarray | None = None  # (N, 3) from SM_0.sm_properties
    is_2d: bool = False
    action_sequence: list = field(default_factory=list)

    @classmethod
    def from_episode_dict(cls, episode_dict):
        lm = episode_dict["LM_0"]

        locations = np.asarray(lm["locations"]["patch"], float)
        n_steps = len(locations)
        is_2d = bool(np.allclose(locations[:, 2], 0.0))

        raw_features = lm["patch"]
        features = {}

        for key, vals in raw_features.items():
            arr = np.asarray(vals, float)
            if key == "pose_vectors":
                # (N, 9) -> (N, 3, 3)
                features[key] = arr.reshape(n_steps, 3, 3)
            elif key in ("on_object", "pose_fully_defined"):
                # Single-element lists -> scalar per step
                features[key] = arr.reshape(n_steps)
            elif key in ("edge_strength", "coherence"):
                vals_flat = arr.flatten()
                if len(vals_flat) < n_steps:
                    # Early steps may lack edge data; zero-pad at front
                    pad = np.zeros(n_steps - len(vals_flat))
                    vals_flat = np.concatenate([pad, vals_flat])
                features[key] = vals_flat[:n_steps]
            elif key == "pose_from_edge":
                features[key] = arr.reshape(n_steps)
            elif key == "hsv":
                features[key] = arr.reshape(n_steps, -1)
            elif key in ("principal_curvatures_log", "principal_curvatures"):
                features[key] = arr.reshape(n_steps, -1)
            else:
                features[key] = arr

        # Object name and pose from target metadata
        target = lm.get("target", {})
        object_name = target.get("object", "unknown")
        object_position = np.asarray(target.get("position", [0, 0, 0]), float)
        object_rotation = np.asarray(
            target.get("quat_rotation", [0, 0, 0, 1]), float
        )

        stepwise_targets = lm.get("stepwise_targets_list", [])

        # Extract per-step sensor positions from SM_0.sm_properties
        sensor_positions = None
        sm_props = episode_dict.get("SM_0", {}).get("sm_properties", [])
        if sm_props:
            sensor_positions = np.array(
                [p["sm_location"] for p in sm_props], float
            )

        # Extract action_sequence (used for sensor fallback and 2D gaze)
        action_seq = episode_dict.get(
            "motor_system", {}
        ).get("action_sequence", [])

        # Fallback: constant agent position from motor_system.action_sequence
        if sensor_positions is None and action_seq:
            state = action_seq[0][1]
            agent_id = next(iter(state))
            agent_pos = np.array(state[agent_id]["position"], float)
            sensor_positions = np.tile(agent_pos, (n_steps, 1))

        # Determine available color modes
        modes = ["step_order"]
        for feat in ("edge_strength", "coherence"):
            if feat in features:
                modes.append(feat)
        for feat in ("on_object", "pose_from_edge"):
            if feat in features:
                modes.append(feat)
        if "hsv" in features:
            modes.append("hsv_hue")

        return cls(
            object_name=object_name,
            locations=locations,
            features=features,
            stepwise_targets=stepwise_targets,
            n_steps=n_steps,
            available_color_modes=modes,
            object_position=object_position,
            object_rotation=object_rotation,
            sensor_positions=sensor_positions,
            is_2d=is_2d,
            action_sequence=action_seq,
        )


def load_episode(json_path, episode=None):
    """Load one episode from detailed_run_stats.json (JSONL format).

    Returns (episode_key, EpisodeData).
    """
    json_path = Path(json_path).expanduser()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Count available episodes and show summary
    episode_keys = []
    with json_path.open() as f:
        for i, line in enumerate(f):
            episode_keys.append(i)
    print(f"Found {len(episode_keys)} episode(s) in {json_path.name}")

    if episode is not None:
        if episode not in episode_keys:
            raise ValueError(
                f"Episode {episode} not found. Available: {episode_keys}"
            )
        target_episode = episode
    else:
        target_episode = episode_keys[0]

    print(f"Loading episode {target_episode}...")

    # Read the specific line
    with json_path.open() as f:
        for i, line in enumerate(f):
            if i == target_episode:
                raw = json.loads(line)
                # The line has a single key (the episode number)
                ep_key = next(iter(raw.keys()))
                return ep_key, EpisodeData.from_episode_dict(raw[ep_key])

    raise RuntimeError(f"Failed to read episode {target_episode}")


def load_object_mesh(object_name, mesh_dir, object_rotation, alpha=1.0):
    """Load .glb mesh via trimesh with native UV texture mapping.

    Args:
        object_name: Folder name under mesh_dir (e.g. "040_grid_cube_12cm").
        mesh_dir: Base directory containing per-object mesh folders.
        object_rotation: [x, y, z, w] quaternion rotation (scipy convention).
        alpha: Mesh opacity (0 = invisible, 1 = opaque).

    Returns:
        (vedo.Mesh, trimesh.Trimesh) or (None, None) if the file is not found.
        The trimesh geometry has the same rotation applied as the vedo mesh.
    """
    glb_path = Path(mesh_dir).expanduser() / object_name / "textured.glb"
    if not glb_path.exists():
        print(f"[mesh] Not found: {glb_path}")
        return None, None

    scene = trimesh.load(str(glb_path))

    # Extract the first geometry from the scene
    if hasattr(scene, "geometry") and scene.geometry:
        geom = next(iter(scene.geometry.values()))
    elif hasattr(scene, "vertices"):
        geom = scene
    else:
        print(f"[mesh] No geometry found in {glb_path}")
        return None, None

    verts = np.asarray(geom.vertices, float)
    faces = np.asarray(geom.faces, int)

    # Apply rotation — mesh stays at origin to match the point cloud
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(object_rotation)  # xyzw
    verts = rot.apply(verts)

    # Build trimesh with rotated vertices (for raycasting)
    tri_geom = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    mesh = Mesh([verts, faces])

    # Apply UV texture natively (same approach as YCBMeshLoader)
    has_texture = (
        hasattr(geom.visual, "uv")
        and geom.visual.uv is not None
        and hasattr(geom.visual, "material")
        and getattr(geom.visual.material, "baseColorTexture", None) is not None
    )
    if has_texture:
        mesh.texture(
            tname=np.array(geom.visual.material.baseColorTexture),
            tcoords=geom.visual.uv,
        )
    else:
        mesh.color("lightblue")

    mesh.alpha(alpha)

    print(f"[mesh] Loaded {object_name}: {len(verts)} verts, {len(faces)} faces")
    return mesh, tri_geom


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class TrainingStepVisualizer:
    """Interactive step-through visualizer for training episodes."""

    def __init__(self, data: EpisodeData, mesh=None, sensor_positions=None):
        self.data = data
        self.mesh = mesh
        self.sensor_positions = (
            np.asarray(sensor_positions, float)
            if sensor_positions is not None
            else None
        )
        self.plotter = Plotter(size=(1400, 1000), title=f"Training: {data.object_name}")

        self._dynamic_actors = []
        self._current_step = 0
        self._color_mode_idx = 0
        self._color_modes = data.available_color_modes
        self._button = None
        self._path_button = None
        self._show_path = True

    def _get_colors(self, step):
        """Compute per-point RGB colors for locations[:step+1]."""
        n = step + 1
        mode = self._color_modes[self._color_mode_idx]

        if mode == "step_order":
            return scalar_to_rgb(np.arange(n), vmin=0, vmax=max(self.data.n_steps - 1, 1))
        elif mode in ("edge_strength", "coherence"):
            vals = self.data.features[mode][:n]
            return scalar_to_rgb(vals, vmin=0, vmax=1)
        elif mode in ("on_object", "pose_from_edge"):
            vals = self.data.features[mode][:n]
            return binary_to_rgb(vals)
        elif mode == "hsv_hue":
            hue = self.data.features["hsv"][:n, 0]
            return scalar_to_rgb(hue, vmin=0, vmax=1)
        else:
            return scalar_to_rgb(np.arange(n), vmin=0, vmax=max(n - 1, 1))

    def _format_info(self, step):
        """Build info text for the current step."""
        lines = [
            f"Object: {self.data.object_name}",
            f"Step: {step + 1} / {self.data.n_steps}",
            f"Color: {self._color_modes[self._color_mode_idx]}",
        ]

        if self.sensor_positions is not None:
            p = self.sensor_positions[step]
            lines.append(f"Agent:  [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
        loc = self.data.locations[step]
        lines.append(f"Point:  [{loc[0]:.4f}, {loc[1]:.4f}, {loc[2]:.4f}]")

        lines.append("")

        # Feature values at current step
        for key in ("on_object", "edge_strength", "coherence", "pose_from_edge"):
            if key in self.data.features:
                val = self.data.features[key][step]
                lines.append(f"  {key}: {val:.3f}")

        if "hsv" in self.data.features:
            h, s, v = self.data.features["hsv"][step]
            lines.append(f"  hsv: ({h:.2f}, {s:.2f}, {v:.2f})")

        pc_key = next(
            (k for k in ("principal_curvatures_log", "principal_curvatures")
             if k in self.data.features), None
        )
        if pc_key is not None:
            pc = self.data.features[pc_key][step]
            lines.append(f"  curvatures: ({pc[0]:.3f}, {pc[1]:.3f})")

        if self.data.stepwise_targets:
            idx = min(step, len(self.data.stepwise_targets) - 1)
            lines.append(f"  target: {self.data.stepwise_targets[idx]}")

        return "\n".join(lines)

    def _redraw(self, step):
        """Rebuild all dynamic objects for the given step."""
        # Remove previous dynamic actors
        if self._dynamic_actors:
            self.plotter.remove(*self._dynamic_actors)
        self._dynamic_actors = []

        locs = self.data.locations
        n = step + 1

        # Accumulated point cloud and trajectory (togglable)
        if self._show_path:
            colors = self._get_colors(step)
            pc = Points(locs[:n], r=6)
            pc.pointcolors = colors
            self._dynamic_actors.append(pc)

            if n >= 2:
                traj = Line(locs[:n], c="gray", alpha=0.4, lw=2)
                self._dynamic_actors.append(traj)

        # Gaze intersection sphere on the surface
        gaze_hit = Sphere(pos=tuple(locs[step]), r=0.002).color("red")
        self._dynamic_actors.append(gaze_hit)

        # Edge tangent arrows
        feats = self.data.features
        if "pose_from_edge" in feats and "pose_vectors" in feats:
            edge_mask = feats["pose_from_edge"][:n].astype(bool)
            edge_indices = np.where(edge_mask)[0]
            if len(edge_indices) > 0:
                tangents = _normalize_rows(feats["pose_vectors"][edge_indices, 1, :])
                coherence_vals = (
                    feats["coherence"][edge_indices]
                    if "coherence" in feats
                    else np.ones(len(edge_indices))
                )
                arrow_scale = 0.0025
                for i, idx in enumerate(edge_indices):
                    t_vec = tangents[i]
                    if not np.isfinite(t_vec).all():
                        continue
                    half = (arrow_scale * coherence_vals[i]) / 2
                    p = locs[idx]
                    seg = Line(
                        p - half * t_vec, p + half * t_vec,
                        c="black", lw=2, alpha=0.7,
                    )
                    self._dynamic_actors.append(seg)

        # Agent marker and gaze line (per-step sensor position)
        if self.sensor_positions is not None:
            spos = self.sensor_positions[step]
            agent_marker = Sphere(pos=tuple(spos), r=0.005).color("orange")
            self._dynamic_actors.append(agent_marker)

            gaze = Line(
                spos, locs[step],
                c="yellow", alpha=0.6, lw=1,
            )
            self._dynamic_actors.append(gaze)

        # Info panel
        info = Text2D(
            self._format_info(step),
            pos="top-left",
            s=0.7,
            font="Courier",
            c="black",
        )
        self._dynamic_actors.append(info)

        self.plotter.add(*self._dynamic_actors)

    def _on_slider(self, widget, _event):
        step = int(round(widget.value))
        step = max(0, min(step, self.data.n_steps - 1))
        if step != self._current_step:
            self._current_step = step
            self._redraw(step)
            self.plotter.render()

    def _on_color_toggle(self, _obj, _event):
        self._color_mode_idx = (self._color_mode_idx + 1) % len(self._color_modes)
        self._button.switch()
        self._redraw(self._current_step)
        self.plotter.render()

    def _on_path_toggle(self, _obj, _event):
        self._show_path = not self._show_path
        self._path_button.switch()
        self._redraw(self._current_step)
        self.plotter.render()

    def show(self):
        """Build scene, add widgets, launch interactive plotter."""
        # Static scene elements (persist across redraws)
        if self.mesh is not None:
            self.plotter.add(self.mesh)

        # Sensor trajectory (faint line showing all positions)
        if self.sensor_positions is not None and len(self.sensor_positions) >= 2:
            sensor_traj = Line(
                self.sensor_positions, c="orange", alpha=0.3, lw=1,
            )
            self.plotter.add(sensor_traj)

        # Initial draw
        self._redraw(0)

        # Step slider
        self.plotter.add_slider(
            self._on_slider,
            xmin=0,
            xmax=self.data.n_steps - 1,
            value=0,
            pos=[(0.15, 0.05), (0.85, 0.05)],
            title="Step",
        )

        # Color mode toggle button
        n_modes = len(self._color_modes)
        self._button = self.plotter.add_button(
            self._on_color_toggle,
            states=self._color_modes,
            c=["w"] * n_modes,
            bc=["green4"] * n_modes,
            pos=(0.15, 0.12),
            size=16,
            bold=True,
        )

        # Path visibility toggle button
        self._path_button = self.plotter.add_button(
            self._on_path_toggle,
            states=["Path: ON", "Path: OFF"],
            c=["w", "w"],
            bc=["green4", "red4"],
            pos=(0.85, 0.12),
            size=16,
            bold=True,
        )

        # Camera along depth axis
        locs = self.data.locations
        ax0, ax1, depth_ax = _detect_grid_axes(locs)
        center = locs.mean(axis=0)
        max_range = (locs.max(axis=0) - locs.min(axis=0)).max()
        camera_distance = max_range * 1.5
        camera_pos = center.copy()
        camera_pos[depth_ax] += camera_distance

        viewup = [0, 0, 0]
        viewup[ax1] = 1

        print(f"Grid axes: {ax0}, {ax1}; depth axis: {depth_ax}")
        print(f"Center: {center}")
        print(f"Color modes: {self._color_modes}")

        self.plotter.show(
            axes=dict(xtitle="X", ytitle="Y", ztitle="Z"),
            viewup=viewup,
            camera=dict(
                pos=tuple(camera_pos),
                focal_point=tuple(center),
                view_angle=45,
            ),
            interactive=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MESH_DIR = "~/tbp/data/compositional_objects/meshes"


def main():
    parser = argparse.ArgumentParser(
        description="Interactive training step visualizer for 2D pretraining."
    )
    parser.add_argument(
        "json_path",
        help="Path to detailed_run_stats.json",
    )
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Episode number to visualize (default: first available)",
    )
    parser.add_argument(
        "--mesh_dir", default=DEFAULT_MESH_DIR,
        help="Base directory containing per-object mesh folders "
             f"(default: {DEFAULT_MESH_DIR})",
    )
    parser.add_argument(
        "--no_mesh", action="store_true",
        help="Disable mesh overlay",
    )
    parser.add_argument(
        "--agent_pos", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Override agent position (world coords). Used as fallback when "
             "SM_0 sensor data is not available in the JSON.",
    )
    parser.add_argument(
        "--no_agent", action="store_true",
        help="Disable agent marker and gaze line",
    )
    args = parser.parse_args()

    ep_key, data = load_episode(args.json_path, episode=args.episode)
    print(f"Episode {ep_key}: {data.object_name}, {data.n_steps} steps")
    print(f"Features: {list(data.features.keys())}")
    print(f"Object position: {data.object_position}")

    # Shift point cloud into object-local frame so it aligns with the mesh
    # at origin. For 3D SMs (HabitatSM) locations are in world coords;
    # for 2D SMs (TwoDPoseSM) they are already near origin.
    data.locations = data.locations - data.object_position
    print(f"Points (object-local): center={data.locations.mean(axis=0)}")

    # Load mesh at origin
    mesh = None
    tri_geom = None
    if not args.no_mesh:
        mesh, tri_geom = load_object_mesh(
            data.object_name,
            args.mesh_dir,
            data.object_rotation,
        )

    # Resolve sensor positions: prefer SM_0 data, fall back to --agent_pos
    sensor_positions = None
    if not args.no_agent:
        if data.sensor_positions is not None:
            sensor_positions = data.sensor_positions - data.object_position
            print(
                f"Sensor positions from SM_0: {len(sensor_positions)} steps, "
                f"first={sensor_positions[0]}"
            )
        elif args.agent_pos is not None:
            # Broadcast static position to all steps
            static = np.asarray(args.agent_pos) - data.object_position
            sensor_positions = np.tile(static, (data.n_steps, 1))
            print(f"Agent (static override, object-local): {static}")
        else:
            print(
                "No sensor position data "
                "(SM_0, motor_system missing and --agent_pos not set)"
            )

    # 2D gaze projection: replace surface-parametric [u,v,0] locations with
    # actual mesh intersection points computed from agent gaze directions
    if data.is_2d:
        print("2D data detected (all z=0). Projecting gaze onto mesh...")
        can_project = (
            tri_geom is not None
            and data.action_sequence
            and sensor_positions is not None
        )
        if can_project:
            gaze_dirs = _compute_gaze_directions(
                data.action_sequence, data.n_steps
            )

            agent_local = sensor_positions[0]
            origins = np.tile(agent_local, (data.n_steps, 1))
            hit_points, hit_mask = _raycast_closest(
                origins, gaze_dirs,
                np.asarray(tri_geom.vertices, float),
                np.asarray(tri_geom.faces, int),
            )
            n_hits = int(hit_mask.sum())
            print(f"Gaze raycast: {n_hits}/{data.n_steps} hits")

            # Filter all per-step arrays to keep only hits
            data.locations = hit_points[hit_mask]
            for key in data.features:
                data.features[key] = data.features[key][hit_mask]
            if data.stepwise_targets:
                hit_indices = np.where(hit_mask)[0]
                data.stepwise_targets = [
                    data.stepwise_targets[i] for i in hit_indices
                ]
            sensor_positions = sensor_positions[hit_mask]
            data.n_steps = n_hits
        else:
            reasons = []
            if tri_geom is None:
                reasons.append("no mesh")
            if not data.action_sequence:
                reasons.append("no action_sequence")
            if sensor_positions is None:
                reasons.append("no sensor position")
            print(f"Warning: cannot project 2D gaze ({', '.join(reasons)})")

    viz = TrainingStepVisualizer(data, mesh=mesh, sensor_positions=sensor_positions)
    viz.show()


if __name__ == "__main__":
    main()
