"""Mesh face normal vs TLS-estimated normal deviation visualizer.

Loads a 3D pretrained model checkpoint and the corresponding cylinder mesh,
computes the angular deviation between TLS-estimated surface normals (stored
in pose_vectors[:,0]) and the ground-truth mesh face normals, then visualizes
the results interactively with vedo.

Usage:
    python sandbox/visualize_normal_deviation.py
    python sandbox/visualize_normal_deviation.py --object 042_grid_cylinder_6cm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
import torch
from vedo import Line, Mesh, Plotter, Points, Text2D

from model_loading_utils import load_object_model


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

    h = 0.66 * (1.0 - t)  # blue -> red
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


def load_cylinder_mesh(object_name, mesh_dir):
    """Load a cylinder .glb mesh and return (vedo.Mesh, trimesh.Trimesh).

    No rotation is applied — cylinder meshes are axis-aligned at origin.
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
        vmesh.color("lightblue")

    print(f"[mesh] {object_name}: {len(verts)} verts, {len(faces)} faces")
    return vmesh, tri_geom


# ---------------------------------------------------------------------------
# Deviation computation
# ---------------------------------------------------------------------------

def compute_deviations(points, tls_normals, tri_geom):
    """Compute angular deviations between TLS normals and mesh/analytical normals.

    Args:
        points: (N, 3) object-local point positions.
        tls_normals: (N, 3) TLS-estimated surface normals.
        tri_geom: trimesh.Trimesh of the cylinder mesh.

    Returns:
        dict with keys:
            face_normals: (N, 3) mesh face normals at closest faces
            analytical_normals: (N, 3) radial outward normals [x, 0, z]/|[x, 0, z]|
            dev_face_deg: (N,) angular deviation vs face normal (degrees)
            dev_analytical_deg: (N,) angular deviation vs analytical normal (degrees)
            closest_face_ids: (N,) face index per point
    """
    # Brute-force nearest face via centroids (128 faces, fast enough)
    centroids = tri_geom.triangles.mean(axis=1)  # (F, 3)
    dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    face_ids = np.argmin(dists, axis=1)

    face_normals = np.asarray(tri_geom.face_normals[face_ids], float)
    face_normals = _normalize_rows(face_normals)

    # Analytical cylinder normal: radial outward in XZ, zero Y
    radial = points.copy()
    radial[:, 1] = 0.0
    analytical_normals = _normalize_rows(radial)

    tls_normals = _normalize_rows(tls_normals)

    def _angular_deviation(a, b):
        dots = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(np.abs(dots)))

    dev_face = _angular_deviation(tls_normals, face_normals)
    dev_analytical = _angular_deviation(tls_normals, analytical_normals)

    return {
        "face_normals": face_normals,
        "analytical_normals": analytical_normals,
        "dev_face_deg": dev_face,
        "dev_analytical_deg": dev_analytical,
        "closest_face_ids": face_ids,
    }


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

OBJECT_POSITION = np.array([0.0, 1.5, 0.0])
ARROW_LENGTH = 0.008


class NormalDeviationVisualizer:
    """Interactive vedo visualization of TLS vs mesh normal deviations."""

    def __init__(self, model_path, mesh_dir, object_names):
        self.model_path = Path(model_path).expanduser()
        self.mesh_dir = Path(mesh_dir).expanduser()
        self.object_names = object_names
        self.obj_idx = 0

        self.plotter = Plotter(size=(1600, 1000), title="Normal Deviation")
        self._static_actors = []
        self._dynamic_actors = []
        self._show_face_normals = True
        self._show_analytical = False

    def _load_object(self, object_name):
        """Load model data and mesh for a single object."""
        model_data = load_object_model(self.model_path, object_name)
        points = model_data["points"] - OBJECT_POSITION

        pv = np.asarray(model_data["features"]["pose_vectors"], float)
        if pv.ndim == 2 and pv.shape[1] == 9:
            pv = pv.reshape(-1, 3, 3)
        tls_normals = pv[:, 0, :]

        vmesh, tri_geom = load_cylinder_mesh(object_name, self.mesh_dir)

        devs = compute_deviations(points, tls_normals, tri_geom)

        return {
            "object_name": object_name,
            "points": points,
            "tls_normals": _normalize_rows(tls_normals),
            "vmesh": vmesh,
            "tri_geom": tri_geom,
            **devs,
        }

    def _clear_dynamic(self):
        if self._dynamic_actors:
            self.plotter.remove(*self._dynamic_actors)
        self._dynamic_actors = []

    def _clear_static(self):
        if self._static_actors:
            self.plotter.remove(*self._static_actors)
        self._static_actors = []

    def _build_scene(self, data):
        """Build all scene actors for a given object's data."""
        self._clear_dynamic()
        self._clear_static()

        points = data["points"]
        n = len(points)
        dev_face = data["dev_face_deg"]
        tls_normals = data["tls_normals"]
        face_normals = data["face_normals"]
        analytical_normals = data["analytical_normals"]

        # Semi-transparent mesh
        vmesh = data["vmesh"]
        vmesh.alpha(0.15)
        self._static_actors.append(vmesh)

        # Color-coded point spheres (blue=0 deg, red=max deg)
        colors = scalar_to_rgb(dev_face, vmin=0, vmax=max(dev_face.max(), 0.5))
        pc = Points(points, r=5)
        pc.pointcolors = colors
        self._dynamic_actors.append(pc)

        # TLS normal arrows (red)
        for i in range(n):
            tip = points[i] + ARROW_LENGTH * tls_normals[i]
            arr = Line(points[i], tip, c="red", lw=2)
            self._dynamic_actors.append(arr)

        # Face normal arrows (cyan, togglable)
        if self._show_face_normals:
            for i in range(n):
                tip = points[i] + ARROW_LENGTH * face_normals[i]
                arr = Line(points[i], tip, c="cyan", lw=2, alpha=0.7)
                self._dynamic_actors.append(arr)

        # Analytical normal arrows (green, togglable)
        if self._show_analytical:
            for i in range(n):
                tip = points[i] + ARROW_LENGTH * analytical_normals[i]
                arr = Line(points[i], tip, c="green", lw=2, alpha=0.7)
                self._dynamic_actors.append(arr)

        # Stats overlay
        dev_a = data["dev_analytical_deg"]
        tls_y = np.abs(data["tls_normals"][:, 1])
        stats_text = (
            f"Object: {data['object_name']}\n"
            f"Points: {n}\n"
            f"\n"
            f"vs Face Normal (deg)\n"
            f"  mean: {dev_face.mean():.2f}\n"
            f"  max:  {dev_face.max():.2f}\n"
            f"  std:  {dev_face.std():.2f}\n"
            f"\n"
            f"vs Analytical Normal (deg)\n"
            f"  mean: {dev_a.mean():.2f}\n"
            f"  max:  {dev_a.max():.2f}\n"
            f"  std:  {dev_a.std():.2f}\n"
            f"\n"
            f"TLS Normal |Y| Component\n"
            f"  mean: {tls_y.mean():.6f}\n"
            f"  max:  {tls_y.max():.6f}\n"
            f"  => max tilt: {np.degrees(np.arcsin(np.clip(tls_y.max(), 0, 1))):.2f} deg"
        )
        info = Text2D(stats_text, pos="top-left", s=0.65, font="Courier", c="black")
        self._dynamic_actors.append(info)

        # Legend
        legend = Text2D(
            "Red = TLS  |  Cyan = Face  |  Green = Analytical",
            pos="bottom-left", s=0.6, font="Courier", c="black",
        )
        self._dynamic_actors.append(legend)

        self.plotter.add(*self._static_actors)
        self.plotter.add(*self._dynamic_actors)

    def _on_cycle_object(self, _obj, _event):
        self.obj_idx = (self.obj_idx + 1) % len(self.object_names)
        name = self.object_names[self.obj_idx]
        print(f"[viz] Switching to {name}...")
        data = self._load_object(name)
        self._current_data = data
        self._build_scene(data)
        self.plotter.render()

    def _on_toggle_face(self, _obj, _event):
        self._show_face_normals = not self._show_face_normals
        self._build_scene(self._current_data)
        self.plotter.render()

    def _on_toggle_analytical(self, _obj, _event):
        self._show_analytical = not self._show_analytical
        self._build_scene(self._current_data)
        self.plotter.render()

    def show(self, initial_object=None):
        """Load initial object and launch the interactive viewer."""
        if initial_object and initial_object in self.object_names:
            self.obj_idx = self.object_names.index(initial_object)

        name = self.object_names[self.obj_idx]
        print(f"[viz] Loading {name}...")
        self._current_data = self._load_object(name)
        self._build_scene(self._current_data)

        # Buttons
        btn_y = 0.06
        if len(self.object_names) > 1:
            labels = [f" {n} " for n in self.object_names]
            self.plotter.add_button(
                self._on_cycle_object,
                pos=(0.85, btn_y),
                states=labels,
                size=16,
                font="Calco",
            )
            btn_y += 0.07

        self.plotter.add_button(
            self._on_toggle_face,
            pos=(0.85, btn_y),
            states=[" Face: ON ", " Face: OFF "],
            size=16,
            font="Calco",
        )
        btn_y += 0.07

        self.plotter.add_button(
            self._on_toggle_analytical,
            pos=(0.85, btn_y),
            states=[" Analytical: OFF ", " Analytical: ON "],
            size=16,
            font="Calco",
        )

        # Camera: look along +Z toward cylinder front face
        pts = self._current_data["points"]
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

DEFAULT_BASE_PATH = (
    "~/tbp/results/monty/pretrained_models/2d_sensor"
)
DEFAULT_MESH_DIR = "~/tbp/data/compositional_objects/meshes"


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TLS normal deviation against mesh face normals."
    )
    parser.add_argument(
        "--run_name", default="cylinder_grid_learning_3d",
        help="Pretraining run name (default: cylinder_grid_learning_3d)",
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

    model_path = (
        Path(args.base_path).expanduser()
        / args.run_name
        / "pretrained"
        / "model.pt"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Discover available objects
    state = torch.load(model_path, map_location="cpu")
    gm = state["lm_dict"][0]["graph_memory"]
    object_names = sorted(gm.keys())
    print(f"Objects: {object_names}")
    del state

    viz = NormalDeviationVisualizer(model_path, args.mesh_dir, object_names)
    viz.show(initial_object=args.object)


if __name__ == "__main__":
    main()
