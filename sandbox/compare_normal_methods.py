#!/usr/bin/env python
"""Compare surface normal estimation methods on a tessellated cylinder.

Generates synthetic point cloud patches at positions spanning two mesh faces
of a tessellated cylinder and compares angular errors against the known
analytical normal.

Methods compared:
- Naive: cross-product of tangent vectors from cardinal pixel offsets
- OLS: ordinary least-squares depth regression in sensor frame
- TLS: total least-squares plane fit (unweighted)
- TLS-W: Gaussian-weighted TLS
- Jet: degree-2 polynomial fit, normal from first-order tilt (CGAL-equivalent)

Usage:
    python sandbox/compare_normal_methods.py
"""

import sys

sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt

from tbp.monty.frameworks.utils.sensor_processing import (
    principal_curvatures,
    surface_normal_naive,
    surface_normal_ordinary_least_squares,
    surface_normal_total_least_squares,
)

# -- Cylinder parameters --
CYLINDER_RADIUS = 0.04  # 4 cm
CYLINDER_CENTER_Y = 1.5
N_MESH_FACES = 24  # flat strips around circumference (15 deg/face)
PATCH_WIDTH = 64
# Approximate pixel pitch for zoom=10 at ~0.5m distance.
# Patch footprint: 64 * 0.0003 = 19.2 mm ~ 1.8 face widths.
PIXEL_PITCH = 0.0003  # metres per pixel at the surface


def angle_between(a, b):
    """Angular difference between two unit vectors, in degrees."""
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def orient(normal, reference):
    """Flip *normal* to face the same hemisphere as *reference*."""
    if np.dot(normal, reference) < 0:
        return -normal
    return normal


# ---------------------------------------------------------------------------
# Synthetic tessellated-cylinder patch generation
# ---------------------------------------------------------------------------

def make_patch(theta_center):
    """Build a 64x64 point cloud patch on a tessellated cylinder.

    The cylinder is divided into N_MESH_FACES vertical flat strips (chords).
    Each 3D point is placed on its strip's flat surface rather than on the
    smooth cylinder, faithfully modelling what a renderer returns for a
    tessellated mesh.

    Returns:
        world_pc : ndarray (N, 4) — [x, y, z, semantic_id] in world frame
        sensor_pc : ndarray (N, 4) — same in sensor frame (for OLS)
        world_camera : ndarray (4, 4) — sensor-to-world transform
        analytical_normal : ndarray (3,) — true cylinder normal at centre
        center_id : int
        view_dir : ndarray (3,) — outward radial direction (== analytical_normal)
    """
    n_points = PATCH_WIDTH ** 2
    half = PATCH_WIDTH // 2
    center_id = half + PATCH_WIDTH * half
    face_angle = 2 * np.pi / N_MESH_FACES

    # Pixel grid in row-major order
    cols, rows = np.meshgrid(np.arange(PATCH_WIDTH), np.arange(PATCH_WIDTH))
    dc = cols.ravel() - half  # horizontal offset (theta)
    dr = rows.ravel() - half  # vertical offset (y)

    thetas = theta_center + dc * PIXEL_PITCH / CYLINDER_RADIUS
    ys = CYLINDER_CENTER_Y + dr * PIXEL_PITCH

    # Assign each point to its flat face and interpolate along the chord
    face_idx = np.floor(thetas / face_angle).astype(int)
    face_start = face_idx * face_angle
    face_end = face_start + face_angle
    t = (thetas - face_start) / face_angle  # [0, 1] within face

    xs = (
        CYLINDER_RADIUS * np.cos(face_start)
        + t * CYLINDER_RADIUS * (np.cos(face_end) - np.cos(face_start))
    )
    zs = (
        CYLINDER_RADIUS * np.sin(face_start)
        + t * CYLINDER_RADIUS * (np.sin(face_end) - np.sin(face_start))
    )

    semantic_ids = np.ones(n_points)
    world_pc = np.column_stack([xs, ys, zs, semantic_ids])

    # Analytical normal at centre (smooth cylinder)
    analytical_normal = np.array([
        np.cos(theta_center), 0.0, np.sin(theta_center),
    ])
    view_dir = analytical_normal.copy()

    # --- Build sensor frame for OLS ---
    z_cam = analytical_normal.copy()
    y_cam = np.array([0.0, 1.0, 0.0])
    x_cam = np.cross(y_cam, z_cam)
    xn = np.linalg.norm(x_cam)
    if xn < 1e-10:
        y_cam = np.array([0.0, 0.0, 1.0])
        x_cam = np.cross(y_cam, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)

    R = np.column_stack([x_cam, y_cam, z_cam])
    world_camera = np.eye(4)
    world_camera[:3, :3] = R

    sensor_xyz = (R.T @ world_pc[:, :3].T).T
    sensor_pc = np.column_stack([sensor_xyz, semantic_ids])

    return world_pc, sensor_pc, world_camera, analytical_normal, center_id, view_dir


# ---------------------------------------------------------------------------
# Run all methods on one patch
# ---------------------------------------------------------------------------

def evaluate_at(theta_center):
    """Return dict of angular errors (degrees) for each method."""
    wpc, spc, wcam, ana, cid, vdir = make_patch(theta_center)
    errors = {}

    # Naive
    n, ok = surface_normal_naive(wpc.copy())
    errors["Naive"] = angle_between(orient(n, vdir), ana) if ok else np.nan

    # OLS
    n, ok = surface_normal_ordinary_least_squares(spc.copy(), wcam, cid)
    errors["OLS"] = angle_between(orient(n, vdir), ana) if ok else np.nan

    # TLS
    n, ok = surface_normal_total_least_squares(wpc.copy(), cid, vdir)
    errors["TLS"] = angle_between(orient(n, vdir), ana) if ok else np.nan

    # TLS Weighted
    n, ok = surface_normal_total_least_squares(
        wpc.copy(), cid, vdir, weighted=True,
    )
    errors["TLS-W"] = angle_between(orient(n, vdir), ana) if ok else np.nan

    # Jet (seeded from unweighted TLS — matches the production pipeline)
    tls_n, _ = surface_normal_total_least_squares(wpc.copy(), cid, vdir)
    _, _, _, _, ok, jet_n = principal_curvatures(wpc.copy(), cid, tls_n)
    if jet_n is not None:
        errors["Jet"] = angle_between(orient(jet_n, vdir), ana)
    else:
        errors["Jet"] = np.nan

    return errors


# ---------------------------------------------------------------------------
# Sweep and plot
# ---------------------------------------------------------------------------

def main():
    face_angle = 2 * np.pi / N_MESH_FACES
    face_deg = np.degrees(face_angle)

    # Sweep across 2 full faces centred on a face boundary
    n_samples = 300
    boundary = face_angle  # boundary between face-0 and face-1
    offsets = np.linspace(-face_angle, face_angle, n_samples)
    thetas = boundary + offsets
    offsets_deg = np.degrees(offsets)

    methods = ["Naive", "OLS", "TLS", "TLS-W", "Jet"]
    all_errors = {m: np.empty(n_samples) for m in methods}

    for i, theta in enumerate(thetas):
        errs = evaluate_at(theta)
        for m in methods:
            all_errors[m][i] = errs[m]

    # ---- Figure ----
    fig, (ax_line, ax_bar) = plt.subplots(
        2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 1]},
    )

    palette = {
        "Naive": "#e74c3c",
        "OLS": "#e67e22",
        "TLS": "#3498db",
        "TLS-W": "#2ecc71",
        "Jet": "#9b59b6",
    }

    for m in methods:
        ax_line.plot(
            offsets_deg, all_errors[m],
            label=m, color=palette[m], linewidth=1.5,
        )

    # Face boundary markers
    for bd in np.arange(-2, 3) * face_deg:
        ax_line.axvline(bd, color="grey", ls="--", alpha=0.3, lw=0.8)
    ax_line.set_xlabel("Angular offset from face boundary (degrees)")
    ax_line.set_ylabel("Angular error vs analytical normal (degrees)")
    ax_line.set_title(
        f"Surface Normal Estimation Error — Tessellated Cylinder\n"
        f"r = {CYLINDER_RADIUS * 100:.0f} cm, {N_MESH_FACES} faces "
        f"({face_deg:.1f} deg/face), "
        f"{PATCH_WIDTH}x{PATCH_WIDTH} patch, "
        f"pixel pitch = {PIXEL_PITCH * 1000:.2f} mm"
    )
    ax_line.legend(loc="upper right")
    ax_line.set_xlim(offsets_deg[0], offsets_deg[-1])
    ax_line.grid(True, alpha=0.2)

    # Summary bar chart
    means = [np.nanmean(all_errors[m]) for m in methods]
    maxes = [np.nanmax(all_errors[m]) for m in methods]
    x = np.arange(len(methods))
    w = 0.35
    bars_mean = ax_bar.bar(
        x - w / 2, means, w, label="Mean",
        color=[palette[m] for m in methods], alpha=0.7,
    )
    bars_max = ax_bar.bar(
        x + w / 2, maxes, w, label="Max",
        color=[palette[m] for m in methods], alpha=0.3, edgecolor="black",
    )
    ax_bar.bar_label(bars_mean, fmt="%.2f", fontsize=8, padding=2)
    ax_bar.bar_label(bars_max, fmt="%.2f", fontsize=8, padding=2)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(methods)
    ax_bar.set_ylabel("Error (deg)")
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = "sandbox/normal_method_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    plt.show()

    # Console summary
    print(f"\n{'Method':<10} {'Mean':>8} {'Max':>8} {'Std':>8}")
    print("-" * 38)
    for m in methods:
        e = all_errors[m][~np.isnan(all_errors[m])]
        print(f"{m:<10} {np.mean(e):8.3f} {np.max(e):8.3f} {np.std(e):8.3f}")


if __name__ == "__main__":
    main()
