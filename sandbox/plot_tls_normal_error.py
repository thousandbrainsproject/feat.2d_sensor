#!/usr/bin/env python3
"""Plot spatial structure of TLS normal estimation errors on a cylinder.

Loads diagnostics saved by AnalyticalNormalCylinderSM and produces four plots:
1. Error heatmap: angular error as f(theta, y) with face boundary overlay
2. Error histogram: distribution with mean/median markers
3. Error decomposition: tangential vs axial components against theta
4. Face boundary correlation: error vs distance to nearest face boundary

Usage:
    python sandbox/plot_tls_normal_error.py
    python sandbox/plot_tls_normal_error.py --n_faces 128 --save_dir ~/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_diagnostics(data_path: Path) -> dict[str, np.ndarray]:
    npz = np.load(data_path)
    return {k: npz[k] for k in npz.files}


def to_cylinder_coords(
    locations: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3D locations to cylinder surface coordinates (theta, y).

    Args:
        locations: (N, 3) array of 3D points.
        center: (3,) cylinder center.

    Returns:
        theta: (N,) azimuthal angle in radians via atan2(z - cz, x - cx).
        y: (N,) axial coordinate (world Y minus center Y).
    """
    local = locations - center
    theta = np.arctan2(local[:, 2], local[:, 0])
    y = local[:, 1]
    return theta, y


def decompose_error(
    tls_normals: np.ndarray,
    analytical_normals: np.ndarray,
    locations: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose normal error into tangential and axial components.

    Tangential (circumferential) errors cause parallel transport drift.
    Axial (Y-direction) errors are mostly benign for 2D mapping.

    Args:
        tls_normals: (N, 3) TLS-estimated normals.
        analytical_normals: (N, 3) ground-truth analytical normals.
        locations: (N, 3) surface points.
        center: (3,) cylinder center.

    Returns:
        tangential_err_deg: (N,) signed tangential error in degrees.
        axial_err_deg: (N,) signed axial error in degrees.
    """
    error_vec = tls_normals - analytical_normals
    theta, _ = to_cylinder_coords(locations, center)

    # Tangential direction: [-sin(theta), 0, cos(theta)]
    tang_dir = np.column_stack([-np.sin(theta), np.zeros(len(theta)),
                                np.cos(theta)])
    # Axial direction: [0, 1, 0]
    axial_dir = np.array([0.0, 1.0, 0.0])

    tang_component = np.sum(error_vec * tang_dir, axis=1)
    axial_component = np.dot(error_vec, axial_dir)

    # Convert from normal-vector difference to approximate angular error.
    # For small angles, |delta_n| ~ sin(delta_angle) ~ delta_angle
    tangential_err_deg = np.degrees(np.arcsin(
        np.clip(tang_component, -1.0, 1.0)
    ))
    axial_err_deg = np.degrees(np.arcsin(
        np.clip(axial_component, -1.0, 1.0)
    ))

    return tangential_err_deg, axial_err_deg


def distance_to_nearest_boundary(
    theta: np.ndarray,
    n_faces: int,
) -> np.ndarray:
    """Angular distance from each theta to the nearest mesh face boundary.

    Face boundaries are at theta = k * (2*pi / n_faces) for k = 0..n_faces-1.

    Returns:
        dist_deg: (N,) distance in degrees to the nearest boundary.
    """
    face_width = 2 * np.pi / n_faces
    # Distance within current face sector [0, face_width)
    theta_in_sector = theta % face_width
    # Minimum of distance to left boundary (0) or right boundary (face_width)
    dist_rad = np.minimum(theta_in_sector, face_width - theta_in_sector)
    return np.degrees(dist_rad)


def plot_error_heatmap(
    ax: plt.Axes,
    theta_deg: np.ndarray,
    y_mm: np.ndarray,
    errors: np.ndarray,
    n_faces: int,
) -> None:
    """Scatter plot of angular error colored by magnitude, with face boundaries."""
    sc = ax.scatter(theta_deg, y_mm, c=errors, s=4, cmap="hot",
                    vmin=0, vmax=np.percentile(errors, 95), edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Angular error (deg)")

    # Overlay face boundary lines
    boundary_angles = np.linspace(-180, 180, n_faces + 1)
    for b in boundary_angles:
        ax.axvline(b, color="cyan", alpha=0.15, linewidth=0.5)

    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("TLS Normal Error Heatmap")


def plot_error_histogram(
    ax: plt.Axes,
    errors: np.ndarray,
) -> None:
    """Histogram of angular errors with mean/median markers."""
    ax.hist(errors, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    ax.axvline(mean_err, color="red", linestyle="--",
               label=f"Mean: {mean_err:.3f} deg")
    ax.axvline(median_err, color="orange", linestyle="-",
               label=f"Median: {median_err:.3f} deg")
    ax.set_xlabel("Angular error (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()


def plot_error_decomposition(
    ax: plt.Axes,
    theta_deg: np.ndarray,
    tang_err: np.ndarray,
    axial_err: np.ndarray,
) -> None:
    """Tangential vs axial error components plotted against theta."""
    ax.scatter(theta_deg, tang_err, s=2, alpha=0.5, label="Tangential",
               color="red")
    ax.scatter(theta_deg, axial_err, s=2, alpha=0.5, label="Axial",
               color="blue")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Signed error (deg)")
    ax.set_title("Error Decomposition: Tangential vs Axial")
    ax.legend(markerscale=4)

    tang_rms = np.sqrt(np.mean(tang_err**2))
    axial_rms = np.sqrt(np.mean(axial_err**2))
    ax.text(0.02, 0.98,
            f"Tang RMS: {tang_rms:.3f} deg\nAxial RMS: {axial_rms:.3f} deg",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))


def plot_face_boundary_correlation(
    ax: plt.Axes,
    theta: np.ndarray,
    errors: np.ndarray,
    n_faces: int,
) -> None:
    """Error vs angular distance to nearest face boundary with binned mean."""
    dist_deg = distance_to_nearest_boundary(theta, n_faces)
    max_dist = np.degrees(np.pi / n_faces)

    ax.scatter(dist_deg, errors, s=2, alpha=0.3, color="gray")

    # Binned mean overlay
    n_bins = 30
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (dist_deg >= bin_edges[i]) & (dist_deg < bin_edges[i + 1])
        if mask.any():
            bin_means[i] = np.mean(errors[mask])

    ax.plot(bin_centers, bin_means, "r-o", markersize=3, linewidth=1.5,
            label="Binned mean")
    ax.set_xlabel("Distance to nearest face boundary (deg)")
    ax.set_ylabel("Angular error (deg)")
    ax.set_title("Face Boundary Correlation")
    ax.legend()


def main():
    parser = argparse.ArgumentParser(
        description="Plot TLS normal error diagnostics from cylinder experiment"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="~/tbp/results/tls_diagnostics/normal_error.npz",
        help="Path to normal_error.npz diagnostics file",
    )
    parser.add_argument(
        "--n_faces",
        type=int,
        default=256,
        help="Number of mesh faces around the cylinder circumference",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save figures. If omitted, shows interactive plots.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Diagnostics file not found: {data_path}\n"
            "Run the analytical cylinder experiment first to generate it."
        )

    data = load_diagnostics(data_path)
    locations = data["locations"]
    tls_normals = data["tls_normals"]
    analytical_normals = data["analytical_normals"]
    errors = data["angular_errors_deg"]
    center = data["cylinder_center"]

    theta, y = to_cylinder_coords(locations, center)
    theta_deg = np.degrees(theta)
    y_mm = y * 1000  # meters to mm

    tang_err, axial_err = decompose_error(
        tls_normals, analytical_normals, locations, center
    )

    print(f"Loaded {len(errors)} samples")
    print(f"  Mean error:   {np.mean(errors):.4f} deg")
    print(f"  Median error: {np.median(errors):.4f} deg")
    print(f"  Max error:    {np.max(errors):.4f} deg")
    print(f"  Tang RMS:     {np.sqrt(np.mean(tang_err**2)):.4f} deg")
    print(f"  Axial RMS:    {np.sqrt(np.mean(axial_err**2)):.4f} deg")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TLS Normal Estimation Error — Cylinder Surface", fontsize=14)

    plot_error_heatmap(axes[0, 0], theta_deg, y_mm, errors, args.n_faces)
    plot_error_histogram(axes[0, 1], errors)
    plot_error_decomposition(axes[1, 0], theta_deg, tang_err, axial_err)
    plot_face_boundary_correlation(axes[1, 1], theta, errors, args.n_faces)

    plt.tight_layout()

    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "tls_normal_error.png", dpi=200,
                    bbox_inches="tight")
        fig.savefig(save_dir / "tls_normal_error.pdf", bbox_inches="tight")
        print(f"Saved figures to {save_dir}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
