#!/usr/bin/env python
"""Analyze spurious holonomy on a tessellated mesh.

Computes the discrete Gaussian curvature (angular deficit) at each vertex
and shows how it deviates from the expected smooth-surface curvature.

For a cylinder (K=0 everywhere on the smooth surface), any nonzero angular
deficit at interior vertices represents spurious holonomy that will cause
parallel-transport drift.

Theory:
-------
On a polyhedral surface, the discrete Gaussian curvature at vertex v is:

    K_v = 2pi - sum(theta_i)

where theta_i are the interior angles of all faces incident on v. This is
the "angular deficit" and represents concentrated Gaussian curvature.

By the discrete Gauss-Bonnet theorem:
    sum_v(K_v) + sum_boundary(kappa_g) = 2pi * chi(M)

For an open cylinder mesh (chi=1 for a disk-like topology, chi=0 for an
annular topology), any spurious curvature at interior vertices will cause
holonomy when the parallel-transported frame passes near those vertices.

Usage:
    python sandbox/mesh_holonomy.py path/to/mesh.glb [--expected-K 0.0]

    # Analyze a specific grid cylinder:
    python sandbox/mesh_holonomy.py \
        ~/tbp/data/compositional_objects/meshes/044_grid_cylinder_4cm_256/textured.glb

    # Compare all grid cylinders:
    python sandbox/mesh_holonomy.py --compare-cylinders
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def compute_angular_deficit(mesh: trimesh.Trimesh) -> dict:
    """Compute discrete Gaussian curvature at every vertex.

    Returns a dict with:
        - vertex_curvature: array of K_v for each vertex
        - is_boundary: boolean array marking boundary vertices
        - total_curvature_interior: sum of K_v for interior vertices
        - total_curvature_boundary: sum of K_v for boundary vertices
        - total_curvature: sum of all K_v
        - euler_characteristic: V - E + F
        - face_angles: (n_faces, 3) array of interior angles per face
    """
    vertices = mesh.vertices
    faces = mesh.faces
    n_verts = len(vertices)
    n_faces = len(faces)

    # Compute interior angles for each face
    face_angles = np.zeros((n_faces, 3))
    for fi in range(n_faces):
        v0, v1, v2 = vertices[faces[fi]]
        edges = [v1 - v0, v2 - v1, v0 - v2]
        for j in range(3):
            e_in = -edges[(j - 1) % 3]  # edge arriving at vertex j
            e_out = edges[j]             # edge leaving vertex j
            cos_angle = np.dot(e_in, e_out) / (
                np.linalg.norm(e_in) * np.linalg.norm(e_out) + 1e-30
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            face_angles[fi, j] = np.arccos(cos_angle)

    # Sum angles at each vertex
    angle_sum = np.zeros(n_verts)
    for fi in range(n_faces):
        for j in range(3):
            angle_sum[faces[fi, j]] += face_angles[fi, j]

    # Identify boundary vertices (boundary edge appears in only one face)
    edge_count: dict[tuple[int, int], int] = {}
    for fi in range(n_faces):
        for j in range(3):
            e = tuple(sorted([faces[fi, j], faces[fi, (j + 1) % 3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = {e for e, c in edge_count.items() if c == 1}
    boundary_verts: set[int] = set()
    for e in boundary_edges:
        boundary_verts.add(e[0])
        boundary_verts.add(e[1])

    is_boundary = np.array([i in boundary_verts for i in range(n_verts)])

    # Angular deficit: K_v = 2pi - sum(angles) for interior vertices
    # For boundary vertices: K_v = pi - sum(angles)
    vertex_curvature = np.zeros(n_verts)
    for i in range(n_verts):
        if is_boundary[i]:
            vertex_curvature[i] = np.pi - angle_sum[i]
        else:
            vertex_curvature[i] = 2.0 * np.pi - angle_sum[i]

    n_edges = len(edge_count)
    euler = n_verts - n_edges + n_faces

    return dict(
        vertex_curvature=vertex_curvature,
        angle_sum=angle_sum,
        is_boundary=is_boundary,
        total_curvature_interior=float(np.sum(vertex_curvature[~is_boundary])),
        total_curvature_boundary=float(np.sum(vertex_curvature[is_boundary])),
        total_curvature=float(np.sum(vertex_curvature)),
        euler_characteristic=euler,
        expected_gauss_bonnet=2.0 * np.pi * euler,
        n_interior=int(np.sum(~is_boundary)),
        n_boundary=int(np.sum(is_boundary)),
        face_angles=face_angles,
    )


def estimate_holonomy_for_path(
    mesh: trimesh.Trimesh,
    result: dict,
    axis: str = "y",
    n_bands: int = 10,
) -> dict:
    """Estimate holonomy for circumferential loops at different heights.

    For a cylinder aligned along `axis`, we bin interior vertices into
    height bands and sum the angular deficit in each band. A circumferential
    loop passing through a band accumulates roughly that band's total
    curvature as holonomy (frame rotation error).

    Args:
        mesh: The trimesh object.
        result: Output from compute_angular_deficit.
        axis: Cylinder axis ('x', 'y', or 'z').
        n_bands: Number of height bands.

    Returns:
        Dict with per-band holonomy estimates.
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    verts = mesh.vertices
    heights = verts[:, axis_idx]
    K = result["vertex_curvature"]

    # Only look at interior vertices
    is_interior = ~result["is_boundary"]
    interior_heights = heights[is_interior]
    interior_K = K[is_interior]

    if len(interior_heights) == 0:
        return dict(
            bands=[], band_holonomy=[], band_holonomy_deg=[],
            total_interior_holonomy=0.0, total_interior_holonomy_deg=0.0,
        )

    h_min, h_max = interior_heights.min(), interior_heights.max()
    band_edges = np.linspace(h_min, h_max, n_bands + 1)

    bands = []
    band_holonomy = []
    for i in range(n_bands):
        mask = (interior_heights >= band_edges[i]) & (
            interior_heights < band_edges[i + 1]
        )
        if i == n_bands - 1:  # include right edge in last band
            mask |= interior_heights == band_edges[i + 1]

        band_K = interior_K[mask]
        h_center = (band_edges[i] + band_edges[i + 1]) / 2
        bands.append(h_center)
        band_holonomy.append(float(np.sum(band_K)))

    return dict(
        bands=bands,
        band_holonomy=band_holonomy,
        band_holonomy_deg=[np.degrees(h) for h in band_holonomy],
        total_interior_holonomy=float(np.sum(interior_K)),
        total_interior_holonomy_deg=float(np.degrees(np.sum(interior_K))),
    )


def print_report(
    mesh: trimesh.Trimesh,
    result: dict,
    holonomy: dict,
    expected_K: float = 0.0,
    mesh_name: str = "",
) -> None:
    """Print a human-readable analysis report."""
    K = result["vertex_curvature"]
    is_bnd = result["is_boundary"]
    interior_K = K[~is_bnd]

    print("=" * 70)
    print(f"MESH HOLONOMY ANALYSIS{f': {mesh_name}' if mesh_name else ''}")
    print("=" * 70)

    print(f"\nMesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Euler characteristic: {result['euler_characteristic']}")
    print(f"Boundary vertices: {result['n_boundary']}")
    print(f"Interior vertices: {result['n_interior']}")

    print("\n--- Discrete Gauss-Bonnet Check ---")
    print(f"Total curvature (all vertices):  {result['total_curvature']:.6f} rad"
          f"  ({np.degrees(result['total_curvature']):.4f} deg)")
    print(f"Expected (2pi*chi):              {result['expected_gauss_bonnet']:.6f} rad"
          f"  ({np.degrees(result['expected_gauss_bonnet']):.4f} deg)")
    residual = abs(result["total_curvature"] - result["expected_gauss_bonnet"])
    print(f"Gauss-Bonnet residual:           {residual:.2e} rad"
          f"  (should be ~0)")

    print(f"\n--- Interior Vertex Curvature (expected: {expected_K:.4f}) ---")
    if len(interior_K) > 0:
        print(f"Total interior curvature:  {result['total_curvature_interior']:.6f} rad"
              f"  ({np.degrees(result['total_curvature_interior']):.4f} deg)")
        print(f"Mean |K_v| at interior:    {np.mean(np.abs(interior_K)):.6e} rad")
        print(f"Max  |K_v| at interior:    {np.max(np.abs(interior_K)):.6e} rad")
        print(f"Std  K_v at interior:      {np.std(interior_K):.6e} rad")
        print(f"RMS  K_v at interior:      "
              f"{np.sqrt(np.mean(interior_K**2)):.6e} rad"
              f"  ({np.degrees(np.sqrt(np.mean(interior_K**2))):.4f} deg)")

        if expected_K == 0.0:
            print(f"\nFor a cylinder (K=0), ALL interior curvature is spurious.")
            print(f"Spurious curvature budget: "
                  f"{np.degrees(result['total_curvature_interior']):.4f} deg")
    else:
        print("No interior vertices found (all vertices are on the boundary).")

    print("\n--- Boundary Vertex Curvature ---")
    boundary_K = K[is_bnd]
    if len(boundary_K) > 0:
        print(f"Total boundary curvature:  {result['total_curvature_boundary']:.6f} rad"
              f"  ({np.degrees(result['total_curvature_boundary']):.4f} deg)")
        print(f"Mean |K_v| at boundary:    {np.mean(np.abs(boundary_K)):.6e} rad")

    print("\n--- Holonomy Estimates for Circumferential Loops ---")
    if holonomy["bands"]:
        print(f"{'Band Height':>12s} | {'sum K_v (rad)':>13s} | {'sum K_v (deg)':>13s}")
        print("-" * 44)
        for h, kr, kd in zip(
            holonomy["bands"], holonomy["band_holonomy"], holonomy["band_holonomy_deg"]
        ):
            print(f"{h:12.4f} | {kr:13.6f} | {kd:13.4f}")
        print("-" * 44)
        print(f"{'TOTAL':>12s} | "
              f"{holonomy['total_interior_holonomy']:13.6f} | "
              f"{holonomy['total_interior_holonomy_deg']:13.4f}")

        max_band = max(holonomy["band_holonomy_deg"], key=abs)
        print(f"\nWorst-case single-band holonomy: {max_band:.4f} deg")
        print("A circumferential loop accumulates frame rotation ~ band holonomy.")
        print("After N loops, frame error ~ N * band holonomy.")
    else:
        print("No interior vertices to analyze.")

    print("\n--- Interpretation ---")
    total_spurious_deg = abs(np.degrees(result["total_curvature_interior"]))
    if total_spurious_deg < 0.1:
        print("[OK] Negligible spurious holonomy. Normal estimation is likely fine.")
    elif total_spurious_deg < 5.0:
        print("[WARN] Moderate spurious holonomy. May cause visible drift over many "
              "loops.")
    else:
        print("[BAD] Significant spurious holonomy. Will cause substantial map "
              "distortion.")
    print(f"  Total spurious budget: {total_spurious_deg:.4f} deg")


def compute_face_normal_analysis(
    mesh: trimesh.Trimesh,
    axis: str = "y",
    radius: float | None = None,
) -> dict:
    """Analyze face normals vs analytical cylinder normals and dihedral angles.

    For cylinder meshes where all vertices are boundary (no interior angular
    deficit), the transport drift comes from:
    1. Face normals that differ from the smooth-surface analytical normal
    2. Normal discontinuities at edges (dihedral angles) that cause jumps
       when a sensor patch straddles two faces

    Args:
        mesh: The trimesh object.
        axis: Cylinder axis ('x', 'y', or 'z').
        radius: Cylinder radius. If None, estimated from the mesh.
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    radial_axes = [i for i in range(3) if i != axis_idx]

    vertices = mesh.vertices
    faces = mesh.faces
    n_faces = len(faces)

    # Estimate cylinder radius from vertices if not provided
    if radius is None:
        radial_coords = vertices[:, radial_axes]
        radius = float(np.mean(np.linalg.norm(radial_coords, axis=1)))

    # Compute face normals (outward-pointing)
    face_normals = mesh.face_normals.copy()
    # Ensure outward-pointing: face centroid radial direction should agree
    centroids = mesh.triangles_center
    for fi in range(n_faces):
        radial = np.zeros(3)
        radial[radial_axes] = centroids[fi, radial_axes]
        if np.dot(face_normals[fi], radial) < 0:
            face_normals[fi] *= -1

    # Analytical normals at face centroids
    analytical_normals = np.zeros((n_faces, 3))
    for fi in range(n_faces):
        radial = np.zeros(3)
        radial[radial_axes] = centroids[fi, radial_axes]
        norm = np.linalg.norm(radial)
        if norm > 1e-10:
            analytical_normals[fi] = radial / norm

    # Face normal deviation from analytical
    cos_angles = np.sum(face_normals * analytical_normals, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    face_deviation_rad = np.arccos(cos_angles)
    face_deviation_deg = np.degrees(face_deviation_rad)

    # Dihedral angles at edges between adjacent faces
    # Build face adjacency: edge -> list of face indices
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        for j in range(3):
            e = tuple(sorted([faces[fi, j], faces[fi, (j + 1) % 3]]))
            edge_faces.setdefault(e, []).append(fi)

    dihedral_angles_rad = []
    dihedral_edges = []
    for e, flist in edge_faces.items():
        if len(flist) == 2:
            n1 = face_normals[flist[0]]
            n2 = face_normals[flist[1]]
            cos_a = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle = np.arccos(cos_a)
            dihedral_angles_rad.append(angle)
            dihedral_edges.append(e)

    dihedral_angles_rad = np.array(dihedral_angles_rad)
    dihedral_angles_deg = np.degrees(dihedral_angles_rad)

    # Separate longitudinal edges (along axis) from circumferential edges
    # Longitudinal edges connect vertices at different heights but same angle
    # Circumferential edges connect vertices at same height but different angles
    is_longitudinal = []
    for e in dihedral_edges:
        v0, v1 = vertices[e[0]], vertices[e[1]]
        height_diff = abs(v0[axis_idx] - v1[axis_idx])
        radial_diff = np.linalg.norm(
            v0[radial_axes] - v1[radial_axes]
        )
        # Longitudinal if height change dominates
        is_longitudinal.append(height_diff > radial_diff)
    is_longitudinal = np.array(is_longitudinal)

    long_dihedrals = dihedral_angles_deg[is_longitudinal]
    circ_dihedrals = dihedral_angles_deg[~is_longitudinal]

    # Count circumferential segments (number of longitudinal edges at one height)
    n_circum_segments = len(long_dihedrals) if len(long_dihedrals) > 0 else 0
    # For a single-row cylinder strip, n_longitudinal_edges = n_circumferential_segments
    # The theoretical dihedral angle between adjacent faces = 2*pi/N
    if n_circum_segments > 0:
        theoretical_dihedral = 360.0 / n_circum_segments
    else:
        theoretical_dihedral = 0.0

    return dict(
        face_deviation_deg=face_deviation_deg,
        face_deviation_mean=float(np.mean(face_deviation_deg)),
        face_deviation_max=float(np.max(face_deviation_deg)),
        face_deviation_std=float(np.std(face_deviation_deg)),
        dihedral_angles_deg=dihedral_angles_deg,
        dihedral_mean=float(np.mean(dihedral_angles_deg))
        if len(dihedral_angles_deg) > 0 else 0.0,
        dihedral_max=float(np.max(dihedral_angles_deg))
        if len(dihedral_angles_deg) > 0 else 0.0,
        long_dihedral_mean=float(np.mean(long_dihedrals))
        if len(long_dihedrals) > 0 else 0.0,
        long_dihedral_max=float(np.max(long_dihedrals))
        if len(long_dihedrals) > 0 else 0.0,
        circ_dihedral_mean=float(np.mean(circ_dihedrals))
        if len(circ_dihedrals) > 0 else 0.0,
        circ_dihedral_max=float(np.max(circ_dihedrals))
        if len(circ_dihedrals) > 0 else 0.0,
        n_longitudinal_edges=int(np.sum(is_longitudinal)),
        n_circumferential_edges=int(np.sum(~is_longitudinal)),
        theoretical_dihedral_deg=theoretical_dihedral,
        estimated_radius=radius,
    )


def print_face_normal_report(face_result: dict, mesh_name: str = "") -> None:
    """Print face normal deviation and dihedral angle analysis."""
    print("\n" + "=" * 70)
    print(f"FACE NORMAL ANALYSIS{f': {mesh_name}' if mesh_name else ''}")
    print("=" * 70)

    print("\n--- Face Normal vs Analytical Cylinder Normal ---")
    print(f"Mean deviation:  {face_result['face_deviation_mean']:.4f} deg")
    print(f"Max deviation:   {face_result['face_deviation_max']:.4f} deg")
    print(f"Std deviation:   {face_result['face_deviation_std']:.4f} deg")
    print(f"Estimated radius: {face_result['estimated_radius']*100:.2f} cm")

    print("\n--- Dihedral Angles (normal jump at edges) ---")
    print(f"Longitudinal edges (along axis):   {face_result['n_longitudinal_edges']}")
    print(f"  Mean dihedral: {face_result['long_dihedral_mean']:.4f} deg")
    print(f"  Max dihedral:  {face_result['long_dihedral_max']:.4f} deg")
    n_seg = face_result['n_longitudinal_edges']
    if n_seg > 0:
        print(f"  Theoretical (360/{n_seg}): "
              f"{face_result['theoretical_dihedral_deg']:.4f} deg")

    print(f"Circumferential edges (around axis): "
          f"{face_result['n_circumferential_edges']}")
    print(f"  Mean dihedral: {face_result['circ_dihedral_mean']:.4f} deg")
    print(f"  Max dihedral:  {face_result['circ_dihedral_max']:.4f} deg")

    print("\n--- Transport Implications ---")
    long_dihedral = face_result['long_dihedral_mean']
    if long_dihedral > 0:
        print(f"Each circumferential face crossing rotates the normal by "
              f"~{long_dihedral:.4f} deg.")
        n_seg = face_result['n_longitudinal_edges']
        print(f"A full loop crosses {n_seg} boundaries, total rotation budget: "
              f"{n_seg * long_dihedral:.2f} deg")
        print(f"(Should sum to ~0 for a regular polygon -> errors are in TLS "
              f"estimation, not geometry)")
        print(f"A 64px patch at zoom=10 spans ~{64*0.0003*100:.1f}mm, "
              f"covering ~{64*0.0003 / (2*np.pi*face_result['estimated_radius']) * n_seg:.1f} faces")
    else:
        print("No longitudinal edges found.")


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load a mesh from a GLB/OBJ file, concatenating scenes if needed."""
    scene_or_mesh = trimesh.load(str(mesh_path))

    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = list(scene_or_mesh.geometry.values())
        if len(meshes) == 0:
            print(f"Error: No geometry found in {mesh_path}")
            sys.exit(1)
        if len(meshes) > 1:
            print(f"Scene contains {len(meshes)} meshes, concatenating")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    return mesh


def analyze_single(
    mesh_path: Path,
    expected_K: float,
    axis: str,
    n_bands: int,
    save_npz: bool = True,
) -> dict:
    """Run full analysis on a single mesh file."""
    print(f"Loading mesh: {mesh_path}")
    mesh = load_mesh(mesh_path)
    print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    result = compute_angular_deficit(mesh)
    holonomy = estimate_holonomy_for_path(mesh, result, axis=axis, n_bands=n_bands)
    print_report(
        mesh, result, holonomy,
        expected_K=expected_K, mesh_name=mesh_path.parent.name,
    )

    face_result = compute_face_normal_analysis(mesh, axis=axis)
    print_face_normal_report(face_result, mesh_name=mesh_path.parent.name)

    if save_npz:
        out_path = mesh_path.with_suffix(".holonomy.npz")
        np.savez(
            out_path,
            vertex_curvature=result["vertex_curvature"],
            is_boundary=result["is_boundary"],
            vertex_positions=mesh.vertices,
            band_heights=np.array(holonomy["bands"]),
            band_holonomy=np.array(holonomy["band_holonomy"]),
        )
        print(f"\nRaw data saved to: {out_path}")

    return dict(result=result, holonomy=holonomy, mesh=mesh)


def compare_cylinders(axis: str, n_bands: int) -> None:
    """Compare holonomy across all available grid cylinder meshes."""
    import os

    data_root = Path(
        os.environ.get("MONTY_DATA", os.path.expanduser("~/tbp/data"))
    )
    mesh_dir = data_root / "compositional_objects" / "meshes"

    cylinder_dirs = sorted(mesh_dir.glob("*cylinder*"))
    if not cylinder_dirs:
        print(f"No cylinder meshes found in {mesh_dir}")
        sys.exit(1)

    print(f"Found {len(cylinder_dirs)} cylinder meshes in {mesh_dir}\n")

    summaries = []
    for d in cylinder_dirs:
        glb = d / "textured.glb"
        if not glb.exists():
            print(f"  Skipping {d.name}: no textured.glb")
            continue

        mesh = load_mesh(glb)
        result = compute_angular_deficit(mesh)
        holonomy = estimate_holonomy_for_path(mesh, result, axis=axis, n_bands=n_bands)
        face_result = compute_face_normal_analysis(mesh, axis=axis)

        interior_K = result["vertex_curvature"][~result["is_boundary"]]
        summaries.append(dict(
            name=d.name,
            n_verts=len(mesh.vertices),
            n_faces=len(mesh.faces),
            n_interior=result["n_interior"],
            euler=result["euler_characteristic"],
            total_interior_deg=np.degrees(result["total_curvature_interior"]),
            face_dev_mean=face_result["face_deviation_mean"],
            face_dev_max=face_result["face_deviation_max"],
            long_dihedral=face_result["long_dihedral_mean"],
            n_segments=face_result["n_longitudinal_edges"],
            radius_cm=face_result["estimated_radius"] * 100,
        ))

    # Print comparison table
    print("=" * 110)
    print("CYLINDER MESH HOLONOMY COMPARISON")
    print("=" * 110)
    print(f"{'Mesh':<32s} {'Faces':>6s} {'r(cm)':>6s} {'Nseg':>5s} "
          f"{'FaceDev':>8s} {'FaceMax':>8s} {'Dihedral':>9s} "
          f"{'PatchSpan':>10s} {'IntK(deg)':>10s}")
    print("-" * 110)
    for s in summaries:
        r = s["radius_cm"] / 100  # back to meters
        n_seg = s["n_segments"]
        patch_span_faces = 64 * 0.0003 / (2 * np.pi * r) * n_seg if r > 0 else 0
        print(f"{s['name']:<32s} {s['n_faces']:6d} {s['radius_cm']:6.1f} "
              f"{n_seg:5d} "
              f"{s['face_dev_mean']:7.3f}d {s['face_dev_max']:7.3f}d "
              f"{s['long_dihedral']:8.3f}d "
              f"{patch_span_faces:9.1f}f "
              f"{s['total_interior_deg']:9.4f}")
    print("-" * 110)
    print("\nFaceDev: mean face-normal deviation from analytical (deg)")
    print("FaceMax: max face-normal deviation (deg)")
    print("Dihedral: mean dihedral angle at longitudinal edges (deg)")
    print("PatchSpan: number of faces a 64px patch covers at zoom=10")
    print("IntK: total interior angular deficit (deg) — 0 means all vertices are "
          "boundary")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze mesh holonomy from angular deficit."
    )
    parser.add_argument(
        "mesh_path", type=str, nargs="?", default=None,
        help="Path to mesh file (.glb, .obj, ...)",
    )
    parser.add_argument(
        "--expected-K",
        type=float,
        default=0.0,
        help="Expected Gaussian curvature of the smooth surface (default: 0 for "
             "cylinder)",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="y",
        choices=["x", "y", "z"],
        help="Cylinder axis for band analysis (default: y)",
    )
    parser.add_argument(
        "--n-bands",
        type=int,
        default=10,
        help="Number of height bands for holonomy estimation (default: 10)",
    )
    parser.add_argument(
        "--compare-cylinders",
        action="store_true",
        help="Compare holonomy across all grid cylinder meshes in MONTY_DATA",
    )
    args = parser.parse_args()

    if args.compare_cylinders:
        compare_cylinders(axis=args.axis, n_bands=args.n_bands)
        return

    if args.mesh_path is None:
        parser.error("mesh_path is required unless --compare-cylinders is used")

    mesh_path = Path(args.mesh_path)
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found")
        sys.exit(1)

    analyze_single(mesh_path, args.expected_K, args.axis, args.n_bands)


if __name__ == "__main__":
    main()
