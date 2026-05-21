"""Visualize compositional object meshes with Vedo.

Examples:
    PYTHONPATH=/Users/hlee/tbp/feat.2d_sensor/src \
        /Users/hlee/miniconda3/bin/conda run -n tbp.monty python \
        sandbox/visualize_compositional_meshes.py

    PYTHONPATH=/Users/hlee/tbp/feat.2d_sensor/src \
        /Users/hlee/miniconda3/bin/conda run -n tbp.monty python \
        sandbox/visualize_compositional_meshes.py --object cube --page-size 6
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np  # noqa: E402
import vtk  # noqa: E402
from vedo import Mesh, Plotter, Text2D, Text3D  # noqa: E402


DEFAULT_MESH_DIR = Path("/Users/hlee/tbp/data/compositional_objects_1.1/meshes")
DEFAULT_WINDOW_SIZE = (1600, 1000)
TEXTURED_MESH_FILE = "textured.glb"
MESH_COLORS = {
    "cube": (0.00, 0.45, 0.85),
    "disk": (0.85, 0.15, 0.65),
    "cylinder": (0.10, 0.55, 0.28),
    "sphere": (0.95, 0.65, 0.10),
    "logo": (0.45, 0.20, 0.75),
    "mug": (0.85, 0.25, 0.15),
}


@dataclass(frozen=True)
class MeshRecord:
    """Path and display metadata for one compositional-object mesh."""

    name: str
    path: Path


def discover_meshes(mesh_dir: Path) -> list[MeshRecord]:
    """Return mesh records for all object folders containing textured.glb."""
    mesh_dir = mesh_dir.expanduser()
    records = [
        MeshRecord(name=path.parent.name, path=path)
        for path in sorted(mesh_dir.glob(f"*/{TEXTURED_MESH_FILE}"))
    ]
    return sorted(records, key=lambda record: record.name)


def select_meshes(records: list[MeshRecord], filters: list[str]) -> list[MeshRecord]:
    """Filter records by exact name or case-insensitive substring."""
    if not filters:
        return records

    selected = []
    lowered_filters = [item.lower() for item in filters]
    for record in records:
        lowered_name = record.name.lower()
        if any(item == lowered_name or item in lowered_name for item in lowered_filters):
            selected.append(record)
    return selected


def _load_glb_polydata(path: Path) -> vtk.vtkPolyData:
    """Load GLB geometry with VTK and flatten it to polydata for Vedo."""
    reader = vtk.vtkGLTFReader()
    reader.SetFileName(str(path))
    reader.Update()

    geometry_filter = vtk.vtkCompositeDataGeometryFilter()
    geometry_filter.SetInputData(reader.GetOutput())
    geometry_filter.Update()

    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(geometry_filter.GetOutput())
    if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfCells() == 0:
        raise ValueError(f"No renderable geometry found in {path}")
    return polydata


def _color_for_name(name: str) -> tuple[float, float, float]:
    """Choose a stable display color from the object name."""
    lowered = name.lower()
    for key, color in MESH_COLORS.items():
        if key in lowered:
            return color
    return (0.68, 0.85, 0.90)


def _center_scale_and_place_mesh(
    mesh: Mesh, *, center: np.ndarray, cell_size: float, normalize: bool
) -> Mesh:
    """Center one Vedo mesh and place it in a grid cell."""
    bounds = np.asarray(mesh.bounds(), dtype=float)
    mesh_center = np.array(
        [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]
    )
    mesh.shift(*(-mesh_center))

    if normalize:
        extents = np.array(
            [
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            ]
        )
        max_extent = float(extents.max())
        if max_extent > 0:
            mesh.scale(cell_size / max_extent)

    mesh.shift(*center)
    return mesh


def build_vedo_mesh(
    record: MeshRecord,
    *,
    center: np.ndarray,
    cell_size: float,
    normalize: bool,
    alpha: float,
) -> Mesh:
    """Create a Vedo mesh actor from one GLB."""
    mesh = Mesh(_load_glb_polydata(record.path))
    _center_scale_and_place_mesh(
        mesh,
        center=center,
        cell_size=cell_size,
        normalize=normalize,
    )
    mesh.color(_color_for_name(record.name), alpha=alpha)
    mesh.name = record.name
    return mesh


def build_label(record: MeshRecord, *, center: np.ndarray, cell_size: float) -> Text3D:
    """Create a small 3D label under one mesh."""
    label_pos = (
        center[0] - cell_size * 0.48,
        center[1] - cell_size * 0.62,
        center[2] - cell_size * 0.52,
    )
    return Text3D(record.name, pos=label_pos, s=cell_size * 0.055, c="black")


def page_records(
    records: list[MeshRecord], *, page: int, page_size: int
) -> list[MeshRecord]:
    """Select one 1-based page of records."""
    if page_size <= 0:
        return records

    start = (page - 1) * page_size
    end = start + page_size
    return records[start:end]


def build_scene_actors(
    records: list[MeshRecord],
    *,
    columns: int,
    normalize: bool,
    alpha: float,
    cell_spacing: float,
) -> list[object]:
    """Build mesh and label actors arranged in a regular grid."""
    actors: list[object] = []
    columns = max(1, columns)
    cell_size = 1.0

    for index, record in enumerate(records):
        row, col = divmod(index, columns)
        center = np.array([col * cell_spacing, -row * cell_spacing, 0.0])
        actors.append(
            build_vedo_mesh(
                record,
                center=center,
                cell_size=cell_size,
                normalize=normalize,
                alpha=alpha,
            )
        )
        actors.append(build_label(record, center=center, cell_size=cell_size))

    return actors


def summarize_selection(records: list[MeshRecord], *, page: int, total: int) -> str:
    """Return text shown in the plot and terminal."""
    names = ", ".join(record.name for record in records)
    return f"Showing {len(records)} of {total} meshes | page {page}\n{names}"


def show_meshes(
    records: list[MeshRecord],
    *,
    page: int,
    total: int,
    columns: int,
    normalize: bool,
    alpha: float,
) -> None:
    """Open an interactive Vedo plotter."""
    actors = build_scene_actors(
        records,
        columns=columns,
        normalize=normalize,
        alpha=alpha,
        cell_spacing=1.6,
    )
    summary = summarize_selection(records, page=page, total=total)
    actors.append(Text2D(summary, pos="top-left", s=0.65, c="black"))

    plotter = Plotter(
        size=DEFAULT_WINDOW_SIZE,
        title="Compositional object meshes",
    )
    plotter.add(*actors)
    plotter.show(
        axes=dict(xtitle="X", ytitle="Y", ztitle="Z"),
        viewup="z",
        interactive=True,
    )
    plotter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize compositional object GLB meshes with Vedo."
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help=f"Directory containing object folders with {TEXTURED_MESH_FILE}.",
    )
    parser.add_argument(
        "--object",
        action="append",
        default=[],
        help="Object name or substring to show. May be passed multiple times.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered object names and exit.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="One-based page number when showing many meshes.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=12,
        help="Number of meshes to show per page. Use 0 to show all.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of grid columns in the scene.",
    )
    parser.add_argument(
        "--preserve-scale",
        action="store_true",
        help="Keep original mesh scale instead of normalizing each mesh to a cell.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Mesh opacity from 0 to 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_dir = args.mesh_dir.expanduser()

    if args.page < 1:
        raise SystemExit("--page must be 1 or greater")
    if not mesh_dir.exists():
        raise SystemExit(f"Mesh directory not found: {mesh_dir}")

    records = discover_meshes(mesh_dir)
    if not records:
        raise SystemExit(f"No {TEXTURED_MESH_FILE} files found under {mesh_dir}")

    if args.list:
        for record in records:
            print(record.name)
        return

    selected = select_meshes(records, args.object)
    if not selected:
        filters = ", ".join(args.object)
        raise SystemExit(f"No meshes matched: {filters}")

    total_pages = (
        1
        if args.page_size <= 0
        else max(1, math.ceil(len(selected) / args.page_size))
    )
    if args.page > total_pages:
        raise SystemExit(f"--page {args.page} exceeds {total_pages} available pages")

    page = page_records(selected, page=args.page, page_size=args.page_size)
    summary = summarize_selection(page, page=args.page, total=len(selected))
    print(summary)

    show_meshes(
        page,
        page=args.page,
        total=len(selected),
        columns=args.columns,
        normalize=not args.preserve_scale,
        alpha=max(0.0, min(1.0, args.alpha)),
    )


if __name__ == "__main__":
    main()
