"""Render compositional object meshes to PNG files on a black background.

Examples:
    PYTHONPATH=/Users/hlee/tbp/feat.2d_sensor/src \
        /Users/hlee/miniconda3/bin/conda run -n tbp.monty python \
        sandbox/render_compositional_mesh_pngs.py --object cube

    PYTHONPATH=/Users/hlee/tbp/feat.2d_sensor/src \
        /Users/hlee/miniconda3/bin/conda run -n tbp.monty python \
        sandbox/render_compositional_mesh_pngs.py \
        --object cube --output-dir /private/tmp/compositional_mesh_pngs
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np  # noqa: E402
from vedo import Mesh, Plotter  # noqa: E402

try:
    from sandbox.visualize_compositional_meshes import (  # noqa: E402
        DEFAULT_MESH_DIR,
        TEXTURED_MESH_FILE,
        MeshRecord,
        build_vedo_mesh,
        discover_meshes,
        select_meshes,
    )
except ModuleNotFoundError:
    from visualize_compositional_meshes import (  # noqa: E402
        DEFAULT_MESH_DIR,
        TEXTURED_MESH_FILE,
        MeshRecord,
        build_vedo_mesh,
        discover_meshes,
        select_meshes,
    )


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "compositional_mesh_pngs"
DEFAULT_IMAGE_SIZE = 1024
FRONT_CAMERA_DIRECTION = np.array([0.12, 0.22, 1.0])


def compute_camera(mesh: Mesh) -> dict:
    """Return a stable, mostly front-on camera with slight up/side angle."""
    bounds = np.asarray(mesh.bounds(), dtype=float)
    center = np.array(
        [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]
    )
    extents = np.array(
        [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ]
    )
    max_extent = max(float(extents.max()), 1e-6)
    camera_direction = FRONT_CAMERA_DIRECTION.copy()
    camera_direction /= np.linalg.norm(camera_direction)
    camera_pos = center + camera_direction * max_extent * 3.0

    return {
        "pos": tuple(float(value) for value in camera_pos),
        "focal_point": tuple(float(value) for value in center),
        "view_angle": 35,
    }


def render_mesh_png(
    record: MeshRecord,
    *,
    output_dir: Path,
    image_size: int,
    normalize: bool,
    alpha: float,
    color_mode: str,
) -> Path:
    """Render one mesh record to a PNG file and return the written path."""
    mesh = build_vedo_mesh(
        record,
        center=np.zeros(3),
        cell_size=1.0,
        normalize=normalize,
        alpha=alpha,
        color_mode=color_mode,
    )
    output_path = output_dir / f"{record.name}.png"
    camera = compute_camera(mesh)

    plotter = Plotter(
        size=(image_size, image_size),
        title=record.name,
        bg="black",
        axes=0,
        offscreen=True,
        interactive=False,
    )
    plotter.show(
        mesh,
        axes=0,
        viewup="z",
        camera=camera,
        interactive=False,
    )
    plotter.screenshot(str(output_path))
    plotter.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render compositional object GLB meshes to PNG files."
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
        help="Object name or substring to render. May be passed multiple times.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered object names and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG files will be saved.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square output image size in pixels.",
    )
    parser.add_argument(
        "--preserve-scale",
        action="store_true",
        help="Keep original mesh scale instead of normalizing each mesh.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Mesh opacity from 0 to 1.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("texture", "object", "family"),
        default="texture",
        help="Use GLB texture/RGB data, per-object debug colors, or family colors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_dir = args.mesh_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if args.image_size <= 0:
        raise SystemExit("--image-size must be greater than 0")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = max(0.0, min(1.0, args.alpha))
    for record in selected:
        output_path = render_mesh_png(
            record,
            output_dir=output_dir,
            image_size=args.image_size,
            normalize=not args.preserve_scale,
            alpha=alpha,
            color_mode=args.color_mode,
        )
        print(output_path)


if __name__ == "__main__":
    main()
