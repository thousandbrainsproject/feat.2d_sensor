# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Plot saved 2D edge debug patches at their world x/y coordinates.

Usage from the project root:

    PYTHONPATH=/Users/hlee/tbp/feat.2d_sensor/src \
        /Users/hlee/miniconda3/bin/conda run -n tbp.monty python \
        sandbox/plot_edge_patch_manifest.py /path/to/edge_patch_manifest.csv

    PYTHONPATH=/Users/hlee/tbp/feat.2d_sensor/src \
        /Users/hlee/miniconda3/bin/conda run -n tbp.monty python \
        sandbox/plot_edge_patch_manifest.py /path/to/edge_patch_manifest.csv \
        --output /path/to/world_xy.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


@dataclass(frozen=True)
class EdgePatchManifestRow:
    """One saved debug edge patch and its world-space center."""

    index: int
    filename: str
    image_path: Path
    sensor_id: str
    world_x: float
    world_y: float
    world_z: float
    angle_degrees: float
    edge_strength: float
    coherence: float
    is_geometric_edge: bool
    has_edge: bool


def parse_bool(value: str) -> bool:
    """Parse boolean values written by csv.DictWriter.

    Returns:
        Parsed boolean value.

    Raises:
        ValueError: If value is not a supported boolean string.
    """
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected True or False, got {value!r}")


def load_manifest_rows(manifest_path: Path | str) -> list[EdgePatchManifestRow]:
    """Load edge patch manifest rows and resolve image paths.

    Returns:
        Manifest rows with image paths resolved relative to the manifest.
    """
    manifest_path = Path(manifest_path).expanduser()
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            rows.append(
                EdgePatchManifestRow(
                    index=int(row["index"]),
                    filename=filename,
                    image_path=manifest_path.parent / filename,
                    sensor_id=row["sensor_id"],
                    world_x=float(row["world_x"]),
                    world_y=float(row["world_y"]),
                    world_z=float(row["world_z"]),
                    angle_degrees=float(row["angle_degrees"]),
                    edge_strength=float(row["edge_strength"]),
                    coherence=float(row["coherence"]),
                    is_geometric_edge=parse_bool(row["is_geometric_edge"]),
                    has_edge=parse_bool(row["has_edge"]),
                )
            )
    return rows


def plot_world_xy(
    rows: list[EdgePatchManifestRow],
    output_path: Path | str,
    *,
    zoom: float = 0.75,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> None:
    """Save a world x/y thumbnail scatter plot for saved edge patches.

    Raises:
        FileNotFoundError: If a manifest row references a missing image.
        ValueError: If no rows are provided.
    """
    if not rows:
        raise ValueError("Manifest contains no rows to plot.")

    output_path = Path(output_path).expanduser()
    fig, ax = plt.subplots(figsize=figsize)
    xs = [row.world_x for row in rows]
    ys = [row.world_y for row in rows]

    for row in rows:
        if not row.image_path.exists():
            raise FileNotFoundError(
                f"Image referenced by manifest not found: {row.image_path}"
            )
        image = plt.imread(row.image_path)
        imagebox = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(
            imagebox,
            (row.world_x, row.world_y),
            frameon=True,
            pad=0.1,
            bboxprops={"edgecolor": "black", "linewidth": 0.5},
        )
        ax.add_artist(ab)

    ax.scatter(xs, ys, s=8, c="black", alpha=0.5, zorder=3)
    ax.set_xlabel("World x")
    ax.set_ylabel("World y")
    ax.set_title("Edge Debug Patches in World XY")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.15)
    ax.grid(visible=True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arrange saved 2D edge debug patches by world x/y coordinates."
    )
    parser.add_argument(
        "manifest_csv",
        type=Path,
        help="Path to edge_patch_manifest.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to edge_patch_world_xy.png next to the "
            "manifest."
        ),
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=0.75,
        help="Thumbnail zoom passed to matplotlib OffsetImage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_csv.expanduser()
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else manifest_path.parent / "edge_patch_world_xy.png"
    )
    rows = load_manifest_rows(manifest_path)
    plot_world_xy(rows, output_path, zoom=args.zoom)
    print(f"Saved world-XY edge patch plot to: {output_path}")


if __name__ == "__main__":
    main()
