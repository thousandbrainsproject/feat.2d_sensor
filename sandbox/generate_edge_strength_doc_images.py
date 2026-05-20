# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Generate edge-strength example images for the 2D sensor documentation.

The examples isolate three ways a clear 90-degree white line can produce lower
edge-strength scores: reduced contrast, increased blur, and movement away from
the center-weighted detector support. Detector scores are computed from the raw
patches first. The saved PNGs and contact sheet then receive a visual arrow
overlay showing the detected edge angle so the annotation does not affect the
measurement.

Usage from the project root:

    PYTHONPATH=/Users/hlee/tbp/feat.2d_sensor/src \
    /Users/hlee/miniconda3/bin/conda run -n tbp.monty \
        python sandbox/generate_edge_strength_doc_images.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.utils.edge_detection import EdgeDetector, EdgeFeatures

IMAGE_SIZE = 64
LINE_ANGLE_DEG = 90.0
LINE_THICKNESS = 2
LINE_RGB = (255, 255, 255)
ARROW_RGB = (255, 214, 0)
DEFAULT_OUTPUT_DIR = Path("write_ups/figures/edge_strength_examples")


@dataclass(frozen=True)
class StrengthCase:
    """Parameters for one edge-strength documentation patch."""

    row: str
    name: str
    label: str
    background: int
    blur_sigma: float = 0.0
    x_position: int = IMAGE_SIZE // 2


@dataclass(frozen=True)
class ScoredStrengthCase:
    """Generated documentation image plus detector output and file metadata."""

    index: int
    case: StrengthCase
    filename: str
    raw_image: np.ndarray
    annotated_image: np.ndarray
    edge: EdgeFeatures


CASES = (
    StrengthCase(
        row="contrast",
        name="contrast_black",
        label="contrast: bg 0",
        background=0,
    ),
    StrengthCase(
        row="contrast",
        name="contrast_dark_gray",
        label="contrast: bg 64",
        background=64,
    ),
    StrengthCase(
        row="contrast",
        name="contrast_gray",
        label="contrast: bg 128",
        background=128,
    ),
    StrengthCase(
        row="blur",
        name="blur_none",
        label="blur: sigma 0",
        background=0,
        blur_sigma=0.0,
    ),
    StrengthCase(
        row="blur",
        name="blur_moderate",
        label="blur: sigma 1",
        background=0,
        blur_sigma=1.0,
    ),
    StrengthCase(
        row="blur",
        name="blur_strong",
        label="blur: sigma 4",
        background=0,
        blur_sigma=4.0,
    ),
    StrengthCase(
        row="off_center",
        name="off_center_centered",
        label="off-center: x 32",
        background=0,
        x_position=32,
    ),
    StrengthCase(
        row="off_center",
        name="off_center_near_edge",
        label="off-center: x 48",
        background=0,
        x_position=48,
    ),
    StrengthCase(
        row="off_center",
        name="off_center_outside",
        label="off-center: x 50",
        background=0,
        x_position=50,
    ),
)


def render_raw_patch(case: StrengthCase) -> np.ndarray:
    """Render an unannotated 64x64 RGB line patch.

    Returns:
        Raw RGB patch used as EdgeDetector input.
    """
    image = np.full(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        case.background,
        dtype=np.uint8,
    )
    cv2.line(
        image,
        (case.x_position, -IMAGE_SIZE // 2),
        (case.x_position, IMAGE_SIZE + IMAGE_SIZE // 2),
        LINE_RGB,
        LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    if case.blur_sigma > 0.0:
        image = cv2.GaussianBlur(image, (0, 0), case.blur_sigma)

    return image


def sensor_observation(image: np.ndarray) -> SensorObservation:
    """Build the minimal observation needed by EdgeDetector.

    Returns:
        Sensor observation containing RGB-D data and an identity camera pose.
    """
    alpha = np.full((IMAGE_SIZE, IMAGE_SIZE, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([image, alpha], axis=-1)
    depth = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return SensorObservation(
        rgba=rgba,
        depth=depth,
        cam_to_world=np.identity(4),
    )


def annotate_detected_angle(image: np.ndarray, angle_rad: float | None) -> np.ndarray:
    """Overlay the detected edge angle as a display-only arrow.

    Returns:
        RGB patch annotated with an arrow, or a copy of the input if no angle was
        detected.
    """
    annotated = image.copy()
    if angle_rad is None:
        return annotated

    center = np.array([13.0, 47.0])
    length = 22.0
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    start = center - 0.5 * length * direction
    end = center + 0.5 * length * direction

    cv2.arrowedLine(
        annotated,
        tuple(np.round(start).astype(int)),
        tuple(np.round(end).astype(int)),
        ARROW_RGB,
        2,
        line_type=cv2.LINE_AA,
        tipLength=0.35,
    )
    return annotated


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    """Save an RGB image as PNG via OpenCV.

    Raises:
        OSError: If OpenCV fails to write the PNG file.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Failed to save image to {path}")


def format_angle_deg(angle_rad: float | None) -> str:
    """Format detector angle for labels and CSV.

    Returns:
        Angle in degrees, or an empty string when no angle was detected.
    """
    if angle_rad is None:
        return ""
    return f"{np.degrees(angle_rad):.1f}"


def generate_examples(output_dir: Path) -> list[ScoredStrengthCase]:
    """Generate all patches, annotated PNGs, and detector scores.

    Returns:
        Scored edge-strength examples with raw and annotated images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = EdgeDetector()

    examples = []
    for index, case in enumerate(CASES, start=1):
        raw_image = render_raw_patch(case)
        edge = detector(sensor_observation(raw_image))
        annotated_image = annotate_detected_angle(raw_image, edge.angle)
        filename = f"{index:02d}_{case.name}.png"
        save_rgb_png(output_dir / filename, annotated_image)
        examples.append(
            ScoredStrengthCase(
                index=index,
                case=case,
                filename=filename,
                raw_image=raw_image,
                annotated_image=annotated_image,
                edge=edge,
            )
        )

    return examples


def write_manifest(path: Path, examples: list[ScoredStrengthCase]) -> None:
    """Write measured detector outputs to CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "filename",
                "row",
                "name",
                "label",
                "background",
                "blur_sigma",
                "x_position",
                "edge_strength",
                "coherence",
                "angle_degrees",
                "has_edge",
                "is_geometric_edge",
            ],
        )
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "index": example.index,
                    "filename": example.filename,
                    "row": example.case.row,
                    "name": example.case.name,
                    "label": example.case.label,
                    "background": example.case.background,
                    "blur_sigma": example.case.blur_sigma,
                    "x_position": example.case.x_position,
                    "edge_strength": f"{example.edge.strength:.6f}",
                    "coherence": f"{example.edge.coherence:.6f}",
                    "angle_degrees": format_angle_deg(example.edge.angle),
                    "has_edge": example.edge.has_edge,
                    "is_geometric_edge": example.edge.is_geometric_edge,
                }
            )


def make_contact_sheet(examples: list[ScoredStrengthCase]) -> np.ndarray:
    """Render a 3x3 LaTeX-ready contact sheet.

    Returns:
        RGB image containing annotated patches and measured detector scores.
    """
    columns = 3
    rows = 3
    scale = 3
    patch_size = IMAGE_SIZE * scale
    label_height = 92
    panel_padding = 14
    panel_width = patch_size + 2 * panel_padding
    panel_height = patch_size + label_height + 2 * panel_padding

    sheet = np.full(
        (rows * panel_height, columns * panel_width, 3),
        255,
        dtype=np.uint8,
    )

    for example in examples:
        row = (example.index - 1) // columns
        col = (example.index - 1) % columns
        x0 = col * panel_width
        y0 = row * panel_height

        panel = sheet[y0 : y0 + panel_height, x0 : x0 + panel_width]
        cv2.rectangle(
            panel,
            (0, 0),
            (panel_width - 1, panel_height - 1),
            (210, 210, 210),
            1,
        )

        patch = cv2.resize(
            example.annotated_image,
            (patch_size, patch_size),
            interpolation=cv2.INTER_NEAREST,
        )
        px = panel_padding
        py = panel_padding
        panel[py : py + patch_size, px : px + patch_size] = patch

        angle = format_angle_deg(example.edge.angle) or "None"
        label_y = py + patch_size + 22
        label_lines = [
            f"{example.index}. {example.case.label}",
            f"strength={example.edge.strength:.2f}",
            f"coherence={example.edge.coherence:.2f}",
            f"edge={example.edge.has_edge}  angle={angle}",
        ]
        for line_index, label in enumerate(label_lines):
            cv2.putText(
                panel,
                label,
                (panel_padding, label_y + line_index * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

    return sheet


def latex_snippet() -> str:
    """Return the figure snippet for the 2D sensor LaTeX write-up."""
    return r"""
\begin{figure}[h]
  \centering
  \includegraphics[width=0.82\textwidth]{../figures/edge_strength_examples/edge_strength_contact_sheet.png}
  \caption{Synthetic 64x64 texture patches illustrating how \texttt{EdgeDetector}
  edge strength decreases as a 90-degree white line loses contrast, becomes
  blurred, or moves outside the center-weighted support. Yellow arrows are drawn
  after scoring to show the detected edge angle without affecting the reported
  values.}
  \label{fig:edge-strength-examples}
\end{figure}
""".strip()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate edge-strength example images for 2D sensor docs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory for images and manifest "
            "(default: write_ups/figures/edge_strength_examples)."
        ),
    )
    parser.add_argument(
        "--no-latex-snippet",
        action="store_true",
        help="Do not print a LaTeX figure snippet after generating assets.",
    )
    return parser.parse_args()


def main(
    output_dir: Path,
    print_latex_snippet: bool = True,
) -> list[ScoredStrengthCase]:
    """Generate documentation assets.

    Returns:
        Generated examples with detector outputs and file metadata.
    """
    examples = generate_examples(output_dir)
    write_manifest(output_dir / "edge_strength_examples.csv", examples)
    save_rgb_png(
        output_dir / "edge_strength_contact_sheet.png",
        make_contact_sheet(examples),
    )

    for example in examples:
        print(
            f"{example.filename}: strength={example.edge.strength:.3f}, "
            f"coherence={example.edge.coherence:.3f}, "
            f"angle={format_angle_deg(example.edge.angle) or 'None'}, "
            f"has_edge={example.edge.has_edge}"
        )

    print(f"\nWrote edge-strength documentation assets to {output_dir.resolve()}")
    if print_latex_snippet:
        print("\nLaTeX snippet:\n")
        print(latex_snippet())

    return examples


if __name__ == "__main__":
    args = parse_args()
    main(args.output_dir, print_latex_snippet=not args.no_latex_snippet)
