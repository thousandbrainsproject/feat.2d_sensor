# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Generate edge-quality example images for the 2D sensor documentation.

This script creates deterministic 64x64 RGB patches that illustrate how
``EdgeDetector`` edge strength and coherence respond to clean lines, weak lines,
blurred lines, ambiguous intersections, and noise. It writes individual PNG
patches, a contact sheet, and a CSV manifest with the measured detector outputs.

Usage from the project root:

    PYTHONPATH=/Users/hlee/tbp/feat.2d_sensor/src \
    /Users/hlee/miniconda3/bin/conda run -n tbp.monty \
        python sandbox/generate_edge_quality_doc_images.py
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
BACKGROUND_RGB = (128, 128, 128)
DEFAULT_FOREGROUND_RGB = (255, 255, 255)
DEFAULT_OUTPUT_DIR = Path("write_ups/figures/edge_quality_examples")
RNG_SEED = 7


@dataclass(frozen=True)
class SecondaryLine:
    """An optional second line used to create cross or distraction examples."""

    angle_deg: float
    thickness: int
    foreground: tuple[int, int, int] = DEFAULT_FOREGROUND_RGB
    alpha: float = 1.0


@dataclass(frozen=True)
class ExampleCase:
    """Parameters for one documentation patch."""

    name: str
    description: str
    angle_deg: float | None
    thickness: int = 2
    foreground: tuple[int, int, int] = DEFAULT_FOREGROUND_RGB
    alpha: float = 1.0
    blur_sigma: float = 0.0
    secondary_line: SecondaryLine | None = None
    noise_std: float = 0.0


@dataclass(frozen=True)
class ScoredExample:
    """Generated image plus detector output and file metadata."""

    index: int
    case: ExampleCase
    filename: str
    image: np.ndarray
    edge: EdgeFeatures


CASES = (
    ExampleCase(
        name="uniform_gray",
        description="Uniform gray patch with no oriented edge evidence.",
        angle_deg=None,
    ),
    ExampleCase(
        name="weak_coherent_line",
        description="Low-contrast centered line: low strength, high coherence.",
        angle_deg=90.0,
        foreground=(150, 150, 150),
    ),
    ExampleCase(
        name="strong_coherent_line",
        description="Sharp centered white line: high strength, high coherence.",
        angle_deg=90.0,
    ),
    ExampleCase(
        name="blurred_coherent_line",
        description="Blurred white line: reduced strength, high coherence.",
        angle_deg=90.0,
        blur_sigma=2.0,
    ),
    ExampleCase(
        name="dominant_plus_weak_cross",
        description=(
            "Strong line with a weak crossing line: dominant orientation remains."
        ),
        angle_deg=90.0,
        secondary_line=SecondaryLine(
            angle_deg=0.0,
            thickness=2,
            foreground=(175, 175, 175),
        ),
    ),
    ExampleCase(
        name="equal_cross",
        description=(
            "Equal-strength cross: strong gradients but low orientation coherence."
        ),
        angle_deg=90.0,
        secondary_line=SecondaryLine(angle_deg=0.0, thickness=2),
    ),
    ExampleCase(
        name="diagonal_line",
        description=(
            "Clean diagonal line: orientation changes while quality remains high."
        ),
        angle_deg=45.0,
    ),
    ExampleCase(
        name="noisy_line",
        description="Centered line with deterministic image noise.",
        angle_deg=90.0,
        noise_std=45.0,
    ),
)

DISPLAY_NAMES = {
    "uniform_gray": "uniform gray",
    "weak_coherent_line": "weak line",
    "strong_coherent_line": "strong line",
    "blurred_coherent_line": "blurred line",
    "dominant_plus_weak_cross": "dominant cross",
    "equal_cross": "equal cross",
    "diagonal_line": "diagonal line",
    "noisy_line": "noisy line",
}


def create_background() -> np.ndarray:
    """Return a gray RGB image."""
    image = np.empty((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[:, :] = BACKGROUND_RGB
    return image


def draw_center_line(
    image: np.ndarray,
    angle_deg: float,
    thickness: int,
    foreground: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Draw one anti-aliased line through the patch center.

    Returns:
        RGB image with the requested line blended onto it.
    """
    center = np.array([IMAGE_SIZE // 2, IMAGE_SIZE // 2], dtype=np.float32)
    angle_rad = np.deg2rad(angle_deg)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)
    half_length = float(IMAGE_SIZE)
    start = center - half_length * direction
    end = center + half_length * direction

    overlay = image.copy()
    cv2.line(
        overlay,
        tuple(np.round(start).astype(int)),
        tuple(np.round(end).astype(int)),
        foreground,
        thickness,
        lineType=cv2.LINE_AA,
    )
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def render_case(case: ExampleCase, rng: np.random.Generator) -> np.ndarray:
    """Render one example patch from case parameters.

    Returns:
        RGB image containing the requested synthetic line pattern.
    """
    image = create_background()

    if case.angle_deg is not None:
        image = draw_center_line(
            image,
            case.angle_deg,
            case.thickness,
            case.foreground,
            case.alpha,
        )

    if case.secondary_line is not None:
        image = draw_center_line(
            image,
            case.secondary_line.angle_deg,
            case.secondary_line.thickness,
            case.secondary_line.foreground,
            case.secondary_line.alpha,
        )

    if case.blur_sigma > 0.0:
        image = cv2.GaussianBlur(image, (0, 0), case.blur_sigma)

    if case.noise_std > 0.0:
        noise = rng.normal(0.0, case.noise_std, size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

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


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    """Save an RGB image as PNG via OpenCV.

    Raises:
        OSError: If OpenCV fails to write the PNG file.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Failed to save image to {path}")


def generate_examples(output_dir: Path) -> list[ScoredExample]:
    """Generate all patches and save individual PNGs.

    Returns:
        Generated examples with detector outputs and file metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = EdgeDetector()
    rng = np.random.default_rng(RNG_SEED)

    examples = []
    for index, case in enumerate(CASES, start=1):
        image = render_case(case, rng)
        edge = detector(sensor_observation(image))
        filename = f"{index:02d}_{case.name}.png"
        save_rgb_png(output_dir / filename, image)
        examples.append(
            ScoredExample(
                index=index,
                case=case,
                filename=filename,
                image=image,
                edge=edge,
            )
        )

    return examples


def format_angle_deg(angle_rad: float | None) -> str:
    """Format detector angle for labels and CSV.

    Returns:
        Angle in degrees, or an empty string when no angle was detected.
    """
    if angle_rad is None:
        return ""
    return f"{np.degrees(angle_rad):.1f}"


def write_manifest(path: Path, examples: list[ScoredExample]) -> None:
    """Write measured detector outputs to CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "filename",
                "name",
                "description",
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
                    "name": example.case.name,
                    "description": example.case.description,
                    "edge_strength": f"{example.edge.strength:.6f}",
                    "coherence": f"{example.edge.coherence:.6f}",
                    "angle_degrees": format_angle_deg(example.edge.angle),
                    "has_edge": example.edge.has_edge,
                    "is_geometric_edge": example.edge.is_geometric_edge,
                }
            )


def make_contact_sheet(examples: list[ScoredExample]) -> np.ndarray:
    """Render a LaTeX-ready contact sheet.

    Returns:
        RGB image containing all patches and measured detector scores.
    """
    columns = 4
    rows = int(np.ceil(len(examples) / columns))
    scale = 3
    patch_size = IMAGE_SIZE * scale
    label_height = 84
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
            example.image,
            (patch_size, patch_size),
            interpolation=cv2.INTER_NEAREST,
        )
        px = panel_padding
        py = panel_padding
        panel[py : py + patch_size, px : px + patch_size] = patch

        label_y = py + patch_size + 22
        label_lines = [
            f"{example.index}. {DISPLAY_NAMES[example.case.name]}",
            f"s={example.edge.strength:.2f}  c={example.edge.coherence:.2f}",
            (
                f"edge={example.edge.has_edge}  "
                f"angle={format_angle_deg(example.edge.angle) or 'None'}"
            ),
        ]
        for line_index, label in enumerate(label_lines):
            cv2.putText(
                panel,
                label,
                (panel_padding, label_y + line_index * 20),
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
  \includegraphics[width=\textwidth]{../figures/edge_quality_examples/edge_quality_contact_sheet.png}
  \caption{Synthetic 64x64 texture patches illustrating how \texttt{EdgeDetector}
  reports edge strength and coherence for weak, clean, blurred, ambiguous, and
  noisy line evidence. Scores are measured by the current implementation using
  the default detector parameters.}
  \label{fig:edge-quality-examples}
\end{figure}
""".strip()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate edge-quality example images for 2D sensor docs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory for images and manifest "
            "(default: write_ups/figures/edge_quality_examples)."
        ),
    )
    parser.add_argument(
        "--no-latex-snippet",
        action="store_true",
        help="Do not print a LaTeX figure snippet after generating assets.",
    )
    return parser.parse_args()


def main(output_dir: Path, print_latex_snippet: bool = True) -> list[ScoredExample]:
    """Generate documentation assets.

    Returns:
        Generated examples with detector outputs and file metadata.
    """
    examples = generate_examples(output_dir)
    write_manifest(output_dir / "edge_quality_examples.csv", examples)
    save_rgb_png(
        output_dir / "edge_quality_contact_sheet.png",
        make_contact_sheet(examples),
    )

    for example in examples:
        print(
            f"{example.filename}: strength={example.edge.strength:.3f}, "
            f"coherence={example.edge.coherence:.3f}, "
            f"angle={format_angle_deg(example.edge.angle) or 'None'}, "
            f"has_edge={example.edge.has_edge}"
        )

    print(f"\nWrote edge-quality documentation assets to {output_dir.resolve()}")
    if print_latex_snippet:
        print("\nLaTeX snippet:\n")
        print(latex_snippet())

    return examples


if __name__ == "__main__":
    args = parse_args()
    main(args.output_dir, print_latex_snippet=not args.no_latex_snippet)
