# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Generate synthetic test images for comparing edge detection methods.

By default this script writes high-contrast 64x64 RGB PNGs to
``data/synthetic_edge_test_images`` under the project root.

Usage from the project root:

    conda activate tbp.monty
    export PYTHONPATH="$(pwd)/src"
    python sandbox/generate_synthetic_edge_images.py

    python sandbox/generate_synthetic_edge_images.py --contrast low
    python sandbox/generate_synthetic_edge_images.py \
        --suite thickness,angled_intersections \
        --output-dir /tmp/synthetic_edge_test_images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# ============================================================================
# Configuration Constants (easily modifiable)
# ============================================================================

# Image dimensions
IMAGE_SIZE = 64
CENTER_X = IMAGE_SIZE // 2  # 32
CENTER_Y = IMAGE_SIZE // 2  # 32

# Contrast levels
HIGH_CONTRAST_BG = (0, 0, 0)  # Black background
HIGH_CONTRAST_EDGE = (255, 255, 255)  # White edge

LOW_CONTRAST_BG = (64, 64, 64)  # Dark gray background
LOW_CONTRAST_EDGE = (192, 192, 192)  # Light gray edge

# Test Suite 1: Thickness values to test (in pixels)
THICKNESSES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24, 28, 32]

# Test Suite 2: Offset values to test (in pixels from center)
OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
OFFSET_EDGE_THICKNESS = 2  # Fixed thickness for offset test suite
MAIN_EDGE_THICKNESS = 2  # Fixed thickness for centered single-line suites

# Test Suite 3: Angled lines through center
ANGLES = list(range(0, 181, 2))  # 0 to 180 degrees in 2-degree increments

# Test Suites 4 & 5: Distraction offsets
DISTRACTION_OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

# Test Suite 6: Angled intersections
ANGLED_INTERSECTION_OFFSETS = [
    0,
    2,
    4,
    8,
    16,
    24,
]  # Y offset below center for intersection
ANGLED_INTERSECTION_ANGLES = list(range(0, 166, 15))  # 0 to 165 in 15-deg increments
ANGLED_INTERSECTION_THICKNESSES = [2, 4, 8, 16]
ANGLED_INTERSECTION_MAIN_THICKNESS = 2  # Fixed main vertical line thickness

# Test Suite 7: Single angled lines offset from center
SINGLE_ANGLED_LINE_OFFSETS = [8, 16, 24]  # Y offset from center
SINGLE_ANGLED_LINE_ANGLES = list(range(0, 166, 15))  # 0 to 165 in 15-deg increments
SINGLE_ANGLED_LINE_THICKNESSES = [2, 4, 8, 16]

# Output directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_edge_test_images"

SUITE_NAMES = [
    "thickness",
    "offset",
    "angled_lines",
    "horizontal_distraction_offset",
    "vertical_distraction_offset",
    "angled_intersections",
    "single_angled_lines",
]

SUITE_ALIASES = {
    str(index): suite_name for index, suite_name in enumerate(SUITE_NAMES, start=1)
}
SUITE_ALIASES.update({suite_name: suite_name for suite_name in SUITE_NAMES})


# ============================================================================
# Helper Functions
# ============================================================================


def create_image(background_color: tuple[int, int, int]) -> np.ndarray:
    """Create a base 64x64x3 RGB image with specified background color.

    Args:
        background_color: RGB tuple (R, G, B) for background color

    Returns:
        64x64x3 numpy array of uint8
    """
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[:, :] = background_color
    return image


def create_vertical_edge(
    image: np.ndarray, x_position: int, thickness: int, color: tuple[int, int, int]
) -> np.ndarray:
    """Draw a vertical edge on the image.

    Args:
        image: RGB image array to modify
        x_position: X coordinate of edge center
        thickness: Thickness of edge in pixels
        color: RGB tuple (R, G, B) for edge color

    Returns:
        Modified image array
    """
    half_thickness = thickness // 2
    x_start = max(0, x_position - half_thickness)
    x_end = min(IMAGE_SIZE, x_position + half_thickness + (thickness % 2))
    image[:, x_start:x_end] = color
    return image


def create_horizontal_line(
    image: np.ndarray, y_position: int, thickness: int, color: tuple[int, int, int]
) -> np.ndarray:
    """Draw a horizontal line on the image.

    Args:
        image: RGB image array to modify
        y_position: Y coordinate of line center
        thickness: Thickness of line in pixels
        color: RGB tuple (R, G, B) for line color

    Returns:
        Modified image array
    """
    half_thickness = thickness // 2
    y_start = max(0, y_position - half_thickness)
    y_end = min(IMAGE_SIZE, y_position + half_thickness + (thickness % 2))
    image[y_start:y_end, :] = color
    return image


def create_diagonal_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    thickness: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw a diagonal line on the image using OpenCV.

    Args:
        image: RGB image array to modify
        start: (x, y) start coordinates
        end: (x, y) end coordinates
        thickness: Thickness of line in pixels
        color: RGB tuple (R, G, B) for line color (OpenCV uses BGR, so we'll convert)

    Returns:
        Modified image array
    """
    # OpenCV uses BGR format
    bgr_color = (color[2], color[1], color[0])
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.line(image_bgr, start, end, bgr_color, thickness)
    # Convert back to RGB
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    """Save an RGB image as a PNG using OpenCV's BGR channel order.

    Raises:
        OSError: If OpenCV fails to write the image.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Failed to save image to {path}")


# ============================================================================
# Input Parsing and Validation Functions
# ============================================================================


def parse_suite_selection(input_str: str) -> list[str]:
    """Parse and validate suite selection input.

    Args:
        input_str: Comma or space-separated suite names or numbers.

    Returns:
        List of valid suite names.

    Raises:
        ValueError: If input contains invalid suite names or numbers.
    """
    # Replace commas with spaces and split
    input_str = input_str.replace(",", " ").strip()
    if not input_str:
        raise ValueError("Empty input provided")

    parts = input_str.split()
    suites = []

    for part in parts:
        suite_name = SUITE_ALIASES.get(part)
        if suite_name is None:
            valid_options = ", ".join(SUITE_NAMES)
            raise ValueError(
                f"Invalid suite '{part}'. Use 1-{len(SUITE_NAMES)} or one of: "
                f"{valid_options}."
            )
        suites.append(suite_name)

    # Remove duplicates while preserving order
    seen = set()
    unique_suites = []
    for suite in suites:
        if suite not in seen:
            seen.add(suite)
            unique_suites.append(suite)

    return unique_suites


def get_contrast_colors(
    contrast_level: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Get background and edge colors for a contrast level.

    Returns:
        Background color and edge color.
    """
    if contrast_level == "high":
        return HIGH_CONTRAST_BG, HIGH_CONTRAST_EDGE
    return LOW_CONTRAST_BG, LOW_CONTRAST_EDGE


# ============================================================================
# Test Suite Generation Functions
# ============================================================================


def generate_thickness_test_images(output_dir: Path, contrast_level: str) -> None:
    """Generate test suite 1: vertical edges at center with varying thicknesses.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for thickness in THICKNESSES:
        # Create image with background
        image = create_image(bg_color)

        # Draw vertical edge at center
        create_vertical_edge(image, CENTER_X, thickness, edge_color)

        # Save image
        filename = f"thickness_{thickness}px_center_{contrast_level}.png"
        filepath = output_dir / filename
        save_rgb_png(filepath, image)
        print(f"Generated: {filepath}")


def generate_offset_test_images(output_dir: Path, contrast_level: str) -> None:
    """Generate test suite 2: vertical edges at different off-center positions.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for offset in OFFSETS:
        # Create image with background
        image = create_image(bg_color)

        # Calculate edge position: center + offset
        # Offset can be positive (right) or we'll use absolute value
        x_position = CENTER_X + offset

        # Only generate if edge is within image bounds
        if 0 <= x_position < IMAGE_SIZE:
            # Draw vertical edge at offset position
            create_vertical_edge(image, x_position, OFFSET_EDGE_THICKNESS, edge_color)

            # Save image
            filename = (
                f"offset_{offset}px_thickness_{OFFSET_EDGE_THICKNESS}px_"
                f"{contrast_level}.png"
            )
            filepath = output_dir / filename
            save_rgb_png(filepath, image)
            print(f"Generated: {filepath}")


def generate_angled_lines_test_images(output_dir: Path, contrast_level: str) -> None:
    """Generate test suite 3: angled lines through center at different angles.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for angle_deg in ANGLES:
        # Create image with background
        image = create_image(bg_color)

        # Convert angle to radians (angle from horizontal, positive = counterclockwise)
        angle_rad = np.deg2rad(angle_deg)

        # Calculate direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Find intersections with image boundaries
        # Line through center: (x, y) = (CENTER_X, CENTER_Y) + t * (dx, dy)
        t_values = []

        # Intersection with left edge (x = 0)
        if abs(dx) > 1e-6:
            t = (0 - CENTER_X) / dx
            y = CENTER_Y + t * dy
            if 0 <= y < IMAGE_SIZE:
                t_values.append(t)

        # Intersection with right edge (x = IMAGE_SIZE - 1)
        if abs(dx) > 1e-6:
            t = (IMAGE_SIZE - 1 - CENTER_X) / dx
            y = CENTER_Y + t * dy
            if 0 <= y < IMAGE_SIZE:
                t_values.append(t)

        # Intersection with top edge (y = 0)
        if abs(dy) > 1e-6:
            t = (0 - CENTER_Y) / dy
            x = CENTER_X + t * dx
            if 0 <= x < IMAGE_SIZE:
                t_values.append(t)

        # Intersection with bottom edge (y = IMAGE_SIZE - 1)
        if abs(dy) > 1e-6:
            t = (IMAGE_SIZE - 1 - CENTER_Y) / dy
            x = CENTER_X + t * dx
            if 0 <= x < IMAGE_SIZE:
                t_values.append(t)

        # Get the two extreme t values (min and max)
        if len(t_values) >= 2:
            t_min = min(t_values)
            t_max = max(t_values)

            start = (
                int(CENTER_X + t_min * dx),
                int(CENTER_Y + t_min * dy),
            )
            end = (
                int(CENTER_X + t_max * dx),
                int(CENTER_Y + t_max * dy),
            )

            # Clamp to image bounds
            start = (
                max(0, min(IMAGE_SIZE - 1, start[0])),
                max(0, min(IMAGE_SIZE - 1, start[1])),
            )
            end = (
                max(0, min(IMAGE_SIZE - 1, end[0])),
                max(0, min(IMAGE_SIZE - 1, end[1])),
            )

            image = create_diagonal_line(
                image, start, end, MAIN_EDGE_THICKNESS, edge_color
            )
        elif angle_deg in {0, 180}:
            # Horizontal line
            image = create_horizontal_line(
                image, CENTER_Y, MAIN_EDGE_THICKNESS, edge_color
            )
        elif angle_deg == 90:
            # Vertical line
            image = create_vertical_edge(
                image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color
            )

        # Save image
        filename = f"angled_{angle_deg}deg_{contrast_level}.png"
        filepath = output_dir / filename
        save_rgb_png(filepath, image)
        print(f"Generated: {filepath}")


def generate_horizontal_distraction_offset_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 4: horizontal distraction lines with offsets.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    # Define thickness combinations
    # Same thickness: (main, distraction)
    same_thickness = [(2, 2), (4, 4), (8, 8), (14, 14)]
    # Main thicker: (main, distraction)
    main_thicker = [(4, 2), (8, 2), (8, 4), (14, 2), (14, 4), (14, 8)]
    # Distraction thicker: (main, distraction)
    dist_thicker = [(2, 4), (2, 8), (2, 14), (4, 8), (4, 14), (8, 14)]

    thickness_combinations = same_thickness + main_thicker + dist_thicker

    for offset in DISTRACTION_OFFSETS:
        for main_thick, dist_thick in thickness_combinations:
            # Create image with background
            image = create_image(bg_color)

            # Draw main vertical edge at center
            create_vertical_edge(image, CENTER_X, main_thick, edge_color)

            # Draw horizontal distraction line at center Y, offset by offset pixels
            y_position = CENTER_Y + offset
            # Only generate if line is within image bounds
            if 0 <= y_position < IMAGE_SIZE:
                create_horizontal_line(image, y_position, dist_thick, edge_color)

                # Save image
                filename = (
                    f"horizontal_distraction_offset{offset}_"
                    f"main{main_thick}_dist{dist_thick}_{contrast_level}.png"
                )
                filepath = output_dir / filename
                save_rgb_png(filepath, image)
                print(f"Generated: {filepath}")


def generate_vertical_distraction_offset_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 5: vertical distraction lines with offsets.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    # Define thickness combinations (same as suite 4)
    same_thickness = [(2, 2), (4, 4), (8, 8), (14, 14)]
    main_thicker = [(4, 2), (8, 2), (8, 4), (14, 2), (14, 4), (14, 8)]
    dist_thicker = [(2, 4), (2, 8), (2, 14), (4, 8), (4, 14), (8, 14)]

    thickness_combinations = same_thickness + main_thicker + dist_thicker

    for offset in DISTRACTION_OFFSETS:
        for main_thick, dist_thick in thickness_combinations:
            # Create image with background
            image = create_image(bg_color)

            # Draw main vertical edge at center
            create_vertical_edge(image, CENTER_X, main_thick, edge_color)

            # Draw vertical distraction line at center X, offset by offset pixels
            x_position = CENTER_X + offset
            # Only generate if line is within image bounds
            if 0 <= x_position < IMAGE_SIZE:
                create_vertical_edge(image, x_position, dist_thick, edge_color)

                # Save image
                filename = (
                    f"vertical_distraction_offset{offset}_"
                    f"main{main_thick}_dist{dist_thick}_{contrast_level}.png"
                )
                filepath = output_dir / filename
                save_rgb_png(filepath, image)
                print(f"Generated: {filepath}")


def generate_angled_intersections_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 6: angled line intersections with vertical line.

    Creates images with a main vertical line at center and an intersecting line
    at various angles, with the intersection point offset below center.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for offset in ANGLED_INTERSECTION_OFFSETS:
        # Intersection point is at center X, offset below center Y
        intersection_x = CENTER_X
        intersection_y = CENTER_Y + offset

        for angle_deg in ANGLED_INTERSECTION_ANGLES:
            for thickness in ANGLED_INTERSECTION_THICKNESSES:
                # Create image with background
                image = create_image(bg_color)

                # Draw main vertical edge at center with fixed thickness
                create_vertical_edge(
                    image, CENTER_X, ANGLED_INTERSECTION_MAIN_THICKNESS, edge_color
                )

                # Draw intersecting line at the specified angle through the
                # intersection point.
                # Convert angle to radians (angle from horizontal)
                angle_rad = np.deg2rad(angle_deg)

                # Calculate direction vector
                dx = np.cos(angle_rad)
                dy = np.sin(angle_rad)

                # Find intersections with image boundaries
                # Line through intersection point: (x, y) = (ix, iy) + t * (dx, dy)
                t_values = []

                # Intersection with left edge (x = 0)
                if abs(dx) > 1e-6:
                    t = (0 - intersection_x) / dx
                    y = intersection_y + t * dy
                    if 0 <= y < IMAGE_SIZE:
                        t_values.append(t)

                # Intersection with right edge (x = IMAGE_SIZE - 1)
                if abs(dx) > 1e-6:
                    t = (IMAGE_SIZE - 1 - intersection_x) / dx
                    y = intersection_y + t * dy
                    if 0 <= y < IMAGE_SIZE:
                        t_values.append(t)

                # Intersection with top edge (y = 0)
                if abs(dy) > 1e-6:
                    t = (0 - intersection_y) / dy
                    x = intersection_x + t * dx
                    if 0 <= x < IMAGE_SIZE:
                        t_values.append(t)

                # Intersection with bottom edge (y = IMAGE_SIZE - 1)
                if abs(dy) > 1e-6:
                    t = (IMAGE_SIZE - 1 - intersection_y) / dy
                    x = intersection_x + t * dx
                    if 0 <= x < IMAGE_SIZE:
                        t_values.append(t)

                # Get the two extreme t values (min and max)
                if len(t_values) >= 2:
                    t_min = min(t_values)
                    t_max = max(t_values)

                    start = (
                        int(intersection_x + t_min * dx),
                        int(intersection_y + t_min * dy),
                    )
                    end = (
                        int(intersection_x + t_max * dx),
                        int(intersection_y + t_max * dy),
                    )

                    # Clamp to image bounds
                    start = (
                        max(0, min(IMAGE_SIZE - 1, start[0])),
                        max(0, min(IMAGE_SIZE - 1, start[1])),
                    )
                    end = (
                        max(0, min(IMAGE_SIZE - 1, end[0])),
                        max(0, min(IMAGE_SIZE - 1, end[1])),
                    )

                    image = create_diagonal_line(
                        image, start, end, thickness, edge_color
                    )
                elif angle_deg == 90:
                    # Special case: vertical line (same as main line, creates overlap)
                    create_vertical_edge(image, intersection_x, thickness, edge_color)

                # Save image
                filename = (
                    f"angled_intersection_offset{offset}_"
                    f"angle{angle_deg}_thick{thickness}_{contrast_level}.png"
                )
                filepath = output_dir / filename
                save_rgb_png(filepath, image)
                print(f"Generated: {filepath}")


def generate_single_angled_line_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 7: single angled lines offset from center.

    Creates images with a single angled line where a point on the line is offset
    from the image center. Similar to suite 6 but without the vertical line.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for offset in SINGLE_ANGLED_LINE_OFFSETS:
        # Point on line is at center X, offset from center Y
        line_point_x = CENTER_X
        line_point_y = CENTER_Y + offset

        for angle_deg in SINGLE_ANGLED_LINE_ANGLES:
            for thickness in SINGLE_ANGLED_LINE_THICKNESSES:
                # Create image with background
                image = create_image(bg_color)

                # Draw single angled line at the specified angle through line point
                # Convert angle to radians (angle from horizontal)
                angle_rad = np.deg2rad(angle_deg)

                # Calculate direction vector
                dx = np.cos(angle_rad)
                dy = np.sin(angle_rad)

                # Find intersections with image boundaries
                # Line through line point: (x, y) = (px, py) + t * (dx, dy)
                t_values = []

                # Intersection with left edge (x = 0)
                if abs(dx) > 1e-6:
                    t = (0 - line_point_x) / dx
                    y = line_point_y + t * dy
                    if 0 <= y < IMAGE_SIZE:
                        t_values.append(t)

                # Intersection with right edge (x = IMAGE_SIZE - 1)
                if abs(dx) > 1e-6:
                    t = (IMAGE_SIZE - 1 - line_point_x) / dx
                    y = line_point_y + t * dy
                    if 0 <= y < IMAGE_SIZE:
                        t_values.append(t)

                # Intersection with top edge (y = 0)
                if abs(dy) > 1e-6:
                    t = (0 - line_point_y) / dy
                    x = line_point_x + t * dx
                    if 0 <= x < IMAGE_SIZE:
                        t_values.append(t)

                # Intersection with bottom edge (y = IMAGE_SIZE - 1)
                if abs(dy) > 1e-6:
                    t = (IMAGE_SIZE - 1 - line_point_y) / dy
                    x = line_point_x + t * dx
                    if 0 <= x < IMAGE_SIZE:
                        t_values.append(t)

                # Get the two extreme t values (min and max)
                if len(t_values) >= 2:
                    t_min = min(t_values)
                    t_max = max(t_values)

                    start = (
                        int(line_point_x + t_min * dx),
                        int(line_point_y + t_min * dy),
                    )
                    end = (
                        int(line_point_x + t_max * dx),
                        int(line_point_y + t_max * dy),
                    )

                    # Clamp to image bounds
                    start = (
                        max(0, min(IMAGE_SIZE - 1, start[0])),
                        max(0, min(IMAGE_SIZE - 1, start[1])),
                    )
                    end = (
                        max(0, min(IMAGE_SIZE - 1, end[0])),
                        max(0, min(IMAGE_SIZE - 1, end[1])),
                    )

                    image = create_diagonal_line(
                        image, start, end, thickness, edge_color
                    )
                elif angle_deg == 90:
                    # Special case: vertical line
                    create_vertical_edge(image, line_point_x, thickness, edge_color)

                # Save image
                filename = (
                    f"single_angled_line_offset{offset}_"
                    f"angle{angle_deg}_thick{thickness}_{contrast_level}.png"
                )
                filepath = output_dir / filename
                save_rgb_png(filepath, image)
                print(f"Generated: {filepath}")


# ============================================================================
# Main Function
# ============================================================================


def main(
    suites: list[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    contrast_level: str = "high",
) -> None:
    """Main function to generate selected test suites.

    Args:
        suites: List of suite names to generate. If None, all suites are generated.
        output_dir: Dataset output directory.
        contrast_level: 'high' or 'low'.
    """
    if suites is None:
        suites = list(SUITE_NAMES)

    # Define output directories for each suite
    suite_dirs = {
        "thickness": output_dir / "thickness",
        "offset": output_dir / "offset",
        "angled_lines": output_dir / "angled_lines",
        "horizontal_distraction_offset": output_dir / "horizontal_distraction_offset",
        "vertical_distraction_offset": output_dir / "vertical_distraction_offset",
        "angled_intersections": output_dir / "angled_intersections",
        "single_angled_lines": output_dir / "single_angled_lines",
    }

    suite_generators = {
        "thickness": generate_thickness_test_images,
        "offset": generate_offset_test_images,
        "angled_lines": generate_angled_lines_test_images,
        "horizontal_distraction_offset": (
            generate_horizontal_distraction_offset_test_images
        ),
        "vertical_distraction_offset": generate_vertical_distraction_offset_test_images,
        "angled_intersections": generate_angled_intersections_test_images,
        "single_angled_lines": generate_single_angled_line_test_images,
    }

    # Check if any selected suite's output directory exists
    existing_dirs = [suite_dirs[s] for s in suites if suite_dirs[s].exists()]
    if existing_dirs:
        print("\nThe following output directories already exist:")
        for d in existing_dirs:
            print(f"  - {d.absolute()}")
        response = (
            input("Do you want to override existing files? (y/n): ").strip().lower()
        )
        if response not in ("y", "yes"):
            print("Aborting. No files were modified.")
            return

    print("=" * 60)
    print("Generating synthetic edge detection test images")
    print(f"Selected test suites: {', '.join(suites)}")
    print(f"Contrast: {contrast_level}")
    print("=" * 60)

    for suite in suites:
        print(f"\nGenerating {suite} ({contrast_level} contrast)...")
        suite_generators[suite](suite_dirs[suite], contrast_level)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic test images for comparing edge detection methods. "
            "By default, all retained suites are generated in high contrast."
        ),
        epilog=(
            "Suites:\n"
            "  1 thickness\n"
            "  2 offset\n"
            "  3 angled_lines\n"
            "  4 horizontal_distraction_offset\n"
            "  5 vertical_distraction_offset\n"
            "  6 angled_intersections\n"
            "  7 single_angled_lines\n\n"
            "Examples:\n"
            "  python sandbox/generate_synthetic_edge_images.py\n"
            "  python sandbox/generate_synthetic_edge_images.py --contrast low\n"
            "  python sandbox/generate_synthetic_edge_images.py "
            "--suite thickness,angled_intersections "
            "--output-dir /tmp/synthetic_edge_test_images"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        "-s",
        nargs="+",
        default=None,
        help=(
            "Suites to generate (comma or space separated). "
            "Accepts names or numbers 1-7. Default: all suites."
        ),
    )
    parser.add_argument(
        "--contrast",
        choices=("high", "low"),
        default="high",
        help="Contrast level to generate (default: high).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="PATH",
        help=(
            "Dataset output directory "
            "(default: <project root>/data/synthetic_edge_test_images)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse suite selection from command line if provided
    suites = None
    if args.suite:
        try:
            suites = parse_suite_selection(" ".join(args.suite))
        except ValueError as e:
            raise SystemExit(f"Error parsing suite selection: {e}") from e

    main(suites=suites, output_dir=args.output_dir, contrast_level=args.contrast)
