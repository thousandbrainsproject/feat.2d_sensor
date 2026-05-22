# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sandbox.plot_edge_patch_manifest import load_manifest_rows, plot_world_xy


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Failed to save image to {path}")


class PlotEdgePatchManifestTest(unittest.TestCase):
    def test_load_manifest_rows_resolves_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_dir = Path(tempdir)
            image_path = manifest_dir / "patch.png"
            save_rgb_png(image_path, np.full((4, 4, 3), 255, dtype=np.uint8))
            manifest_path = manifest_dir / "edge_patch_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "index",
                        "filename",
                        "sensor_id",
                        "world_x",
                        "world_y",
                        "world_z",
                        "angle_degrees",
                        "edge_strength",
                        "coherence",
                        "is_geometric_edge",
                        "has_edge",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "index": "0",
                        "filename": "patch.png",
                        "sensor_id": "patch",
                        "world_x": "1.5",
                        "world_y": "-2.5",
                        "world_z": "3.5",
                        "angle_degrees": "90.0",
                        "edge_strength": "2.5",
                        "coherence": "0.75",
                        "is_geometric_edge": "False",
                        "has_edge": "True",
                    }
                )

            rows = load_manifest_rows(manifest_path)

            assert len(rows) == 1
            assert rows[0].image_path == image_path
            assert rows[0].world_x == 1.5
            assert rows[0].world_y == -2.5
            assert rows[0].angle_degrees == 90.0

    def test_plot_world_xy_saves_png(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_dir = Path(tempdir)
            for index, color in enumerate([(255, 0, 0), (0, 255, 0)]):
                image = np.zeros((4, 4, 3), dtype=np.uint8)
                image[:, :] = color
                save_rgb_png(manifest_dir / f"patch_{index}.png", image)

            manifest_path = manifest_dir / "edge_patch_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "index",
                        "filename",
                        "sensor_id",
                        "world_x",
                        "world_y",
                        "world_z",
                        "angle_degrees",
                        "edge_strength",
                        "coherence",
                        "is_geometric_edge",
                        "has_edge",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "index": "0",
                        "filename": "patch_0.png",
                        "sensor_id": "patch",
                        "world_x": "0.0",
                        "world_y": "0.0",
                        "world_z": "0.0",
                        "angle_degrees": "0.0",
                        "edge_strength": "1.0",
                        "coherence": "0.5",
                        "is_geometric_edge": "False",
                        "has_edge": "True",
                    }
                )
                writer.writerow(
                    {
                        "index": "1",
                        "filename": "patch_1.png",
                        "sensor_id": "patch",
                        "world_x": "1.0",
                        "world_y": "1.0",
                        "world_z": "0.0",
                        "angle_degrees": "90.0",
                        "edge_strength": "1.0",
                        "coherence": "0.5",
                        "is_geometric_edge": "False",
                        "has_edge": "True",
                    }
                )
            output_path = manifest_dir / "world_xy.png"

            rows = load_manifest_rows(manifest_path)
            plot_world_xy(rows, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0


if __name__ == "__main__":
    unittest.main()
