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
from unittest.mock import Mock

import numpy as np

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.utils.edge_detection import EdgeFeatures
from tests.unit.frameworks.models.two_d_sensor_module_test import (
    SURFACE_NORMAL_3D,
    make_2d_sm,
    make_message,
    make_no_edge,
    make_raw_observation,
)


def make_edge(*, is_geometric_edge: bool = False) -> EdgeFeatures:
    return EdgeFeatures(
        angle=np.pi / 2,
        strength=2.5,
        coherence=0.75,
        is_geometric_edge=is_geometric_edge,
        has_edge=True,
    )


def read_manifest_rows(debug_dir: Path) -> list[dict[str, str]]:
    manifest_path = debug_dir / "edge_patch_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TwoDSensorModuleDebugManifestTest(unittest.TestCase):
    def test_debug_edge_manifest_not_written_when_directory_is_not_set(self) -> None:
        observation = make_raw_observation(
            center_location=np.array([1.0, 2.0, 3.0]),
            semantic_id=1,
        )
        two_d_sm = make_2d_sm(edge_detector=Mock(return_value=make_edge()))
        two_d_sm._update_tangent_frame(surface_normal_3d=SURFACE_NORMAL_3D)

        two_d_sm._extract_2d_edge(
            make_message(location=np.array([1.0, 2.0, 3.0])),
            observation,
            SURFACE_NORMAL_3D,
        )

        assert two_d_sm._debug_edge_patch_dir is None

    def test_detected_edge_writes_png_and_manifest_row(self) -> None:
        observation = make_raw_observation(
            center_location=np.array([99.0, 99.0, 99.0]),
            semantic_id=1,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            debug_dir = Path(tempdir)
            two_d_sm = make_2d_sm(
                sensor_module_id="patch/0",
                edge_detector=Mock(return_value=make_edge()),
                debug_edge_patch_dir=debug_dir,
                debug_edge_patch_prefix="debug",
            )
            percept = make_message(location=np.array([1.25, -2.5, 3.75]))
            two_d_sm._observation_processor.process = Mock(return_value=percept)

            msg = two_d_sm.step(
                ctx=RuntimeContext(rng=np.random.RandomState()),
                observation=observation,
                motor_only_step=False,
            )

            np.testing.assert_allclose(msg.location, [1.25, -2.5, 0.0])
            saved_paths = list(debug_dir.glob("*.png"))
            assert len(saved_paths) == 1
            assert saved_paths[0].name == "debug_patch_0_000000_angle_90.0.png"

            rows = read_manifest_rows(debug_dir)
            assert rows == [
                {
                    "index": "0",
                    "filename": "debug_patch_0_000000_angle_90.0.png",
                    "sensor_id": "patch_0",
                    "world_x": "1.250000",
                    "world_y": "-2.500000",
                    "world_z": "3.750000",
                    "angle_degrees": "90.000000",
                    "edge_strength": "2.500000",
                    "coherence": "0.750000",
                    "is_geometric_edge": "False",
                    "has_edge": "True",
                }
            ]

    def test_geometric_edge_is_included_in_manifest_before_rejection(self) -> None:
        observation = make_raw_observation(
            center_location=np.zeros(3),
            semantic_id=1,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            debug_dir = Path(tempdir)
            two_d_sm = make_2d_sm(
                edge_detector=Mock(return_value=make_edge(is_geometric_edge=True)),
                debug_edge_patch_dir=debug_dir,
            )
            two_d_sm._update_tangent_frame(surface_normal_3d=SURFACE_NORMAL_3D)
            percept = make_message(location=np.array([4.0, 5.0, 6.0]))

            msg = two_d_sm._extract_2d_edge(percept, observation, SURFACE_NORMAL_3D)

            assert msg.morphological_features["pose_fully_defined"] is False
            rows = read_manifest_rows(debug_dir)
            assert rows[0]["is_geometric_edge"] == "True"
            assert rows[0]["world_x"] == "4.000000"
            assert rows[0]["world_y"] == "5.000000"
            assert rows[0]["world_z"] == "6.000000"

    def test_no_edge_writes_neither_png_nor_manifest(self) -> None:
        observation = make_raw_observation(
            center_location=np.zeros(3),
            semantic_id=1,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            debug_dir = Path(tempdir)
            two_d_sm = make_2d_sm(
                edge_detector=Mock(return_value=make_no_edge()),
                debug_edge_patch_dir=debug_dir,
            )
            two_d_sm._update_tangent_frame(surface_normal_3d=SURFACE_NORMAL_3D)

            two_d_sm._extract_2d_edge(make_message(), observation, SURFACE_NORMAL_3D)

            assert list(debug_dir.glob("*.png")) == []
            assert not (debug_dir / "edge_patch_manifest.csv").exists()


if __name__ == "__main__":
    unittest.main()
