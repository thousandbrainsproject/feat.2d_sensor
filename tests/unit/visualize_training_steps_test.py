# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np

from sandbox import visualize_training_steps as viz


def test_default_mesh_dir_uses_compositional_objects_v1_1():
    """Logo learning configs train on compositional_objects_1.1 meshes."""
    assert viz.DEFAULT_MESH_DIR.endswith("compositional_objects_1.1/meshes")


def test_compute_gaze_directions_uses_patch_sensor_key():
    """2D detailed logs store the patch rotation under sensors.patch."""
    action_sequence = [
        [
            [],
            {
                "agent_id_0": {
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "sensors": {
                        "patch": {
                            "position": [0.0, 0.0, 0.03],
                            "rotation": [1.0, 0.0, 0.0, 0.0],
                        },
                        "view_finder": {
                            "position": [0.0, 0.0, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0],
                        },
                    },
                }
            },
        ]
    ]

    directions = viz._compute_gaze_directions(action_sequence, n_steps=1)

    np.testing.assert_allclose(directions, [[0.0, 0.0, -1.0]])


def test_project_2d_gaze_keeps_original_points_when_raycast_misses():
    data = viz.EpisodeData(
        object_name="object",
        locations=np.array([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]]),
        features={"on_object": np.array([1.0, 1.0])},
        stepwise_targets=["object", "object"],
        n_steps=2,
        sensor_positions=np.array([[0.0, 0.0, 0.03], [0.0, 0.0, 0.03]]),
        is_2d=True,
        action_sequence=[],
    )
    original_locations = data.locations.copy()
    original_sensor_positions = data.sensor_positions.copy()

    sensor_positions, n_hits = viz._apply_gaze_projection(
        data,
        hit_points=np.full((2, 3), np.nan),
        hit_mask=np.array([False, False]),
        sensor_positions=data.sensor_positions,
    )

    assert n_hits == 0
    np.testing.assert_allclose(data.locations, original_locations)
    np.testing.assert_allclose(sensor_positions, original_sensor_positions)
    assert data.n_steps == 2
    assert data.stepwise_targets == ["object", "object"]
