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
