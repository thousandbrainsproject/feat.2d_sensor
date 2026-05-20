# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import hydra

from tests import HYDRA_ROOT


def test_disk_2d_inference_delta_threshold_features_are_extracted() -> None:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
        config = hydra.compose(
            config_name="experiment",
            overrides=["experiment=2d_sm/inference/disk_inference_2d_on_2d"],
        )

    sensor_config = config.experiment.config.monty_config.sensor_modules.sensor_module_0
    threshold_features = set(sensor_config.delta_thresholds) - {"distance", "n_steps"}

    assert threshold_features <= set(sensor_config.features)
