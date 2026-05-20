# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import fields

import pytest

from sandbox import visualize_learned_model as viz


def test_cli_uses_fixed_edge_filtering_and_edge_lengths(monkeypatch):
    """Learned-model visualization keeps edge thresholds and lengths internal."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "visualize_learned_model.py",
            "--model-path",
            "model.pt",
        ],
    )

    args = viz.parse_args()

    assert not hasattr(args, "edge_strength_threshold")
    assert not hasattr(args, "coherence_threshold")
    assert not hasattr(args, "arrow_scale")
    assert not hasattr(args, "hide_unscaled_edge_lines")


@pytest.mark.parametrize(
    "removed_flag",
    [
        "--edge-strength-threshold",
        "--coherence-threshold",
        "--arrow-scale",
        "--hide-unscaled-edge-lines",
    ],
)
def test_removed_cli_flags_are_rejected(monkeypatch, removed_flag):
    """Removed tuning flags are no longer accepted by argparse."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "visualize_learned_model.py",
            "--model-path",
            "model.pt",
            removed_flag,
            "1.0",
        ],
    )

    with pytest.raises(SystemExit):
        viz.parse_args()


def test_visualization_config_has_no_scaled_edge_arrow_setting():
    """Only fixed-length reference edge lines remain configurable."""
    config_fields = {field.name for field in fields(viz.VisualizationConfig)}

    assert "arrow_scale" not in config_fields
    assert "show_scaled_edge_lines" not in config_fields
    assert "unscaled_edge_scale" in config_fields
    assert "show_unscaled_edge_lines" not in config_fields


def test_control_specs_show_reference_edges_and_reset_camera():
    """Controls expose edge filtering, fixed edge lines, normals, and camera reset."""
    visualizer = object.__new__(viz.LearnedModelVisualizer)

    specs = visualizer._control_specs()
    callbacks = [spec.callback_name for spec in specs]
    labels = {spec.callback_name: (spec.label_off, spec.label_on) for spec in specs}

    assert "scaled_edges" not in callbacks
    assert "unscaled_edges" in callbacks
    assert "_reset_camera" in callbacks
    assert labels["unscaled_edges"] == (" Edges Off ", " Edges On ")
    assert labels["_reset_camera"] == (" Reset Camera ", " Reset Camera ")


def test_control_positions_are_horizontal():
    """Controls are arranged left-to-right along one row."""
    visualizer = object.__new__(viz.LearnedModelVisualizer)

    first = visualizer._control_position(0)
    second = visualizer._control_position(1)
    third = visualizer._control_position(2)

    assert second[0] > first[0]
    assert third[0] > second[0]
    assert first[1] == second[1] == third[1]
