# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
import numpy.testing as nptest
import quaternion as qt
from parameterized import parameterized_class

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.edge_detection import EdgeDetectionConfig

MODULE_PATH = "tbp.monty.frameworks.models.two_d_sensor_module"


def make_state(**overrides):
    """Create a State with sensible defaults.

    use_state=False by default to skip _check_all_attributes() validation.
    """
    defaults = dict(
        location=np.array([1.0, 2.0, 3.0]),
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": False,
        },
        non_morphological_features={},
        confidence=1.0,
        use_state=False,
        sender_id="test_sm",
        sender_type="SM",
    )
    defaults.update(overrides)
    return State(**defaults)


def make_module(**overrides):
    """Create a TwoDSensorModule with minimal valid args."""
    defaults = dict(
        sensor_module_id="test_sm",
        features=["edge_strength", "coherence", "on_object"],
    )
    defaults.update(overrides)
    return TwoDSensorModule(**defaults)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit(unittest.TestCase):
    def test_default_edge_config_when_none(self):
        sm = make_module(edge_detection_config=None)
        expected = EdgeDetectionConfig()
        self.assertEqual(sm.edge_detection_config, expected)

    def test_custom_edge_config_stored(self):
        config = EdgeDetectionConfig(strength_threshold=0.5)
        sm = make_module(edge_detection_config=config)
        self.assertIs(sm.edge_detection_config, config)

    def test_warns_missing_edge_features(self):
        logger = logging.getLogger(MODULE_PATH)
        with self.assertLogs(logger, level="WARNING") as cm:
            make_module(features=["on_object"])
        self.assertTrue(
            any("edge_strength" in msg or "coherence" in msg for msg in cm.output)
        )

    def test_no_warning_when_edge_features_present(self):
        logger = logging.getLogger(MODULE_PATH)
        with self.assertLogs(logger, level="DEBUG") as cm:
            logger.debug("sentinel")
            make_module(features=["edge_strength", "coherence", "on_object"])
        warnings = [m for m in cm.output if "WARNING" in m]
        self.assertEqual(warnings, [])

    def test_initial_internal_state(self):
        sm = make_module()
        self.assertIsNone(sm._previous_3d_location)
        self.assertIsNone(sm._tangent_frame)
        nptest.assert_array_equal(sm._previous_2d_location, np.zeros(2))


# ---------------------------------------------------------------------------
# TestExtract2dEdge
# ---------------------------------------------------------------------------


class TestExtract2dEdge(unittest.TestCase):
    def setUp(self):
        self.sm = make_module()
        self.rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        self.world_camera = np.eye(4)
        self.depth = np.ones((64, 64), dtype=np.float32)

    def _base_state(self, **morph_overrides):
        morph = {"pose_vectors": np.eye(3), "pose_fully_defined": False}
        morph.update(morph_overrides)
        return make_state(morphological_features=morph)

    def test_no_pose_vectors_returns_unchanged(self):
        state = make_state(
            morphological_features={"pose_fully_defined": False},
            use_state=False,
        )
        result = self.sm._extract_2d_edge(state, self.rgba, self.world_camera)
        self.assertIs(result, state)

    @patch(f"{MODULE_PATH}.is_geometric_edge", return_value=True)
    @patch(
        f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
        return_value=(0.5, 0.8, 1.0),
    )
    def test_geometric_edge_filtered_out(self, _mock_compute, _mock_geo):
        state = self._base_state()
        result = self.sm._extract_2d_edge(
            state, self.rgba, self.world_camera, depth_image=self.depth
        )
        self.assertIs(result, state)

    @patch(
        f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
        return_value=(0.01, 0.9, 0.5),
    )
    def test_below_strength_threshold(self, _mock_compute):
        state = self._base_state()
        result = self.sm._extract_2d_edge(state, self.rgba, self.world_camera)
        self.assertFalse(state.morphological_features["pose_fully_defined"])
        self.assertIs(result, state)

    @patch(
        f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
        return_value=(0.5, 0.1, 0.5),
    )
    def test_below_coherence_threshold(self, _mock_compute):
        state = self._base_state()
        result = self.sm._extract_2d_edge(state, self.rgba, self.world_camera)
        self.assertFalse(state.morphological_features["pose_fully_defined"])
        self.assertIs(result, state)

    @patch(
        f"{MODULE_PATH}.edge_angle_to_2d_pose",
        return_value=np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float),
    )
    @patch(f"{MODULE_PATH}.is_geometric_edge", return_value=False)
    @patch(
        f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
        return_value=(0.5, 0.8, 0.7),
    )
    def test_successful_edge_updates_pose(self, _mc, _mg, _mt):
        state = self._base_state()
        result = self.sm._extract_2d_edge(
            state, self.rgba, self.world_camera, depth_image=self.depth
        )
        self.assertTrue(result.morphological_features["pose_fully_defined"])
        pose = result.morphological_features["pose_vectors"]
        # Row 0: normal always [0,0,1] in 2D plane
        nptest.assert_allclose(pose[0], [0, 0, 1], atol=1e-7)
        # Row 1: edge tangent in xy-plane
        nptest.assert_allclose(pose[1], [0, 1, 0], atol=1e-7)
        # Row 2: edge perp in xy-plane
        nptest.assert_allclose(pose[2], [-1, 0, 0], atol=1e-7)
        self.assertAlmostEqual(result.non_morphological_features["edge_strength"], 0.5)
        self.assertAlmostEqual(result.non_morphological_features["coherence"], 0.8)

    @patch(f"{MODULE_PATH}.edge_angle_to_2d_pose", return_value=np.eye(3))
    @patch(f"{MODULE_PATH}.is_geometric_edge", return_value=False)
    @patch(
        f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
        return_value=(0.5, 0.8, 0.7),
    )
    def test_strips_alpha_channel(self, mock_compute, _mg, _mt):
        state = self._base_state()
        rgba_4ch = np.zeros((32, 32, 4), dtype=np.uint8)
        self.sm._extract_2d_edge(
            state, rgba_4ch, self.world_camera, depth_image=self.depth
        )
        patch_arg = mock_compute.call_args[0][0]
        self.assertEqual(patch_arg.shape[2], 3)

    @patch(f"{MODULE_PATH}.is_geometric_edge")
    @patch(f"{MODULE_PATH}.edge_angle_to_2d_pose", return_value=np.eye(3))
    @patch(
        f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
        return_value=(0.5, 0.8, 0.7),
    )
    def test_no_depth_skips_geometric_check(self, _mc, _mt, mock_geo):
        state = self._base_state()
        self.sm._extract_2d_edge(state, self.rgba, self.world_camera, depth_image=None)
        mock_geo.assert_not_called()

    @patch(
        f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
        return_value=(0.5, 0.8, np.pi / 4),
    )
    def test_edge_pose_uses_2d_rotation(self, _mc):
        """Verify edge_angle_to_2d_pose is called with orientation and camera."""
        state = self._base_state()
        with patch(f"{MODULE_PATH}.edge_angle_to_2d_pose") as mock_pose:
            s2 = np.sqrt(2) / 2
            mock_pose.return_value = np.array([[0, 0, 1], [s2, s2, 0], [-s2, s2, 0]])
            result = self.sm._extract_2d_edge(state, self.rgba, self.world_camera)
            mock_pose.assert_called_once_with(np.pi / 4, self.world_camera)
        self.assertTrue(result.morphological_features["pose_fully_defined"])
        nptest.assert_allclose(
            result.morphological_features["pose_vectors"][0], [0, 0, 1], atol=1e-7
        )

    @patch(f"{MODULE_PATH}.edge_angle_to_2d_pose", return_value=np.eye(3))
    @patch(
        f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
        return_value=(0.5, 0.8, 0.7),
    )
    def test_omits_features_not_in_list(self, _mc, _mt):
        sm = make_module(features=["coherence", "on_object"])
        state = self._base_state()
        result = sm._extract_2d_edge(state, self.rgba, self.world_camera)
        self.assertNotIn("edge_strength", result.non_morphological_features)
        self.assertIn("coherence", result.non_morphological_features)


# ---------------------------------------------------------------------------
# TestUpdate2dPositionAndDisplacement
# ---------------------------------------------------------------------------


class TestUpdate2dPositionAndDisplacement(unittest.TestCase):
    def setUp(self):
        self.sm = make_module()

    def _make_mock_tangent_frame(self):
        frame = MagicMock()
        frame.basis_u = np.array([1.0, 0.0, 0.0])
        frame.basis_v = np.array([0.0, 1.0, 0.0])
        frame.transport = MagicMock()
        return frame

    def test_off_object_zero_displacement(self):
        state = make_state(
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": False,
                "on_object": False,
            }
        )
        result = self.sm._update_2d_position_and_displacement(state, None, None)
        nptest.assert_array_equal(result.displacement["displacement"], np.zeros(3))

    def test_first_obs_initializes_from_world_xy(self):
        state = make_state(location=np.array([4.0, 5.0, 6.0]))
        self.sm._previous_3d_location = None
        sn = state.get_surface_normal()

        result = self.sm._update_2d_position_and_displacement(state, None, sn)
        nptest.assert_array_equal(result.location, [4.0, 5.0, 0.0])
        nptest.assert_array_equal(self.sm._previous_2d_location, [4.0, 5.0])
        self.assertIsNotNone(self.sm._tangent_frame)

    def test_zero_tangent_displacement(self):
        loc = np.array([1.0, 2.0, 3.0])
        self.sm._previous_3d_location = loc.copy()
        self.sm._previous_2d_location = np.array([1.0, 2.0])
        self.sm._tangent_frame = self._make_mock_tangent_frame()

        state = make_state(location=loc.copy())
        sn = state.get_surface_normal()
        result = self.sm._update_2d_position_and_displacement(state, None, sn)
        nptest.assert_array_equal(result.displacement["displacement"], np.zeros(3))
        nptest.assert_array_equal(result.location, [1.0, 2.0, 0.0])

    def test_displacement_without_curvature(self):
        self.sm._previous_3d_location = np.array([0.0, 0.0, 0.0])
        self.sm._previous_2d_location = np.array([0.0, 0.0])
        self.sm._tangent_frame = self._make_mock_tangent_frame()

        pose = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        state = make_state(
            location=np.array([0.3, 0.4, 0.0]),
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": False,
            },
        )
        sn = state.get_surface_normal()
        result = self.sm._update_2d_position_and_displacement(
            state, curvature_pose_vectors=None, surface_normal=sn
        )
        nptest.assert_allclose(
            result.displacement["displacement"], [0.3, 0.4, 0.0], atol=1e-10
        )
        nptest.assert_allclose(result.location, [0.3, 0.4, 0.0], atol=1e-10)

    @patch(
        f"{MODULE_PATH}.arc_length_corrected_displacement", return_value=(0.35, 0.45)
    )
    def test_displacement_with_arc_length_correction(self, mock_arc):
        self.sm._previous_3d_location = np.array([0.0, 0.0, 0.0])
        self.sm._previous_2d_location = np.array([0.0, 0.0])
        self.sm._tangent_frame = self._make_mock_tangent_frame()

        pose = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        curvature_pv = pose.copy()
        state = make_state(
            location=np.array([0.3, 0.4, 0.0]),
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": False,
                "principal_curvatures": np.array([0.1, 0.2]),
            },
        )
        sn = state.get_surface_normal()
        result = self.sm._update_2d_position_and_displacement(
            state, curvature_pose_vectors=curvature_pv, surface_normal=sn
        )
        mock_arc.assert_called_once()
        nptest.assert_allclose(
            result.displacement["displacement"], [0.35, 0.45, 0.0], atol=1e-10
        )

    def test_displacement_accumulates(self):
        self.sm._previous_3d_location = np.array([0.0, 0.0, 0.0])
        self.sm._previous_2d_location = np.array([0.0, 0.0])
        self.sm._tangent_frame = self._make_mock_tangent_frame()

        pose = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        state1 = make_state(
            location=np.array([1.0, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": pose.copy(),
                "pose_fully_defined": False,
            },
        )
        sn1 = state1.get_surface_normal()
        self.sm._update_2d_position_and_displacement(state1, None, sn1)

        state2 = make_state(
            location=np.array([1.0, 2.0, 0.0]),
            morphological_features={
                "pose_vectors": pose.copy(),
                "pose_fully_defined": False,
            },
        )
        sn2 = state2.get_surface_normal()
        result = self.sm._update_2d_position_and_displacement(state2, None, sn2)
        nptest.assert_allclose(self.sm._previous_2d_location, [1.0, 2.0], atol=1e-10)
        nptest.assert_allclose(result.location, [1.0, 2.0, 0.0], atol=1e-10)

    def test_z_always_zero(self):
        self.sm._previous_3d_location = np.array([0.0, 0.0, 0.0])
        self.sm._previous_2d_location = np.array([0.0, 0.0])
        self.sm._tangent_frame = self._make_mock_tangent_frame()

        pose = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        state = make_state(
            location=np.array([1.0, 2.0, 3.0]),
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": False,
            },
        )
        sn = state.get_surface_normal()
        result = self.sm._update_2d_position_and_displacement(state, None, sn)
        self.assertEqual(result.location[2], 0.0)


if __name__ == "__main__":
    unittest.main()
