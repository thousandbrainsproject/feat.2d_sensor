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
from tbp.monty.frameworks.models.sensor_modules import (
    DefaultMessageNoise,
    FeatureChangeFilter,
    NoMessageNoise,
    PassthroughStateFilter,
)
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

    def test_no_noise_when_params_none(self):
        sm = make_module(noise_params=None)
        self.assertIsInstance(sm._message_noise, NoMessageNoise)

    def test_default_noise_when_params_given(self):
        sm = make_module(noise_params={"hsv": 0.1})
        self.assertIsInstance(sm._message_noise, DefaultMessageNoise)

    def test_passthrough_filter_when_no_thresholds(self):
        sm = make_module(delta_thresholds=None)
        self.assertIsInstance(sm._state_filter, PassthroughStateFilter)

    def test_feature_change_filter_when_thresholds_given(self):
        sm = make_module(delta_thresholds={"hsv": 0.1})
        self.assertIsInstance(sm._state_filter, FeatureChangeFilter)

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
        self.assertTrue(any("edge_strength" in msg or "coherence" in msg
                            for msg in cm.output))

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
# TestPreEpisode
# ---------------------------------------------------------------------------


class TestPreEpisode(unittest.TestCase):

    def setUp(self):
        self.sm = make_module()
        # Mutate internal state so we can verify reset.
        self.sm._previous_3d_location = np.array([1.0, 2.0, 3.0])
        self.sm._tangent_frame = MagicMock()
        self.sm._previous_2d_location = np.array([5.0, 6.0])
        self.sm.processed_obs = [{"x": 1}]
        self.sm.states = [MagicMock()]
        self.sm.is_exploring = True
        self.sm._snapshot_telemetry = MagicMock()
        self.sm._state_filter = MagicMock()

        self.sm.pre_episode()

    def test_resets_telemetry(self):
        self.sm._snapshot_telemetry.reset.assert_called_once()

    def test_resets_state_filter(self):
        self.sm._state_filter.reset.assert_called_once()

    def test_clears_processed_obs_and_states(self):
        self.assertEqual(self.sm.processed_obs, [])
        self.assertEqual(self.sm.states, [])

    def test_resets_exploring_flag(self):
        self.assertFalse(self.sm.is_exploring)

    def test_resets_spatial_state(self):
        self.assertIsNone(self.sm._previous_3d_location)
        self.assertIsNone(self.sm._tangent_frame)
        nptest.assert_array_equal(self.sm._previous_2d_location, np.zeros(2))


# ---------------------------------------------------------------------------
# TestUpdateState
# ---------------------------------------------------------------------------


class TestUpdateState(unittest.TestCase):

    def _make_agent(self, agent_pos, agent_rot, sensor_pos, sensor_rot):
        sensor_state = SensorState(position=sensor_pos, rotation=sensor_rot)
        return AgentState(
            sensors={SensorID("test_sm"): sensor_state},
            position=agent_pos,
            rotation=agent_rot,
        )

    def test_identity_rotation(self):
        sm = make_module()
        identity = qt.quaternion(1, 0, 0, 0)
        agent = self._make_agent(
            np.array([1.0, 2.0, 3.0]), identity,
            np.array([0.1, 0.2, 0.3]), identity,
        )
        sm.update_state(agent)
        nptest.assert_allclose(sm.state.position, [1.1, 2.2, 3.3], atol=1e-10)

    def test_agent_rotation_applied_to_sensor_offset(self):
        sm = make_module()
        # 90-degree rotation around Z axis
        rot_z_90 = qt.from_euler_angles(0, 0, np.pi / 2)
        identity = qt.quaternion(1, 0, 0, 0)
        agent = self._make_agent(
            np.zeros(3), rot_z_90,
            np.array([1.0, 0.0, 0.0]), identity,
        )
        sm.update_state(agent)
        # [1,0,0] rotated 90 degrees around Z -> [0,1,0]
        nptest.assert_allclose(sm.state.position, [0, 1, 0], atol=1e-7)

    def test_rotation_composition(self):
        sm = make_module()
        r1 = qt.from_euler_angles(0.1, 0.2, 0.3)
        r2 = qt.from_euler_angles(0.4, 0.5, 0.6)
        agent = self._make_agent(np.zeros(3), r1, np.zeros(3), r2)
        sm.update_state(agent)
        expected = r1 * r2
        self.assertTrue(np.isclose(
            qt.as_float_array(sm.state.rotation),
            qt.as_float_array(expected),
            atol=1e-10,
        ).all())


# ---------------------------------------------------------------------------
# TestStateDict
# ---------------------------------------------------------------------------


class TestStateDict(unittest.TestCase):

    def test_merges_telemetry_with_processed_obs(self):
        sm = make_module()
        sm._snapshot_telemetry = MagicMock()
        sm._snapshot_telemetry.state_dict.return_value = {"raw": []}
        sm.processed_obs = [{"a": 1}]

        result = sm.state_dict()
        self.assertEqual(result, {"raw": [], "processed_observations": [{"a": 1}]})


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

    def test_zero_normal_returns_unchanged(self):
        state = self._base_state(
            pose_vectors=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                                  dtype=float)
        )
        result = self.sm._extract_2d_edge(state, self.rgba, self.world_camera)
        self.assertIs(result, state)

    @patch(f"{MODULE_PATH}.is_geometric_edge", return_value=True)
    @patch(f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
           return_value=(0.5, 0.8, 1.0))
    def test_geometric_edge_filtered_out(self, _mock_compute, _mock_geo):
        state = self._base_state()
        result = self.sm._extract_2d_edge(
            state, self.rgba, self.world_camera, depth_image=self.depth
        )
        self.assertIs(result, state)

    @patch(f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
           return_value=(0.01, 0.9, 0.5))
    def test_below_strength_threshold(self, _mock_compute):
        state = self._base_state()
        result = self.sm._extract_2d_edge(state, self.rgba, self.world_camera)
        self.assertFalse(state.morphological_features["pose_fully_defined"])
        self.assertIs(result, state)

    @patch(f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
           return_value=(0.5, 0.1, 0.5))
    def test_below_coherence_threshold(self, _mock_compute):
        state = self._base_state()
        result = self.sm._extract_2d_edge(state, self.rgba, self.world_camera)
        self.assertFalse(state.morphological_features["pose_fully_defined"])
        self.assertIs(result, state)

    @patch(f"{MODULE_PATH}.edge_angle_to_3d_tangent",
           return_value=np.array([0.0, 1.0, 0.0]))
    @patch(f"{MODULE_PATH}.is_geometric_edge", return_value=False)
    @patch(f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
           return_value=(0.5, 0.8, 0.7))
    def test_successful_edge_updates_pose(self, _mc, _mg, _mt):
        state = self._base_state()
        result = self.sm._extract_2d_edge(
            state, self.rgba, self.world_camera, depth_image=self.depth
        )
        self.assertTrue(result.morphological_features["pose_fully_defined"])
        pose = result.morphological_features["pose_vectors"]
        # Row 0: surface normal [1,0,0] (eye(3) row 0, normalized)
        nptest.assert_allclose(pose[0], [1, 0, 0], atol=1e-7)
        # Row 1: edge tangent (normalized [0,1,0])
        nptest.assert_allclose(pose[1], [0, 1, 0], atol=1e-7)
        # Row 2: cross(normal, tangent) = cross([1,0,0],[0,1,0]) = [0,0,1]
        nptest.assert_allclose(pose[2], [0, 0, 1], atol=1e-7)
        self.assertAlmostEqual(
            result.non_morphological_features["edge_strength"], 0.5
        )
        self.assertAlmostEqual(
            result.non_morphological_features["coherence"], 0.8
        )

    @patch(f"{MODULE_PATH}.edge_angle_to_3d_tangent",
           return_value=np.array([0.0, 1.0, 0.0]))
    @patch(f"{MODULE_PATH}.is_geometric_edge", return_value=False)
    @patch(f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
           return_value=(0.5, 0.8, 0.7))
    def test_strips_alpha_channel(self, mock_compute, _mg, _mt):
        state = self._base_state()
        rgba_4ch = np.zeros((32, 32, 4), dtype=np.uint8)
        self.sm._extract_2d_edge(
            state, rgba_4ch, self.world_camera, depth_image=self.depth
        )
        patch_arg = mock_compute.call_args[0][0]
        self.assertEqual(patch_arg.shape[2], 3)

    @patch(f"{MODULE_PATH}.is_geometric_edge")
    @patch(f"{MODULE_PATH}.edge_angle_to_3d_tangent",
           return_value=np.array([0.0, 1.0, 0.0]))
    @patch(f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
           return_value=(0.5, 0.8, 0.7))
    def test_no_depth_skips_geometric_check(self, _mc, _mt, mock_geo):
        state = self._base_state()
        self.sm._extract_2d_edge(state, self.rgba, self.world_camera,
                                 depth_image=None)
        mock_geo.assert_not_called()

    @patch(f"{MODULE_PATH}.edge_angle_to_3d_tangent",
           return_value=np.array([0.0, 0.0, 0.0]))
    @patch(f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
           return_value=(0.5, 0.8, 0.7))
    def test_zero_edge_tangent_returns_unchanged(self, _mc, _mt):
        state = self._base_state()
        result = self.sm._extract_2d_edge(state, self.rgba, self.world_camera)
        self.assertFalse(state.morphological_features["pose_fully_defined"])
        self.assertIs(result, state)

    @patch(f"{MODULE_PATH}.edge_angle_to_3d_tangent",
           return_value=np.array([0.0, 1.0, 0.0]))
    @patch(f"{MODULE_PATH}.compute_weighted_structure_tensor_edge_features",
           return_value=(0.5, 0.8, 0.7))
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
        result = self.sm._update_2d_position_and_displacement(state, None)
        nptest.assert_array_equal(
            result.displacement["displacement"], np.zeros(3)
        )

    def test_first_obs_initializes_from_world_xy(self):
        state = make_state(location=np.array([4.0, 5.0, 6.0]))
        self.sm._previous_3d_location = None

        result = self.sm._update_2d_position_and_displacement(state, None)
        nptest.assert_array_equal(result.location, [4.0, 5.0, 0.0])
        nptest.assert_array_equal(self.sm._previous_2d_location, [4.0, 5.0])
        self.assertIsNotNone(self.sm._tangent_frame)

    def test_first_obs_creates_tangent_frame_from_normal(self):
        state = make_state(location=np.array([4.0, 5.0, 6.0]))
        self.sm._previous_3d_location = None

        self.sm._update_2d_position_and_displacement(state, None)
        # Surface normal is [1,0,0] from eye(3), so tangent frame should exist
        self.assertIsNotNone(self.sm._tangent_frame)

    def test_zero_tangent_displacement(self):
        loc = np.array([1.0, 2.0, 3.0])
        self.sm._previous_3d_location = loc.copy()
        self.sm._previous_2d_location = np.array([1.0, 2.0])
        self.sm._tangent_frame = self._make_mock_tangent_frame()

        state = make_state(location=loc.copy())
        result = self.sm._update_2d_position_and_displacement(state, None)
        nptest.assert_array_equal(
            result.displacement["displacement"], np.zeros(3)
        )
        nptest.assert_array_equal(result.location, [1.0, 2.0, 0.0])

    def test_displacement_without_curvature(self):
        self.sm._previous_3d_location = np.array([0.0, 0.0, 0.0])
        self.sm._previous_2d_location = np.array([0.0, 0.0])
        self.sm._tangent_frame = self._make_mock_tangent_frame()

        # Normal = [0,0,1] so tangent plane is xy and mock frame aligns
        pose = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        state = make_state(
            location=np.array([0.3, 0.4, 0.0]),
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": False,
            },
        )
        result = self.sm._update_2d_position_and_displacement(
            state, curvature_pose_vectors=None
        )
        # d_tan projects [0.3, 0.4, 0] onto plane perp to [0,0,1] = [0.3, 0.4, 0]
        # du = dot([0.3,0.4,0], [1,0,0]) = 0.3
        # dv = dot([0.3,0.4,0], [0,1,0]) = 0.4
        nptest.assert_allclose(
            result.displacement["displacement"], [0.3, 0.4, 0.0], atol=1e-10
        )
        nptest.assert_allclose(result.location, [0.3, 0.4, 0.0], atol=1e-10)

    @patch(f"{MODULE_PATH}.arc_length_corrected_displacement",
           return_value=(0.35, 0.45))
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
        result = self.sm._update_2d_position_and_displacement(
            state, curvature_pose_vectors=curvature_pv
        )
        mock_arc.assert_called_once()
        nptest.assert_allclose(
            result.displacement["displacement"], [0.35, 0.45, 0.0], atol=1e-10
        )

    def test_displacement_accumulates(self):
        self.sm._previous_3d_location = np.array([0.0, 0.0, 0.0])
        self.sm._previous_2d_location = np.array([0.0, 0.0])
        self.sm._tangent_frame = self._make_mock_tangent_frame()

        # Normal = [0,0,1] so tangent plane is xy
        pose = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        state1 = make_state(
            location=np.array([1.0, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": pose.copy(),
                "pose_fully_defined": False,
            },
        )
        self.sm._update_2d_position_and_displacement(state1, None)

        state2 = make_state(
            location=np.array([1.0, 2.0, 0.0]),
            morphological_features={
                "pose_vectors": pose.copy(),
                "pose_fully_defined": False,
            },
        )
        result = self.sm._update_2d_position_and_displacement(state2, None)
        nptest.assert_allclose(
            self.sm._previous_2d_location, [1.0, 2.0], atol=1e-10
        )
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
        result = self.sm._update_2d_position_and_displacement(state, None)
        self.assertEqual(result.location[2], 0.0)


# ---------------------------------------------------------------------------
# TestStep
# ---------------------------------------------------------------------------


class TestStep(unittest.TestCase):

    def setUp(self):
        self.sm = make_module()
        self.sm.is_exploring = False
        self.sm.state = SensorState(
            position=np.zeros(3),
            rotation=qt.quaternion(1, 0, 0, 0),
        )
        self.ctx = RuntimeContext(rng=np.random.RandomState(42))
        self.data = {
            "rgba": np.zeros((64, 64, 4), dtype=np.uint8),
            "world_camera": np.eye(4),
            "depth": np.ones((64, 64), dtype=np.float32),
        }

    def _make_on_object_state(self):
        return make_state(
            use_state=False,
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": False,
                "on_object": True,
            },
        )

    def _make_off_object_state(self):
        return make_state(
            use_state=False,
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": False,
                "on_object": False,
            },
        )

    def test_pipeline_call_order(self):
        state_after_process = self._make_on_object_state()
        state_after_process.use_state = True
        state_after_edge = self._make_on_object_state()
        state_after_edge.use_state = True
        state_after_noise = self._make_on_object_state()
        state_after_noise.use_state = True
        state_after_2d = self._make_on_object_state()
        state_after_filter = self._make_on_object_state()

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state_after_process
        self.sm._extract_2d_edge = MagicMock(return_value=state_after_edge)
        self.sm._message_noise = MagicMock(return_value=state_after_noise)
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state_after_2d
        )
        self.sm._state_filter = MagicMock(return_value=state_after_filter)

        result = self.sm.step(self.ctx, self.data)

        self.sm._observation_processor.process.assert_called_once_with(self.data)
        self.sm._extract_2d_edge.assert_called_once()
        self.sm._message_noise.assert_called_once()
        self.sm._update_2d_position_and_displacement.assert_called_once()
        self.sm._state_filter.assert_called_once_with(state_after_2d)
        self.assertIs(result, state_after_filter)

    def test_skips_edge_when_not_on_object(self):
        state = self._make_off_object_state()
        state.use_state = True

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state
        self.sm._extract_2d_edge = MagicMock()
        self.sm._message_noise = MagicMock(return_value=state)
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state
        )
        self.sm._state_filter = MagicMock(return_value=state)

        self.sm.step(self.ctx, self.data)
        self.sm._extract_2d_edge.assert_not_called()

    def test_skips_edge_when_use_state_false(self):
        state = self._make_on_object_state()
        state.use_state = False

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state
        self.sm._extract_2d_edge = MagicMock()
        self.sm._message_noise = MagicMock(return_value=state)
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state
        )
        self.sm._state_filter = MagicMock(return_value=state)

        self.sm.step(self.ctx, self.data)
        self.sm._extract_2d_edge.assert_not_called()

    def test_skips_noise_when_use_state_false(self):
        state = self._make_on_object_state()
        state.use_state = False

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state
        self.sm._message_noise = MagicMock()
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state
        )
        self.sm._state_filter = MagicMock(return_value=state)

        self.sm.step(self.ctx, self.data)
        self.sm._message_noise.assert_not_called()

    def test_motor_only_sets_use_state_false(self):
        state = self._make_on_object_state()
        state.use_state = True

        state_after_edge = self._make_on_object_state()
        state_after_edge.use_state = True

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state
        self.sm._extract_2d_edge = MagicMock(return_value=state_after_edge)
        self.sm._message_noise = MagicMock(return_value=state_after_edge)
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state_after_edge
        )
        self.sm._state_filter = MagicMock(return_value=state_after_edge)

        self.sm.step(self.ctx, self.data, motor_only_step=True)
        # motor_only_step sets use_state to False on the state after noise
        self.assertFalse(state_after_edge.use_state)

    def test_appends_when_not_exploring(self):
        state = self._make_on_object_state()
        self.sm.is_exploring = False

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state
        self.sm._message_noise = MagicMock(return_value=state)
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state
        )
        self.sm._state_filter = MagicMock(return_value=state)

        self.sm.step(self.ctx, self.data)
        self.assertEqual(len(self.sm.processed_obs), 1)
        self.assertEqual(len(self.sm.states), 1)

    def test_no_append_when_exploring(self):
        state = self._make_on_object_state()
        self.sm.is_exploring = True

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state
        self.sm._message_noise = MagicMock(return_value=state)
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state
        )
        self.sm._state_filter = MagicMock(return_value=state)

        self.sm.step(self.ctx, self.data)
        self.assertEqual(len(self.sm.processed_obs), 0)
        self.assertEqual(len(self.sm.states), 0)

    def test_curvature_pose_vectors_copied_before_edge(self):
        original_pv = np.eye(3)
        state = self._make_on_object_state()
        state.use_state = True
        state.morphological_features["pose_vectors"] = original_pv

        def mutate_edge(s, *args, **kwargs):
            s.morphological_features["pose_vectors"] = np.zeros((3, 3))
            return s

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state
        self.sm._extract_2d_edge = MagicMock(side_effect=mutate_edge)
        self.sm._message_noise = MagicMock(return_value=state)
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state
        )
        self.sm._state_filter = MagicMock(return_value=state)

        self.sm.step(self.ctx, self.data)

        # The curvature_pose_vectors passed to _update_2d should be the original
        call_args = self.sm._update_2d_position_and_displacement.call_args
        passed_pv = call_args[0][1]
        nptest.assert_array_equal(passed_pv, original_pv)

    def test_pose_fully_defined_reset_before_edge_detection(self):
        state = self._make_on_object_state()
        state.use_state = True
        state.morphological_features["pose_fully_defined"] = True

        def check_flag(s, *args, **kwargs):
            # By the time _extract_2d_edge is called, the flag should be False
            self.assertFalse(s.morphological_features["pose_fully_defined"])
            return s

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state
        self.sm._extract_2d_edge = MagicMock(side_effect=check_flag)
        self.sm._message_noise = MagicMock(return_value=state)
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state
        )
        self.sm._state_filter = MagicMock(return_value=state)

        self.sm.step(self.ctx, self.data)

    def test_returns_filter_output(self):
        state = self._make_on_object_state()
        filter_output = self._make_on_object_state()

        self.sm._observation_processor = MagicMock()
        self.sm._observation_processor.process.return_value = state
        self.sm._message_noise = MagicMock(return_value=state)
        self.sm._update_2d_position_and_displacement = MagicMock(
            return_value=state
        )
        self.sm._state_filter = MagicMock(return_value=filter_output)

        result = self.sm.step(self.ctx, self.data)
        self.assertIs(result, filter_output)


# ---------------------------------------------------------------------------
# TestStepSnapshot (parameterized)
# ---------------------------------------------------------------------------


@parameterized_class(
    ("save_raw_obs", "is_exploring", "should_snapshot"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
class TestStepSnapshot(unittest.TestCase):

    save_raw_obs: bool
    is_exploring: bool
    should_snapshot: bool

    def test_step_snapshots_raw_observation_as_needed(self):
        sm = make_module(save_raw_obs=self.save_raw_obs)
        sm.is_exploring = self.is_exploring
        sm.state = SensorState(
            position=np.zeros(3),
            rotation=qt.quaternion(1, 0, 0, 0),
        )
        sm._snapshot_telemetry = MagicMock()

        state = make_state(
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": False,
                "on_object": True,
            },
        )
        sm._observation_processor = MagicMock()
        sm._observation_processor.process.return_value = state
        sm._message_noise = MagicMock(return_value=state)
        sm._update_2d_position_and_displacement = MagicMock(return_value=state)
        sm._state_filter = MagicMock(return_value=state)

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        data = {
            "rgba": np.zeros((64, 64, 4), dtype=np.uint8),
            "world_camera": np.eye(4),
            "depth": np.ones((64, 64), dtype=np.float32),
        }
        sm.step(ctx, data)

        if self.should_snapshot:
            sm._snapshot_telemetry.raw_observation.assert_called_once()
        else:
            sm._snapshot_telemetry.raw_observation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
