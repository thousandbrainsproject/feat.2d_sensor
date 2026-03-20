# Copyright 2025-2026  Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import quaternion as qt

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    SensorState,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DefaultMessageNoise,
    FeatureChangeFilter,
    MessageNoise,
    NoMessageNoise,
    ObservationProcessor,
    PassthroughStateFilter,
    SnapshotTelemetry,
    StateFilter,
)
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.edge_detection import (
    EdgeDetectionConfig,
    compute_weighted_structure_tensor_edge_features,
    edge_angle_to_3d_tangent,
    is_geometric_edge,
)
from tbp.monty.frameworks.utils.sensor_processing import (
    arc_length_corrected_displacement,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    TangentFrame,
    normalize,
    project_onto_tangent_plane,
)

logger = logging.getLogger(__name__)


class TwoDSensorModule(SensorModule):
    """Sensor Module that extracts edges and other features at 2D locations.

    Extends the base sensor module to detect edges on an object's surface (e.g.
    edges of a logo on a cup) and build a corresponding 2D model of the 3D
    object. The 2D model is constructed by unrolling the 3D surface into a
    local tangent plane, which requires extracting surface normals and principal
    curvatures at each observation point to perform the projection correctly.

    The 2D position is initialized to the world x,y coordinates. A 2D model is
    built by accumulating tangent plane displacements.
    """

    def __init__(
        self,
        sensor_module_id: str,
        features: list[str],
        save_raw_obs: bool = False,
        pc1_is_pc2_threshold: int = 10,
        is_surface_sm: bool = False,
        edge_detection_config: EdgeDetectionConfig | None = None,
        noise_params: dict[str, Any] | None = None,
        delta_thresholds: dict[str, Any] | None = None,
    ):
        """Initialize 2D Sensor Module.

        Args:
            sensor_module_id: Name of sensor module.
            features: Which features to extract.
            save_raw_obs: Whether to save raw sensory input for logging.
            pc1_is_pc2_threshold: Maximum difference between pc1 and pc2 to be
                classified as being roughly the same (ignore curvature directions).
                Defaults to 10.
            is_surface_sm: Surface SMs do not require that the central pixel is
                "on object" in order to process the observation (i.e., extract
                features). Defaults to False.
            edge_detection_config: Configuration for structure-tensor edge
                detection. If None, uses EdgeDetectionConfig defaults.
            noise_params: Dictionary of noise amount for each feature.
            delta_thresholds: If given, a FeatureChangeFilter will be used to
                check whether the current state's features are significantly different
                from the previous with tolerances set according to `delta_thresholds`.
                Defaults to None.
        """
        self._observation_processor = ObservationProcessor(
            features=features,
            sensor_module_id=sensor_module_id,
            pc1_is_pc2_threshold=pc1_is_pc2_threshold,
            is_surface_sm=is_surface_sm,
        )
        if noise_params:
            self._message_noise: MessageNoise = DefaultMessageNoise(
                noise_params=noise_params
            )
        else:
            self._message_noise = NoMessageNoise()
        if delta_thresholds:
            self._state_filter: StateFilter = FeatureChangeFilter(
                delta_thresholds=delta_thresholds
            )
        else:
            self._state_filter = PassthroughStateFilter()
        self._snapshot_telemetry = SnapshotTelemetry()

        self.features = features
        self.processed_obs = []
        self.states = []
        self.sensor_module_id = sensor_module_id
        self.save_raw_obs = save_raw_obs
        self.edge_detection_config = edge_detection_config or EdgeDetectionConfig()

        edge_features = {"edge_strength", "coherence"}
        missing = edge_features - set(features)
        if missing:
            logger.warning(
                f"TwoDSensorModule '{sensor_module_id}' does not include {missing} "
                f"in its features list. Edge features will not appear in "
                f"observations."
            )

        self._previous_3d_location: np.ndarray | None = None
        self._tangent_frame: TangentFrame | None = None
        self._previous_2d_location: np.ndarray = np.zeros(2)

    def pre_episode(self) -> None:
        self._snapshot_telemetry.reset()
        self._state_filter.reset()
        self.is_exploring = False
        self.processed_obs = []
        self.states = []
        self._previous_3d_location = None
        self._tangent_frame = None
        self._previous_2d_location = np.zeros(2)

    def update_state(self, agent: AgentState):
        """Update information about the sensor's location and rotation."""
        sensor = agent.sensors[SensorID(self.sensor_module_id)]
        self.state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )

    def state_dict(self):
        state_dict = self._snapshot_telemetry.state_dict()
        state_dict.update(processed_observations=self.processed_obs)
        return state_dict

    def step(self, ctx: RuntimeContext, data, motor_only_step: bool = False) -> State:
        """Turn raw observations into dict of features at location.

        Args:
            ctx: The runtime context.
            data: Raw observations.

        Returns:
            State with features and morphological features. Noise may be added.
            use_state flag may be set.
        """
        if self.save_raw_obs and not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                data, self.state.rotation, self.state.position
            )

        # TODO: Consider a FeatureRegistry to replace the two-step
        # process() + _extract_2d_edge() pipeline.

        observed_state = self._observation_processor.process(data)

        curvature_pose_vectors = observed_state.morphological_features.get(
            "pose_vectors"
        )
        if curvature_pose_vectors is not None:
            curvature_pose_vectors = curvature_pose_vectors.copy()

        if observed_state.use_state and observed_state.get_on_object():
            observed_state = self._extract_2d_edge(
                observed_state,
                data["rgba"],
                data["world_camera"],
                depth_image=data.get("depth"),
            )

        if observed_state.use_state:
            observed_state = self._message_noise(observed_state, rng=ctx.rng)

        if motor_only_step:
            # Set interesting-features flag to False, as should not be passed to
            # LM, even in e.g. pre-training experiments that might otherwise do so
            observed_state.use_state = False

        observed_state = self._update_2d_position_and_displacement(
            observed_state, curvature_pose_vectors
        )

        observed_state = self._state_filter(observed_state)

        if not self.is_exploring:
            self.processed_obs.append(observed_state.__dict__)
            self.states.append(self.state)

        return observed_state

    def _extract_2d_edge(
        self,
        state: State,
        rgba_image: np.ndarray,
        world_camera: np.ndarray,
        depth_image: np.ndarray | None = None,
    ) -> State:
        """Extract 2D edge-based pose if edge is detected.

        This method attempts to create a fully-defined pose (normal + 2 tangents)
        using edge detection, replacing the standard curvature-based tangents.

        Args:
            state: State with standard features from ObservationProcessor
            rgba_image: RGBA image patch
            world_camera: World to camera transformation matrix
            depth_image: Optional depth image patch for filtering geometric edges.
                If provided, edges at depth discontinuities (object boundaries,
                surface creases) are filtered out, keeping only texture edges.

        Returns:
            State with edge-based pose vectors if edge detected,
            otherwise returns the original state unchanged.
        """
        if "pose_vectors" not in state.morphological_features:
            return state

        try:
            surface_normal = normalize(state.morphological_features["pose_vectors"][0])
        except ValueError:
            return state

        if rgba_image.shape[2] == 4:
            patch = rgba_image[:, :, :3]
        else:
            patch = rgba_image

        edge_strength, coherence, edge_orientation = (
            compute_weighted_structure_tensor_edge_features(
                patch,
                edge_detection_config=self.edge_detection_config,
            )
        )

        if (
            edge_strength > 0
            and depth_image is not None
            and is_geometric_edge(
                depth_image,
                edge_orientation,
                self.edge_detection_config.depth_edge_threshold,
            )
        ):
            return state

        strength_threshold = self.edge_detection_config.edge_threshold
        coherence_threshold = self.edge_detection_config.coherence_threshold
        has_edge = (edge_strength > strength_threshold) and (
            coherence > coherence_threshold
        )

        if not has_edge:
            return state

        edge_tangent = edge_angle_to_3d_tangent(
            edge_orientation, surface_normal, world_camera
        )
        try:
            edge_tangent = normalize(edge_tangent)
            edge_perp = normalize(np.cross(surface_normal, edge_tangent))
        except ValueError:
            return state

        state.morphological_features["pose_vectors"] = np.vstack(
            [
                surface_normal,
                edge_tangent,
                edge_perp,
            ]
        )
        state.morphological_features["pose_fully_defined"] = True

        if "edge_strength" in self.features:
            state.non_morphological_features["edge_strength"] = edge_strength
        if "coherence" in self.features:
            state.non_morphological_features["coherence"] = coherence

        return state

    def _update_2d_position_and_displacement(
        self,
        observed_state: State,
        curvature_pose_vectors: np.ndarray | None,
    ) -> State:
        """Project the 3D step onto the tangent plane to get a 2D displacement.

        Args:
            observed_state: State to update with 2D displacement and position.
            curvature_pose_vectors: Curvature-based pose vectors used for
                arc-length correction. If None, chord length is used as-is.

        Returns:
            The updated state with 2D position and displacement.
        """
        if not observed_state.get_on_object():
            observed_state.set_displacement(np.zeros(3))
            return observed_state

        current_location = observed_state.location.copy()
        surface_normal = observed_state.get_surface_normal()

        if self._previous_3d_location is None or surface_normal is None:
            if surface_normal is not None:
                self._tangent_frame = TangentFrame(surface_normal)
            self._previous_3d_location = current_location
            # Initialize 2D position to world x,y.
            self._previous_2d_location = current_location[:2].copy()
            observed_state.location = np.array(
                [current_location[0], current_location[1], 0.0]
            )
            return observed_state
        displacement_3d = current_location - self._previous_3d_location
        d_tan = project_onto_tangent_plane(displacement_3d, surface_normal)

        if np.linalg.norm(d_tan) < 1e-12:
            self._previous_3d_location = current_location.copy()
            observed_state.set_displacement(np.zeros(3))
            observed_state.location = np.array(
                [
                    self._previous_2d_location[0],
                    self._previous_2d_location[1],
                    0.0,
                ]
            )
            return observed_state

        self._tangent_frame.transport(surface_normal)

        du = float(np.dot(d_tan, self._tangent_frame.basis_u))
        dv = float(np.dot(d_tan, self._tangent_frame.basis_v))

        principal_curvatures = observed_state.morphological_features.get(
            "principal_curvatures"
        )
        if principal_curvatures is not None and curvature_pose_vectors is not None:
            du, dv = arc_length_corrected_displacement(
                du,
                dv,
                self._tangent_frame.basis_u,
                self._tangent_frame.basis_v,
                principal_curvatures,
                curvature_pose_vectors,
            )

        self._previous_2d_location += [du, dv]
        observed_state.set_displacement(np.array([du, dv, 0.0]))
        observed_state.location = np.array(
            [
                self._previous_2d_location[0],
                self._previous_2d_location[1],
                0.0,
            ]
        )
        self._previous_3d_location = current_location.copy()
        return observed_state
