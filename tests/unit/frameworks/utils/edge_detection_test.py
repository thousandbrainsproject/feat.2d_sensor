# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
import numpy.testing as npt
from hypothesis import assume, example, given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.edge_detection import (
    EdgeDetectionConfig,
    StructureTensor,
    compute_weighted_structure_tensor_edge_features,
    edge_angle_to_2d_pose,
    gradient_to_tangent_angle,
    is_geometric_edge,
)
from tbp.monty.math import DEFAULT_TOLERANCE

angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi)
a_scalar = st.floats(min_value=DEFAULT_TOLERANCE, max_value=100.0)


@st.composite
def structure_tensors(draw, max_value=100.0, allow_zero_matrix=True):
    """Generate valid PSD structure tensors.

    Args:
        max_value: Maximum value for Jxx, Jyy.
        allow_zero_matrix: If True, allows zero/near-zero tensors.

    Returns:
        PSD StructureTensor satisfying Jxy^2 <= Jxx * Jyy.
    """
    min_val = 0.0 if allow_zero_matrix else DEFAULT_TOLERANCE
    Jxx = draw(st.floats(min_value=min_val, max_value=max_value).filter(lambda x: abs(x) > DEFAULT_TOLERANCE))
    Jyy = draw(st.floats(min_value=min_val, max_value=max_value).filter(lambda x: abs(x) > DEFAULT_TOLERANCE))
    # Cauchy-Schwarz bound: |Jxy| <= sqrt(Jxx * Jyy) guarantees det(J) >= 0
    max_Jxy = np.sqrt(Jxx * Jyy)
    Jxy = draw(st.floats(min_value=-max_Jxy, max_value=max_Jxy).filter(lambda x: abs(x) > DEFAULT_TOLERANCE))
    return StructureTensor(Jxx=Jxx, Jyy=Jyy, Jxy=Jxy)


@st.composite
def rotation_3x3(draw):
    """Draw a uniformly random SO(3) rotation matrix.

    Returns:
        rot: 3x3 rotation matrix.
    """
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    return Rotation.random(random_state=rng).as_matrix()


@st.composite
def camera_4x4(draw):
    """Draw a random 4x4 world-to-camera matrix with arbitrary translation.

    Returns:
        cam: 4x4 world-to-camera matrix with arbitrary translation.
    """
    R = draw(rotation_3x3())  # noqa: N806
    tx, ty, tz = (draw(st.floats(min_value=-100.0, max_value=100.0)) for _ in range(3))
    cam = np.eye(4)
    cam[:3, :3] = R
    cam[:3, 3] = [tx, ty, tz]
    return cam


PATCH_SIZE = 64

positive_thresholds = st.floats(min_value=1e-8, max_value=10.0)

STEP_EDGE_IMAGE = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
STEP_EDGE_IMAGE[:, : PATCH_SIZE // 2] = 1.0


@st.composite
def flat_depth_image(draw):
    """Generate a constant-depth patch with a random depth value.

    Returns:
        Float32 array of shape (PATCH_SIZE, PATCH_SIZE) filled with a constant depth.
    """
    depth = draw(st.floats(min_value=0.01, max_value=100.0))
    return np.full((PATCH_SIZE, PATCH_SIZE), depth, dtype=np.float32)


class GradientToTangentAngleTest(unittest.TestCase):
    @given(gradient_angle=angles)
    def test_result_in_range(self, gradient_angle):
        result = gradient_to_tangent_angle(gradient_angle)
        assert 0.0 <= result < 2 * np.pi

    @given(gradient_angle=angles)
    def test_perpendicularity(self, gradient_angle):
        result = gradient_to_tangent_angle(gradient_angle)
        remainder = (result - gradient_angle) % np.pi
        npt.assert_allclose(remainder, np.pi / 2, atol=DEFAULT_TOLERANCE)


class IsGeometricEdgeTest(unittest.TestCase):
    @given(patch=flat_depth_image(), theta=angles, threshold=positive_thresholds)
    @example(
        patch=np.full((PATCH_SIZE, PATCH_SIZE), 1.0, dtype=np.float32),
        theta=0.0,
        threshold=0.01,
    )
    def test_flat_depth_returns_false(self, patch, theta, threshold):
        self.assertFalse(is_geometric_edge(patch, theta, threshold))

    @given(theta=angles)
    @example(theta=np.pi / 2)
    @example(theta=0.0)
    def test_theta_has_period_pi(self, theta):
        result_base = is_geometric_edge(STEP_EDGE_IMAGE, theta)
        result_shifted = is_geometric_edge(STEP_EDGE_IMAGE, theta + np.pi)
        self.assertEqual(result_base, result_shifted)

    @given(theta=angles, t_low=positive_thresholds, t_high=positive_thresholds)
    @example(theta=np.pi / 2, t_low=1e-6, t_high=1.0)
    def test_lower_threshold_preserves_detection(self, theta, t_low, t_high):
        assume(t_low < t_high)
        if is_geometric_edge(STEP_EDGE_IMAGE, theta, t_high):
            self.assertTrue(is_geometric_edge(STEP_EDGE_IMAGE, theta, t_low))


class EdgeAngleTo2dPoseTest(unittest.TestCase):
    def test_identity_camera_theta_zero(self):
        """Canonical reference: identity camera, theta=0 aligns with world x-axis."""
        pose = edge_angle_to_2d_pose(theta=0.0, world_camera=np.eye(4))
        npt.assert_allclose(pose[0], [0, 0, 1])
        npt.assert_allclose(pose[1], [1, 0, 0])
        npt.assert_allclose(pose[2], [0, 1, 0])

    def test_tilted_camera_90_yaw(self):
        """Camera yawed 90 degrees CCW shifts world_theta by pi/2."""
        # R = Rz(pi/2) so R.T @ [1,0,0] = [0, 1, 0], ref_angle = pi/2.
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)  # noqa: N806
        cam = np.eye(4)
        cam[:3, :3] = R
        pose = edge_angle_to_2d_pose(theta=0.0, world_camera=cam)
        npt.assert_allclose(pose[1], [0, 1, 0], atol=DEFAULT_TOLERANCE)
        npt.assert_allclose(pose[2], [-1, 0, 0], atol=DEFAULT_TOLERANCE)

    @given(theta=angles, cam=camera_4x4())
    @example(theta=0.0, cam=np.eye(4))
    def test_normal_is_001(self, theta, cam):
        """Row 0 is always the world z-axis, regardless of theta or camera."""
        pose = edge_angle_to_2d_pose(theta, cam)
        npt.assert_allclose(pose[0], [0.0, 0.0, 1.0])

    @given(theta=angles, cam=camera_4x4())
    def test_tangent_and_perp_lie_in_xy_plane(self, theta, cam):
        """Rows 1 and 2 always have zero z-component."""
        pose = edge_angle_to_2d_pose(theta, cam)
        npt.assert_allclose(pose[1][2], 0.0, atol=DEFAULT_TOLERANCE)
        npt.assert_allclose(pose[2][2], 0.0, atol=DEFAULT_TOLERANCE)

    @given(theta=angles, cam=camera_4x4())
    @example(theta=0.0, cam=np.eye(4))
    def test_orthonormality(self, theta, cam):
        """Result is always an orthonormal frame (unit rows, mutually orthogonal)."""
        pose = edge_angle_to_2d_pose(theta, cam)
        for i in range(3):
            npt.assert_allclose(np.linalg.norm(pose[i]), 1.0, atol=DEFAULT_TOLERANCE)
        for i in range(3):
            for j in range(i + 1, 3):
                npt.assert_allclose(
                    np.dot(pose[i], pose[j]), 0.0, atol=DEFAULT_TOLERANCE
                )

    @given(theta=angles, R=rotation_3x3())
    def test_translation_invariance(self, theta, R):  # noqa: N803
        """Translation component of the camera matrix does not affect the result."""
        cam_no_t = np.eye(4)
        cam_no_t[:3, :3] = R
        cam_with_t = cam_no_t.copy()
        cam_with_t[:3, 3] = [10.0, 20.0, 30.0]
        npt.assert_allclose(
            edge_angle_to_2d_pose(theta, cam_with_t),
            edge_angle_to_2d_pose(theta, cam_no_t),
        )

    @given(theta=angles, cam=camera_4x4())
    def test_theta_periodicity_2pi(self, theta, cam):
        """Shifting theta by 2*pi returns the identical pose."""
        tol = max(DEFAULT_TOLERANCE * abs(theta), DEFAULT_TOLERANCE)
        npt.assert_allclose(
            edge_angle_to_2d_pose(theta, cam),
            edge_angle_to_2d_pose(theta + 2 * np.pi, cam),
            atol=tol,
        )

    @given(theta=angles, cam=camera_4x4())
    def test_theta_shift_pi_negates_tangent_and_perp(self, theta, cam):
        """Shifting theta by pi negates the tangent and perp rows (normal unchanged)."""
        tol = max(DEFAULT_TOLERANCE * abs(theta), DEFAULT_TOLERANCE)
        pose = edge_angle_to_2d_pose(theta, cam)
        pose_shifted = edge_angle_to_2d_pose(theta + np.pi, cam)
        npt.assert_allclose(pose[0], pose_shifted[0], atol=tol)
        npt.assert_allclose(pose[1], -pose_shifted[1], atol=tol)
        npt.assert_allclose(pose[2], -pose_shifted[2], atol=tol)


class StructureTensorTest(unittest.TestCase):
    def test_eigenvalues_match_analytical(self):
        t = StructureTensor(Jxx=3.0, Jyy=1.0, Jxy=1.0)
        lambda_min, lambda_max = t.eigenvalues
        npt.assert_allclose(lambda_min, 2.0 - np.sqrt(2.0), atol=DEFAULT_TOLERANCE)
        npt.assert_allclose(lambda_max, 2.0 + np.sqrt(2.0), atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors())
    @example(t=StructureTensor(Jxx=0.0, Jyy=0.0, Jxy=0.0))
    def test_eigenvalues_ordered(self, t):
        lambda_min, lambda_max = t.eigenvalues
        assert lambda_min <= lambda_max

    @given(t=structure_tensors())
    @example(t=StructureTensor(Jxx=0.0, Jyy=9.0, Jxy=0.0))
    def test_edge_strength_nonnegative(self, t):
        assert t.edge_strength >= 0.0

    @given(t=structure_tensors())
    @example(t=StructureTensor(Jxx=4.0, Jyy=0.0, Jxy=0.0))
    def test_coherence_in_unit_interval(self, t):
        assert 0.0 <= t.coherence <= 1.0

    @given(t=structure_tensors())
    @example(t=StructureTensor(Jxx=4.0, Jyy=0.0, Jxy=0.0))
    def test_edge_orientation_range(self, t):
        assert 0.0 <= t.edge_orientation <= np.pi

    @given(t=structure_tensors())
    def test_eigenvalue_trace_equals_jxx_plus_jyy(self, t):
        lambda_min, lambda_max = t.eigenvalues
        npt.assert_allclose(lambda_min + lambda_max, t.Jxx + t.Jyy, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors())
    def test_eigenvalue_product_equals_determinant(self, t):
        lambda_min, lambda_max = t.eigenvalues
        npt.assert_allclose(lambda_min * lambda_max, t.Jxx * t.Jyy - t.Jxy**2, atol=DEFAULT_TOLERANCE)

    @given(k=a_scalar)
    def test_isotropic_coherence_is_zero(self, k):
        t = StructureTensor(Jxx=k, Jyy=k, Jxy=0.0)
        npt.assert_allclose(t.coherence, 0.0, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors(), k=a_scalar)
    def test_scaling_multiplies_edge_strength(self, t, k):
        scaled = StructureTensor(Jxx=k * t.Jxx, Jyy=k * t.Jyy, Jxy=k * t.Jxy)
        npt.assert_allclose(scaled.edge_strength, np.sqrt(k) * t.edge_strength, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors(), k=a_scalar)
    @example(t=StructureTensor(Jxx=4.0, Jyy=0.0, Jxy=0.0), k=2.0)
    @example(t=StructureTensor(Jxx=0.0, Jyy=9.0, Jxy=0.0), k=3.0)
    def test_scaling_preserves_gradient_theta(self, t, k):
        scaled = StructureTensor(Jxx=k * t.Jxx, Jyy=k * t.Jyy, Jxy=k * t.Jxy)
        npt.assert_allclose(scaled.gradient_theta, t.gradient_theta, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors())
    def test_edge_strength_equals_sqrt_lambda_max(self, t):
        _, lambda_max = t.eigenvalues
        npt.assert_allclose(t.edge_strength, np.sqrt(max(lambda_max, 0.0)), atol=1e-10)


class ComputeWeightedStructureTensorEdgeFeaturesTest(unittest.TestCase):
    @staticmethod
    def _make_rgb_patch(size, pattern) -> np.ndarray:
        """Generate synthetic RGB patches for testing.

        Args:
            size: Patch dimension (square).
            pattern: One of "uniform", "vertical_edge", "horizontal_edge",
                "diagonal_edge".

        Returns:
            uint8 RGB array of shape (size, size, 3).

        Raises:
            ValueError: If pattern is not recognized.
        """
        if pattern == "uniform":
            return np.full((size, size, 3), 128, dtype=np.uint8)
        if pattern == "vertical_edge":
            patch = np.zeros((size, size, 3), dtype=np.uint8)
            patch[:, size // 2 :] = 255
            return patch
        if pattern == "horizontal_edge":
            patch = np.zeros((size, size, 3), dtype=np.uint8)
            patch[size // 2 :, :] = 255
            return patch
        if pattern == "diagonal_edge":
            patch = np.zeros((size, size, 3), dtype=np.uint8)
            for r in range(size):
                patch[r, r:] = 255
            return patch
        raise ValueError(f"Unknown pattern: {pattern}")

    def test_uniform_patch_returns_zero_strength(self):
        patch = self._make_rgb_patch(32, "uniform")
        strength, _coherence, _theta = compute_weighted_structure_tensor_edge_features(
            patch
        )
        self.assertAlmostEqual(strength, 0.0)

    def test_vertical_edge_detected(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        strength, coherence, _ = compute_weighted_structure_tensor_edge_features(patch)
        self.assertGreater(strength, 0.0)
        self.assertGreater(coherence, 0.0)

    def test_vertical_edge_orientation(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        _, _, theta = compute_weighted_structure_tensor_edge_features(patch)
        # Vertical edge tangent should be near pi/2 or 3*pi/2
        angle_to_vertical = min(abs(theta - np.pi / 2), abs(theta - 3 * np.pi / 2))
        self.assertLess(angle_to_vertical, 0.3)

    def test_horizontal_edge_orientation(self):
        patch = self._make_rgb_patch(32, "horizontal_edge")
        _, _, theta = compute_weighted_structure_tensor_edge_features(patch)
        # Horizontal edge tangent should be near 0 or pi
        angle_to_horizontal = min(abs(theta), abs(theta - np.pi))
        self.assertLess(angle_to_horizontal, 0.3)

    def test_default_params_used_when_none(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        result = compute_weighted_structure_tensor_edge_features(
            patch, edge_detection_config=None
        )
        self.assertEqual(len(result), 3)

    def test_returns_three_floats(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        result = compute_weighted_structure_tensor_edge_features(patch)
        self.assertEqual(len(result), 3)
        for val in result:
            self.assertIsInstance(val, float)

    def test_center_offset_rejects_off_center_edge(self):
        # Edge at right boundary, not at center
        patch = np.full((32, 32, 3), 0, dtype=np.uint8)
        patch[:, 28:] = 255
        config = EdgeDetectionConfig(max_center_offset=1)
        strength, coherence, theta = compute_weighted_structure_tensor_edge_features(
            patch, config
        )
        self.assertAlmostEqual(strength, 0.0)
        self.assertAlmostEqual(coherence, 0.0)
        self.assertIsNone(theta)

    def test_coherence_in_zero_one_range(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        _, coherence, _ = compute_weighted_structure_tensor_edge_features(patch)
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)

    def test_tangent_theta_in_valid_range(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        _, _, theta = compute_weighted_structure_tensor_edge_features(patch)
        self.assertGreaterEqual(theta, 0.0)
        self.assertLess(theta, 2 * np.pi)
