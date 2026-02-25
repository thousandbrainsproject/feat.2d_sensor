# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import math
import unittest

import numpy as np

from tbp.monty.frameworks.utils.sensor_processing import (
    arc_length_corrected_displacement,
    compute_arc_from_tangent_projection,
    directional_curvature,
)


class ComputeArcFromTangentProjectionTest(unittest.TestCase):
    """Unit tests for the compute_arc_from_tangent_projection function."""

    def test_zero_curvature_returns_projection(self):
        result = compute_arc_from_tangent_projection(0.5, curvature=0.0)
        self.assertEqual(result, 0.5)

    def test_zero_projection_returns_zero(self):
        result = compute_arc_from_tangent_projection(0.0, curvature=5.0)
        self.assertEqual(result, 0.0)

    def test_both_zero_returns_zero(self):
        result = compute_arc_from_tangent_projection(0.0, curvature=0.0)
        self.assertEqual(result, 0.0)

    def test_small_kp_below_threshold_returns_projection(self):
        result = compute_arc_from_tangent_projection(0.01, curvature=0.01)
        self.assertEqual(result, 0.01)

    def test_kp_at_threshold_triggers_correction(self):
        # k=1.0, p=0.001 => kp = 0.001, which is NOT < 0.001, so correction fires
        result = compute_arc_from_tangent_projection(0.001, curvature=1.0)
        expected = math.asin(0.001) / 1.0
        self.assertAlmostEqual(result, expected)

    def test_known_correction(self):
        # k=1, p=0.5 => arcsin(0.5)/1 = pi/6
        result = compute_arc_from_tangent_projection(0.5, curvature=1.0)
        self.assertAlmostEqual(result, math.pi / 6)

    def test_negative_projection_yields_negative_arc(self):
        result = compute_arc_from_tangent_projection(-0.5, curvature=1.0)
        self.assertAlmostEqual(result, -math.pi / 6)

    def test_negative_curvature_preserves_projection_sign(self):
        result = compute_arc_from_tangent_projection(0.5, curvature=-1.0)
        self.assertAlmostEqual(result, math.pi / 6)

    def test_domain_guard_kp_equals_one(self):
        # kp = 1.0 => >= 1.0 guard triggers, returns projection unchanged
        result = compute_arc_from_tangent_projection(1.0, curvature=1.0)
        self.assertEqual(result, 1.0)

    def test_domain_guard_kp_above_one(self):
        result = compute_arc_from_tangent_projection(0.5, curvature=3.0)
        self.assertEqual(result, 0.5)

    def test_arc_longer_than_projection(self):
        # On a curved surface, arc length >= chord projection
        projection = 0.3
        result = compute_arc_from_tangent_projection(projection, curvature=2.0)
        self.assertGreaterEqual(abs(result), abs(projection))

    def test_sign_symmetry(self):
        pos = compute_arc_from_tangent_projection(0.4, curvature=1.5)
        neg = compute_arc_from_tangent_projection(-0.4, curvature=1.5)
        self.assertAlmostEqual(neg, -pos)


class DirectionalCurvatureTest(unittest.TestCase):
    """Unit tests for the directional_curvature function."""

    def setUp(self):
        self.dir1 = np.array([1.0, 0.0, 0.0])
        self.dir2 = np.array([0.0, 1.0, 0.0])

    def test_zero_direction_returns_zero(self):
        result = directional_curvature(
            np.array([0.0, 0.0, 0.0]),
            k1=5.0,
            k2=3.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        self.assertEqual(result, 0.0)

    def test_near_zero_direction_returns_zero(self):
        result = directional_curvature(
            np.array([1e-15, 0.0, 0.0]),
            k1=5.0,
            k2=3.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        self.assertEqual(result, 0.0)

    def test_aligned_with_dir1_returns_k1(self):
        result = directional_curvature(
            np.array([1.0, 0.0, 0.0]),
            k1=4.0,
            k2=2.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        self.assertAlmostEqual(result, 4.0)

    def test_perpendicular_to_dir1_returns_k2(self):
        result = directional_curvature(
            np.array([0.0, 1.0, 0.0]),
            k1=4.0,
            k2=2.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        self.assertAlmostEqual(result, 2.0)

    def test_equal_curvatures_returns_that_value(self):
        result = directional_curvature(
            np.array([1.0, 1.0, 0.0]),
            k1=3.0,
            k2=3.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        self.assertAlmostEqual(result, 3.0)

    def test_45_degrees_gives_average(self):
        # At 45 deg from dir1: cos^2 = sin^2 = 0.5
        result = directional_curvature(
            np.array([1.0, 1.0, 0.0]),
            k1=4.0,
            k2=2.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        self.assertAlmostEqual(result, 3.0)

    def test_non_orthogonal_dirs_raises(self):
        bad_dir2 = np.array([0.5, 0.5, 0.0])
        with self.assertRaises(ValueError):
            directional_curvature(
                np.array([1.0, 0.0, 0.0]),
                k1=4.0,
                k2=2.0,
                dir1=self.dir1,
                dir2=bad_dir2,
            )

    def test_non_unit_direction_normalizes(self):
        result = directional_curvature(
            np.array([10.0, 0.0, 0.0]),
            k1=7.0,
            k2=1.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        self.assertAlmostEqual(result, 7.0)

    def test_negative_curvatures(self):
        result = directional_curvature(
            np.array([1.0, 0.0, 0.0]),
            k1=-3.0,
            k2=-1.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        self.assertAlmostEqual(result, -3.0)

    def test_opposite_direction_same_result(self):
        fwd = directional_curvature(
            np.array([1.0, 1.0, 0.0]),
            k1=5.0,
            k2=2.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        bwd = directional_curvature(
            np.array([-1.0, -1.0, 0.0]),
            k1=5.0,
            k2=2.0,
            dir1=self.dir1,
            dir2=self.dir2,
        )
        self.assertAlmostEqual(fwd, bwd)


class ArcLengthCorrectedDisplacementTest(unittest.TestCase):
    """Unit tests for the arc_length_corrected_displacement function."""

    def setUp(self):
        self.basis_u = np.array([1.0, 0.0, 0.0])
        self.basis_v = np.array([0.0, 1.0, 0.0])
        # Principal directions aligned with basis vectors
        self.pose_vectors = np.array(
            [
                [0.0, 0.0, 1.0],  # row 0: normal (unused by function)
                [1.0, 0.0, 0.0],  # row 1: principal dir 1
                [0.0, 1.0, 0.0],  # row 2: principal dir 2
            ]
        )

    def test_zero_curvature_returns_unchanged(self):
        result = arc_length_corrected_displacement(
            0.5,
            0.3,
            self.basis_u,
            self.basis_v,
            np.array([0.0, 0.0]),
            self.pose_vectors,
        )
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 0.3)

    def test_axes_corrected_independently(self):
        # k1=1.0 along basis_u, k2=0.0 along basis_v
        arc_u, arc_v = arc_length_corrected_displacement(
            0.5,
            0.5,
            self.basis_u,
            self.basis_v,
            np.array([1.0, 0.0]),
            self.pose_vectors,
        )
        # u axis should be corrected (k=1, p=0.5 => arcsin(0.5)/1 = pi/6)
        self.assertAlmostEqual(arc_u, math.pi / 6)
        # v axis has zero curvature, unchanged
        self.assertAlmostEqual(arc_v, 0.5)

    def test_symmetric_curvature_corrects_both(self):
        # k1 = k2 = 1.0, so both axes get same correction
        arc_u, arc_v = arc_length_corrected_displacement(
            0.5,
            0.5,
            self.basis_u,
            self.basis_v,
            np.array([1.0, 1.0]),
            self.pose_vectors,
        )
        expected = math.pi / 6
        self.assertAlmostEqual(arc_u, expected)
        self.assertAlmostEqual(arc_v, expected)

    def test_zero_displacement_returns_zero(self):
        arc_u, arc_v = arc_length_corrected_displacement(
            0.0,
            0.0,
            self.basis_u,
            self.basis_v,
            np.array([5.0, 5.0]),
            self.pose_vectors,
        )
        self.assertEqual(arc_u, 0.0)
        self.assertEqual(arc_v, 0.0)

    def test_negative_displacement_preserved(self):
        arc_u, arc_v = arc_length_corrected_displacement(
            -0.5,
            0.5,
            self.basis_u,
            self.basis_v,
            np.array([1.0, 1.0]),
            self.pose_vectors,
        )
        self.assertAlmostEqual(arc_u, -math.pi / 6)
        self.assertAlmostEqual(arc_v, math.pi / 6)

    def test_arc_at_least_as_long_as_chord(self):
        arc_u, arc_v = arc_length_corrected_displacement(
            0.3,
            0.4,
            self.basis_u,
            self.basis_v,
            np.array([2.0, 2.0]),
            self.pose_vectors,
        )
        self.assertGreaterEqual(abs(arc_u), 0.3)
        self.assertGreaterEqual(abs(arc_v), 0.4)


if __name__ == "__main__":
    unittest.main()
