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

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    TangentFrame,
    normalize,
    project_onto_tangent_plane,
)


class NormalizeTest(unittest.TestCase):
    """Unit tests for the normalize function."""

    def test_unit_vector_along_axis(self):
        result = normalize(np.array([3.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_3d_diagonal_vector(self):
        result = normalize(np.array([1.0, 1.0, 1.0]))
        expected = np.array([1, 1, 1]) / np.sqrt(3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_already_unit_vector(self):
        v = np.array([0.0, 1.0, 0.0])
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_negative_components(self):
        result = normalize(np.array([-3.0, 4.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [-0.6, 0.8, 0.0])

    def test_near_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            normalize(np.array([1e-13, 0.0, 0.0]))

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            normalize(np.array([0.0, 0.0, 0.0]))

    def test_custom_eps(self):
        with self.assertRaises(ValueError):
            normalize(np.array([1e-8, 0.0, 0.0]), eps=1e-6)

    def test_result_has_unit_norm(self):
        v = np.array([2.5, -7.1, 3.3])
        result = normalize(v)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0)


class ProjectOntoTangentPlaneTest(unittest.TestCase):
    """Unit tests for the project_onto_tangent_plane function."""

    def test_parallel_to_normal(self):
        n = normalize(np.array([0.0, 0.0, 1.0]))
        result = project_onto_tangent_plane(n, n)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_perpendicular_to_normal(self):
        v = np.array([1.0, 0.0, 0.0])
        n = np.array([0.0, 0.0, 1.0])
        result = project_onto_tangent_plane(v, n)
        np.testing.assert_array_almost_equal(result, v)

    def test_general_oblique_case(self):
        v = np.array([1.0, 1.0, 0.0])
        n = np.array([0.0, 0.0, 1.0])
        result = project_onto_tangent_plane(v, n)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 0.0])

    def test_result_orthogonal_to_normal(self):
        v = np.array([3.0, -2.0, 5.0])
        n = normalize(np.array([1.0, 1.0, 1.0]))
        result = project_onto_tangent_plane(v, n)
        self.assertAlmostEqual(np.dot(result, n), 0.0)

    def test_zero_vector_input(self):
        v = np.array([0.0, 0.0, 0.0])
        n = np.array([0.0, 0.0, 1.0])
        result = project_onto_tangent_plane(v, n)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])


class TangentFrameTest(unittest.TestCase):
    """Unit tests for the TangentFrame class."""

    def _assert_orthonormal_frame(self, frame, normal):
        """Assert (basis_u, basis_v, normal) form an orthonormal right-handed frame."""
        u, v = frame.basis_u, frame.basis_v
        self.assertAlmostEqual(np.linalg.norm(u), 1.0, places=10)
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=10)
        self.assertAlmostEqual(np.dot(u, v), 0.0, places=10)
        self.assertAlmostEqual(np.dot(u, normal), 0.0, places=10)
        self.assertAlmostEqual(np.dot(v, normal), 0.0, places=10)
        # Right-handed: n x u ~ v (or u x v ~ n)
        np.testing.assert_array_almost_equal(np.cross(u, v), normal, decimal=10)

    def test_construction_with_z_normal(self):
        n = np.array([0.0, 0.0, 1.0])
        frame = TangentFrame(n)
        self._assert_orthonormal_frame(frame, n)

    def test_construction_with_y_aligned_normal_triggers_fallback(self):
        n = normalize(np.array([0.0, 1.0, 0.01]))
        frame = TangentFrame(n)
        self._assert_orthonormal_frame(frame, n)

    def test_transport_to_same_normal_is_noop(self):
        n = normalize(np.array([1.0, 1.0, 1.0]))
        frame = TangentFrame(n)
        u_before, v_before = frame.basis_u.copy(), frame.basis_v.copy()
        frame.transport(n)
        np.testing.assert_array_almost_equal(frame.basis_u, u_before)
        np.testing.assert_array_almost_equal(frame.basis_v, v_before)

    def test_transport_preserves_orthonormality(self):
        n1 = np.array([0.0, 0.0, 1.0])
        n2 = normalize(np.array([0.1, 0.0, 1.0]))
        frame = TangentFrame(n1)
        frame.transport(n2)
        self._assert_orthonormal_frame(frame, n2)

    def test_transport_anti_parallel(self):
        n1 = np.array([0.0, 0.0, 1.0])
        n2 = np.array([0.0, 0.0, -1.0])
        frame = TangentFrame(n1)
        frame.transport(n2)
        u, v = frame.basis_u, frame.basis_v
        self.assertAlmostEqual(np.linalg.norm(u), 1.0, places=10)
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=10)
        self.assertAlmostEqual(np.dot(u, n2), 0.0, places=10)
        self.assertAlmostEqual(np.dot(v, n2), 0.0, places=10)

    def test_accumulated_transports_stay_orthonormal(self):
        rng = np.random.RandomState(42)
        n = normalize(rng.randn(3))
        frame = TangentFrame(n)
        for _ in range(100):
            n_new = normalize(n + 0.05 * rng.randn(3))
            frame.transport(n_new)
            n = n_new
        self._assert_orthonormal_frame(frame, n)


if __name__ == "__main__":
    unittest.main()
