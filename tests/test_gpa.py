import numpy as np
import pytest

from gmm_core.gpa import (
    centroid_size,
    generalized_procrustes,
    optimal_rotation,
    pairwise_procrustes_distances,
)


def _base_shape() -> np.ndarray:
    # Deliberately asymmetric, non-planar 6-landmark shape: nothing here
    # should be mappable onto its own mirror image by a proper rotation.
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.3, 0.1],
            [1.0, 2.0, 0.4],
            [0.2, 1.1, 2.0],
            [1.5, 1.5, 1.0],
            [0.7, 0.2, 1.3],
        ]
    )


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


def test_translation_invariance():
    base = _base_shape()
    translated = base + np.array([5.0, -3.0, 2.0])
    shapes = np.stack([base, translated])
    result = generalized_procrustes(shapes, scale=True)
    assert np.allclose(result.aligned[0], result.aligned[1], atol=1e-6)


def test_rotation_invariance():
    base = _base_shape()
    rotation = _rotation_matrix(np.array([0.2, 0.7, 0.3]), 1.1)
    rotated = base @ rotation
    shapes = np.stack([base, rotated])
    result = generalized_procrustes(shapes, scale=True)
    assert np.allclose(result.aligned[0], result.aligned[1], atol=1e-6)


def test_scale_invariance_under_full_gpa():
    base = _base_shape()
    scaled = base * 3.7
    shapes = np.stack([base, scaled])
    result = generalized_procrustes(shapes, scale=True)
    assert np.allclose(result.aligned[0], result.aligned[1], atol=1e-6)
    assert np.isclose(centroid_size(result.aligned[0]), 1.0, atol=1e-6)


def test_noscale_gpa_retains_original_size():
    base = _base_shape()
    scaled = base * 2.0
    shapes = np.stack([base, scaled])
    result = generalized_procrustes(shapes, scale=False)
    cs0 = centroid_size(result.aligned[0])
    cs1 = centroid_size(result.aligned[1])
    assert not np.isclose(cs0, cs1, rtol=0.05)
    assert np.isclose(cs1 / cs0, 2.0, atol=1e-3)


def test_reflection_is_not_silently_accepted():
    base = _base_shape()
    mirrored = base.copy()
    mirrored[:, 0] *= -1  # a true reflection (det = -1), not reachable by rotation
    shapes = np.stack([base, mirrored])
    result = generalized_procrustes(shapes, scale=True)
    # Reflections are disabled by default: GPA cannot rotate a mirror image
    # onto the original, so the residual distance must stay large.
    assert result.procrustes_distances.max() > 0.2


def test_landmark_permutation_is_not_silently_aligned_away():
    base = _base_shape()
    permuted = base[[1, 0, 2, 3, 4, 5]]  # swap the first two landmarks
    shapes = np.stack([base, permuted])
    result = generalized_procrustes(shapes, scale=True)
    # GPA has no correspondence-fixing step; mislabeled landmarks are a data
    # problem that must show up as a large residual, not vanish.
    assert result.procrustes_distances.max() > 0.05


def test_convergence_tracking_reports_sane_values():
    rng = np.random.default_rng(42)
    base = _base_shape()
    shapes = np.stack([base + rng.normal(scale=0.05, size=base.shape) for _ in range(8)])
    result = generalized_procrustes(shapes, scale=True, max_iter=200, tol=1e-8)
    assert result.converged
    assert result.iterations >= 1
    assert result.final_delta < 1e-8
    assert result.convergence_threshold == 1e-8
    assert result.warnings == []


def test_zero_centroid_size_raises():
    base = _base_shape()
    degenerate = np.tile(
        base[0], (base.shape[0], 1)
    )  # every landmark identical -> zero centroid size
    shapes = np.stack([base, degenerate])
    with pytest.raises(ValueError):
        generalized_procrustes(shapes, scale=True)


def test_optimal_rotation_is_always_proper():
    base = _base_shape()
    mirrored = base.copy()
    mirrored[:, 0] *= -1
    rotation = optimal_rotation(base - base.mean(axis=0), mirrored - mirrored.mean(axis=0))
    assert np.linalg.det(rotation) > 0


def test_pairwise_procrustes_distances_symmetric_zero_diagonal():
    base = _base_shape()
    shapes = np.stack([base, base + 0.01, base - 0.01])
    result = generalized_procrustes(shapes, scale=True)
    distances = pairwise_procrustes_distances(result.aligned)
    assert np.allclose(np.diag(distances), 0, atol=1e-6)
    assert np.allclose(distances, distances.T)


def test_requires_at_least_two_specimens():
    base = _base_shape()
    with pytest.raises(ValueError):
        generalized_procrustes(base[np.newaxis, :, :], scale=True)
