import numpy as np
import pytest

from gmm_core.mesh_warp import fit_similarity_transform, warp_points


def _random_proper_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


def test_fit_similarity_transform_recovers_known_transform():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(10, 3))
    R_true = _random_proper_rotation(rng)
    s_true = 2.3
    t_true = np.array([1.0, -2.0, 0.5])
    target = (s_true * (source @ R_true.T)) + t_true

    transform = fit_similarity_transform(source, target)
    assert transform.scale == pytest.approx(s_true, abs=1e-6)
    assert np.allclose(transform.rotation, R_true, atol=1e-6)
    assert np.allclose(transform.translation, t_true, atol=1e-6)


def test_fit_similarity_transform_applies_correctly_to_new_points():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(8, 3))
    R_true = _random_proper_rotation(rng)
    s_true = 0.5
    t_true = np.array([3.0, 1.0, -1.0])
    target = (s_true * (source @ R_true.T)) + t_true

    transform = fit_similarity_transform(source, target)
    reapplied = transform.apply(source)
    assert np.allclose(reapplied, target, atol=1e-6)


def test_invert_round_trips():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(12, 3))
    R_true = _random_proper_rotation(rng)
    target = (1.7 * (source @ R_true.T)) + np.array([0.2, 0.4, -0.6])

    transform = fit_similarity_transform(source, target)
    inv = transform.invert()
    recovered_source = inv.apply(transform.apply(source))
    assert np.allclose(recovered_source, source, atol=1e-6)


def test_never_returns_a_reflection():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(10, 3))
    # Mirror the target across one axis - without correction this would fit
    # a reflection (det=-1) rather than a proper rotation.
    mirror = np.diag([1.0, 1.0, -1.0])
    target = source @ mirror.T

    transform = fit_similarity_transform(source, target)
    assert np.linalg.det(transform.rotation) == pytest.approx(1.0, abs=1e-6)


def test_rejects_mismatched_shapes():
    source = np.zeros((5, 3))
    target = np.zeros((4, 3))
    with pytest.raises(ValueError, match="same shape"):
        fit_similarity_transform(source, target)


def test_rejects_zero_spread_source():
    source = np.ones((5, 3)) * 3.0  # all identical points, zero spread
    target = np.zeros((5, 3))
    with pytest.raises(ValueError, match="zero spread"):
        fit_similarity_transform(source, target)


def test_warp_points_reproduces_target_exactly_at_control_points():
    rng = np.random.default_rng(5)
    source_landmarks = rng.normal(size=(15, 3))
    target_landmarks = source_landmarks + rng.normal(scale=1.5, size=(15, 3))

    warped = warp_points(source_landmarks, source_landmarks, target_landmarks)
    assert np.allclose(warped, target_landmarks, atol=1e-4)


def test_warp_points_identity_when_source_equals_target():
    rng = np.random.default_rng(6)
    landmarks = rng.normal(size=(10, 3))
    other_points = rng.normal(size=(50, 3))

    warped = warp_points(other_points, landmarks, landmarks)
    assert np.allclose(warped, other_points, atol=1e-6)


def test_warp_points_rejects_mismatched_landmark_shapes():
    source_landmarks = np.zeros((5, 3))
    target_landmarks = np.zeros((4, 3))
    points = np.zeros((10, 3))
    with pytest.raises(ValueError, match="same shape"):
        warp_points(points, source_landmarks, target_landmarks)
