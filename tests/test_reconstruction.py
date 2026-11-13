import numpy as np
import pytest

from gmm_core.reconstruction import reconstruct_missing_landmarks


def _make_reference_and_target(rng, n_landmarks=20, n_dim=3, n_reference=60, target_score=2.5):
    """A reference sample dominated by one true axis of variation (PC1),
    plus a target specimen drawn from the same model in its own (rotated/
    scaled/translated) raw coordinate frame - the standard setup for proving
    a reconstruction method actually recovers real signal, not noise.
    """
    consensus = rng.normal(scale=2.0, size=(n_landmarks, n_dim))
    direction = rng.normal(size=(n_landmarks, n_dim))
    direction /= np.linalg.norm(direction)

    scores = rng.normal(scale=1.0, size=n_reference)
    reference_aligned = np.array(
        [
            consensus + s * direction + rng.normal(scale=0.02, size=(n_landmarks, n_dim))
            for s in scores
        ]
    )

    target_aligned = (
        consensus + target_score * direction + rng.normal(scale=0.02, size=(n_landmarks, n_dim))
    )

    # An arbitrary known rigid+scale transform simulating the target's own,
    # unrelated raw digitizing frame.
    theta = 0.7
    c, s_ = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    scale = 3.3
    translation = np.array([15.0, -4.0, 2.0])
    target_raw_full = scale * (target_aligned @ rot.T) + translation

    return consensus, direction, reference_aligned, target_aligned, target_raw_full


def test_recovers_hidden_landmarks_better_than_naive_mean_guess():
    rng = np.random.default_rng(1)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng)

    hidden_indices = [2, 5, 8]
    target_raw_missing = target_raw_full.copy()
    target_raw_missing[hidden_indices] = np.nan

    result = reconstruct_missing_landmarks(reference_aligned, target_raw_missing, n_pcs=1)

    naive_guess_raw = result.transform.invert().apply(reference_aligned.mean(axis=0))
    naive_error = np.linalg.norm(naive_guess_raw[hidden_indices] - target_raw_full[hidden_indices])
    recon_error = np.linalg.norm(
        result.reconstructed_raw[hidden_indices] - target_raw_full[hidden_indices]
    )

    # The iterative PCA-based reconstruction should recover substantially more
    # of the true signal than just guessing the reference consensus's own
    # values - this is the whole value proposition of the method, verified
    # directly rather than assumed.
    assert recon_error < naive_error * 0.7
    assert result.converged


def test_present_landmarks_round_trip_exactly():
    rng = np.random.default_rng(2)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng)
    present_indices = [0, 1, 3, 4, 6, 7, 9]
    target_raw_missing = target_raw_full.copy()
    target_raw_missing[[2, 5, 8]] = np.nan

    result = reconstruct_missing_landmarks(reference_aligned, target_raw_missing, n_pcs=2)

    np.testing.assert_allclose(
        result.reconstructed_raw[present_indices], target_raw_full[present_indices], atol=1e-8
    )


def test_converges_within_default_iterations():
    rng = np.random.default_rng(3)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng)
    target_raw_missing = target_raw_full.copy()
    target_raw_missing[[1, 4]] = np.nan

    result = reconstruct_missing_landmarks(reference_aligned, target_raw_missing, n_pcs=3)
    assert result.converged
    assert result.iterations >= 1


def test_multiple_missing_landmarks_all_reconstructed():
    rng = np.random.default_rng(4)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng)
    hidden = [0, 3, 6, 9]
    target_raw_missing = target_raw_full.copy()
    target_raw_missing[hidden] = np.nan

    result = reconstruct_missing_landmarks(reference_aligned, target_raw_missing, n_pcs=1)
    assert set(result.missing_indices) == set(hidden)
    assert not np.isnan(result.reconstructed_raw).any()


def test_rejects_target_with_no_missing_landmarks():
    rng = np.random.default_rng(5)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng)
    with pytest.raises(ValueError, match="no missing landmarks"):
        reconstruct_missing_landmarks(reference_aligned, target_raw_full, n_pcs=1)


def test_rejects_too_few_present_landmarks():
    rng = np.random.default_rng(6)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng)
    target_raw_missing = target_raw_full.copy()
    target_raw_missing[2:] = np.nan  # only 2 present landmarks left
    with pytest.raises(ValueError, match="at least 3 present landmarks"):
        reconstruct_missing_landmarks(reference_aligned, target_raw_missing, n_pcs=1)


def test_rejects_reference_with_missing_values():
    rng = np.random.default_rng(7)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng)
    reference_aligned[0, 0, 0] = np.nan
    target_raw_missing = target_raw_full.copy()
    target_raw_missing[[2]] = np.nan
    with pytest.raises(ValueError, match="reference sample must be complete"):
        reconstruct_missing_landmarks(reference_aligned, target_raw_missing, n_pcs=1)


def test_rejects_landmark_shape_mismatch():
    rng = np.random.default_rng(8)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng, n_landmarks=10)
    target_raw_missing = target_raw_full[:8].copy()  # wrong landmark count
    target_raw_missing[[2]] = np.nan
    with pytest.raises(ValueError, match="must be"):
        reconstruct_missing_landmarks(reference_aligned, target_raw_missing, n_pcs=1)


def test_n_pcs_capped_to_available_components():
    rng = np.random.default_rng(9)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng, n_reference=5)
    target_raw_missing = target_raw_full.copy()
    target_raw_missing[[2]] = np.nan

    result = reconstruct_missing_landmarks(reference_aligned, target_raw_missing, n_pcs=99)
    assert result.n_pcs_requested == 99
    assert result.n_pcs_used == 4  # n_reference - 1


def test_rejects_n_pcs_less_than_one():
    rng = np.random.default_rng(10)
    _, _, reference_aligned, _, target_raw_full = _make_reference_and_target(rng)
    target_raw_missing = target_raw_full.copy()
    target_raw_missing[[2]] = np.nan
    with pytest.raises(ValueError, match="n_pcs must be at least 1"):
        reconstruct_missing_landmarks(reference_aligned, target_raw_missing, n_pcs=0)
