"""Thin-plate-spline warping of a reference specimen's 3D mesh.

The standard way to show a hypothetical shape (a PC extreme, a group-mean
difference) on a real surface:

1. The reference specimen's landmarks were digitised on its own mesh, so its
   raw landmark coordinates are already in the mesh's frame.
2. Its GPA-aligned landmarks differ from the raw ones only by a similarity
   transform. Fitting that transform (:func:`fit_similarity_transform`) and
   inverting it brings any aligned-space shape back into the mesh's frame.
3. A thin-plate spline from the reference's raw landmarks to that shape gives
   a smooth 3D deformation field, which :func:`warp_points` applies to every
   mesh vertex.

Depends only on NumPy and SciPy; mesh file I/O is left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass
class SimilarityTransform:
    rotation: np.ndarray  # (n_dim, n_dim), proper rotation (det=+1)
    scale: float
    translation: np.ndarray  # (n_dim,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (self.scale * (points @ self.rotation.T)) + self.translation

    def invert(self) -> SimilarityTransform:
        inv_rotation = self.rotation.T
        inv_scale = 1.0 / self.scale
        inv_translation = -inv_scale * (inv_rotation @ self.translation)
        return SimilarityTransform(
            rotation=inv_rotation, scale=inv_scale, translation=inv_translation
        )


def fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    """Least-squares rigid+uniform-scale transform mapping `source` onto
    `target` (both (n_points, n_dim), point-for-point correspondence).

    Proper rotation only (reflections never silently introduced), consistent
    with :mod:`gmm_core.gpa`. Intended for a *single specimen's*
    own raw vs. GPA-aligned landmarks, where an exact (near-zero-residual)
    similarity transform exists by construction - not a general shape-
    comparison tool.
    """
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape.")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    denom = float(np.sum(source_centered**2))
    if denom <= 0:
        raise ValueError("Source landmarks have zero spread - cannot fit a similarity transform.")

    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        S = S.copy()
        S[-1] *= -1
        R = Vt.T @ U.T

    scale = float(np.sum(S) / denom)
    translation = target_mean - scale * (R @ source_mean)

    return SimilarityTransform(rotation=R, scale=scale, translation=translation)


def warp_points(
    points: np.ndarray, source_landmarks: np.ndarray, target_landmarks: np.ndarray
) -> np.ndarray:
    """Thin-plate-spline warp: fit a smooth deformation mapping
    `source_landmarks` exactly onto `target_landmarks` (both (n_landmarks,
    n_dim), same coordinate frame as `points`), then apply it to every row
    of `points` (mesh vertices, or anything else in that same frame).
    """
    if source_landmarks.shape != target_landmarks.shape:
        raise ValueError("source_landmarks and target_landmarks must have the same shape.")
    tps = RBFInterpolator(source_landmarks, target_landmarks, kernel="thin_plate_spline")
    return tps(points)
