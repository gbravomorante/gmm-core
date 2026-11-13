"""Generalized Procrustes Analysis, implemented from scratch.

Deliberately not a wrapper around a black-box library: every step is short
enough to read and check.

Two modes share the same rotation-fitting loop:

    scale=True   full GPA: translate, rotate and scale to unit centroid size.
    scale=False  rigid alignment only: translate and rotate, keeping each
                 specimen's original centroid size.

Reflections are never produced: every fitted rotation is constrained to a
proper rotation (det = +1). Semilandmark sliding is a separate operation and
is not performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MAX_ITER = 200
DEFAULT_TOLERANCE = 1e-8


@dataclass
class GPARunResult:
    #: (n, k, d) aligned shapes: unit centroid size if scale=True, original size otherwise
    aligned: np.ndarray
    #: (k, d) consensus shape, at unit centroid size
    consensus: np.ndarray
    #: (n,) original centroid sizes, before any GPA scaling
    centroid_sizes: np.ndarray
    #: (n,) full Procrustes distance to the consensus, always in unit-size space
    procrustes_distances: np.ndarray
    iterations: int
    converged: bool
    final_delta: float
    convergence_threshold: float
    max_iterations: int
    scale: bool
    reflections_allowed: bool = False
    warnings: list[str] = field(default_factory=list)


def centroid_size(shape: np.ndarray) -> float:
    centered = shape - shape.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.sum(centered**2)))


def center_shape(shape: np.ndarray) -> np.ndarray:
    return shape - shape.mean(axis=0, keepdims=True)


def optimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes rotation R (source @ R ~= target).

    Restricted to proper rotations only (det(R) = +1): if the unconstrained
    SVD solution is a reflection, the sign of the last singular vector is
    flipped, per the standard proper-Procrustes correction. This is
    scale-invariant, so the same R applies whether `source`/`target` are
    unit-centroid-size or at original scale.
    """
    h = source.T @ target
    u, _s, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u = u.copy()
        u[:, -1] *= -1
        r = u @ vt
    return r


def generalized_procrustes(
    raw_shapes: np.ndarray,
    scale: bool = True,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOLERANCE,
) -> GPARunResult:
    """Run GPA on (n, k, d) raw landmark configurations.

    Every shape is centered on its own centroid first. Rotation fitting is
    always performed in unit-centroid-size space (mathematically equivalent
    to fitting at any other uniform scale, since Procrustes rotation is
    scale-invariant - see `optimal_rotation`); the `scale` flag only controls
    whether the *returned* aligned coordinates are rescaled to unit size or
    left at their original centroid size.

    Raises ValueError if any input shape has (near-)zero centroid size, since
    GPA cannot orient a degenerate configuration - this is a data problem to
    surface, not something to paper over with a fallback.
    """
    raw_shapes = np.asarray(raw_shapes, dtype=float)
    if raw_shapes.ndim != 3:
        raise ValueError(f"Expected raw_shapes with shape (n, k, d); got {raw_shapes.shape}.")
    if not np.isfinite(raw_shapes).all():
        raise ValueError(
            "raw_shapes contains NaN/inf values; exclude or complete these rows before running GPA."
        )

    n = raw_shapes.shape[0]
    if n < 2:
        raise ValueError("GPA requires at least 2 specimens.")

    centered = raw_shapes - raw_shapes.mean(axis=1, keepdims=True)
    orig_centroid_sizes = np.sqrt(np.sum(centered**2, axis=(1, 2)))
    zero_size = np.flatnonzero(orig_centroid_sizes <= 1e-12)
    if zero_size.size:
        raise ValueError(
            f"Zero (or near-zero) centroid size at row index(es) {zero_size.tolist()}; cannot run GPA."
        )

    normalized = centered / orig_centroid_sizes[:, None, None]

    reference = normalized[0].copy()
    reference -= reference.mean(axis=0, keepdims=True)
    ref_norm = np.linalg.norm(reference)
    reference /= ref_norm

    aligned_normalized = normalized.copy()
    warnings_list: list[str] = []
    converged = False
    final_delta = float("inf")
    iterations_done = 0

    for iteration in range(1, max_iter + 1):
        iterations_done = iteration
        rotations = [optimal_rotation(normalized[i], reference) for i in range(n)]
        aligned_normalized = np.stack([normalized[i] @ rotations[i] for i in range(n)])

        new_reference = aligned_normalized.mean(axis=0)
        new_reference -= new_reference.mean(axis=0, keepdims=True)
        new_ref_norm = np.linalg.norm(new_reference)
        if new_ref_norm <= 1e-12:
            raise ValueError(
                "GPA consensus collapsed to zero size; check input landmarks for degeneracy."
            )
        new_reference /= new_ref_norm

        final_delta = float(np.linalg.norm(new_reference - reference))
        reference = new_reference
        if final_delta < tol:
            converged = True
            break

    if not converged:
        warnings_list.append(
            f"GPA did not converge within {max_iter} iterations "
            f"(final delta={final_delta:.3e}, threshold={tol:.3e}). "
            "Results are still returned, but treat the consensus as provisional."
        )

    # Final rotations against the converged (or last) consensus, applied at
    # whichever scale the caller asked for.
    rotations = [optimal_rotation(normalized[i], reference) for i in range(n)]
    aligned_normalized = np.stack([normalized[i] @ rotations[i] for i in range(n)])

    if scale:
        aligned = aligned_normalized
    else:
        aligned = np.stack([centered[i] @ rotations[i] for i in range(n)])

    procrustes_distances = np.linalg.norm(aligned_normalized - reference[None, :, :], axis=(1, 2))

    return GPARunResult(
        aligned=aligned,
        consensus=reference,
        centroid_sizes=orig_centroid_sizes,
        procrustes_distances=procrustes_distances,
        iterations=iterations_done,
        converged=converged,
        final_delta=final_delta,
        convergence_threshold=tol,
        max_iterations=max_iter,
        scale=scale,
        reflections_allowed=False,
        warnings=warnings_list,
    )


def pairwise_procrustes_distances(unit_size_shapes: np.ndarray) -> np.ndarray:
    """Full (n, n) Procrustes distance matrix between already-aligned,
    unit-centroid-size shapes (e.g. GPARunResult.aligned when scale=True).

    Distances are computed directly (no further rotation) since all inputs
    share one consensus orientation already.
    """
    flat = unit_size_shapes.reshape(unit_size_shapes.shape[0], -1)
    squared_norms = np.sum(flat**2, axis=1)
    d2 = squared_norms[:, None] + squared_norms[None, :] - 2.0 * flat @ flat.T
    return np.sqrt(np.maximum(d2, 0.0))
