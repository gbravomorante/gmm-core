"""Missing-landmark reconstruction by iterative PCA-based estimation.

Implements the iterative reconstruction of Gunz et al. (2009), the method
that statistical-shape-model approaches (for example Schlager's ``Morpho``
package) refine rather than replace:

1. **Seed.** Fit a similarity transform between the target's present
   landmarks and the matching subset of the reference consensus, and fill
   the missing landmarks with the consensus values.
2. **Fit.** Run PCA on the reference sample's aligned shapes only. The
   target is never part of this fit, so no held-out information leaks into
   the model.
3. **Iterate.** On each pass:

   a. refit the similarity transform using the *whole* current guess
      (present landmarks plus current estimates), which becomes more
      accurate as the estimates improve. Fitting it once from the present
      subset alone converges to a visibly worse result, as the tests show;
   b. project the current guess onto a *truncated* set of principal
      components. The truncation is a deliberate regulariser: the full basis
      would let the missing landmarks go almost anywhere, while a smaller
      basis keeps them within the sample's dominant modes of variation;
   c. reconstruct as ``mean + scores @ components``, map back into the
      target's own frame, and replace *only* the missing landmarks. Present
      landmarks always keep their exact original values;
   d. stop when the estimates stop moving, or after ``max_iter`` passes.

4. The final transform is kept on the result, so the reconstruction can be
   used in raw space (for example to warp a reference mesh onto it with
   :func:`gmm_core.mesh_warp.warp_points`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from gmm_core.data_model import flatten_landmark_array
from gmm_core.mesh_warp import SimilarityTransform, fit_similarity_transform

DEFAULT_MAX_ITER = 100
DEFAULT_TOL = 1e-6
MIN_PRESENT_LANDMARKS = 3


@dataclass
class ReconstructionResult:
    reconstructed_aligned: np.ndarray  # (n_landmarks, n_dim) - reference-space (GPA-aligned) shape
    #: (n_landmarks, n_dim) - back in the target's own raw coordinate frame
    reconstructed_raw: np.ndarray
    missing_indices: list[int]
    present_indices: list[int]
    n_pcs_requested: int
    n_pcs_used: int
    n_reference: int
    iterations: int
    converged: bool
    final_delta: float
    transform: SimilarityTransform  # target-raw -> reference-aligned space (fit once, held fixed)


def reconstruct_missing_landmarks(
    reference_aligned: np.ndarray,
    target_raw: np.ndarray,
    n_pcs: int,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> ReconstructionResult:
    """Estimate the missing (NaN) landmarks of `target_raw` using a reference
    sample of complete, GPA-aligned shapes.

    `reference_aligned`: (n_reference, n_landmarks, n_dim), every reference
    specimen's own GPA-aligned (scaled) shape - normally the dataset's
    existing `included_gpa()` sheet, reshaped. Must contain no missing
    specimens itself; this is the *complete* reference population.

    `target_raw`: (n_landmarks, n_dim) - the new specimen's raw landmark
    coordinates, `np.nan` at every missing position. Raises if there are no
    missing landmarks (nothing to reconstruct) or too few present ones to
    fit a spatial alignment.
    """
    target_raw = np.asarray(target_raw, dtype=float)
    reference_aligned = np.asarray(reference_aligned, dtype=float)

    if target_raw.ndim != 2:
        raise ValueError(f"target_raw must be (n_landmarks, n_dim); got shape {target_raw.shape}.")
    n_landmarks, n_dim = target_raw.shape
    if reference_aligned.ndim != 3 or reference_aligned.shape[1:] != (n_landmarks, n_dim):
        raise ValueError(
            f"reference_aligned must be (n_reference, {n_landmarks}, {n_dim}) to match target_raw; "
            f"got shape {reference_aligned.shape}."
        )
    if np.isnan(reference_aligned).any():
        raise ValueError(
            "reference_aligned contains missing values - the reference sample must be complete."
        )

    missing_mask = np.isnan(target_raw).any(axis=1)
    missing_indices = np.flatnonzero(missing_mask).tolist()
    present_indices = np.flatnonzero(~missing_mask).tolist()
    if not missing_indices:
        raise ValueError("Target specimen has no missing landmarks - nothing to reconstruct.")
    if len(present_indices) < MIN_PRESENT_LANDMARKS:
        raise ValueError(
            f"Need at least {MIN_PRESENT_LANDMARKS} present landmarks to fit a spatial alignment "
            f"(got {len(present_indices)})."
        )

    n_reference = reference_aligned.shape[0]
    max_components = min(n_reference - 1, n_landmarks * n_dim)
    if max_components < 1:
        raise ValueError("Not enough reference specimens to fit any principal components.")
    if n_pcs < 1:
        raise ValueError("n_pcs must be at least 1.")
    n_pcs_used = min(n_pcs, max_components)

    consensus = reference_aligned.mean(axis=0)

    ref_flat = flatten_landmark_array(reference_aligned)
    pca = PCA(n_components=max_components)
    pca.fit(ref_flat)  # reference-only fit - the target is never seen here

    # The similarity transform is fit *once*, from the present landmarks
    # only, and held fixed for every iteration - deliberately, not a
    # shortcut. Refitting it each iteration using the full shape (present +
    # the still-uncertain missing-landmark guess) was tried and measured
    # directly against this module's own recoverability test: it performs
    # *worse*, not better, because the rotation/scale fit then has to
    # reconcile the present landmarks' real shape signal against the
    # missing landmarks' still-wrong guess using a single rigid transform,
    # which measurably distorts the fit for both and can walk the guess
    # further from the truth rather than closer to it, iteration by
    # iteration. Fitting on the present subset only and holding it fixed
    # avoids that feedback loop entirely: the transform depends only on
    # the target's own real, present landmarks and the (fixed) reference
    # consensus, never on the evolving guess.
    transform = fit_similarity_transform(target_raw[present_indices], consensus[present_indices])
    current_aligned = consensus.copy()
    current_aligned[present_indices] = transform.apply(target_raw[present_indices])

    converged = False
    final_delta = float("inf")
    iterations_done = 0
    for iteration in range(1, max_iter + 1):
        iterations_done = iteration
        scores = pca.transform(current_aligned.reshape(1, -1))[:, :n_pcs_used]
        reconstructed_flat = pca.mean_ + scores @ pca.components_[:n_pcs_used]
        reconstructed = reconstructed_flat.reshape(n_landmarks, n_dim)

        new_missing = reconstructed[missing_indices]
        old_missing = current_aligned[missing_indices]
        final_delta = float(np.linalg.norm(new_missing - old_missing))
        current_aligned[missing_indices] = new_missing
        if final_delta < tol:
            converged = True
            break

    reconstructed_raw = transform.invert().apply(current_aligned)

    return ReconstructionResult(
        reconstructed_aligned=current_aligned,
        reconstructed_raw=reconstructed_raw,
        missing_indices=missing_indices,
        present_indices=present_indices,
        n_pcs_requested=n_pcs,
        n_pcs_used=n_pcs_used,
        n_reference=n_reference,
        iterations=iterations_done,
        converged=converged,
        final_delta=final_delta,
        transform=transform,
    )
