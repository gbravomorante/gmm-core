"""Working with incomplete specimens.

Standard GPA drops any specimen with a missing landmark, and fossils are
precisely the specimens with missing landmarks. This module offers
alternatives, each of which keeps estimated values clearly separate from
observed ones:

* :func:`coverage_tables` and :func:`greedy_landmark_drop`: how much data each
  landmark costs, and which to give up so that more specimens become
  complete.
* :func:`shared_distance_matrix`: Procrustes distances between specimens on
  the landmarks each pair actually shares, so nothing is estimated at all.
* :func:`observed_only_test`: is an incomplete specimen unusual compared with
  a complete reference sample, judged on its observed landmarks only?
* :func:`estimate_missing_tps` and :func:`reconstruct_batch`: estimation for a
  whole sample, optionally in stages (for example, reconstruct one fossil
  from a modern reference, then a second fossil from the first), with
  non-convergence reported.
* :func:`pattern_validation`: the error to expect for a given pattern of
  damage, measured by imposing exactly that pattern on complete specimens
  whose true landmarks are known.

Estimates are always flagged (``estimated_mask``), never mixed silently with
observed coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

from gmm_core.gpa import generalized_procrustes
from gmm_core.mesh_warp import fit_similarity_transform
from gmm_core.reconstruction import reconstruct_missing_landmarks

MIN_SHARED_LANDMARKS = 4


def present_mask(shapes: np.ndarray) -> np.ndarray:
    """(n_specimens, n_landmarks) True where a landmark is fully observed."""
    return ~np.isnan(np.asarray(shapes, dtype=float)).any(axis=2)


def coverage_tables(
    shapes: np.ndarray, ids: Sequence[str], landmark_names: Sequence[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Missingness per landmark and per specimen."""
    mask = present_mask(shapes)
    names = (
        list(landmark_names)
        if landmark_names is not None
        else [f"LM{i}" for i in range(mask.shape[1])]
    )
    landmarks = pd.DataFrame(
        {
            "Landmark": names,
            "N_present": mask.sum(axis=0),
            "Percent_present": 100 * mask.mean(axis=0),
        }
    )
    specimens = pd.DataFrame(
        {
            "ID": list(ids),
            "N_present": mask.sum(axis=1),
            "N_missing": (~mask).sum(axis=1),
            "Percent_present": 100 * mask.mean(axis=1),
        }
    )
    return {
        "by_landmark": landmarks.sort_values("N_present").reset_index(drop=True),
        "by_specimen": specimens.sort_values("N_present").reset_index(drop=True),
    }


def greedy_landmark_drop(mask: np.ndarray, max_drop: int | None = None) -> pd.DataFrame:
    """Drop landmarks one at a time, each time the one that makes the most
    specimens complete. Ties go to the landmark present in the fewest specimens.

    Returns one row per step: what was dropped and how many specimens are then
    complete. Row 0 is the starting point.
    """
    mask = np.asarray(mask, dtype=bool)
    n_landmarks = mask.shape[1]
    limit = n_landmarks - 3 if max_drop is None else min(max_drop, n_landmarks - 3)
    kept = list(range(n_landmarks))
    rows = [
        {
            "N_dropped": 0,
            "Dropped": "",
            "N_landmarks": n_landmarks,
            "N_complete": int(mask.all(axis=1).sum()),
        }
    ]
    for step in range(1, limit + 1):
        best, best_key = None, None
        for lm in kept:
            others = [k for k in kept if k != lm]
            key = (int(mask[:, others].all(axis=1).sum()), -int(mask[:, lm].sum()))
            if best_key is None or key > best_key:
                best, best_key = lm, key
        kept.remove(best)
        rows.append(
            {
                "N_dropped": step,
                "Dropped": best,
                "N_landmarks": len(kept),
                "N_complete": best_key[0],
            }
        )
    return pd.DataFrame(rows)


def _unit(shape: np.ndarray) -> np.ndarray:
    centred = shape - shape.mean(axis=0)
    size = float(np.sqrt((centred**2).sum()))
    if size == 0:
        raise ValueError("A configuration has zero centroid size.")
    return centred / size


def partial_procrustes_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two complete configurations, proper rotations only."""
    a, b = _unit(a), _unit(b)
    u, s, vt = np.linalg.svd(a.T @ b)
    s = s.copy()
    if np.linalg.det(u @ vt) < 0:
        s[-1] = -s[-1]
    return float(np.sqrt(max(2.0 - 2.0 * s.sum(), 0.0)))


def shared_distance_matrix(
    shapes: np.ndarray, min_shared: int = MIN_SHARED_LANDMARKS
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise partial Procrustes distance using only the landmarks each pair
    shares, plus the shared count. Pairs below `min_shared` are NaN.

    Distances from different landmark sets are not strictly comparable with
    each other: a pair sharing the skull's most variable region looks further
    apart than one sharing its stable region. `observed_only_test` is the
    calibrated version of this.
    """
    shapes = np.asarray(shapes, dtype=float)
    mask = present_mask(shapes)
    n = len(shapes)
    distances = np.full((n, n), np.nan)
    shared = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            both = mask[i] & mask[j]
            shared[i, j] = shared[j, i] = int(both.sum())
            if both.sum() >= min_shared:
                distances[i, j] = distances[j, i] = (
                    0.0 if i == j else partial_procrustes_distance(shapes[i][both], shapes[j][both])
                )
    return distances, shared


@dataclass
class ObservedOnlyResult:
    """An incomplete specimen against a complete reference, nothing estimated."""

    distance: float
    null_distances: np.ndarray  # each reference specimen against the others
    percentile: float  # of the reference null lying below `distance`
    upper_tail: float  # (k + 1) / (n + 1)
    n_landmarks: int


def observed_only_test(reference: np.ndarray, target: np.ndarray) -> ObservedOnlyResult:
    """Distance of `target` to the reference mean, on `target`'s observed
    landmarks only, against the reference's own spread on those landmarks.

    Each reference specimen is scored against a mean fitted WITHOUT it, as the
    target is, because otherwise the null is too tight and every target looks
    unusual. The GPA is refitted per left-out specimen, so this costs n GPAs.
    """
    reference = np.asarray(reference, dtype=float)
    target = np.asarray(target, dtype=float)
    observed = ~np.isnan(target).any(axis=1)
    if observed.sum() < MIN_SHARED_LANDMARKS:
        raise ValueError(
            f"Only {int(observed.sum())} observed landmarks; at least {MIN_SHARED_LANDMARKS} are needed."
        )
    if np.isnan(reference[:, observed]).any():
        raise ValueError("The reference has missing values on the target's observed landmarks.")
    subset = reference[:, observed]
    n = len(subset)
    if n < 4:
        raise ValueError("The reference needs at least 4 specimens.")
    null = np.empty(n)
    for k in range(n):
        consensus = generalized_procrustes(np.delete(subset, k, axis=0), scale=True).consensus
        null[k] = partial_procrustes_distance(subset[k], consensus)
    consensus = generalized_procrustes(subset, scale=True).consensus
    distance = partial_procrustes_distance(target[observed], consensus)
    return ObservedOnlyResult(
        distance=distance,
        null_distances=null,
        percentile=float((null < distance).mean() * 100),
        upper_tail=float(((null >= distance).sum() + 1) / (n + 1)),
        n_landmarks=int(observed.sum()),
    )


def estimate_missing_tps(
    reference: np.ndarray, target: np.ndarray, n_closest: int = 3
) -> np.ndarray:
    """Fill a specimen's gaps by thin-plate spline from similar complete specimens.

    The same idea as `Morpho::fixLMtps`: find the reference specimens closest
    to the target on its observed landmarks, take their mean (nearer ones count
    more), and warp that mean onto the target through the observed landmarks.
    Returns a complete copy; observed coordinates are unchanged.

    `reference` may be in any frame and any size; each is fitted to the target.
    """
    target = np.asarray(target, dtype=float)
    reference = np.asarray(reference, dtype=float)
    observed = ~np.isnan(target).any(axis=1)
    if observed.sum() < target.shape[1] + 2:
        raise ValueError(f"Only {int(observed.sum())} observed landmarks; too few for a spline.")
    if observed.all():
        return target.copy()
    if np.isnan(reference).any():
        raise ValueError("The reference sample must be complete.")

    fitted, distances = [], []
    for candidate in reference:
        transform = fit_similarity_transform(candidate[observed], target[observed])
        placed = transform.apply(candidate)
        fitted.append(placed)
        distances.append(float(np.sqrt(((placed[observed] - target[observed]) ** 2).sum())))
    distances = np.array(distances)
    order = np.argsort(distances)[: max(1, min(n_closest, len(reference)))]
    weights = 1.0 / np.maximum(distances[order], 1e-12)
    weights /= weights.sum()
    mean_shape = (np.stack([fitted[i] for i in order]) * weights[:, None, None]).sum(axis=0)

    spline = RBFInterpolator(mean_shape[observed], target[observed], kernel="thin_plate_spline")
    out = target.copy()
    out[~observed] = spline(mean_shape[~observed])
    return out


def estimate_missing_pca(
    reference: np.ndarray, target: np.ndarray, n_pcs: int = 5, max_iter: int = 500
) -> tuple[np.ndarray, bool, int]:
    """Iterative PCA estimation (Gunz et al., 2009), returned in the target's own frame."""
    reference = np.asarray(reference, dtype=float)
    aligned = generalized_procrustes(reference, scale=True).aligned
    result = reconstruct_missing_landmarks(
        aligned, target, n_pcs=min(n_pcs, len(reference) - 1), max_iter=max_iter
    )
    return result.reconstructed_raw, bool(result.converged), int(result.iterations)


@dataclass
class BatchResult:
    """A sample with its gaps filled. Observed values are never changed."""

    complete: np.ndarray  # (n, p, k)
    estimated_mask: np.ndarray  # (n, p) True where estimated
    report: pd.DataFrame  # ID, stage, reference used, missing, converged
    warnings: list[str] = field(default_factory=list)


def reconstruct_batch(
    shapes: np.ndarray,
    ids: Sequence[str],
    stages: list[Sequence[str]] | None = None,
    method: str = "tps",
    n_closest: int = 3,
    n_pcs: int = 5,
) -> BatchResult:
    """Estimate every incomplete specimen, optionally in stages.

    With no `stages`, all incomplete specimens are estimated from the complete
    ones. Each entry of `stages` is a list of IDs estimated from the reference
    as it stands at that stage, after which they join it. Incomplete specimens
    named in no stage form a last stage of their own.
    """
    if method not in ("tps", "pca"):
        raise ValueError(f"Unknown method {method!r}: use 'tps' or 'pca'.")
    shapes = np.array(shapes, dtype=float)
    ids = [str(i) for i in ids]
    mask = present_mask(shapes)
    complete_now = set(i for i, full in enumerate(mask.all(axis=1)) if full)
    incomplete = [i for i in range(len(ids)) if i not in complete_now]
    if not complete_now:
        raise ValueError("No complete specimen to use as a reference.")
    if stages is None:
        groups = [incomplete]
    else:
        index = {name: i for i, name in enumerate(ids)}
        unknown = [name for stage in stages for name in stage if name not in index]
        if unknown:
            raise ValueError(f"Unknown IDs in stages: {unknown}")
        groups = [[index[name] for name in stage if index[name] in incomplete] for stage in stages]
        placed = {i for group in groups for i in group}
        rest = [i for i in incomplete if i not in placed]
        if rest:
            groups.append(rest)

    rows, warnings = [], []
    out = shapes.copy()
    for stage_no, group in enumerate(groups, start=1):
        reference_ids = sorted(complete_now)
        reference = out[reference_ids]
        for i in group:
            converged, iterations = True, 0
            if method == "tps":
                out[i] = estimate_missing_tps(
                    reference, shapes[i], n_closest=min(n_closest, len(reference))
                )
            else:
                out[i], converged, iterations = estimate_missing_pca(
                    reference, shapes[i], n_pcs=n_pcs
                )
            if not converged:
                warnings.append(
                    f"{ids[i]}: the estimate did not converge in {iterations} iterations; "
                    f"{int((~mask[i]).sum())} of {mask.shape[1]} landmarks are missing."
                )
            rows.append(
                {
                    "ID": ids[i],
                    "Stage": stage_no,
                    "Method": method,
                    "N_reference": len(reference_ids),
                    "N_missing": int((~mask[i]).sum()),
                    "Converged": converged,
                    "Iterations": iterations,
                }
            )
        complete_now.update(group)
    return BatchResult(
        complete=out, estimated_mask=~mask, report=pd.DataFrame(rows), warnings=warnings
    )


def pattern_validation(
    complete: np.ndarray,
    missing_landmarks: np.ndarray,
    method: str = "tps",
    n_closest: int = 3,
    n_pcs: int = 5,
) -> pd.DataFrame:
    """Impose one damage pattern on each complete specimen in turn, estimate it
    from the others, and compare with its true coordinates.

    `missing_landmarks` is a boolean per landmark. Returns one row per specimen:
    mean and largest landmark error in the specimen's own units, the estimate's
    Procrustes distance from the truth, and how far the true shape lies from
    the reference mean, the scale against which that error should be read.
    """
    complete = np.asarray(complete, dtype=float)
    missing = np.asarray(missing_landmarks, dtype=bool)
    if not missing.any():
        raise ValueError("The pattern has no missing landmarks.")
    rows = []
    for k in range(len(complete)):
        reference = np.delete(complete, k, axis=0)
        damaged = complete[k].copy()
        damaged[missing] = np.nan
        if method == "tps":
            estimate = estimate_missing_tps(reference, damaged, n_closest=n_closest)
        else:
            estimate, _, _ = estimate_missing_pca(reference, damaged, n_pcs=n_pcs)
        errors = np.sqrt(((estimate[missing] - complete[k][missing]) ** 2).sum(axis=1))
        size = float(np.sqrt(((complete[k] - complete[k].mean(axis=0)) ** 2).sum()))
        consensus = generalized_procrustes(reference, scale=True).consensus
        rows.append(
            {
                "Specimen": k,
                "Mean_error": float(errors.mean()),
                "Max_error": float(errors.max()),
                "Mean_error_percent_size": float(errors.mean() / size * 100),
                "Procrustes_error": partial_procrustes_distance(estimate, complete[k]),
                "Distance_truth_to_mean": partial_procrustes_distance(complete[k], consensus),
                "Distance_estimate_to_mean": partial_procrustes_distance(estimate, consensus),
            }
        )
    return pd.DataFrame(rows)


ESTIMATE_COUNT_COL = "N_estimated_landmarks"
BACKUP_KEY = "raw_before_estimation"


def apply_estimates(dataset, coordinates: pd.DataFrame, method: str) -> list[str]:
    """Put estimated coordinates into the dataset, so these specimens can go on to GPA and PCA.

    `coordinates` holds ID and the landmark columns, for instance the Coordinates_with_estimates
    table of a batch estimation. Only cells that were missing are filled: an observed value is never
    replaced, even if the table differs. Which cells were filled is kept in
    `dataset.estimated_landmarks` and in the N_estimated_landmarks column, the original sheet is
    kept for `revert_estimates`, and any alignment is cleared, since it no longer matches.
    Returns notes for the person.
    """
    ids = dataset.raw["ID"].astype(str)
    if ids.duplicated().any():
        raise ValueError(
            "Estimates are matched by ID, and this dataset has duplicate IDs. Resolve those first."
        )
    table = coordinates.copy()
    table["ID"] = table["ID"].astype(str)
    table = table.set_index("ID")
    missing_columns = [c for c in dataset.landmark_cols if c not in table.columns]
    if missing_columns:
        raise ValueError(
            f"The table lacks {len(missing_columns)} landmark columns, starting with {missing_columns[0]}."
        )

    outputs = dataset.module_outputs
    outputs.setdefault(BACKUP_KEY, dataset.raw.copy())
    raw = dataset.raw.copy()
    cols = list(dataset.landmark_cols)
    n_dim = dataset.n_dim
    filled_total = 0
    estimated = dict(dataset.estimated_landmarks)
    counts = (
        raw[ESTIMATE_COUNT_COL].tolist() if ESTIMATE_COUNT_COL in raw.columns else [0] * len(raw)
    )
    values = raw[cols].to_numpy(dtype=float, copy=True)
    for row, specimen in enumerate(ids):
        if specimen not in table.index:
            continue
        gaps = np.isnan(values[row])
        if not gaps.any():
            continue
        replacement = table.loc[specimen, cols].to_numpy(dtype=float)
        usable = gaps & np.isfinite(replacement)
        if not usable.any():
            continue
        values[row][usable] = replacement[usable]
        landmarks = sorted({int(i) // n_dim for i in np.flatnonzero(usable)})
        estimated[specimen] = sorted(set(estimated.get(specimen, [])) | set(landmarks))
        counts[row] = len(estimated[specimen])
        filled_total += len(landmarks)
    if not filled_total:
        return ["Nothing was changed: no missing landmark had an estimate."]
    raw[cols] = values
    raw[ESTIMATE_COUNT_COL] = counts
    dataset.raw = raw
    dataset.estimated_landmarks = estimated
    dataset.run_settings["Estimated_landmarks_method"] = method
    had_alignment = dataset.gpa is not None
    dataset.gpa = None
    dataset.gpa_noscale = None
    dataset.gpa_settings = {}
    notes = [
        f"{filled_total} landmarks in {len(estimated)} specimens now hold estimates ({method}). They are recorded per "
        f"specimen and in the {ESTIMATE_COUNT_COL} column."
    ]
    notes.append(
        "The alignment was cleared because it no longer matches; run it again."
        if had_alignment
        else "Run the alignment next."
    )
    return notes


def revert_estimates(dataset) -> list[str]:
    """Back to the sheet as it was before any estimates. Alignment is cleared, as it would no longer match."""
    backup = dataset.module_outputs.pop(BACKUP_KEY, None)
    if backup is None:
        return ["There are no estimates to undo."]
    dataset.raw = backup
    dataset.estimated_landmarks = {}
    dataset.run_settings.pop("Estimated_landmarks_method", None)
    dataset.gpa = None
    dataset.gpa_noscale = None
    dataset.gpa_settings = {}
    return ["The estimates were removed and the alignment cleared."]
