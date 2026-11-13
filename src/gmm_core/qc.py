"""Post-alignment quality-control diagnostics.

Everything here *flags* candidates for review; nothing removes a specimen,
because an outlier is not necessarily an error. Output rows use the same
schema as :mod:`gmm_core.validation`, so all checks end up in one table.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gmm_core.validation import STATUS_WARNING, VALIDATION_COLUMNS, empty_validation_table

DEFAULT_ROBUST_Z_THRESHOLD = 3.5


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD-based z-score; more resistant to the outliers it's meant to
    detect than a mean/SD z-score would be."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad <= 1e-12:
        return np.zeros_like(values)
    return 0.6744897501960817 * (values - median) / mad


def centroid_size_outliers(
    ids: Sequence[str],
    centroid_sizes: np.ndarray,
    z_threshold: float = DEFAULT_ROBUST_Z_THRESHOLD,
) -> pd.DataFrame:
    """Flag specimens whose centroid size is a robust-z outlier within this sample."""
    if len(ids) < 4:
        return empty_validation_table()
    z = _robust_z(np.asarray(centroid_sizes, dtype=float))
    rows: list[dict] = []
    median = float(np.median(centroid_sizes))
    for i, spec_id in enumerate(ids):
        if abs(z[i]) >= z_threshold:
            rows.append(
                {
                    "Row": i,
                    "ID": spec_id,
                    "Status": STATUS_WARNING,
                    "Reason": (
                        f"Centroid size ({centroid_sizes[i]:.4g}) is an outlier within this sample "
                        f"(robust z={z[i]:.2f}, sample median={median:.4g})."
                    ),
                    "Missing_landmarks": "",
                    "Suggested_action": "Inspect this specimen's raw shape; could be a measurement, scale, or digitizing issue - or a genuine biological extreme.",
                }
            )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS) if rows else empty_validation_table()


SCALE_GAP_DECADES = 2.0
SIZE_GAP_MIN_RATIO = 1.5  # the empty gap, as a ratio of sizes
SIZE_GAP_SPREADS = 6.0  # and in units of the majority's own spread
SIZE_GAP_MAX_MINORITY_SHARE = 0.05  # the set-apart group is at most this share of the sample
SIZE_GAP_MAX_MINORITY_COUNT = 3  # or this many, for small samples


def scale_anomaly(
    ids: Sequence[str],
    centroid_sizes: np.ndarray,
    gap_decades: float = SCALE_GAP_DECADES,
) -> dict:
    """Do the centroid sizes look like MIXED UNITS rather than biology?

    Landmark files carry no unit unless the format declares one, and this
    project has already been bitten by a real sample mixing nanometres with
    millimetres (see the Amira importer's `MM_PER_UNIT`). Excel has no
    declaration at all, so nothing catches it on that path.

    The decisive signature is NOT a large range - a comparative mammal sample
    genuinely spans shrew to elephant. It is an **empty gap**: biological size
    varies continuously, so a real sample cannot have two populated bands with
    two or more decades of nothing between them. A unit change of 10^3 or 10^6
    produces exactly that.

    Returns {} when nothing looks wrong. Flags, never removes - the smaller
    band may be the correct one.
    """
    sizes = np.asarray(centroid_sizes, dtype=float)
    finite = np.isfinite(sizes) & (sizes > 0)
    if finite.sum() < 4:
        return {}

    order = np.argsort(sizes[finite])
    logs = np.log10(sizes[finite][order])
    gaps = np.diff(logs)
    if not len(gaps):
        return {}
    widest = int(np.argmax(gaps))
    if gaps[widest] < gap_decades:
        return {}

    idx = np.flatnonzero(finite)[order]
    lower, upper = idx[: widest + 1], idx[widest + 1 :]
    minority, majority = (upper, lower) if len(upper) <= len(lower) else (lower, upper)
    # The offset between the two bands is the ratio of their MEDIANS, not the
    # width of the gap: each band spans a real biological range of its own, so
    # the gap understates how far apart the two scales actually are.
    factor = float(np.median(sizes[minority]) / np.median(sizes[majority]))
    if factor < 1.0:
        factor = 1.0 / factor
    return {
        "gap_decades": float(gaps[widest]),
        "factor": factor,
        "n_minority": int(len(minority)),
        "n_majority": int(len(majority)),
        "minority_ids": [str(ids[i]) for i in minority],
        "minority_is_larger": bool(minority is upper),
        "minority_range": (float(sizes[minority].min()), float(sizes[minority].max())),
        "majority_range": (float(sizes[majority].min()), float(sizes[majority].max())),
    }


def scale_anomaly_message(anomaly: dict) -> str:
    """One sentence a person or a model can act on. Empty when nothing is wrong."""
    if not anomaly:
        return ""
    side = "LARGER" if anomaly["minority_is_larger"] else "SMALLER"
    lo, hi = anomaly["minority_range"]
    mlo, mhi = anomaly["majority_range"]
    likely = ""
    for name, ratio in (
        ("mm-to-um or m-to-mm", 1e3),
        ("cm-to-um", 1e4),
        ("mm-to-nm", 1e6),
        ("m-to-um", 1e6),
    ):
        if 0.3 <= anomaly["factor"] / ratio <= 3.0:
            likely = (
                f" The two bands sit roughly {ratio:.0g}x apart, which is the "
                f"{name} conversion - worth checking first, though the exact "
                f"factor is obscured by real size variation within each band."
            )
            break
    return (
        f"Centroid sizes fall into two separated bands with NOTHING in between "
        f"({anomaly['gap_decades']:.1f} orders of magnitude of empty range): "
        f"{anomaly['n_majority']} specimens at {mlo:.4g}-{mhi:.4g} and "
        f"{anomaly['n_minority']} at {lo:.4g}-{hi:.4g}, about {anomaly['factor']:.3g}x "
        f"{side}. Biological size varies continuously and cannot leave a gap like this, "
        f"so this is very likely a UNIT MISMATCH - some files digitised in different "
        f"units.{likely} Fix the units before interpreting anything size-related: "
        f"centroid size, allometry, and any size covariate are all invalid until then."
    )


def scale_anomaly_rows(
    ids: Sequence[str],
    centroid_sizes: np.ndarray,
    gap_decades: float = SCALE_GAP_DECADES,
) -> pd.DataFrame:
    """The scale-anomaly finding as validation-table rows, one per odd specimen."""
    anomaly = scale_anomaly(ids, centroid_sizes, gap_decades)
    if not anomaly:
        return empty_validation_table()
    message = scale_anomaly_message(anomaly)
    id_to_row = {str(v): i for i, v in enumerate(ids)}
    rows = [
        {
            "Row": id_to_row.get(spec_id, -1),
            "ID": spec_id,
            "Status": STATUS_WARNING,
            "Reason": message,
            "Missing_landmarks": "",
            "Suggested_action": (
                "Check what unit this specimen's coordinates were digitised in and convert "
                "it to match the rest of the sample. Do not exclude it before checking - the "
                "other band may be the wrong one."
            ),
        }
        for spec_id in anomaly["minority_ids"]
    ]
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def procrustes_distance_outliers(
    ids: Sequence[str],
    procrustes_distances: np.ndarray,
    z_threshold: float = DEFAULT_ROBUST_Z_THRESHOLD,
) -> pd.DataFrame:
    """Flag specimens unusually far (in aligned shape space) from the GPA consensus."""
    if len(ids) < 4:
        return empty_validation_table()
    z = _robust_z(np.asarray(procrustes_distances, dtype=float))
    rows: list[dict] = []
    median = float(np.median(procrustes_distances))
    for i, spec_id in enumerate(ids):
        if z[i] >= z_threshold:  # one-sided: only "unusually far", not "unusually close"
            rows.append(
                {
                    "Row": i,
                    "ID": spec_id,
                    "Status": STATUS_WARNING,
                    "Reason": (
                        f"Procrustes distance to consensus ({procrustes_distances[i]:.4g}) is unusually "
                        f"large (robust z={z[i]:.2f}, sample median={median:.4g})."
                    ),
                    "Missing_landmarks": "",
                    "Suggested_action": "Inspect this specimen's aligned shape for a possible landmark-placement or labeling error, or note it as a genuine shape outlier.",
                }
            )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS) if rows else empty_validation_table()


def single_landmark_outliers(
    ids: Sequence[str],
    aligned_unit_shapes: np.ndarray,
    consensus: np.ndarray,
    z_threshold: float = DEFAULT_ROBUST_Z_THRESHOLD,
) -> pd.DataFrame:
    """Flag specimens where one landmark sits much farther from its consensus
    position than that same landmark does for the rest of the sample - a
    common signature of a swapped or mislabeled landmark.
    """
    n, k, _d = aligned_unit_shapes.shape
    if n < 4:
        return empty_validation_table()
    displacement = np.linalg.norm(aligned_unit_shapes - consensus[None, :, :], axis=2)  # (n, k)
    rows: list[dict] = []
    for lm in range(k):
        column = displacement[:, lm]
        z = _robust_z(column)
        median = float(np.median(column))
        for i, spec_id in enumerate(ids):
            if z[i] >= z_threshold:
                rows.append(
                    {
                        "Row": i,
                        "ID": spec_id,
                        "Status": STATUS_WARNING,
                        "Reason": (
                            f"Landmark LM{lm} sits unusually far from its consensus position "
                            f"relative to the rest of the sample (robust z={z[i]:.2f}, "
                            f"this specimen's displacement={column[i]:.4g}, sample median={median:.4g})."
                        ),
                        "Missing_landmarks": "",
                        "Suggested_action": f"Check LM{lm} on this specimen for mislabeling, a swapped landmark, or a digitizing error.",
                    }
                )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS) if rows else empty_validation_table()


# Every QC reason starts with one of these, which is how an earlier run's rows are told
# apart from the import-time validation rows they were appended to.
QC_REASON_PREFIXES = (
    "Centroid sizes fall into two separated bands",
    "Centroid size (",
    "Procrustes distance to consensus",
    "Shape is unusually far",
    "Landmark LM",
)


def without_qc_rows(validation: pd.DataFrame | None) -> pd.DataFrame | None:
    """The validation table minus rows a previous QC run added.

    Re-running an alignment appended a second full set of QC warnings to the first, so every
    flag appeared twice, then three times. Import-time rows (missing IDs, unit notes, partial
    landmarks) are kept.
    """
    if validation is None or validation.empty or "Reason" not in validation.columns:
        return validation
    reasons = validation["Reason"].astype(object).fillna("").astype(str)
    is_qc = reasons.str.startswith(QC_REASON_PREFIXES) & (validation["Status"] == STATUS_WARNING)
    return validation.loc[~is_qc].reset_index(drop=True)


# A group needs this many specimens before its own centre and spread mean anything.
MIN_GROUP_FOR_QC = 8
# Family-wise false-alarm rate for the per-landmark covariance check, over EVERY
# specimen x landmark test in the sample (chosen by the planted-swap comparison below).
DEFAULT_LANDMARK_ALPHA = 0.1

LANDMARK_METHOD_ROBUST_Z = "robust_z"
LANDMARK_METHOD_COVARIANCE = "covariance"


def _robust_mean_cov(points: np.ndarray, keep: float = 0.9):
    """Mean and covariance of `points` (n, d), down-weighting the worst-placed tenth.

    One reweighting step from a median start, with the consistency factor that makes a
    covariance fitted to the central `keep` of a normal sample unbiased again. Without the
    trimming the swapped landmarks being searched for would inflate the very spread they are
    judged against.
    """
    from scipy import stats

    d = points.shape[1]
    centre = np.median(points, axis=0)
    cov = np.cov(points.T) if len(points) > d else np.eye(d)
    dist = np.einsum(
        "ij,jk,ik->i", points - centre, np.linalg.pinv(np.atleast_2d(cov)), points - centre
    )
    kept = points[dist <= np.quantile(dist, keep)]
    if len(kept) <= d:
        kept = points
    factor = keep / stats.chi2.cdf(stats.chi2.ppf(keep, d), d + 2)
    return kept.mean(axis=0), np.atleast_2d(np.cov(kept.T)) * factor


def comparative_qc_rows(
    ids: Sequence[str],
    centroid_sizes: np.ndarray,
    aligned_unit_shapes: np.ndarray,
    groups: Sequence[object] | None = None,
    group_label: str = "group",
    z_threshold: float = DEFAULT_ROBUST_Z_THRESHOLD,
    alpha: float = DEFAULT_LANDMARK_ALPHA,
) -> pd.DataFrame:
    """Outlier checks for samples of several taxa: per-landmark covariance, within groups.

    Measured on 2026-10-28 by planting landmark swaps (each landmark exchanged with one of its
    three nearest neighbours) in 15 specimens, four times over, in two real samples:

        hipbones (699, 20 taxa):  today's check caught 58/60 swaps and also flagged 70 other
                                  specimens per run; this one caught 60/60 and flagged 40.
        dogs (146, 4 species):    today's caught 60/60 with 31 others; this one 57/60 with 19.

    So it is better on a comparative sample and a trade-off on a near-uniform one, which is
    why it is an explicit choice and not the default.

    Why it works: the old check compares each landmark's distance from its mean position
    with a robust z-score. A distance is always positive with a long right tail, and with
    26 landmarks x 699 specimens = 18,174 tests a fixed cut-off of 3.5 flags about one percent
    by chance alone. Here each landmark's 2D/3D deviation is scaled by that landmark's own
    covariance (a landmark that varies along a curve is allowed to vary along it), which gives
    a chi-square value, and the cut-off is set so the whole sample has a `alpha` chance of a
    single false alarm.

    With `groups`, every group of at least `MIN_GROUP_FOR_QC` specimens is judged against its
    own centre, size and spread - unusual *for its own species*. Smaller groups (a single
    fossil) are still checked, against the whole sample, and their rows say so. Centroid size
    and overall shape distance are judged the same way, with the usual robust z.
    """
    from scipy import stats

    ids = list(ids)
    sizes = np.asarray(centroid_sizes, dtype=float)
    shapes = np.asarray(aligned_unit_shapes, dtype=float)
    n, k, d = shapes.shape
    rows: list[dict] = []
    if n < 4:
        return empty_validation_table()

    labels = [
        None if g is None or (isinstance(g, float) and np.isnan(g)) else str(g)
        for g in (groups if groups is not None else [None] * n)
    ]
    everyone = np.arange(n)
    blocks = []  # (members, reference, scope text, note)
    if groups is None:
        blocks.append((everyone, everyone, "relative to the whole sample", ""))
    else:
        small = []
        for label in sorted({lab for lab in labels if lab is not None}):
            members = np.array([i for i in everyone if labels[i] == label])
            if len(members) >= MIN_GROUP_FOR_QC:
                blocks.append((members, members, f"for its {group_label} '{label}'", ""))
            else:
                small.extend(members.tolist())
        small.extend(i for i in everyone if labels[i] is None)
        for i in small:
            why = (
                f"; its {group_label} '{labels[i]}' has fewer than {MIN_GROUP_FOR_QC} specimens, "
                f"so it was compared with the whole sample"
                if labels[i] is not None
                else f"; it has no {group_label} value, so it was compared with the whole sample"
            )
            blocks.append((np.array([i]), everyone, "relative to the whole sample", why))

    threshold = float(stats.chi2.ppf(1.0 - alpha / (n * k), d))

    def flag(i, reason, action):
        rows.append(
            {
                "Row": int(i),
                "ID": ids[i],
                "Status": STATUS_WARNING,
                "Reason": reason,
                "Missing_landmarks": "",
                "Suggested_action": action,
            }
        )

    def z_against(values: np.ndarray, reference: np.ndarray, member_positions) -> np.ndarray:
        """Robust z of each member value against the reference values (members included)."""
        median = np.median(reference)
        mad = np.median(np.abs(reference - median))
        if mad <= 1e-12:
            return np.zeros(len(values))
        return 0.6744897501960817 * (values - median) / mad

    for members, reference, scope, note in blocks:
        ref_shapes = shapes[reference]
        mean_shape = ref_shapes.mean(axis=0)
        z_size = z_against(sizes[members], sizes[reference], None)
        dist_ref = np.linalg.norm((ref_shapes - mean_shape).reshape(len(reference), -1), axis=1)
        dist_mem = np.linalg.norm((shapes[members] - mean_shape).reshape(len(members), -1), axis=1)
        z_dist = z_against(dist_mem, dist_ref, None)
        lm_stats = []
        for lm in range(k):
            mu, cov = _robust_mean_cov(ref_shapes[:, lm, :])
            x = shapes[members, lm, :] - mu
            lm_stats.append(np.einsum("ij,jk,ik->i", x, np.linalg.pinv(cov), x))
        lm_stats = np.column_stack(lm_stats)  # (members, k)
        for pos, i in enumerate(members):
            if abs(z_size[pos]) >= z_threshold:
                flag(
                    i,
                    f"Centroid size ({sizes[i]:.4g}) is an outlier {scope} (robust "
                    f"z={z_size[pos]:.2f}){note}.",
                    "Inspect this specimen's raw shape; could be a measurement, scale, or "
                    "digitizing issue - or a genuine biological extreme.",
                )
            if z_dist[pos] >= z_threshold:
                flag(
                    i,
                    f"Shape is unusually far from the mean shape {scope} (Procrustes distance "
                    f"{dist_mem[pos]:.4g}, robust z={z_dist[pos]:.2f}){note}.",
                    "Inspect this specimen's aligned shape for a landmark-placement or labeling "
                    "error, or note it as a genuine shape outlier.",
                )
            for lm in np.flatnonzero(lm_stats[pos] >= threshold):
                flag(
                    i,
                    f"Landmark LM{lm} is unusually placed {scope} (squared Mahalanobis "
                    f"distance {lm_stats[pos, lm]:.1f}; the cut-off is {threshold:.1f}, set so "
                    f"the whole sample has a {alpha:g} chance of one false alarm){note}.",
                    f"Check LM{lm} on this specimen for mislabeling, a swapped landmark, or a "
                    f"digitizing error.",
                )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS) if rows else empty_validation_table()


def size_gap(
    ids: Sequence[str], centroid_sizes: np.ndarray, min_ratio: float = SIZE_GAP_MIN_RATIO
) -> dict:
    """A few specimens set apart from the rest by an empty gap in size.

    Finer than `scale_anomaly`, which looks for a gap of orders of magnitude. Baab's files hold
    three records about 3 times too large among otherwise continuous sizes. A biological sample
    has few gaps wider than its own spread: the gap must be several times the majority's spread,
    and the set-apart group must be a small minority. A real two-species sample has two large
    groups, and does not qualify.
    """
    sizes = np.asarray(centroid_sizes, dtype=float)
    valid = np.isfinite(sizes) & (sizes > 0)
    if valid.sum() < 8:
        return {}
    logs = np.log(sizes[valid])
    order = np.argsort(logs)
    ordered = logs[order]
    gaps = np.diff(ordered)
    cut = int(np.argmax(gaps))
    gap = float(gaps[cut])
    if np.exp(gap) < min_ratio:
        return {}
    low, high = ordered[: cut + 1], ordered[cut + 1 :]
    minority_is_larger = len(high) <= len(low)
    minority, majority = (high, low) if minority_is_larger else (low, high)
    if len(minority) > max(SIZE_GAP_MAX_MINORITY_COUNT, SIZE_GAP_MAX_MINORITY_SHARE * valid.sum()):
        return {}
    spread = float(np.median(np.abs(majority - np.median(majority))) * 1.4826)
    if spread <= 0 or gap < SIZE_GAP_SPREADS * spread:
        return {}
    kept_ids = [str(i) for i, ok in zip(ids, valid) if ok]
    side = order[cut + 1 :] if minority_is_larger else order[: cut + 1]
    return {
        "ids": [kept_ids[i] for i in side],
        "minority_is_larger": minority_is_larger,
        "ratio": float(np.exp(np.median(minority) - np.median(majority))),
        "minority_range": (float(np.exp(minority.min())), float(np.exp(minority.max()))),
        "majority_range": (float(np.exp(majority.min())), float(np.exp(majority.max()))),
    }


def size_gap_message(found: dict) -> str:
    if not found:
        return ""
    lo, hi = found["minority_range"]
    mlo, mhi = found["majority_range"]
    ratio = found["ratio"] if found["ratio"] >= 1 else 1 / found["ratio"]
    return (
        f"{len(found['ids'])} specimen(s) are set apart in size by an empty gap, about {ratio:.1f} times "
        f"{'larger' if found['minority_is_larger'] else 'smaller'} than the rest (centroid size {lo:.3g}-{hi:.3g}, "
        f"against {mlo:.3g}-{mhi:.3g}). A scale error in those files looks like this. It does not affect shape, "
        "since GPA removes scale, but any size analysis would be wrong."
    )


def size_gap_rows(ids: Sequence[str], centroid_sizes: np.ndarray) -> pd.DataFrame:
    """The size-gap finding as validation rows, one per specimen; silent when mixed units already explain it."""
    if scale_anomaly(ids, centroid_sizes):
        return empty_validation_table()
    found = size_gap(ids, centroid_sizes)
    if not found:
        return empty_validation_table()
    message = size_gap_message(found)
    id_to_row = {str(v): i for i, v in enumerate(ids)}
    rows = [
        {
            "Row": id_to_row.get(spec_id, -1),
            "ID": spec_id,
            "Status": STATUS_WARNING,
            "Reason": message,
            "Missing_landmarks": "",
            "Suggested_action": "Check the units and scale of this file. Leave it out of size analyses until resolved.",
        }
        for spec_id in found["ids"]
    ]
    return pd.DataFrame(rows, columns=empty_validation_table().columns)


def run_all_qc_checks(
    ids: Sequence[str],
    centroid_sizes: np.ndarray,
    procrustes_distances: np.ndarray,
    aligned_unit_shapes: np.ndarray,
    consensus: np.ndarray,
    z_threshold: float = DEFAULT_ROBUST_Z_THRESHOLD,
    landmark_method: str = LANDMARK_METHOD_ROBUST_Z,
    groups: Sequence[object] | None = None,
    group_label: str = "group",
) -> pd.DataFrame:
    """All post-GPA checks. `landmark_method="covariance"` (optionally with `groups`) uses
    `comparative_qc_rows`; the default keeps the original whole-sample robust-z checks."""
    from gmm_core.validation import combine_validation_tables

    # First, and always on the whole sample: a unit mix-up is not a property of a group,
    # and if the units are mixed every size-based diagnostic below describes the mistake.
    tables = [scale_anomaly_rows(ids, centroid_sizes), size_gap_rows(ids, centroid_sizes)]
    if landmark_method == LANDMARK_METHOD_COVARIANCE:
        tables.append(
            comparative_qc_rows(
                ids, centroid_sizes, aligned_unit_shapes, groups, group_label, z_threshold
            )
        )
    elif landmark_method == LANDMARK_METHOD_ROBUST_Z:
        tables += [
            centroid_size_outliers(ids, centroid_sizes, z_threshold),
            procrustes_distance_outliers(ids, procrustes_distances, z_threshold),
            single_landmark_outliers(ids, aligned_unit_shapes, consensus, z_threshold),
        ]
    else:
        raise ValueError(f"Unknown landmark_method {landmark_method!r}.")
    return combine_validation_tables(tables)
