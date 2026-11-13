"""Measurement error and repeatability for landmark data.

The Technical Error of Measurement framework (Perini et al., 2005), adapted
to shape data through Procrustes distances between repeat digitisations of
the same physical specimen.

A grouping column identifies which rows are repeats of the same object; row
IDs stay unique per digitisation (for example ``spec001_s1`` and
``spec001_s2``, both grouped as ``spec001``). Repeats are never inferred from
duplicate IDs, which are a separate data-quality problem.

An optional observer column splits the result into intra-observer TEM (pairs
from the same observer) and inter-observer TEM (pairs from different
observers); without it, all replicate pairs are pooled.

    TEM   = sqrt(sum(d_i^2) / (2 * n_pairs)), where d_i is the Procrustes
            distance between the two members of replicate pair i, in the
            same aligned space as the rest of the sample
    %TEM  = TEM / typical between-specimen Procrustes distance * 100: the
            digitising noise relative to the sample's real shape variation
    R     = 1 - TEM^2 / total variance: a shape-space analogue of the
            classical reliability coefficient (not a validated ICC)

Works on raw coordinates and fits its own GPA across all replicate rows,
because measurement error concerns the digitising itself.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gmm_core.data_model import ID_COL, extract_landmark_array, flatten_landmark_array
from gmm_core.gpa import generalized_procrustes, pairwise_procrustes_distances


@dataclass
class ReplicatePair:
    group_id: str
    id_a: str
    id_b: str
    observer_a: str | None
    observer_b: str | None
    kind: str  # "intra" | "inter" | "unspecified" (no Observer column given)
    procrustes_distance: float


@dataclass
class MeasurementErrorResult:
    group_col: str
    observer_col: str | None
    n_groups_with_replicates: int
    n_groups_skipped_single: int
    pairs: list[ReplicatePair]
    pairs_df: pd.DataFrame
    total_variance: float
    n_group_representatives: int
    overall_tem: float
    overall_percent_tem: float
    overall_r: float
    n_excluded_missing_group: int = 0
    intra_tem: float | None = None
    intra_percent_tem: float | None = None
    intra_r: float | None = None
    intra_n_pairs: int = 0
    inter_tem: float | None = None
    inter_percent_tem: float | None = None
    inter_r: float | None = None
    inter_n_pairs: int = 0


def _tem_stats(distances: np.ndarray, total_variance: float) -> tuple[float, float, float]:
    n_pairs = len(distances)
    tem = float(np.sqrt(np.sum(distances**2) / (2 * n_pairs)))
    mean_dist = float(np.mean(distances))
    percent_tem = float(tem / mean_dist * 100) if mean_dist > 0 else float("nan")
    r = float(1 - (tem**2 / total_variance)) if total_variance > 0 else float("nan")
    return tem, percent_tem, r


def run_measurement_error_analysis(
    raw_df: pd.DataFrame,
    landmark_cols: list[str],
    n_dim: int,
    group_col: str,
    observer_col: str | None = None,
) -> MeasurementErrorResult:
    if group_col not in raw_df.columns:
        raise ValueError(f"Column '{group_col}' not found.")
    if observer_col is not None and observer_col not in raw_df.columns:
        raise ValueError(f"Column '{observer_col}' not found.")
    if len(raw_df) < 4:
        raise ValueError(
            "Need at least 4 included rows (with repeat digitizations) to run this analysis."
        )

    # Rows with no specimen-grouping value at all (e.g. comparative specimens not part of any
    # repeat-digitization set) can't be linked to a replicate group - excluded from *this
    # analysis only*, with the count reported, never from dataset.raw. Checked via notna()
    # rather than relying on .astype(str) to normalize missingness: pandas' newer default
    # string dtype leaves a missing cell as a genuine float NaN even in an already-str column,
    # so grouping directly on the stringified column risks silently merging unrelated blank
    # rows into one bogus "nan" group (or, on other pandas versions, keeping each blank as its
    # own float-NaN dict key with fragile identity-based equality).
    valid_group_mask = raw_df[group_col].notna()
    n_excluded_missing_group = int((~valid_group_mask).sum())
    raw_df = raw_df.loc[valid_group_mask].reset_index(drop=True)
    if len(raw_df) < 2:
        raise ValueError(f"Fewer than 2 included rows have a value in column '{group_col}'.")

    shapes = extract_landmark_array(raw_df, landmark_cols, n_dim)
    if np.isnan(shapes).any():
        raise ValueError(
            "Some included specimens have missing landmark coordinates; exclude or "
            "estimate them before running this analysis."
        )

    gpa_result = generalized_procrustes(shapes, scale=True)
    aligned_flat = flatten_landmark_array(gpa_result.aligned)

    ids = raw_df[ID_COL].astype(str).tolist()
    groups = raw_df[group_col].astype(str).tolist()
    observer_values = raw_df[observer_col] if observer_col else None
    observers = (
        [(str(v) if pd.notna(v) else None) for v in observer_values]
        if observer_values is not None
        else [None] * len(raw_df)
    )

    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)

    pairs: list[ReplicatePair] = []
    for g, idxs in by_group.items():
        if len(idxs) < 2:
            continue
        for a, b in itertools.combinations(idxs, 2):
            dist = float(np.linalg.norm(aligned_flat[a] - aligned_flat[b]))
            obs_a, obs_b = observers[a], observers[b]
            if observer_col is None or obs_a is None or obs_b is None:
                kind = "unspecified"
            else:
                kind = "intra" if obs_a == obs_b else "inter"
            pairs.append(
                ReplicatePair(
                    group_id=g,
                    id_a=ids[a],
                    id_b=ids[b],
                    observer_a=obs_a,
                    observer_b=obs_b,
                    kind=kind,
                    procrustes_distance=dist,
                )
            )

    n_groups_with_replicates = sum(1 for idxs in by_group.values() if len(idxs) >= 2)
    n_groups_skipped_single = sum(1 for idxs in by_group.values() if len(idxs) < 2)
    if not pairs:
        raise ValueError(
            f"No group in '{group_col}' has 2 or more repeat digitizations - measurement error "
            "needs at least one specimen digitized more than once."
        )

    # "Total variance" reference: mean squared pairwise Procrustes distance among one
    # representative shape per group (that group's own mean aligned shape) - the sample's
    # typical *between-specimen* shape distance, to compare measurement error against.
    n_landmarks = len(landmark_cols) // n_dim
    representative_shapes = np.array(
        [aligned_flat[idxs].mean(axis=0) for idxs in by_group.values()]
    )
    if len(representative_shapes) >= 2:
        pw = pairwise_procrustes_distances(representative_shapes.reshape(-1, n_landmarks, n_dim))
        iu = np.triu_indices_from(pw, k=1)
        total_variance = float(np.mean(pw[iu] ** 2)) if len(iu[0]) else float("nan")
    else:
        total_variance = float("nan")

    all_distances = np.array([p.procrustes_distance for p in pairs])
    overall_tem, overall_pct, overall_r = _tem_stats(all_distances, total_variance)

    intra_distances = np.array([p.procrustes_distance for p in pairs if p.kind == "intra"])
    inter_distances = np.array([p.procrustes_distance for p in pairs if p.kind == "inter"])

    intra_tem = intra_pct = intra_r = None
    if observer_col is not None and len(intra_distances) > 0:
        intra_tem, intra_pct, intra_r = _tem_stats(intra_distances, total_variance)

    inter_tem = inter_pct = inter_r = None
    if observer_col is not None and len(inter_distances) > 0:
        inter_tem, inter_pct, inter_r = _tem_stats(inter_distances, total_variance)

    pairs_df = pd.DataFrame(
        [
            {
                "Group": p.group_id,
                "ID_A": p.id_a,
                "ID_B": p.id_b,
                "Observer_A": p.observer_a,
                "Observer_B": p.observer_b,
                "Kind": p.kind,
                "Procrustes_distance": p.procrustes_distance,
            }
            for p in pairs
        ]
    )

    return MeasurementErrorResult(
        group_col=group_col,
        observer_col=observer_col,
        n_groups_with_replicates=n_groups_with_replicates,
        n_groups_skipped_single=n_groups_skipped_single,
        pairs=pairs,
        pairs_df=pairs_df,
        total_variance=total_variance,
        n_group_representatives=len(representative_shapes),
        overall_tem=overall_tem,
        overall_percent_tem=overall_pct,
        overall_r=overall_r,
        intra_tem=intra_tem,
        intra_percent_tem=intra_pct,
        intra_r=intra_r,
        intra_n_pairs=len(intra_distances),
        inter_tem=inter_tem,
        inter_percent_tem=inter_pct,
        inter_r=inter_r,
        inter_n_pairs=len(inter_distances),
        n_excluded_missing_group=n_excluded_missing_group,
    )


def measurement_error_summary_to_dataframe(result: MeasurementErrorResult) -> pd.DataFrame:
    rows = [
        {
            "Category": "Overall",
            "N_pairs": len(result.pairs),
            "TEM": result.overall_tem,
            "Percent_TEM": result.overall_percent_tem,
            "R": result.overall_r,
        }
    ]
    if result.observer_col is not None:
        rows.append(
            {
                "Category": "Intra-observer",
                "N_pairs": result.intra_n_pairs,
                "TEM": result.intra_tem,
                "Percent_TEM": result.intra_percent_tem,
                "R": result.intra_r,
            }
        )
        rows.append(
            {
                "Category": "Inter-observer",
                "N_pairs": result.inter_n_pairs,
                "TEM": result.inter_tem,
                "Percent_TEM": result.inter_percent_tem,
                "R": result.inter_r,
            }
        )
    return pd.DataFrame(rows)
