"""Group differences in shape.

A permutation-based omnibus test (RRPP) plus pairwise comparisons between
groups, with optional covariates (age, centroid size, observer) partialled
out first. Classical parametric MANOVA is avoided on purpose, given the small
samples typical of morphometric datasets.

Requires scaled GPA coordinates, for the same reason as
:mod:`gmm_core.analysis.allometry`: a shape difference between groups should
not be a side effect of a size difference between them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gmm_core.data_model import (
    ID_COL,
    LABEL_COL,
    add_display_label_column,
    extract_landmark_array,
    flatten_landmark_array,
    numeric_coercible_columns,
)
from gmm_core.regression import RRPPResult, dummy_encode, rrpp_test

DEFAULT_N_PERMUTATIONS = 999


@dataclass
class GroupTestResult:
    omnibus: RRPPResult
    pairwise: list[RRPPResult]
    scores: pd.DataFrame  # ID, Label, metadata..., Distance_to_group_mean
    model_formula: str
    n_specimens: int
    n_excluded_missing_covariates: int
    n_permutations: int
    seed: int | None
    group_col: str
    covariate_cols: list[str]
    warnings: list[str] = field(default_factory=list)


# --- guardrails -------------------------------------------------------------
# Thresholds are deliberately loose: these exist to make an analyst look, not to
# gatekeep. Every one of them warns and proceeds.
MIN_GROUP_N = 3
IMBALANCE_RATIO = 10.0
SIZE_RATIO_WARN = 1.10  # >10% difference in mean centroid size between groups
CRAMERS_V_WARN = 0.5  # entanglement with another categorical column
MIN_COLUMN_COVERAGE = 0.5
OBSERVER_NAME_RE = re.compile(
    r"observ|student|digiti|operator|rater|collector|technician|measur|scanner|batch|lab\b|session|protocol|operator",
    re.I,
)


def _nesting(groups: pd.Series, values: pd.Series) -> str:
    """How a column sits against the grouping, on specimens with both. "coarser": each group has one
    value, as Genus for Species. "finer": each value occurs in one group, or a one-to-one match. "" otherwise."""
    frame = pd.DataFrame({"g": groups, "v": values}).dropna()
    if frame.empty:
        return ""
    one_per_group = (frame.groupby("g")["v"].nunique() <= 1).all()
    one_per_value = (frame.groupby("v")["g"].nunique() <= 1).all()
    if one_per_group and one_per_value:
        return "finer"  # a one-to-one match can restate the grouping or be a perfect confound
    if one_per_group:
        return "coarser"
    if one_per_value:
        return "finer"
    return ""


def _cramers_v(a: pd.Series, b: pd.Series, corrected: bool = False) -> float:
    """Association between two categorical columns, 0 (independent) to 1 (identical).
    `corrected` applies the Bergsma (2013) bias correction, which matters for sparse tables."""
    table = pd.crosstab(a, b)
    if table.size == 0 or min(table.shape) < 2:
        return 0.0
    total = float(table.values.sum())
    if total <= 0:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((table.values - expected) ** 2 / expected))
    k = min(table.shape) - 1
    if corrected and total > 1:
        r, c = table.shape
        phi2 = max(0.0, chi2 / total - (r - 1) * (c - 1) / (total - 1))
        rc, cc = r - (r - 1) ** 2 / (total - 1), c - (c - 1) ** 2 / (total - 1)
        denominator = min(rc - 1, cc - 1)
        return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0
    return float(np.sqrt(chi2 / (total * k))) if k > 0 else 0.0


def group_test_warnings(
    df: pd.DataFrame,
    group_col: str,
    covariate_cols: Sequence[str],
    meta_cols: Sequence[str] | None = None,
) -> list[str]:
    """What about this comparison should give the analyst pause?

    Four checks, in the order they mislead people:

    1. **Size differences between groups.** The commonest confound in
       morphometrics - a group difference that is really an allometric
       consequence of one group being bigger.
    2. **Entanglement with another recorded variable**, above all the observer.
       On real data this is what turns "populations differ" into "the people who
       measured them differ".
    3. Groups too small to describe.
    4. Severe imbalance.
    """
    warnings: list[str] = []
    if group_col not in df.columns:
        return warnings

    groups = df[group_col].astype("string")
    counts = groups.value_counts()

    # 1. size
    if "CentroidSize" in df.columns:
        sizes = pd.to_numeric(df["CentroidSize"], errors="coerce")

        # Before saying anything about size, check the sizes are believable. A
        # "19.85x size difference" between groups is far more likely a unit
        # conversion error than biology, and advising a size covariate would then
        # be actively wrong. A units problem must be reported as a units problem.
        from gmm_core.qc import scale_anomaly, scale_anomaly_message

        ids = (
            df[ID_COL].astype(str).tolist()
            if ID_COL in df.columns
            else [str(i) for i in range(len(df))]
        )
        anomaly = scale_anomaly(ids, sizes.to_numpy(dtype=float))
        if anomaly:
            warnings.append(
                scale_anomaly_message(anomaly) + " Until that is resolved, treat every "
                "size-based statement below as unreliable, and do NOT add centroid size "
                "as a covariate to 'control for' it - that would be adjusting for an error."
            )

        else:
            from gmm_core.qc import size_gap, size_gap_message

            gap = size_gap(ids, sizes.to_numpy(dtype=float))
            if gap:
                warnings.append(
                    size_gap_message(gap) + " Group size comparisons below include them."
                )

        means = sizes.groupby(groups).mean().dropna()
        if not anomaly and len(means) >= 2 and means.min() > 0:
            ratio = float(means.max() / means.min())
            if ratio >= SIZE_RATIO_WARN:
                biggest, smallest = means.idxmax(), means.idxmin()
                warnings.append(
                    f"The groups differ in SIZE: mean centroid size is "
                    f"{means.max():.1f} for '{biggest}' against {means.min():.1f} for "
                    f"'{smallest}' ({ratio:.2f}x). Shape differences between groups may "
                    f"be an allometric consequence of that size difference rather than a "
                    f"difference in shape as such. "
                    + (
                        "CentroidSize is included as a covariate here; compare with the "
                        "result without it."
                        if "CentroidSize" in covariate_cols
                        else "Consider adding 'CentroidSize' as a covariate, and compare "
                        "the two results."
                    )
                )

    # 2. entanglement with another categorical variable
    candidates = [
        c
        for c in (meta_cols or df.columns)
        if c != group_col and c not in covariate_cols and c in df.columns
    ]
    finer_columns: list[str] = []
    for col in candidates:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        values = series.astype("string")
        n_levels = values.nunique(dropna=True)
        if n_levels < 2 or n_levels > 20 or n_levels >= len(df) * 0.5:
            continue
        # A column filled in for few specimens, such as notes, says nothing about the design.
        if values.notna().mean() < MIN_COLUMN_COVERAGE:
            continue
        relation = _nesting(groups, values)
        # Genus for a Species grouping restates it: the same fact, and no alternative explanation.
        if relation == "coarser":
            continue
        # A finer column (population within species) has the same structure as an observer who
        # worked on one group only, and only the name can tell them apart. An observer always warns;
        # the others are listed once, so a real warning is not buried under restatements.
        if relation == "finer" and not OBSERVER_NAME_RE.search(str(col)):
            finer_columns.append(str(col))
            continue
        v = _cramers_v(groups, values)
        # Few specimens per cell inflate V, so a chance pattern in small groups can look strong.
        if v >= CRAMERS_V_WARN and _cramers_v(groups, values, corrected=True) >= CRAMERS_V_WARN:
            table = pd.crosstab(groups, values)
            dominated = float(table.max(axis=1).sum() / table.values.sum())
            warnings.append(
                f"'{group_col}' is strongly entangled with '{col}' "
                f"(Cramer's V {v:.2f}; {dominated:.0%} of specimens fall in a "
                f"'{group_col}' level dominated by one '{col}' value). The two cannot be "
                f"separated by this design, so a difference attributed to '{group_col}' "
                f"may belong to '{col}'. If '{col}' records who digitised each specimen, "
                f"this is inter-observer error, not biology. Consider adding '{col}' as a "
                f"covariate to see how much of the effect survives."
            )

    if finer_columns:
        warnings.append(
            f"Nested within '{group_col}', so not separable from it: {', '.join(finer_columns)}. "
            "If any of these records who or how specimens were measured, treat it as a possible "
            "explanation of the group difference."
        )

    # 3. small groups
    small = counts[counts < MIN_GROUP_N]
    if len(small):
        listed = ", ".join(f"'{k}' (n={v})" for k, v in small.items())
        warnings.append(
            f"{len(small)} group(s) have fewer than {MIN_GROUP_N} specimens: {listed}. "
            "A mean shape from so few specimens is not a reliable description of the group."
        )

    # 4. imbalance
    if len(counts) >= 2 and counts.min() > 0:
        ratio = float(counts.max() / counts.min())
        if ratio >= IMBALANCE_RATIO:
            warnings.append(
                f"The groups are severely unbalanced ({counts.max()} in "
                f"'{counts.idxmax()}' against {counts.min()} in '{counts.idxmin()}', "
                f"{ratio:.0f}:1). The test is dominated by the larger group and the "
                "smaller one's mean is poorly estimated."
            )

    return warnings


def _build_covariate_matrix(
    df: pd.DataFrame, covariate_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X_covariates_incl_intercept, row_mask): row_mask flags rows
    with usable (non-missing) values for every requested covariate. A
    numeric-coercible covariate is used as-is (numeric); anything else is
    dummy-encoded as categorical.
    """
    n = len(df)
    row_mask = np.ones(n, dtype=bool)
    columns = [np.ones((n, 1))]

    for col in covariate_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or numeric_coercible_columns(df, [col]) == [col]:
            values = pd.to_numeric(df[col], errors="coerce")
            row_mask &= values.notna().to_numpy()
            columns.append(values.to_numpy(dtype=float).reshape(-1, 1))
        else:
            series = df[col].astype(str)
            row_mask &= df[col].notna().to_numpy()
            dummies, _ = dummy_encode(series, drop_first=True)
            columns.append(dummies)

    return np.hstack(columns), row_mask


def run_group_test(
    gpa_df: pd.DataFrame,
    landmark_cols: list[str],
    n_dim: int,
    group_col: str,
    meta_cols: list[str],
    covariate_cols: list[str] | None = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int | None = None,
    id_labels: dict[str, str] | None = None,
) -> GroupTestResult:
    covariate_cols = covariate_cols or []

    shapes_all = extract_landmark_array(gpa_df, landmark_cols, n_dim)
    Y_all = flatten_landmark_array(shapes_all)
    X_covariates_all, row_mask = _build_covariate_matrix(gpa_df, covariate_cols)
    n_excluded = int((~row_mask).sum())

    gpa_used = gpa_df.loc[row_mask].reset_index(drop=True)
    Y = Y_all[row_mask]
    X_reduced = X_covariates_all[row_mask]
    n = Y.shape[0]

    if n < 4:
        raise ValueError(
            "Fewer than 4 specimens have complete data for the chosen covariates - cannot test group differences."
        )

    group_values = gpa_used[group_col].astype(str).to_numpy()
    levels = sorted(set(group_values))
    if len(levels) < 2:
        raise ValueError(
            f"Column '{group_col}' has fewer than 2 levels among specimens with complete "
            "covariate data; cannot test group differences."
        )

    group_dummies, _level_names = dummy_encode(group_values, drop_first=True)
    X_full = np.hstack([X_reduced, group_dummies])

    covariate_note = f" + {' + '.join(covariate_cols)}" if covariate_cols else ""
    model_formula = f"Shape ~{covariate_note} + Group ({group_col})".replace("~ +", "~")

    omnibus = rrpp_test(Y, X_reduced, X_full, f"Group ({group_col})", n_permutations, seed)

    pairwise: list[RRPPResult] = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            g1, g2 = levels[i], levels[j]
            pair_mask = np.isin(group_values, [g1, g2])
            y_sub = Y[pair_mask]
            x_reduced_sub = X_reduced[pair_mask]
            sub_group_values = group_values[pair_mask]
            dummy_sub, _ = dummy_encode(sub_group_values, drop_first=True)
            x_full_sub = np.hstack([x_reduced_sub, dummy_sub])
            pairwise.append(
                rrpp_test(y_sub, x_reduced_sub, x_full_sub, f"{g1} vs {g2}", n_permutations, seed)
            )

    # Descriptive only, not itself the significance test above: each
    # specimen's Procrustes distance to its own group's mean shape (already
    # GPA-aligned unit-size coordinates, so a plain Euclidean norm between
    # configurations is the standard partial Procrustes distance).
    shapes = extract_landmark_array(gpa_used, landmark_cols, n_dim)
    distances = np.empty(n, dtype=float)
    for level in levels:
        idx = np.where(group_values == level)[0]
        mean_shape = shapes[idx].mean(axis=0)
        for i in idx:
            distances[i] = float(np.linalg.norm(shapes[i] - mean_shape))

    gpa_with_label = add_display_label_column(gpa_used, id_labels or {})
    present_meta = [c for c in meta_cols if c in gpa_used.columns]
    scores = gpa_with_label[[ID_COL, LABEL_COL] + present_meta].copy().reset_index(drop=True)
    scores["Distance_to_group_mean"] = distances

    return GroupTestResult(
        warnings=group_test_warnings(gpa_used, group_col, covariate_cols, meta_cols),
        omnibus=omnibus,
        pairwise=pairwise,
        scores=scores,
        model_formula=model_formula,
        n_specimens=n,
        n_excluded_missing_covariates=n_excluded,
        n_permutations=n_permutations,
        seed=seed,
        group_col=group_col,
        covariate_cols=covariate_cols,
    )
