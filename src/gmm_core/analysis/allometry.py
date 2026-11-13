"""Allometry: does shape change with size?

Tests whether Procrustes shape, from which size has been removed by scaling,
still correlates with size carried as a separate variable. "GPA removed size"
and "there is no allometric effect" are different claims, and this module
tests the second. Only scaled GPA coordinates make sense as input; unscaled
coordinates still carry size, which would make a size predictor circular.

**The size variable is the caller's choice.** Centroid size is the standard
default, but a directly measured dimension (a bone's length in millimetres,
body mass) is equally valid and is often what a comparative study reports. It
also has a methodological advantage: it is not derived from the landmarks, so
predictor and response share no measurement. ``size_label`` names the
variable everywhere it appears: the model formula, the RRPP term names and the
scores column (``Log_<size_label>``).

Optionally tests whether groups follow the same allometric trajectory (a
Group x log(size) interaction) instead of assuming one common slope.
Significance is always permutation-based (RRPP, :mod:`gmm_core.regression`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from gmm_core.data_model import (
    ID_COL,
    LABEL_COL,
    add_display_label_column,
    extract_landmark_array,
    flatten_landmark_array,
)
from gmm_core.regression import (
    RRPPResult,
    adjust_p_values,
    dummy_encode,
    fit_ols,
    rrpp_test,
    terms_to_dataframe,
)

__all__ = [
    "AllometryResult",
    "run_allometry",
    "terms_to_dataframe",
    "DEFAULT_N_PERMUTATIONS",
    "PCAllometryResult",
    "run_pc_allometry",
    "DEFAULT_P_ADJUST",
]

DEFAULT_N_PERMUTATIONS = 999
DEFAULT_P_ADJUST = "Benjamini-Hochberg"


DEFAULT_SIZE_LABEL = "CentroidSize"


@dataclass
class AllometryResult:
    terms: list[RRPPResult]
    scores: pd.DataFrame  # ID, Label, metadata..., Log_<size_label>, RegressionScore[, group_col]
    model_formula: str
    n_specimens: int
    n_permutations: int
    seed: int | None
    group_col: str | None = None
    size_label: str = DEFAULT_SIZE_LABEL
    warnings: list[str] = field(default_factory=list)

    @property
    def size_column(self) -> str:
        """Name of the log-size column in `scores` - `Log_CentroidSize` for the
        default, `Log_Hipbone Length` for a measured variable, and so on."""
        return f"Log_{self.size_label}"


# A group whose log(size) span is this much smaller than the widest group's
# cannot support a comparable slope estimate. 2.0 is deliberately forgiving:
# the measured dogs case, where the test was effectively blind, sits at 3.4.
SPAN_RATIO_WARN = 2.0


SIZE_PROXY_WARN_R = 0.5


def size_variable_warnings(
    size_values: np.ndarray,
    centroid_sizes: np.ndarray | None,
    size_label: str,
) -> list[str]:
    """Does the chosen size variable actually behave like a size measure?

    Any numeric column can be handed to this analysis, and nothing about the
    column's NAME says whether it measures size. A real size measure tracks
    centroid size closely; measured on real data, genuine linear dimensions
    correlate about +0.78 to +0.84 with it while columns that merely happen to
    be numeric (a disparity score, an age) sit near +0.15 to +0.22.

    So a weak correlation is a genuine, checkable signal that the wrong column
    was chosen - which turns an allometry result into a regression of shape on
    something that is not size, reported as allometry. Warned about, never
    blocked: a legitimate size measure could correlate weakly in a very
    unusual sample, and that is the analyst's call to make.
    """
    from gmm_core.qc import scale_anomaly, scale_anomaly_message

    warnings: list[str] = []

    # A regression of shape on log(size) is meaningless when "size" is recorded
    # in two different units - the log difference is a conversion constant, not
    # biology, and it will dominate the fit. Checked on the size variable that
    # is actually being regressed, whichever column that is.
    ids = [str(i) for i in range(len(np.asarray(size_values)))]
    anomaly = scale_anomaly(ids, np.asarray(size_values, dtype=float))
    if anomaly:
        warnings.append(
            scale_anomaly_message(anomaly).replace("Centroid sizes", f"'{size_label}' values")
            + f" This allometry is a regression on log('{size_label}'), so the result "
            f"below is largely a measure of that unit difference, not of allometry."
        )

    if centroid_sizes is None:
        return warnings
    size_values = np.asarray(size_values, dtype=float)
    centroid_sizes = np.asarray(centroid_sizes, dtype=float)
    if centroid_sizes.shape != size_values.shape or size_values.size < 3:
        return warnings
    finite = np.isfinite(size_values) & np.isfinite(centroid_sizes)
    if finite.sum() < 3:
        return warnings
    if np.std(size_values[finite]) <= 0 or np.std(centroid_sizes[finite]) <= 0:
        return warnings

    r = float(np.corrcoef(size_values[finite], centroid_sizes[finite])[0, 1])
    if not np.isfinite(r):
        return warnings
    if r < SIZE_PROXY_WARN_R:
        direction = "barely tracks" if r >= 0 else "runs opposite to"
        warnings.append(
            f"'{size_label}' {direction} centroid size (r = {r:+.2f}). A size measure is "
            f"normally strongly correlated with centroid size, so this may not be one - "
            f"in which case the result is a regression of shape on '{size_label}', not an "
            f"allometry. Check that this column really records size."
        )
    return warnings


def allometry_power_warnings(
    log_size: np.ndarray,
    group_values: np.ndarray | None,
    terms: list[RRPPResult],
    size_label: str,
) -> list[str]:
    """Is a non-significant interaction uninformative rather than reassuring?

    The power of an interaction test depends almost entirely on how much SIZE
    RANGE each group spans - a group measured over a narrow range has a poorly
    estimated slope no matter how many specimens it has. So a non-significant
    interaction across groups with very unequal spans means "not detectable
    here", never "the slopes are equal".
    """
    warnings: list[str] = []
    if group_values is None or len(log_size) == 0:
        return warnings

    interaction = next(
        (t for t in terms if ":" in t.term_name or " x " in t.term_name.lower()), None
    )
    if interaction is None:
        return warnings

    spans: dict[str, float] = {}
    flat = np.asarray(log_size).flatten()
    for level in sorted(set(map(str, group_values))):
        idx = np.array([str(g) == level for g in group_values])
        values = flat[idx]
        if len(values) >= 2:
            spans[level] = float(values.max() - values.min())
    if len(spans) < 2:
        return warnings

    widest = max(spans, key=spans.get)
    narrowest = min(spans, key=spans.get)
    if spans[narrowest] <= 0:
        warnings.append(
            f"'{narrowest}' spans no range in {size_label} at all, so no slope can be "
            "estimated for it and the interaction term is meaningless."
        )
        return warnings

    ratio = spans[widest] / spans[narrowest]
    significant = interaction.p_value < 0.05
    if ratio >= SPAN_RATIO_WARN and not significant:
        as_ratio = {k: f"{np.exp(v):.2f}x" for k, v in spans.items()}
        warnings.append(
            f"The interaction is NOT significant (p = {interaction.p_value:.3f}), but the "
            f"groups span very different amounts of {size_label}: "
            + ", ".join(f"'{k}' {v}" for k, v in as_ratio.items())
            + f". '{narrowest}' covers {ratio:.1f}x less range than '{widest}', so its slope "
            "is poorly estimated and this test had little power to detect a difference. "
            "Report this as 'no detectable difference, and the design has little power to "
            "find one' - NOT as evidence that the groups share a common trajectory."
        )
    return warnings


def run_allometry(
    gpa_df: pd.DataFrame,
    landmark_cols: list[str],
    n_dim: int,
    size_values: np.ndarray,
    meta_cols: list[str],
    group_col: str | None = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int | None = None,
    id_labels: dict[str, str] | None = None,
    size_label: str = DEFAULT_SIZE_LABEL,
) -> AllometryResult:
    """Fit shape ~ log(size)[, * group_col] and test each term by RRPP.

    `size_values` is one positive number per specimen - centroid size from the
    GPA sheet by default, or any measured size variable the caller chooses
    (`size_label` names it in the formula, term names and scores column).

    Callers must pass GPA-scaled coordinates (not GPA_noscale) - size has
    already been removed as a shape variable there, so testing whether shape
    still covaries with size is a meaningful, non-circular question.
    """
    shapes = extract_landmark_array(gpa_df, landmark_cols, n_dim)
    Y = flatten_landmark_array(shapes)
    n = Y.shape[0]

    size_values = np.asarray(size_values, dtype=float)
    if size_values.shape[0] != n:
        raise ValueError(f"'{size_label}' must have one value per specimen in gpa_df.")
    if np.any(~np.isfinite(size_values)):
        raise ValueError(f"'{size_label}' has missing or non-numeric values for some specimens.")
    if np.any(size_values <= 0):
        raise ValueError(
            f"'{size_label}' must be a positive number for every specimen to take a log "
            "(allometry is fitted on log size)."
        )

    size_term = f"log({size_label})"
    log_size = np.log(size_values).reshape(n, 1)
    intercept = np.ones((n, 1))
    X_intercept = intercept
    X_common = np.hstack([intercept, log_size])

    terms: list[RRPPResult] = [rrpp_test(Y, X_intercept, X_common, size_term, n_permutations, seed)]

    model_formula = f"Shape ~ {size_term}"
    x_for_score = X_common

    group_values = None
    if group_col is not None:
        group_values = gpa_df[group_col].astype(str).to_numpy()
        group_dummies, _level_names = dummy_encode(group_values, drop_first=True)
        if group_dummies.shape[1] == 0:
            raise ValueError(
                f"Column '{group_col}' has fewer than 2 levels; cannot test group differences."
            )

        x_group_main = np.hstack([X_common, group_dummies])
        terms.append(
            rrpp_test(Y, X_common, x_group_main, f"Group ({group_col})", n_permutations, seed)
        )

        interaction = group_dummies * log_size  # broadcasts across each dummy column
        x_interaction = np.hstack([x_group_main, interaction])
        terms.append(
            rrpp_test(
                Y,
                x_group_main,
                x_interaction,
                f"{size_term} x Group ({group_col})",
                n_permutations,
                seed,
            )
        )

        model_formula = f"Shape ~ {size_term} * {group_col}"
        x_for_score = x_group_main

    # Drake & Klingenberg (2008) regression score: project each specimen's
    # deviation from the grand mean shape onto the (unit-normalized) log-size
    # coefficient direction from the fitted model - a univariate proxy for
    # "how allometric" each specimen's shape is.
    coefficients, _fitted = fit_ols(x_for_score, Y)
    size_coefficient = coefficients[1, :]  # row 1 = the log-size column in every model above
    norm = np.linalg.norm(size_coefficient)
    if norm <= 1e-12:
        raise ValueError(
            f"The fitted {size_term} coefficient is degenerate (zero norm) - check the input data."
        )
    size_axis = size_coefficient / norm
    grand_mean = Y.mean(axis=0, keepdims=True)
    regression_scores = (Y - grand_mean) @ size_axis

    gpa_df_with_label = add_display_label_column(gpa_df, id_labels or {})
    present_meta = [c for c in meta_cols if c in gpa_df.columns]
    scores = gpa_df_with_label[[ID_COL, LABEL_COL] + present_meta].copy().reset_index(drop=True)
    scores[f"Log_{size_label}"] = log_size.flatten()
    scores["RegressionScore"] = regression_scores

    return AllometryResult(
        warnings=(
            size_variable_warnings(
                size_values,
                # Coerced, not cast: a reused external GPA sheet has been seen
                # with text in this column, and a guardrail must not be the
                # thing that crashes the analysis.
                pd.to_numeric(gpa_df["CentroidSize"], errors="coerce").to_numpy(float)
                if "CentroidSize" in gpa_df.columns
                else None,
                size_label,
            )
            + allometry_power_warnings(log_size, group_values, terms, size_label)
        ),
        terms=terms,
        scores=scores,
        model_formula=model_formula,
        n_specimens=n,
        n_permutations=n_permutations,
        seed=seed,
        group_col=group_col,
        size_label=size_label,
    )


@dataclass
class PCAllometryResult:
    """One allometry test per principal component."""

    #: PC, Percent_variance, Slope, R2, P_value, P_adjusted[, group/interaction cols]

    table: pd.DataFrame
    scores: pd.DataFrame  # ID, Label, metadata..., Log_<size>, PC1..PCn
    size_label: str
    n_specimens: int
    n_pcs: int
    n_permutations: int
    seed: int | None
    p_adjust_method: str
    group_col: str | None = None
    total_variance_covered: float = 0.0

    @property
    def size_column(self) -> str:
        return f"Log_{self.size_label}"


def run_pc_allometry(
    gpa_df: pd.DataFrame,
    landmark_cols: list[str],
    n_dim: int,
    size_values: np.ndarray,
    meta_cols: list[str],
    n_pcs: int = 10,
    group_col: str | None = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int | None = None,
    id_labels: dict[str, str] | None = None,
    size_label: str = DEFAULT_SIZE_LABEL,
    p_adjust_method: str = DEFAULT_P_ADJUST,
) -> PCAllometryResult:
    """Test each principal component of shape against log(size) separately.

    The whole-configuration test (`run_allometry`) answers "does shape covary
    with size"; this answers the follow-up "*which axes of shape variation*
    carry that size signal" - it is common for PC1 to be largely size-driven
    while another PC is not, or vice versa.

    The PCA is fitted **once, on every included specimen together**, never
    per group, so the PCs are one shared basis
    and the group comparison below happens within it.

    Each PC score is then a univariate response passed through exactly the
    same RRPP machinery as the multivariate test, so the two are directly
    comparable. Because that is one test per PC, `P_adjusted` carries a
    multiple-comparison correction; report it rather than the raw p-value.
    """
    shapes = extract_landmark_array(gpa_df, landmark_cols, n_dim)
    Y_full = flatten_landmark_array(shapes)
    n = Y_full.shape[0]

    size_values = np.asarray(size_values, dtype=float)
    if size_values.shape[0] != n:
        raise ValueError(f"'{size_label}' must have one value per specimen in gpa_df.")
    if np.any(~np.isfinite(size_values)):
        raise ValueError(f"'{size_label}' has missing or non-numeric values for some specimens.")
    if np.any(size_values <= 0):
        raise ValueError(
            f"'{size_label}' must be a positive number for every specimen to take a log "
            "(allometry is fitted on log size)."
        )
    if n_pcs < 1:
        raise ValueError("Ask for at least 1 principal component.")

    max_components = min(n - 1, Y_full.shape[1])
    if max_components < 1:
        raise ValueError("Not enough specimens to fit any principal components.")
    n_pcs_used = min(n_pcs, max_components)

    pca = PCA(n_components=n_pcs_used)
    pc_scores = pca.fit_transform(Y_full)
    percent_variance = pca.explained_variance_ratio_ * 100.0

    size_term = f"log({size_label})"
    log_size = np.log(size_values).reshape(n, 1)
    intercept = np.ones((n, 1))
    X_common = np.hstack([intercept, log_size])

    group_dummies = None
    if group_col is not None:
        group_dummies, _levels = dummy_encode(
            gpa_df[group_col].astype(str).to_numpy(), drop_first=True
        )
        if group_dummies.shape[1] == 0:
            raise ValueError(
                f"Column '{group_col}' has fewer than 2 levels; cannot test group differences."
            )

    rows = []
    for k in range(n_pcs_used):
        y = pc_scores[:, [k]]
        size_result = rrpp_test(y, intercept, X_common, size_term, n_permutations, seed)
        # Univariate slope of this PC on log size - the sign is meaningful
        # relative to the PC's own (arbitrary) direction, so report both.
        slope = float(fit_ols(X_common, y)[0][1, 0])

        row = {
            "PC": f"PC{k + 1}",
            "Percent_variance": float(percent_variance[k]),
            "Slope_on_log_size": slope,
            "R2": size_result.r2,
            "P_value": size_result.p_value,
        }

        if group_dummies is not None:
            x_group_main = np.hstack([X_common, group_dummies])
            group_result = rrpp_test(
                y, X_common, x_group_main, f"Group ({group_col})", n_permutations, seed
            )
            x_interaction = np.hstack([x_group_main, group_dummies * log_size])
            interaction_result = rrpp_test(
                y,
                x_group_main,
                x_interaction,
                f"{size_term} x Group",
                n_permutations,
                seed,
            )
            row["Group_R2"] = group_result.r2
            row["Group_P_value"] = group_result.p_value
            row["Interaction_R2"] = interaction_result.r2
            row["Interaction_P_value"] = interaction_result.p_value

        rows.append(row)

    table = pd.DataFrame(rows)
    table["P_adjusted"] = adjust_p_values(table["P_value"].to_numpy(), p_adjust_method)
    if group_dummies is not None:
        table["Group_P_adjusted"] = adjust_p_values(
            table["Group_P_value"].to_numpy(), p_adjust_method
        )
        table["Interaction_P_adjusted"] = adjust_p_values(
            table["Interaction_P_value"].to_numpy(), p_adjust_method
        )
        table = table[
            [
                "PC",
                "Percent_variance",
                "Slope_on_log_size",
                "R2",
                "P_value",
                "P_adjusted",
                "Group_R2",
                "Group_P_value",
                "Group_P_adjusted",
                "Interaction_R2",
                "Interaction_P_value",
                "Interaction_P_adjusted",
            ]
        ]
    else:
        table = table[
            ["PC", "Percent_variance", "Slope_on_log_size", "R2", "P_value", "P_adjusted"]
        ]

    gpa_df_with_label = add_display_label_column(gpa_df, id_labels or {})
    present_meta = [c for c in meta_cols if c in gpa_df.columns]
    scores = gpa_df_with_label[[ID_COL, LABEL_COL] + present_meta].copy().reset_index(drop=True)
    scores[f"Log_{size_label}"] = log_size.flatten()
    for k in range(n_pcs_used):
        scores[f"PC{k + 1}"] = pc_scores[:, k]

    return PCAllometryResult(
        table=table,
        scores=scores,
        size_label=size_label,
        n_specimens=n,
        n_pcs=n_pcs_used,
        n_permutations=n_permutations,
        seed=seed,
        p_adjust_method=p_adjust_method,
        group_col=group_col,
        total_variance_covered=float(percent_variance.sum()),
    )
