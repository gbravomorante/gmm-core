"""Permutation-based multivariate regression for shape data.

Implements RRPP (Residual Randomization in Permutation Procedures; Collyer &
Adams), the standard approach for testing a term in a multivariate linear
model of shape. It is used by the allometry and group-difference analyses,
which deliberately rely on permutation-based significance rather than
classical parametric tests such as Wilks' lambda.

Why RRPP rather than permuting raw rows: to test one term while controlling
for every other term in the model (for example a Group x log(centroid size)
interaction after the two main effects), the residuals of the *reduced* model
are permuted and added back to its fitted values. That builds pseudo-data
consistent with the null hypothesis for the term of interest specifically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def fit_ols(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Multivariate OLS: one design matrix X, many response columns Y.
    Returns (coefficients, fitted_values)."""
    coefficients, _residuals, _rank, _s = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ coefficients
    return coefficients, fitted


def sum_of_squares(deviation: np.ndarray) -> float:
    return float(np.sum(deviation**2))


def dummy_encode(values, drop_first: bool = True) -> tuple[np.ndarray, list[str]]:
    """0/1 dummy encoding for a categorical column, sorted levels for
    determinism. Returns (matrix (n, k or k-1), level_names)."""
    series = pd.Series(values).astype(str)
    dummies = pd.get_dummies(series, drop_first=drop_first)
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    return dummies.to_numpy(dtype=float), list(dummies.columns)


@dataclass
class RRPPResult:
    term_name: str
    ss_effect: float
    ss_total: float
    r2: float
    p_value: float
    n_permutations: int
    seed: int | None


def rrpp_test(
    Y: np.ndarray,
    X_reduced: np.ndarray,
    X_full: np.ndarray,
    term_name: str,
    n_permutations: int = 999,
    seed: int | None = None,
) -> RRPPResult:
    """Permutation test for the term(s) present in X_full but not X_reduced.

    Procedure: fit the reduced model, permute its residuals, reattach them to
    its fitted values to build a pseudo-response under the null for this term
    specifically, then refit *both* models to that pseudo-response and compare
    `SS = ||H_full y* - H_reduced y*||^2` against the same quantity computed on
    the real data.

    Refitting both models per permutation is essential, not tidiness - see the
    comment in the loop below.
    """
    n = Y.shape[0]
    if X_reduced.shape[0] != n or X_full.shape[0] != n:
        raise ValueError("X_reduced and X_full must have the same number of rows as Y.")
    rng = np.random.default_rng(seed)

    _coeffs_reduced, fitted_reduced = fit_ols(X_reduced, Y)
    residuals_reduced = Y - fitted_reduced

    _coeffs_full, fitted_full = fit_ols(X_full, Y)
    observed_ss_effect = sum_of_squares(fitted_full - fitted_reduced)

    grand_mean = Y.mean(axis=0, keepdims=True)
    ss_total = sum_of_squares(Y - grand_mean)

    # Both projections are recomputed on each pseudo-response, so the permuted
    # statistic is the same quantity as the observed one.
    #
    # Getting this wrong is subtle and was a real bug here: comparing the full
    # fit of y* against `fitted_reduced` (the reduced fit of the *original* Y)
    # adds an extra, always-positive term. Since
    #     H_red y* = fitted_reduced + H_red (permuted residuals)
    # and that second piece lies in the reduced model's column space - which is
    # orthogonal to the space the tested term lives in - the mistake inflates
    # every permuted draw by ||H_red (permuted residuals)||^2. Permuted
    # residuals are no longer orthogonal to X_reduced, so that term is zero
    # only when X_reduced is intercept-only. The result was a badly conservative
    # test for exactly the terms that need a non-trivial reduced model:
    # interactions, and group effects with a covariate partialled out. On null
    # data it returned p = 1.000 in ~98% of replicates instead of a uniform
    # p-value. Precomputing the pseudo-inverses keeps the corrected version
    # roughly as fast as the original despite the second projection.
    pinv_reduced = np.linalg.pinv(X_reduced)
    pinv_full = np.linalg.pinv(X_full)

    permuted_ss = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        order = rng.permutation(n)
        y_star = fitted_reduced + residuals_reduced[order]
        fitted_full_star = X_full @ (pinv_full @ y_star)
        fitted_reduced_star = X_reduced @ (pinv_reduced @ y_star)
        permuted_ss[i] = sum_of_squares(fitted_full_star - fitted_reduced_star)

    p_value = float((np.sum(permuted_ss >= observed_ss_effect) + 1) / (n_permutations + 1))
    r2 = observed_ss_effect / ss_total if ss_total > 0 else float("nan")

    return RRPPResult(
        term_name=term_name,
        ss_effect=observed_ss_effect,
        ss_total=ss_total,
        r2=r2,
        p_value=p_value,
        n_permutations=n_permutations,
        seed=seed,
    )


def adjust_p_values(p_values, method: str = "Benjamini-Hochberg") -> np.ndarray:
    """Multiple-comparison adjustment for a family of tests.

    Needed as soon as a module tests many things at once (e.g. one allometry
    test per principal component): at 20 PCs and alpha = 0.05 you expect one
    "significant" PC by chance alone, so the raw p-values must not be read as
    if each were the only test.

    - "Benjamini-Hochberg" controls the false discovery rate - the usual
      choice for an exploratory scan where some false positives are tolerable.
    - "Holm" controls the family-wise error rate - stricter, for when any
      false positive is costly.
    - "None" returns the raw values unchanged.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p.copy()
    if method == "None":
        return p.copy()

    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)

    if method == "Holm":
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (n - rank) * p[idx])
            adjusted[idx] = min(running, 1.0)
        return adjusted

    if method == "Benjamini-Hochberg":
        running = 1.0
        for rank in range(n - 1, -1, -1):
            idx = order[rank]
            running = min(running, n * p[idx] / (rank + 1))
            adjusted[idx] = min(running, 1.0)
        return adjusted

    raise ValueError(f"Unknown p-value adjustment method: {method!r}")


def terms_to_dataframe(terms: list[RRPPResult]) -> pd.DataFrame:
    """Shared tabular rendering for a list of RRPP term results - used by
    every module that runs one or more RRPP tests (allometry, group tests,
    and later PLS), so the exported/displayed table always has the same
    columns and formatting.
    """
    return pd.DataFrame(
        [
            {
                "Term": t.term_name,
                "SS_effect": t.ss_effect,
                "SS_total": t.ss_total,
                "R2": t.r2,
                "P_value": t.p_value,
                "N_permutations": t.n_permutations,
                "Random_seed": t.seed if t.seed is not None else "",
            }
            for t in terms
        ]
    )
