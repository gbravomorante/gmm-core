"""Principal component analysis of aligned shape coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from gmm_core.data_model import (
    ID_COL,
    LABEL_COL,
    add_display_label_column,
    add_virtual_metadata_columns,
    extract_landmark_array,
    flatten_landmark_array,
)

# Numeric columns from the GPA sheet worth carrying into `scores` even though
# they aren't metadata, so they're selectable for gradient (continuous)
# coloring alongside real metadata like Age.
_EXTRA_NUMERIC_SCORE_COLS = ["CentroidSize", "ProcrustesDistance_to_consensus"]


@dataclass
class PCAResult:
    model: PCA
    scores: pd.DataFrame  # ID + Label + metadata columns (+ AgeClass, CentroidSize, ...) + PC1..PCn
    loadings: pd.DataFrame  # Landmark_column + PC1..PCn
    #: PC, Explained_variance, Explained_variance_ratio, Cumulative_explained_variance_ratio
    variance: pd.DataFrame


def fit_pca(
    gpa_df: pd.DataFrame,
    landmark_cols: list[str],
    n_dim: int,
    meta_cols: list[str],
    n_components: int | None = None,
    id_labels: dict[str, str] | None = None,
    fit_mask: np.ndarray | None = None,
) -> PCAResult:
    """Fit one PCA on flattened GPA coordinates for the whole sample given.

    Callers are responsible for making sure `gpa_df` already contains every
    group that needs to be compared in the same morphospace - fitting
    separate PCAs per group and comparing scores across them is not valid.

    With `fit_mask`, the axes come from those specimens only and the rest are projected onto them,
    in the same morphospace. For specimens holding estimates, this stops the estimates from shaping
    the axes they are then read on. The scores gain a Used_to_fit_axes column.
    """
    shapes = extract_landmark_array(gpa_df, landmark_cols, n_dim)
    x = flatten_landmark_array(shapes)

    fit_x = x if fit_mask is None else x[np.asarray(fit_mask, dtype=bool)]
    max_components = min(fit_x.shape[0] - 1, fit_x.shape[1])
    if max_components < 1:
        raise ValueError("Not enough specimens to fit a PCA (need at least 2 complete specimens).")
    if n_components is None:
        n_components = max_components
    else:
        n_components = max(1, min(n_components, max_components))

    model = PCA(n_components=n_components)
    model.fit(fit_x)
    scores_arr = model.transform(x)

    gpa_df = add_virtual_metadata_columns(gpa_df)
    gpa_df = add_display_label_column(gpa_df, id_labels or {})
    present_meta = [c for c in meta_cols if c in gpa_df.columns]
    present_numeric = [c for c in _EXTRA_NUMERIC_SCORE_COLS if c in gpa_df.columns]
    scores = (
        gpa_df[[ID_COL, LABEL_COL] + present_meta + present_numeric].copy().reset_index(drop=True)
    )
    if fit_mask is not None:
        scores["Used_to_fit_axes"] = [
            "Yes" if v else "No, projected" for v in np.asarray(fit_mask, dtype=bool)
        ]
    for i in range(scores_arr.shape[1]):
        scores[f"PC{i + 1}"] = scores_arr[:, i]

    loadings = pd.DataFrame(
        model.components_.T,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    loadings.insert(0, "Landmark_column", landmark_cols)

    variance = pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(n_components)],
            "Explained_variance": model.explained_variance_,
            "Explained_variance_ratio": model.explained_variance_ratio_,
            "Cumulative_explained_variance_ratio": np.cumsum(model.explained_variance_ratio_),
        }
    )

    return PCAResult(model=model, scores=scores, loadings=loadings, variance=variance)


def pc_axis_label(variance: pd.DataFrame, pc_number: int) -> str:
    row = variance.loc[variance["PC"] == f"PC{pc_number}"]
    if row.empty:
        return f"PC{pc_number}"
    ratio = float(row["Explained_variance_ratio"].iloc[0])
    return f"PC{pc_number} ({ratio * 100:.1f}%)"
