import numpy as np
import pandas as pd

from gmm_core.analysis.pca_explore import fit_pca
from gmm_core.data_model import AGE_CLASS_COL, ID_COL, LABEL_COL, landmark_columns


def _sample_gpa_df(n=8, n_lm=4):
    rng = np.random.default_rng(0)
    cols = landmark_columns(n_lm, n_dim=3)
    data = {
        "ID": [f"s{i}" for i in range(n)],
        "Sex": ["M", "F"] * (n // 2),
        "Age": ["subadult", 40, 12, "Adult", 50, 8, 30, "subadult"][:n],
        "CentroidSize": rng.normal(loc=10, scale=1, size=n),
    }
    for col in cols:
        data[col] = rng.normal(size=n)
    return pd.DataFrame(data), cols


def test_fit_pca_includes_age_class_in_scores():
    gpa_df, cols = _sample_gpa_df()
    result = fit_pca(gpa_df, cols, n_dim=3, meta_cols=["Sex", "Age", AGE_CLASS_COL])
    assert AGE_CLASS_COL in result.scores.columns
    assert set(result.scores[AGE_CLASS_COL].unique()) <= {"Adult", "Subadult", "Unknown"}


def test_fit_pca_carries_centroid_size_for_gradient_coloring():
    gpa_df, cols = _sample_gpa_df()
    result = fit_pca(gpa_df, cols, n_dim=3, meta_cols=["Sex"])
    assert "CentroidSize" in result.scores.columns
    assert np.allclose(result.scores["CentroidSize"].to_numpy(), gpa_df["CentroidSize"].to_numpy())


def test_fit_pca_does_not_duplicate_columns_when_age_class_already_present():
    gpa_df, cols = _sample_gpa_df()
    result = fit_pca(gpa_df, cols, n_dim=3, meta_cols=["Sex", AGE_CLASS_COL])
    assert list(result.scores.columns).count(AGE_CLASS_COL) == 1


def test_fit_pca_always_includes_label_column_defaulting_to_id():
    gpa_df, cols = _sample_gpa_df()
    result = fit_pca(gpa_df, cols, n_dim=3, meta_cols=["Sex"])
    assert LABEL_COL in result.scores.columns
    assert list(result.scores[LABEL_COL]) == list(result.scores[ID_COL])


def test_fit_pca_applies_id_labels_override():
    gpa_df, cols = _sample_gpa_df()
    result = fit_pca(gpa_df, cols, n_dim=3, meta_cols=["Sex"], id_labels={"s0": "First Specimen"})
    row = result.scores.loc[result.scores[ID_COL] == "s0"].iloc[0]
    assert row[LABEL_COL] == "First Specimen"
    other_row = result.scores.loc[result.scores[ID_COL] == "s1"].iloc[0]
    assert other_row[LABEL_COL] == "s1"
