"""gmm-core: transparent, tested geometric morphometrics in Python.

The most common entry points are re-exported here::

    from gmm_core import import_tabular_files, generalized_procrustes, fit_pca

Everything else lives in the submodules:

* :mod:`gmm_core.io`        importers for Excel, CSV/TXT/TPS, Morphologika, Amira and 3D Slicer
* :mod:`gmm_core.gpa`       generalized Procrustes analysis
* :mod:`gmm_core.qc`        post-alignment quality control
* :mod:`gmm_core.analysis`  PCA, allometry, group tests and measurement error (RRPP)
* :mod:`gmm_core.missing_data`, :mod:`gmm_core.reconstruction`  incomplete specimens
* :mod:`gmm_core.mesh_warp` thin-plate-spline warping of a reference mesh
"""

from gmm_core.analysis.allometry import run_allometry
from gmm_core.analysis.group_tests import run_group_test
from gmm_core.analysis.measurement_error import run_measurement_error_analysis
from gmm_core.analysis.pca_explore import fit_pca
from gmm_core.data_model import (
    extract_landmark_array,
    flatten_landmark_array,
    landmark_columns,
)
from gmm_core.gpa import (
    centroid_size,
    generalized_procrustes,
    pairwise_procrustes_distances,
)
from gmm_core.io.amira_landmarkascii import import_amira_files
from gmm_core.io.excel import import_excel_sheet
from gmm_core.io.morphologika import import_morphologika_files
from gmm_core.io.slicer_json import import_slicer_any
from gmm_core.io.tabular import TabularOptions, import_tabular_files
from gmm_core.mesh_warp import fit_similarity_transform, warp_points
from gmm_core.reconstruction import reconstruct_missing_landmarks

__version__ = "0.1.0"

__all__ = [
    "TabularOptions",
    "__version__",
    "centroid_size",
    "extract_landmark_array",
    "fit_pca",
    "fit_similarity_transform",
    "flatten_landmark_array",
    "generalized_procrustes",
    "import_amira_files",
    "import_excel_sheet",
    "import_morphologika_files",
    "import_slicer_any",
    "import_tabular_files",
    "landmark_columns",
    "pairwise_procrustes_distances",
    "reconstruct_missing_landmarks",
    "run_allometry",
    "run_group_test",
    "run_measurement_error_analysis",
    "warp_points",
]
