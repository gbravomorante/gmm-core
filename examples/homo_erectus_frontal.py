"""A complete analysis on a public dataset: frontal-bone shape in Homo erectus and Homo sapiens.

Data: Baab, K. L. (2021). Reconstructing cranial evolution in an extinct hominin.
Proc. R. Soc. B 288: 20202604. Data on Dryad (doi:10.5061/dryad.5qfttdz32), CC0.
The file is downloaded once from the Zenodo mirror and checked against its SHA-256.

    python examples/homo_erectus_frontal.py

Semilandmarks are treated as fixed points here; sliding them is a separate step and
would sharpen, not change, the picture.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gmm_core import (
    extract_landmark_array,
    fit_pca,
    generalized_procrustes,
    import_tabular_files,
    run_group_test,
)
from gmm_core.data_model import ID_COL

URL = "https://zenodo.org/api/records/4441216/files/all_coord_f2.csv/content"
SHA256 = "c0bb8b3b48d375b257469a56ba2e47f6516484dd8ab4ba1b83619b08c91aa9a9"
CACHE = Path.home() / ".cache" / "gmm-core" / "baab_2021_all_coord_f2.csv"
SPECIES = {"Hsap": "Homo sapiens", "Here": "Homo erectus"}


def italic(name: str) -> str:
    return "$\\it{" + name.replace(" ", "\\ ") + "}$"


def fetch(path: Path | None) -> Path:
    path = path or CACHE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{path} does not match the published checksum")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, help="use a local copy of all_coord_f2.csv")
    args = parser.parse_args()

    # 1. Import: the wide V1, V2, ... layout is detected and read as x, y, z triplets.
    data = import_tabular_files([fetch(args.csv)])
    raw = data.raw.rename(columns={"spp": "Species"})
    raw["Species"] = raw["Species"].map(SPECIES)
    print(
        f"{len(raw)} specimens, {data.n_landmarks} landmarks in {data.n_dim}D, "
        f"{len(data.validation)} validation issues"
    )

    # 2. Generalized Procrustes analysis.
    shapes = extract_landmark_array(raw, data.landmark_cols, data.n_dim)
    gpa = generalized_procrustes(shapes, scale=True)
    aligned = raw[[ID_COL, "Species"]].copy()
    flat = gpa.aligned.reshape(len(raw), -1)
    aligned = pd.concat([aligned, pd.DataFrame(flat, columns=data.landmark_cols)], axis=1)
    aligned["CentroidSize"] = gpa.centroid_sizes
    print(f"GPA converged in {gpa.iterations} iterations")

    # 3. PCA in one shared morphospace.
    pca = fit_pca(aligned, data.landmark_cols, data.n_dim, meta_cols=["Species", "CentroidSize"])
    explained = pca.variance["Explained_variance_ratio"].to_numpy() * 100

    # 4. Do the species differ in shape, after accounting for size? (RRPP, 999 permutations)
    test = run_group_test(
        aligned,
        data.landmark_cols,
        data.n_dim,
        group_col="Species",
        meta_cols=["Species", "CentroidSize"],
        covariate_cols=["CentroidSize"],
        n_permutations=999,
        seed=1,
    )
    o = test.omnibus
    print(
        f"{test.model_formula}: R2 = {o.r2:.3f}, p = {o.p_value:.3f} ({o.n_permutations} permutations)"
    )
    for warning in test.warnings:
        print(f"  note: {warning}")

    # 5. Figure.
    scores = pca.scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    colors = {"Homo sapiens": "#9aa0a6", "Homo erectus": "#e8710a"}
    for name, group in scores.groupby("Species"):
        ax1.scatter(
            group["PC1"],
            group["PC2"],
            s=22 if name == "Homo erectus" else 12,
            c=colors[name],
            alpha=0.95 if name == "Homo erectus" else 0.6,
            edgecolors="white",
            linewidths=0.4,
            label=f"{italic(name)} (n = {len(group)})",
        )
    ax1.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax1.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax1.set_title("Frontal-bone shape space", loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=9, loc="best")
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # Shape difference between the species means, seen in the plane of the consensus'
    # two main axes and magnified 2x so it is easy to read.
    consensus = gpa.consensus
    _, _, vt = np.linalg.svd(consensus - consensus.mean(0))
    plane = vt[[0, 2]].T
    means = {
        name: gpa.aligned[(raw["Species"] == name).to_numpy()].mean(axis=0)
        for name in ("Homo sapiens", "Homo erectus")
    }
    start = means["Homo sapiens"] @ plane
    end = (means["Homo sapiens"] + 2 * (means["Homo erectus"] - means["Homo sapiens"])) @ plane
    ax2.quiver(
        *start.T,
        *(end - start).T,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
        color=colors["Homo erectus"],
        alpha=0.9,
    )
    ax2.scatter(*start.T, s=16, c="#5f6368", zorder=3, label=f"{italic('Homo sapiens')} mean")
    ax2.scatter(
        [],
        [],
        marker=r"$\rightarrow$",
        s=120,
        c=colors["Homo erectus"],
        label=f"towards {italic('Homo erectus')} (x2)",
    )
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("How the mean shape differs", loc="left", fontsize=11)
    ax2.legend(frameon=False, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.22))

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "img" / "homo_erectus_frontal.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
