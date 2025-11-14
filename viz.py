# viz.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from sklearn.random_projection import SparseRandomProjection
from umap import UMAP


# ---------------------------------------------------------------------
# Global plotting config / constants
# ---------------------------------------------------------------------

MODEL_ORDER  = ["A_raw", "B_raw", "B_pca64q4"]
CORPUS_ORDER = ["food", "medical"]  # adjust if needed
COPPER3      = sns.color_palette("copper", len(MODEL_ORDER))


# ---------------------------------------------------------------------
# 1) Stability curves + heatmaps
# ---------------------------------------------------------------------

def prepare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure metrics has consistent categories and helper columns.

    Expected columns:
      - corpus
      - model
      - k
      - tau
      - clean_frac
    """
    df = metrics.copy()
    df["model"]  = pd.Categorical(df["model"].astype(str), MODEL_ORDER, ordered=True)
    df["corpus"] = pd.Categorical(df["corpus"].astype(str), CORPUS_ORDER, ordered=True)
    df["corpus_model"] = df["corpus"].astype(str) + " · " + df["model"].astype(str)
    return df


def plot_stability_curves(
    metrics: pd.DataFrame,
    out_path: str | Path = "images/stability_curves.png",
) -> None:
    """
    Plot fraction of clean anchors vs k, averaged over tau, per corpus/model.
    """
    df = prepare_metrics(metrics)
    df2 = (
        df.groupby(["corpus", "model", "k"], as_index=False)["clean_frac"]
          .mean()
          .sort_values(["corpus", "model", "k"])
    )

    g = sns.FacetGrid(
        df2,
        col="corpus",
        sharey=True,
        height=4,
        aspect=1.2,
        col_order=CORPUS_ORDER,
    )

    for (corp, ax) in zip(CORPUS_ORDER, g.axes.flat):
        sub = df2[df2["corpus"] == corp]
        for m, color in zip(MODEL_ORDER, COPPER3):
            s = sub[sub["model"] == m]
            if s.empty:
                continue
            ax.plot(
                s["k"],
                s["clean_frac"],
                marker="o",
                label=m,
                color=color,
            )
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_title(str(corp))
        ax.set_xlabel("k")
        ax.set_ylabel("fraction of clean anchors")
        ax.legend(title="model", loc="lower left")

    g.fig.suptitle("Stability vs k (mean over τ)", y=1.02, fontsize=13)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_heatmaps_by_tau(
    metrics: pd.DataFrame,
    out_path: str | Path = "images/heatmap.png",
) -> None:
    """
    For each tau, plot a heatmap of clean_frac vs k (columns) and corpus·model (rows).
    """
    df = prepare_metrics(metrics)
    row_labels = [f"{c} · {m}" for c in CORPUS_ORDER for m in MODEL_ORDER]
    taus = sorted(df["tau"].unique())
    ncols = len(taus)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4.2 * ncols + 1.5, 0.75 * len(row_labels) + 1),
        constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]

    for ax, tau in zip(axes, taus):
        sub = df[df["tau"] == tau].copy()
        sub["row"] = sub["corpus"].astype(str) + " · " + sub["model"].astype(str)

        piv = (
            sub.pivot_table(
                index="row",
                columns="k",
                values="clean_frac",
                aggfunc="mean",
            )
            .reindex(row_labels)
        )

        sns.heatmap(
            piv,
            ax=ax,
            cmap="copper",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "clean_frac"},
        )
        ax.set_title(f"τ = {tau:g}")
        ax.set_xlabel("k")
        ax.set_ylabel("" if ax is not axes[0] else "corpus · model")

    fig.suptitle("Embedding Retrieval Stability", y=1.02, fontsize=13)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# 2) Triangle violation scatter
# ---------------------------------------------------------------------

def plot_triangle_violations(
    examples: pd.DataFrame,
    sample_n: int = 4000,
    out_path: str | Path = "images/Triangle_Violations.png",
) -> None:
    """
    Scatter d(i,j)+d(j,k) (lhs) vs d(i,k) (rhs), faceted by model, colored by corpus.

    Expected columns in `examples`:
      - model
      - corpus
      - lhs
      - rhs
    """
    ex = examples.copy()
    ex["model"] = pd.Categorical(ex["model"], MODEL_ORDER, ordered=True)

    if len(ex) > sample_n:
        ex = ex.sample(sample_n, random_state=42)

    g = sns.FacetGrid(
        ex,
        col="model",
        col_order=MODEL_ORDER,
        hue="corpus",
        hue_order=CORPUS_ORDER,
        height=4,
        aspect=1.0,
    )
    g.map_dataframe(sns.scatterplot, x="lhs", y="rhs", s=18, alpha=0.6)

    for ax in g.axes.flat:
        x_max = ax.get_xlim()[1]
        y_max = ax.get_ylim()[1]
        lim = max(x_max, y_max)
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("d(i,j) + d(j,k)")
        ax.set_ylabel("d(i,k)")

    g.add_legend(title="corpus")
    g.fig.suptitle("Triangle Violations by Model", y=1.02)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# 3) UMAP embedding viz
# ---------------------------------------------------------------------

def pad_to_dim(X: np.ndarray, target: int = 768) -> np.ndarray:
    """
    Right-pad embeddings to a common dimensionality.
    """
    D = X.shape[1]
    if D == target:
        return X
    Z = np.zeros((X.shape[0], target), dtype=X.dtype)
    Z[:, :D] = X
    return Z


def build_umap_dataframe(
    E: Dict[Tuple[str, str], np.ndarray],
    sample_per_block: int = 500,
    target_dim: int = 768,
    rp_dim: int = 64,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a 2D UMAP embedding from a dict of embeddings.

    E: mapping (corpus, model) -> np.ndarray [N, D]
       where corpus in CORPUS_ORDER, model in MODEL_ORDER.
    """
    rows = []
    mats = []

    for corpus in CORPUS_ORDER:
        for model in MODEL_ORDER:
            key = (corpus, model)
            if key not in E:
                continue
            X = E[key]
            n = len(X)
            if n == 0:
                continue

            idx = np.random.choice(n, size=min(sample_per_block, n), replace=False)
            Xs = X[idx]
            Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
            Xs = pad_to_dim(Xs, target=target_dim)

            mats.append(Xs)
            rows += [(corpus, model)] * len(Xs)

    if not mats:
        raise ValueError("No embeddings found in E for the given CORPUS_ORDER / MODEL_ORDER.")

    M = np.vstack(mats)

    rp = SparseRandomProjection(
        n_components=rp_dim,
        density="auto",
        random_state=random_state,
    )
    M64 = rp.fit_transform(M)

    um = UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    U = um.fit_transform(M64)

    dfu = pd.DataFrame(
        {
            "x": U[:, 0],
            "y": U[:, 1],
            "corpus": [r[0] for r in rows],
            "model": [r[1] for r in rows],
        }
    )
    dfu["model"] = pd.Categorical(dfu["model"], MODEL_ORDER, ordered=True)
    dfu["corpus"] = pd.Categorical(dfu["corpus"], CORPUS_ORDER, ordered=True)

    return dfu


def plot_umap_with_labels(
    df: pd.DataFrame,
    out_path: str | Path = "images/umap_embeddings.png",
    palette: str = "copper",
) -> None:
    """
    Plot 2D UMAP points with convex hulls per (model, corpus).
    """
    plt.figure(figsize=(10, 7))
    colors = sns.color_palette(palette, df["model"].nunique())

    # scatter
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="model",
        style="corpus",
        palette=colors,
        s=20,
        alpha=0.7,
        linewidth=0,
        hue_order=MODEL_ORDER,
        style_order=CORPUS_ORDER,
    )

    # convex hulls per (model, corpus)
    for (model, corpus), sub in df.groupby(["model", "corpus"]):
        if len(sub) < 5:
            continue
        pts = sub[["x", "y"]].to_numpy()
        hull = ConvexHull(pts)
        color = colors[MODEL_ORDER.index(model)]
        plt.fill(
            pts[hull.vertices, 0],
            pts[hull.vertices, 1],
            alpha=0.08,
            color=color,
            label=None,
        )

        cx, cy = pts.mean(axis=0)
        plt.text(
            cx,
            cy,
            f"{model}\n{corpus}",
            fontsize=9,
            weight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.6, lw=0),
        )

    plt.title("UMAP of embeddings with corpus–model clusters")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(alpha=0.25, lw=0.5)
    plt.legend(title=None, loc="best", frameon=False)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
