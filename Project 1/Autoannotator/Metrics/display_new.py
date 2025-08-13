import re
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import sys 
import os
sys.path.append(os.path.abspath("C:\\Users\\bence\\Projects\\BIO446\\McKinnon-Rosati-Laboratory\\Project 2\\jl_modules"))
import sc_module as sm
WES = list(sm.wes)
from sklearn.metrics import confusion_matrix

# ----------------------------- #
# Helpers
# ----------------------------- #

def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _get_pred_col(adata):
    # normalize name: predicted cell type
    for c in ["predicted_cell_type", "predicted_celltype", "pred", "y_pred"]:
        if c in adata.obs:
            return c
    raise KeyError("No predicted cell-type column found (looked for 'predicted_cell_type'/'predicted_celltype').")

def _get_conf_col(adata):
    for c in ["accuracy", "confidence", "max_prob", "pred_confidence"]:
        if c in adata.obs:
            return c
    return None  # optional

def _get_true_col(adata):
    # your truth lives here
    for c in ["single_cell_types", "single_cell_type", "cell_type", "y_true"]:
        if c in adata.obs:
            return c
    raise KeyError("No true label column found (looked for 'single_cell_types').")

def _per_class_accuracy_from_obs(adata, pred_name_col, true_col):
    """
    Returns:
      per_class_acc: dict {label -> recall on that true class}
      overall_acc: float
      balanced_acc: float (macro avg over classes with support > 0)
      support: dict {label -> true count}
    """
    y_true = adata.obs[true_col].astype(str)
    y_pred = adata.obs[pred_name_col].astype(str)

    # union of labels so confusion_matrix is aligned
    cats = sorted(set(y_true.dropna().unique()) | set(y_pred.dropna().unique()))
    y_true_cat = pd.Categorical(y_true, categories=cats)
    y_pred_cat = pd.Categorical(y_pred, categories=cats)

    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=cats)
    tp = np.diag(cm)
    support_arr = cm.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.where(support_arr > 0, tp / support_arr, np.nan)

    per_class_acc = {lab: acc for lab, acc in zip(cats, per_class)}
    support = {lab: int(s) for lab, s in zip(cats, support_arr)}

    overall_acc = float((y_true_cat == y_pred_cat).mean())
    # balanced accuracy: mean of per-class recall for classes with support > 0
    valid = ~np.isnan(per_class)
    balanced_acc = float(per_class[valid].mean()) if valid.any() else np.nan

    return per_class_acc, overall_acc, balanced_acc, support

def _maybe_map_int_to_names(adata, pred_col):
    """If predictions are ints and class names exist in .uns, map to names."""
    s = adata.obs[pred_col]
    if pd.api.types.is_numeric_dtype(s):
        # try some common uns keys that might hold class names
        candidate_uns_keys = [
            "class_names", "classes", "flat_classes", "hierarchy_classes",
            "logits_source_classes", "target_label_order"
        ]
        names = None
        for k in candidate_uns_keys:
            if k in adata.uns and isinstance(adata.uns[k], (list, tuple)) and len(adata.uns[k]) > int(s.max()):
                names = list(map(str, adata.uns[k]))
                break
        if names is not None:
            return s.astype(int).map({i:n for i, n in enumerate(names)})
        else:
            return s.astype(int).astype(str)  # fallback: keep ints as strings for plotting
    else:
        return s.astype(str)

def _ensure_umap(adata, prefer_aligned=False, n_pcs=50, n_neighbors=15, min_dist=0.5):
    """Use existing X_umap if present; otherwise compute one.
       For transfer runs prefer obsm['X_aligned_to_source'] if available."""
    if "X_umap" in adata.obsm:
        return

    use_rep = None
    if prefer_aligned and "X_aligned_to_source" in adata.obsm:
        use_rep = "X_aligned_to_source"

    # Compute neighbors/umap
    if use_rep is None:
        # compute PCA on .X
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars))
        sc.pp.neighbors(adata, n_pcs=min(n_pcs, adata.n_vars), n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist)

def _abbrev(name: str):
    name = re.sub(r"\(.*?\)", "", str(name))
    words = re.split(r"[ /-]+", name.strip())
    words = [w for w in words if w]
    if not words:
        return "NA"
    if len(words) >= 3:
        return "".join(w[0].upper() for w in words[:3])
    if len(words) == 2:
        first, second = words
        return (first[:2] + second[:1]).upper() if len(first) > 1 else (first[:1] + second[:2]).upper()
    # single word
    return (words[0][:3]).upper()

def _palette_for(categories):
    cats = list(categories)
    if WES:
        # cycle if needed
        from itertools import cycle, islice
        cyc = cycle(WES)
        return {c: next(cyc) for c in cats}
    else:
        import seaborn as sns
        return dict(zip(cats, sns.color_palette("tab20", len(cats))))

def plot_umap_by_prediction(
    adata,
    title,
    prefer_aligned=False,
    annotate_centroids=True,
    show_counts=True,
    show_per_class_acc=True,
    point_size=4,
    alpha=0.7,
    savepath=None,
):
    """
    UMAP colored by predicted cell type.

    Legend shows:
      - per-class accuracy (recall on that true class) as 'acc='
      - counts (n=)
    Title appends overall accuracy and balanced accuracy.
    """

    pred_col = _get_pred_col(adata)
    true_col = _get_true_col(adata)

    # Map predicted integers -> names if needed
    adata.obs["__pred_name__"] = _maybe_map_int_to_names(adata, pred_col)

    # Compute metrics
    per_class_acc, overall_acc, balanced_acc, support = _per_class_accuracy_from_obs(
        adata, "__pred_name__", true_col
    )

    # UMAP (use aligned space for transfer if available)
    _ensure_umap(adata, prefer_aligned=prefer_aligned)

    coords = pd.DataFrame(
        adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names
    )
    coords["celltype"] = adata.obs["__pred_name__"].astype("category")

    cats = list(coords["celltype"].cat.categories)
    pal = _palette_for(cats)

    fig, ax = plt.subplots(figsize=(9, 8))
    for cat, group in coords.groupby("celltype", observed=True):
        bits = [f"{str(cat)}: "]
        if show_per_class_acc:
            acc = per_class_acc.get(str(cat), np.nan)
            bits.append(f"acc={acc:.2f}" if pd.notna(acc) else "acc=NA")
        if show_counts:
            n = int(group.shape[0])
            bits.append(f", n={n}")
        legend_label = "".join(bits)

        ax.scatter(
            group["UMAP1"], group["UMAP2"],
            s=point_size, alpha=alpha, lw=0,
            color=pal.get(cat, "#888888"),
            label=legend_label
        )

        if annotate_centroids:
            x_c = group["UMAP1"].mean()
            y_c = group["UMAP2"].mean()
            ax.text(
                x_c, y_c, _abbrev(cat),
                fontsize=7, fontweight="bold",
                ha="center", va="center",
                color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
            )

    # Append overall + balanced accuracy to title
    suffix = f"  |  acc={overall_acc:.3f},  bal-acc={balanced_acc:.3f}"
    ax.set_title(title + suffix)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Predicted (acc, n)")
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close(fig)

# ----------------------------- #
# File patterns for your runs
# ----------------------------- #

def paths_for_dataset(ds: str, other: str):
    """
    Returns a dict of 4 files for a dataset:
      - original flat
      - original hierarchical
      - transfer flat (from `other`)
      - transfer hierarchical (from `other`)
    Handles a couple common filename variants.
    """
    base = "../../Data"

    orig_flat = _first_existing(
        f"{base}/{ds}_test_anndata_flat_predictions.h5ad",
        f"{base}/{ds}_test_anndata_flat.h5ad"
    )
    orig_hier = _first_existing(
        f"{base}/{ds}_test_anndata_hierarchy_predictions.h5ad",
        f"{base}/{ds}_test_anndata_hierarchy.h5ad"
    )
    xfer_flat = _first_existing(
        f"{base}/{ds}_test_anndata_flat_from_{other}.h5ad",
        f"{base}/{ds}_test_anndata_flat_from_{other}_predictions.h5ad"
    )
    xfer_hier = _first_existing(
        f"{base}/{ds}_test_anndata_hierarchy_from_{other}.h5ad",
        f"{base}/{ds}_test_anndata_hierarchy_from_{other}_predictions.h5ad"
    )

    return {
        "orig_flat": orig_flat,
        "orig_hier": orig_hier,
        "xfer_flat": xfer_flat,
        "xfer_hier": xfer_hier,
    }

# ----------------------------- #
# Driver: make all 8 plots
# ----------------------------- #

def make_umaps_for(ds: str, other: str, outdir: str = "UMAPs"):
    files = paths_for_dataset(ds, other)

    def _load(path):
        if not path:
            print(f"[WARN] File not found for {ds}. Skipping.")
            return None
        print(f"[LOAD] {path}")
        return sc.read_h5ad(path)

    # original (trained and tested on same dataset)
    ad_flat  = _load(files["orig_flat"])
    ad_hier  = _load(files["orig_hier"])

    # transfer (trained on `other`, tested on `ds`)
    ad_xflat = _load(files["xfer_flat"])
    ad_xhier = _load(files["xfer_hier"])

    # Plot: originals
    if ad_flat is not None:
        plot_umap_by_prediction(
            ad_flat,
            title=f"{ds}: original flat",
            prefer_aligned=False,
            savepath=f"{outdir}/{ds}_original_flat.png",
        )
    if ad_hier is not None:
        plot_umap_by_prediction(
            ad_hier,
            title=f"{ds}: original hier",
            prefer_aligned=False,
            savepath=f"{outdir}/{ds}_original_hier.png",
        )

    # Plot: transfers (prefer aligned rep if present)
    if ad_xflat is not None:
        plot_umap_by_prediction(
            ad_xflat,
            title=f"{ds}: transfer flat (from {other})",
            prefer_aligned=True,
            savepath=f"{outdir}/{ds}_transfer_flat_from_{other}.png",
        )
    if ad_xhier is not None:
        plot_umap_by_prediction(
            ad_xhier,
            title=f"{ds}: transfer hier (from {other})",
            prefer_aligned=True,
            savepath=f"{outdir}/{ds}_transfer_hier_from_{other}.png",
        )


make_umaps_for("sc92_final", other="granulomas_final")
make_umaps_for("sc93_final", other="granulomas_final")
