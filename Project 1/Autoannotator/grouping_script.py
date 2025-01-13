import numpy as np
import pandas as pd
import scanpy as sc

# -------------------------------------------------------------------------------------
# This needs to be added to the autoannotator after the annotation dict has been used. 
# It is not a standalone script.
# -------------------------------------------------------------------------------------
fourth_dict = {
    'Blood vessels':    ['CAP1','CAP2','VEC','AEC'],
    'Lymphatic EC':     ['LEC'],
    'Airway epithelium':['Ciliated','Secretory'],
    'Alveolar epithelium': ['AT1','AT2','AT2-t1','AT2-t2'],
    'Fibroblast':       ['AF','Pericyte'],
    'Smooth muscle':    ['SMC'],
    'Mesothelial':      ['Mesothelial'],
    'B lineage':        ['B1','B2'],
    'T lineage':        ['Th1','Tnaive','Treg','Tex'],
    'NK':               ['NK'],
    'Macrophage':       ['AM','M-t1','M-t2','M-C1q','M-lc'],
    'mononuclear':      ['iMon','DC','pDC'],
    'Neutrophil':       ['N1','N2']
}

group_lookup = {}
for broad_cat, fine_types in fourth_dict.items():
    for ft in fine_types:
        group_lookup[ft] = broad_cat

# -------------------------------------------------
# Map each cell’s fine-grained type to broad type
# -------------------------------------------------

adata.obs['broad_celltype'] = adata.obs['cell_type_edit'].map(group_lookup)

# print("Broad celltype value counts:")
# print(adata.obs['broad_celltype'].value_counts(dropna=False))

# -----------------------------------------------------------------------------------
# Use this to actually subset the data
# -----------------------------------------------------------------------------------

# broad_types = adata.obs['broad_celltype'].unique()
# cell_type_subsets = {}

# for cat in broad_types:
#     if pd.isna(cat):
#         continue
    
#     adata_sub = adata[adata.obs['broad_celltype'] == cat].copy()
#     cell_type_subsets[cat] = adata_sub
    
#     print(f"\nCreated subset for broad cell type '{cat}' "
#           f"with {adata_sub.n_obs} cells.")

# ------------------------------------------------------------------------------
# Use this to train model in place for each type of broad celltype
# ------------------------------------------------------------------------------

# for cat, subset_adata in cell_type_subsets.items():
#     # Example of extracting each cell type and training
#     X_sub = subset_adata.X
#     y_sub = subset_adata.obs['broad_celltype'].values  
    
#     # Train your model here ->
#     print(f"Trained model on '{cat}' subset.")
