# Subsetting
adata_hv = adata[:, adata.var['highly_variable'] ].copy()

# Showing dimension and how many genes are hv just to know
print("No. of highly variable genes:", adata.var['highly_variable'].sum())
print("Shape:", adata_hv.shape)
