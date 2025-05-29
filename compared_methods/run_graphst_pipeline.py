import os
import torch
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn import metrics
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from GraphST import GraphST
import paste as pst

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# The data with format required by GraphST are available at: 
# https://drive.google.com/drive/folders/1NVROB3oyBgBEzCt1yH_AJe6uSIjspDLf?usp=sharing


data_dir = "spatialLIBD"

slice_idx_list = [151673, 151674, 151675, 151676]
adata_st_list = []

for slice_idx in slice_idx_list:
    adata_st = sc.read_visium(path=data_dir+"/%d" % slice_idx, count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx)
    anno_df = pd.read_csv(data_dir+'/barcode_level_layer_map.tsv', sep='\t', header=None)
    anno_df = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx)]
    anno_df.columns = ["barcode", "slice_id", "layer"]
    anno_df.index = anno_df['barcode']
    adata_st.obs = adata_st.obs.join(anno_df, how="left")
    adata_st = adata_st[adata_st.obs['layer'].notna()]
    adata_st_add = adata_st.copy()
    adata_st_add.var_names_make_unique()
    adata_st_list.append(adata_st_add)

# Running integrative analysis using GraphST requires a prior spatial registration across sections.
# We use PASTE to prepared the GraphST's required data, which is available at:
# https://drive.google.com/drive/folders/1DC5sfT15OeFuRUGuwfPFKDz5D3k5lJfH?usp=sharing

loc_dir = "paste_pi"

for i in range(len(slice_idx_list)):
    new_slice = sc.read_h5ad(loc_dir + "/new_slices_" + str(i) + ".h5ad")
    adata_st_list[i].obsm["spatial"] = new_slice.obsm["spatial"]
    adata_st_list[i].obs.index = adata_st_list[i].obs.index + "-" + str(i)

adata = ad.concat(adata_st_list)

plt.rcParams["figure.figsize"] = (3, 3)
adata.obsm['spatial'][:, 1] = -1*adata.obsm['spatial'][:, 1]
ax = sc.pl.embedding(adata, basis='spatial',
                color='slice_id',
                show=False)
ax.set_title('Aligned image')

# define model
model = GraphST.GraphST(adata, device=device)

# run model
adata = model.train()

from sklearn.decomposition import PCA

pca = PCA(n_components=20, random_state=42) 
embedding = pca.fit_transform(adata.obsm['emb'].copy())
adata.obsm['emb_pca'] = embedding

### Plotting UMAP after batch effect correction
sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=10)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['layer','slice_id'])

save_dir = "run_graphst/res_graphst"
adata.write(save_dir + "/res_graphst_adata.h5ad")
