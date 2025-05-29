import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import math
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import SpaGCN as spg
import cv2


# The data with format required by SpaGCN are available at: 
# https://drive.google.com/drive/folders/1YJ6LcghneN5nXyBgVcKEeSmfjMImsvNs?usp=sharing


min_dist_ref = 5000

data_dir = "data_format_for_spagcn"

slice_idx_list = [151673,151674,151675,151676]

adata_list = []
for i, slice_idx in enumerate(slice_idx_list):
    adata = sc.read_h5ad(data_dir + "/adata_" + str(slice_idx) + ".h5ad")
    adata.var_names_make_unique()
    adata.obs.index = adata.obs.index + "-" + str(i)
    adata.obs["slice"] = i

    adata.obsm["spatial"] = adata.obs[["x_pixel", "y_pixel"]].values
    
    if i == 0:
        adata_list.append(adata)
    else:
        xmax_1 = np.max(adata_list[i-1].obsm["spatial"][:,0])
        xmin_2 = np.min(adata.obsm["spatial"][:,0])
        ymin_1 = np.max(adata_list[i-1].obsm["spatial"][:,1])
        ymin_2 = np.max(adata.obsm["spatial"][:,1])
    
        adata.obsm["spatial"][:,0] = adata.obsm["spatial"][:,0] + (xmax_1 - xmin_2) + min_dist_ref
        adata.obsm["spatial"][:,1] = adata.obsm["spatial"][:,1] + (ymin_1 - ymin_2)
        
        adata_list.append(adata)

adata_all = ad.concat(adata_list, join="outer")
sc.pl.spatial(adata_all, spot_size = 100)

X = np.array([adata_all.obsm["spatial"][:,0], adata_all.obsm["spatial"][:,1], adata_all.obs["z"]]).T.astype(np.float32)
adj = spg.pairwise_distance(X)

sc.pp.normalize_per_cell(adata_all, min_counts=0)
sc.pp.log1p(adata_all)
p = 0.5 
#Find the l value given p
l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

#For this toy data, we set the number of clusters=7 since this tissue has 7 layers
n_clusters = 7
#Set seed
r_seed = t_seed = n_seed = 1
#Search for suitable resolution
res = spg.search_res(adata_all, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

clf = spg.SpaGCN()
clf.set_l(l)
#Set seed
random.seed(r_seed)
torch.manual_seed(t_seed)
np.random.seed(n_seed)
#Run
clf.train(adata_all, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)

z, q = clf.model.predict(clf.embed, clf.adj_exp)

latent = pd.DataFrame(data = z.detach().numpy())
latent.index = adata_all.obs.index
latent.columns = ["x"+str(i) for i in range(latent.shape[1])]

save_path = "res_spagcn_latent"
latent.to_csv(save_path + "/res_spagcn_sharednet.csv")
adata_all.obs.to_csv(save_path + "/meta.csv")


