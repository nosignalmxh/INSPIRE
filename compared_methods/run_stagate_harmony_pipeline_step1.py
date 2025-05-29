import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys

import STAGATE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# The data with format required by STAGATE are available at: 
# https://drive.google.com/file/d/1dT7bB6JSMr_OAhrFMV2_ABuGWa_uudti/view?usp=sharing

data_path = "data_for_R"
adata = sc.read_h5ad(data_path + "/DLPFC_4slices_largedistance.h5ad")
adata.obsm["spatial"] = np.array(adata.obs[["spatial_1", "spatial_2"]])

sc.pl.spatial(adata, spot_size=100)

adata_hvg = adata.copy()
sample_list = list(set(adata_hvg.obs["slice_id"]))
hvg = adata.var.index
for s in sample_list:
    adata_tmp = adata_hvg[adata_hvg.obs["slice_id"] == s, :]
    sc.pp.highly_variable_genes(adata_tmp, flavor="seurat_v3", n_top_genes=3000)
    hvg = hvg & adata_tmp.var[adata_tmp.var.highly_variable == True].sort_values(by="highly_variable_rank").index

#Normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, hvg]

STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
STAGATE.Stats_Spatial_Net(adata)

adata = STAGATE.train_STAGATE(adata, alpha=0)

save_dir = "run_stagate_harmony"
adata.write(save_dir + "/adata_stagate_sharednet.h5ad")

latent.to_csv(save_dir + "/latent_stagate_sharednet.csv")
adata.obs.to_csv(save_dir + "/meta_stagate_sharednet.csv")



