import os
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import umap

import SpatialGlue


# The data with format required by SpatialGlue are available at: 
# https://drive.google.com/file/d/1dT7bB6JSMr_OAhrFMV2_ABuGWa_uudti/view?usp=sharing


# Environment configuration. SpatialGlue pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = "data_for_R"
adata = sc.read_h5ad(data_path + "/DLPFC_4slices_largedistance.h5ad")
adata.obsm["spatial"] = np.array(adata.obs[["spatial_1", "spatial_2"]])
sc.pl.spatial(adata, spot_size=100)

adata_omics1 = adata.copy()
adata_omics1.var_names_make_unique()

# Specify data type
data_type = '10x'

# Fix random seed
from SpatialGlue.preprocess import fix_seed
random_seed = 1234
fix_seed(random_seed)

## preprocessing data

from SpatialGlue.preprocess import clr_normalize_each_cell, pca

# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=30)

## constructing neighbor graph

from SpatialGlue.preprocess import construct_neighbor_graph_single
data = construct_neighbor_graph_single(adata_omics1, datatype=data_type)

## training the model

# define model
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue_single
model = Train_SpatialGlue_single(data, datatype=data_type, device=device)

# train model
output = model.train()

adata = adata_omics1.copy()
adata.obsm['latent'] = output['emb_latent_omics1'].copy()
adata.obsm['alpha_omics1'] = output['alpha_omics1']


save_dir = "res_spatialglue"
adata.write(save_dir + "/adata_spatialglue.h5ad")

