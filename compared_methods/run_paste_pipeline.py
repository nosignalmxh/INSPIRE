import math
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
import matplotlib
import time
import scanpy as sc
import sklearn
import networkx as nx
import ot
import paste as pst
from paste.helper import to_dense_array
import anndata


# The data with format required by PASTE are available at: 
# https://drive.google.com/drive/folders/1NVROB3oyBgBEzCt1yH_AJe6uSIjspDLf?usp=sharing


data_dir = "spatialLIBD"

slice_idx = 151673
adata_st = sc.read_visium(path=data_dir+"/%d" % slice_idx,
                          count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx)
anno_df = pd.read_csv(data_dir+'/barcode_level_layer_map.tsv', sep='\t', header=None)
anno_df = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx)]
anno_df.columns = ["barcode", "slice_id", "layer"]
anno_df.index = anno_df['barcode']
adata_st.obs = adata_st.obs.join(anno_df, how="left")
adata_st = adata_st[adata_st.obs['layer'].notna()]
adata_st1 = adata_st.copy()
adata_st1.var_names_make_unique()
adata_st1.obs_names_make_unique()
adata_st1.obs.index = adata_st1.obs.index + "-1"

slice_idx = 151674
adata_st = sc.read_visium(path=data_dir+"/%d" % slice_idx,
                          count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx)
anno_df = pd.read_csv(data_dir+'/barcode_level_layer_map.tsv', sep='\t', header=None)
anno_df = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx)]
anno_df.columns = ["barcode", "slice_id", "layer"]
anno_df.index = anno_df['barcode']
adata_st.obs = adata_st.obs.join(anno_df, how="left")
adata_st = adata_st[adata_st.obs['layer'].notna()]
adata_st2 = adata_st.copy()
adata_st2.var_names_make_unique()
adata_st2.obs_names_make_unique()
adata_st2.obs.index = adata_st2.obs.index + "-2"

slice_idx = 151675
adata_st = sc.read_visium(path=data_dir+"/%d" % slice_idx,
                          count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx)
anno_df = pd.read_csv(data_dir+'/barcode_level_layer_map.tsv', sep='\t', header=None)
anno_df = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx)]
anno_df.columns = ["barcode", "slice_id", "layer"]
anno_df.index = anno_df['barcode']
adata_st.obs = adata_st.obs.join(anno_df, how="left")
adata_st = adata_st[adata_st.obs['layer'].notna()]
adata_st3 = adata_st.copy()
adata_st3.var_names_make_unique()
adata_st3.obs_names_make_unique()
adata_st3.obs.index = adata_st3.obs.index + "-3"

slice_idx = 151676
adata_st = sc.read_visium(path=data_dir+"/%d" % slice_idx,
                          count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx)
anno_df = pd.read_csv(data_dir+'/barcode_level_layer_map.tsv', sep='\t', header=None)
anno_df = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx)]
anno_df.columns = ["barcode", "slice_id", "layer"]
anno_df.index = anno_df['barcode']
adata_st.obs = adata_st.obs.join(anno_df, how="left")
adata_st = adata_st[adata_st.obs['layer'].notna()]
adata_st4 = adata_st.copy()
adata_st4.var_names_make_unique()
adata_st4.obs_names_make_unique()
adata_st4.obs.index = adata_st4.obs.index + "-4"

del adata_st

adata_st1.X = adata_st1.X.toarray()
adata_st2.X = adata_st2.X.toarray()
adata_st3.X = adata_st3.X.toarray()
adata_st4.X = adata_st4.X.toarray()


slices = [adata_st1, adata_st2, adata_st3, adata_st4]
hvg_num = 12000
for i in range(len(slices)):
    sc.pp.highly_variable_genes(slices[i], flavor='seurat_v3', n_top_genes=hvg_num)
    hvgs = slices[i].var[slices[i].var.highly_variable == True].sort_values(by="highly_variable_rank").index
    slices[i] = slices[i][:, hvgs]
lmbda = len(slices)*[1/len(slices)]


pst.filter_for_common_genes(slices)


initial_slice = slices[0].copy()
center_slice, pis = pst.center_align(initial_slice, slices, lmbda, random_seed = 1234, alpha=0.1, norm=True)


res_path = "run_paste"
center_slice.write(res_path + "/adata_paste_151673.h5ad")
adata_st = adata_st1.copy()

## similar codes for performing denoising on other tissue sections
