import time, os, sys, pickle, h5py, importlib, gc, copy, re, itertools, json, logging
from tqdm.auto import tqdm, trange
from pathlib import Path
import numpy as np, pandas as pd, scipy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from umap import UMAP
import torch
from matplotlib import pyplot as plt
import seaborn as sns


# -- specify device
context = dict(device='cuda:0', dtype=torch.float64)

# -- specify dataset
path2dataset = Path('data_format_for_spicemix')
repli_list = ["151673", "151674", "151675", "151676"]


from model import SpiceMix
np.random.seed(1234)

K, num_pcs, n_neighbors, res_lo, res_hi = 20, 50, 20, .5, 5.

path2result = path2dataset / 'results' / 'SpiceMix.h5'
os.makedirs(path2result.parent, exist_ok=True)
if os.path.exists(path2result):
    os.remove(path2result)

obj = SpiceMix(
    K=K,
    lambda_Sigma_x_inv=1e-6, power_Sigma_x_inv=2,
    repli_list=repli_list,
    context=context,
    context_Y=context,
    path2result=path2result,
)
obj.load_dataset(path2dataset)

obj.initialize(
    method='louvain', kwargs=dict(num_pcs=num_pcs, n_neighbors=n_neighbors, resolution_boundaries=(res_lo, res_hi), num_rs=10),
)
for iiter in range(10):
    obj.estimate_weights(iiter=iiter, use_spatial=[False]*obj.num_repli)
    obj.estimate_parameters(iiter=iiter, use_spatial=[False]*obj.num_repli)
obj.initialize_Sigma_x_inv()
for iiter in range(1, 201):
    obj.estimate_parameters(iiter=iiter, use_spatial=[True]*obj.num_repli)
    obj.estimate_weights(iiter=iiter, use_spatial=[True]*obj.num_repli)


# --- save result
res_path = "run_spicemix/res_spicemix"
for i in range(len(obj.Xs)):
	np.save(res_path+"/X_"+str(i)+".npy", obj.Xs[i].detach().cpu().numpy())
np.save(res_path+"/M.npy", obj.M.detach().cpu().numpy())


