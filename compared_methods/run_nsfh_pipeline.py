import random
import numpy as np
import scanpy as sc
from os import path
from scipy import sparse
from matplotlib import pyplot as plt
from pandas import get_dummies
from tensorflow_probability import math as tm
tfk = tm.psd_kernels
from models import sf, sfh
from utils import misc,preprocess,training,postprocess,visualize


# # The data with format required by NSFH are available at: 
# https://drive.google.com/file/d/1dT7bB6JSMr_OAhrFMV2_ABuGWa_uudti/view?usp=sharing


## load data
random.seed(1234)
data_path = "data_for_R"
ad = sc.read_h5ad(data_path + "/DLPFC_4slices_largedistance.h5ad")
ad.obsm["spatial"] = np.array(ad.obs[["spatial_1","spatial_2"]])


## preprocess
ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X
sc.pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
sc.pp.log1p(ad)
ad.var['deviance_poisson'] = preprocess.deviancePoisson(ad.layers["counts"])
o = np.argsort(-ad.var['deviance_poisson'])
ad = ad[:,o]
ad = ad[:,:2000]


## before fit model
D,Dval = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1., flip_yaxis=False)
Ntr,J = D["Y"].shape
Xtr = D["X"]
ad = ad[:Ntr,:]
#convert to tensorflow objects
Dtf = preprocess.prepare_datasets_tf(D,Dval=Dval)


## fit model
#%% Initialize inducing points
L = 20 #number of components
Z = misc.kmeans_inducing_pts(Xtr, 500)
M = Z.shape[0] #number of inducing points
ker = tfk.MaternThreeHalves
#%% NSFH
# fit = sf.SpatialFactorization(J, L, Z, psd_kernel=ker, nonneg=True, lik="poi")
fit = sfh.SpatialFactorizationHybrid(Ntr, J, L, Z, lik="poi", nonneg=True, psd_kernel=ker)
fit.init_loadings(D["Y"], X=Xtr, sz=D["sz"])
tro = training.ModelTrainer(fit)
tro.train_model(*Dtf, status_freq=50) #about 3 mins


## after fit model
insf_LDAmodeF = postprocess.interpret_nsfh(fit,Xtr,lda_mode=False)
insf_LDAmodeT = postprocess.interpret_nsfh(fit,Xtr,lda_mode=True)

save_path = "run_nsf"

np.save(save_path+"/ide_LDAmodeF_spatial_factors.npy", np.array(insf_LDAmodeF["spatial"]["factors"]))
np.save(save_path+"/ide_LDAmodeF_spatial_loadings.npy", np.array(insf_LDAmodeF["spatial"]["loadings"]))
np.save(save_path+"/ide_LDAmodeF_nonspatial_factors.npy", np.array(insf_LDAmodeF["nonspatial"]["factors"]))
np.save(save_path+"/ide_LDAmodeF_nonspatial_loadings.npy", np.array(insf_LDAmodeF["nonspatial"]["loadings"]))

np.save(save_path+"/ide_LDAmodeT_spatial_factors.npy", np.array(insf_LDAmodeT["spatial"]["factors"]))
np.save(save_path+"/ide_LDAmodeT_spatial_loadings.npy", np.array(insf_LDAmodeT["spatial"]["loadings"]))
np.save(save_path+"/ide_LDAmodeT_nonspatial_factors.npy", np.array(insf_LDAmodeT["nonspatial"]["factors"]))
np.save(save_path+"/ide_LDAmodeT_nonspatial_loadings.npy", np.array(insf_LDAmodeT["nonspatial"]["loadings"]))


