import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from mofapy2.run.entry_point import entry_point


# We organized the tissue sections as the format of NSFH's input. The organized sections are available at:
# https://drive.google.com/file/d/1dT7bB6JSMr_OAhrFMV2_ABuGWa_uudti/view?usp=sharing


data_path = "data_for_R"
adata = sc.read_h5ad(data_path + "/DLPFC_4slices_largedistance.h5ad")
adata.obsm["spatial"] = np.array(adata.obs[["spatial_1","spatial_2"]])
sc.pl.spatial(adata, color="layer", spot_size=100)

slice_inds = [151673, 151674, 151675, 151676]

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

adata.obs = pd.concat([adata.obs, 
                       pd.DataFrame(adata.obsm["spatial"], columns=["imagerow", "imagecol"], index=adata.obs_names),
                      ], axis=1)

ent = entry_point()
ent.set_data_options(use_float32=True)
ent.set_data_from_anndata(adata, groups_label="slice_id", features_subset="highly_variable")
ent.set_model_options(factors=20)
ent.set_train_options(seed=1234)

cov_list = [adata.obsm["spatial"][adata.obs["slice_id"].values.astype(int) == s, :] for s in slice_inds]

# We use 1000 inducing points to learn spatial covariance patterns
n_inducing = 1000

ent.set_covariates(cov_list, covariates_names=["imagerow", "imagecol"])
ent.set_smooth_options(sparseGP=True, frac_inducing=n_inducing/adata.n_obs,
                       start_opt=10, opt_freq=10)

ent.build()
ent.run()
ent.save("res_mefisto/mefisto_model.hdf5")



