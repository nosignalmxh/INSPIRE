import scanpy as sc
import spatialleiden as sl
import squidpy as sq
import numpy as np
import umap
from multispaeti import MultispatiPCA

# The data with format required by SpatialLeiden are available at: 
# https://drive.google.com/file/d/1g763Z7ClovTDn7aQXj4hJs6cxLFjvQAS/view?usp=sharing

seed = 42

data_path = "data_for_R"
adata = sc.read_h5ad(data_path + "/DLPFC_4slices_largedistance.h5ad")
adata.obsm["spatial"] = np.array(adata.obs[["spatial_1", "spatial_2"]])

## connectivity
sq.gr.spatial_neighbors(adata)
connectivity = adata.obsp["spatial_connectivities"].toarray()

# # A distance matrix should be transformed to connectivities by e.g. calculating :math:`1-d/d_{max}` beforehand.
# from sklearn.metrics.pairwise import pairwise_distances
# dist = pairwise_distances(adata.obsm["spatial"], adata.obsm["spatial"], metric='euclidean')
# connectivity = 1 - dist / np.max(dist)

## svgs
sq.gr.spatial_autocorr(adata)
svg_num = 3000
svg = list(adata.uns["moranI"].iloc[:svg_num, ].index)
adata = adata[:, svg]

## spatialpca
from multispaeti import MultispatiPCA
msPCA = MultispatiPCA(n_components=30, connectivity=connectivity)
msPCA.fit(adata.X)
X_transformed = msPCA.transform(adata.X)

## spatialleiden
adata.obsm["X_pca"] = X_transformed
sc.pp.neighbors(adata, random_state=seed)
# sl.spatialleiden(adata, layer_ratio=1.8, directed=(False, True), seed=seed)

n_clusters = 7
latent_resolution, spatial_resolution = sl.search_resolution(
    adata,
    n_clusters,
    latent_kwargs={"seed": seed},
    spatial_kwargs={"layer_ratio": 1.8, "seed": seed, "directed": (False, True)},
)

sc.pl.spatial(adata, color="spatialleiden", spot_size=150)

reducer = umap.UMAP(n_neighbors=30,
                    n_components=2,
                    metric="correlation",
                    n_epochs=None,
                    learning_rate=1.0,
                    min_dist=0.3,
                    spread=1.0,
                    set_op_mix_ratio=1.0,
                    local_connectivity=1,
                    repulsion_strength=1,
                    negative_sample_rate=5,
                    a=None,
                    b=None,
                    random_state=1234,
                    metric_kwds=None,
                    angular_rp_forest=False,
                    verbose=True)
embedding = reducer.fit_transform(adata.obsm['X_pca'])
adata.obsm["X_umap"] = embedding
adata.obs["slice"] = adata.obs["slice"].values.astype(str)

sc.pl.umap(adata, color=["layer", "spatialleiden", "slice"])

res_path = "run_spatialleiden/res_spatialleiden"
adata.write(res_path + "/adata_res_spatialleiden.h5ad")

