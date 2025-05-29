rm(list = ls())
library(Banksy)
library(SummarizedExperiment)
library(SpatialExperiment)
library(Seurat)

library(scater)
library(cowplot)
library(ggplot2)


# The data with format required by SpatialGlue are available from the spatialLIBD package.


SEED <- 1234

# load the data
library(spatialLIBD)
library(ExperimentHub)
ehub <- ExperimentHub::ExperimentHub()
spe <- spatialLIBD::fetch_data(type = "spe", eh = ehub)

#' Remove NA spots
na_id <- which(is.na(spe$spatialLIBD))
spe <- spe[, -na_id]
unique(spe$spatialLIBD)

#' Trim
imgData(spe) <- NULL
assay(spe, "logcounts") <- NULL
reducedDims(spe) <- NULL
rowData(spe) <- NULL
colData(spe) <- DataFrame(
  sample_id = spe$sample_id,
  clust_annotation = factor(as.numeric(spe$layer_guess_reordered_short)),
  in_tissue = spe$in_tissue,
  row.names = colnames(spe)
)
invisible(gc())


sampleid_list <- c("151673", "151674", "151675", "151676")
spe <- spe[, spe$sample_id %in% sampleid_list]
for (i in 1:length(sampleid_list)) {
  colnames(spe)[spe$sample_id == sampleid_list[i]] <- paste0(colnames(spe)[spe$sample_id == sampleid_list[i]], "-", i-1)
}
sample_names <- unique(spe$sample_id)
sample_names


# #' Stagger spatial coordinates
# locs <- spatialCoords(spe)
# locs <- cbind(locs, sample_id = factor(spe$sample_id))
# locs_dt <- data.table(locs)
# colnames(locs_dt) <- c("sdimx", "sdimy", "group")
# locs_dt[, sdimx := sdimx - min(sdimx), by = group]
# global_max <- max(locs_dt$sdimx) * 1.5
# locs_dt[, sdimx := sdimx + group * global_max]
# locs <- as.matrix(locs_dt[, 1:2])
# ggplot(locs_dt, aes(x=sdimx, y=sdimy)) + geom_point(size=0.1) + coord_fixed() # check


spe_list <- lapply(sampleid_list, function(x) spe[, spe$sample_id == x])
rm(spe)
invisible(gc())
 

# Data preprocessing
#' Normalize data
seu_list <- lapply(spe_list, function(x) {
    x <- as.Seurat(x, data = NULL)
    NormalizeData(x, scale.factor = 5000, normalization.method = 'RC')
})

#' Compute HVGs
hvgs <- lapply(seu_list, function(x) {
    VariableFeatures(FindVariableFeatures(x, nfeatures = 2000))
})
hvgs <- Reduce(union, hvgs)

#' Add data to SpatialExperiment and subset to HVGs
aname <- "normcounts"
spe_list <- Map(function(spe, seu) {
    assay(spe, aname) <- GetAssayData(seu)
    spe[hvgs,]
    }, spe_list, seu_list)
rm(seu_list)
invisible(gc())


# Running BANKSY
compute_agf <- FALSE
k_geom <- 6
spe_list <- lapply(spe_list, computeBanksy, assay_name = aname, 
                   compute_agf = compute_agf, k_geom = k_geom)

spe_joint <- do.call(cbind, spe_list)
rm(spe_list)
invisible(gc())

lambda <- 0.2
use_agf <- FALSE
spe_joint <- runBanksyPCA(spe_joint, use_agf = use_agf, lambda = lambda, group = "sample_id", seed = SEED)


# Run Harmony on BANKSY’s embedding
set.seed(SEED)
latent <- as.matrix(reducedDim(spe_joint, "PCA_M0_lam0.2"))

obj <- CreateSeuratObject(counts = t(latent*0), meta.data = as.data.frame(colData(spe_joint)))
obj@assays$RNA@data <- t(latent*0)
obj@assays$RNA@scale.data <- t(latent*0)
pca.dr <- CreateDimReducObject(
  embeddings = as.matrix(latent),
  key = "PC",
  assay = "RNA"
)
obj[['pca']] <- pca.dr

obj <- RunHarmony(obj, 'sample_id')
harmony_embedding <- obj@reductions$harmony@cell.embeddings


save.dir <- "run_banksy_harmony"
write.csv(harmony_embedding, file = paste0(save.dir, "/banksy_harmony_latent.csv"))
write.csv(as.data.frame(colData(spe_joint)), file = paste0(save.dir, "/meta.csv"))


