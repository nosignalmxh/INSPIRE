rm(list = ls())
library(ggplot2)
library(patchwork)
library(scater)
library(harmony)
library(BayesSpace)
library(Seurat)

set.seed(1234)


# The data with format required by BayesSpace are available at: 
# https://drive.google.com/drive/folders/1NVROB3oyBgBEzCt1yH_AJe6uSIjspDLf?usp=sharing, and
# https://drive.google.com/drive/folders/1qJiZUYhTCVlpYXwFa6xE9JsCnZn8qrMP?usp=sharing


cluster_df <- read.table(file = 'spatialLIBD/barcode_level_layer_map.tsv', sep='\t', col.name=c("barcode", "slice", "layer"))

section_list <- c("151673", "151674", "151675", "151676")
obj_list <- list()

for (i in 1:length(section_list)) {
    s <- section_list[i]
    print(s)

    # load data
    obj <- Load10X_Spatial(
        paste0("spatialLIBD_r/", s),
        filename = paste0(s, "_filtered_feature_bc_matrix.h5"),
        assay = "Spatial",
        slice = "X1")

    # layer anno
    cluster_df_s <- cluster_df[cluster_df$slice == s, ]
    obj <- subset(obj, cells = cluster_df_s$barcode)
    obj@meta.data$layer <- cluster_df_s$layer

    # spatial coor
    coor <- obj@images$X1@coordinates[, c("row", "col")]
    obj <- as.SingleCellExperiment(obj)
    colData(obj)$row = coor$row
    colData(obj)$col = coor$col
    colData(obj)$sample_name = s

    rowData(obj)$is.HVG = NULL 
    rowData(obj)$is.HVG = NULL 

    obj_list[[i]] <- obj
}


# clusterPlot(obj_list[[1]], "layer", color = NA, platform = "Visium", is.enhanced = FALSE)

#Combine into 1 SCE and preprocess
sce.combined = cbind(
    obj_list[[1]], obj_list[[2]], obj_list[[3]], obj_list[[4]],
    deparse.level = 1)
sce.combined = spatialPreprocess(sce.combined, n.PCs = 50) #lognormalize, PCA


#Batch correction
# sce.combined = runUMAP(sce.combined, dimred = "PCA")
# colnames(reducedDim(sce.combined, "UMAP")) = c("UMAP1", "UMAP2")
# ggplot(data.frame(reducedDim(sce.combined, "UMAP")), 
#        aes(x = UMAP1, y = UMAP2, color = factor(sce.combined$sample_name))) +
#   geom_point() +
#   labs(color = "Sample") +
#   theme_bw()
sce.combined = RunHarmony(sce.combined, "sample_name", verbose = TRUE)
sce.combined = runUMAP(sce.combined, dimred = "HARMONY", name = "UMAP.HARMONY")
colnames(reducedDim(sce.combined, "UMAP.HARMONY")) = c("UMAP1", "UMAP2")
# ggplot(data.frame(reducedDim(sce.combined, "UMAP.HARMONY")), 
#        aes(x = UMAP1, y = UMAP2, color = factor(sce.combined$sample_name))) +
#   geom_point() +
#   labs(color = "Sample") +
#   theme_bw()


#Clustering
sce.combined$row[sce.combined$sample_name == "151674"] = 
  100 + sce.combined$row[sce.combined$sample_name == "151674"]
sce.combined$col[sce.combined$sample_name == "151675"] = 
  150 + sce.combined$col[sce.combined$sample_name == "151675"]
sce.combined$row[sce.combined$sample_name == "151676"] = 
  100 + sce.combined$row[sce.combined$sample_name == "151676"]
sce.combined$col[sce.combined$sample_name == "151676"] = 
  150 + sce.combined$col[sce.combined$sample_name == "151676"]

# clusterPlot(sce.combined, "sample_name", color = NA) + #make sure no overlap between samples
#   labs(fill = "Sample", title = "Offset check")


sce.combined = spatialCluster(sce.combined, use.dimred = "HARMONY", q = 7, nrep = 10000) #use HARMONY
# clusterPlot(sce.combined, color = NA) + #plot clusters
#   labs(title = "BayesSpace joint clustering")
setwd("res_harmony_bayesspace")
save(sce.combined, file="sce_combined.RData")

res.df <- data.frame(
    barcode = colnames(sce.combined),
    sample_name = as.vector(sce.combined$sample_name),
    layer = as.vector(sce.combined$layer),
    bayesspace = sce.combined$spatial.cluster)
write.csv(res.df, "res_harmony_bayesspace.csv")

