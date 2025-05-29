rm(list = ls())
library(Seurat)
library(BASS)

set.seed(1234)

cluster_df <- read.table(file = 'spatialLIBD/barcode_level_layer_map.tsv', sep='\t', col.name=c("barcode", "slice", "layer"))

section_list <- c("151673", "151674", "151675", "151676")

cnts <- list()
xy <- list()

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

    # spatial counts, coor
    spatial_count <- obj@assays$Spatial@counts
    spatial_location <- obj@images$X1@coordinates[, c("row", "col")]
    rownames(spatial_location) <- colnames(spatial_count)

    cnts[[i]] <- spatial_count
    xy[[i]] <- spatial_location
}

# hyper-parameters
# We set the number of cell types to a relatively large
# number (20) to capture the expression heterogeneity.
C <- 20
# number of spatial domains
R <- 7

# Set up BASS object
BASS <- createBASSObject(cnts, xy, C = C, R = R,
  beta_method = "SW", init_method = "mclust", 
  nsample = 10000)

# Data pre-processing:
# 1.Library size normalization followed with a log2 transformation
# 2.Select top 3000 spatially expressed genes with SPARK-X
# 3.Dimension reduction with PCA
BASS <- BASS.preprocess(BASS, doLogNormalize = TRUE,
  geneSelect = "sparkx", nSE = 3000, doPCA = TRUE, 
  scaleFeature = FALSE, nPC = 20)

# Run BASS algorithm
BASS <- BASS.run(BASS)

# post-process posterior samples:
# 1.Adjust for label switching with the ECR-1 algorithm
# 2.Summarize the posterior samples to obtain the spatial domain labels
BASS <- BASS.postprocess(BASS)

setwd("run_bass/res_bass")
save(BASS, file="res_bass.RData")

zlabels <- BASS@results$z # spatial domain labels
for (i in 1:length(zlabels)){
  xy_result <- as.data.frame(BASS@xy[[i]])
  xy_result$bass_label <- as.vector(zlabels[[i]])
  xy_result$spotid <- as.vector(rownames(xy_result))
  write.table(xy_result, file=paste0("res_bass_", i, ".txt"), row.names=FALSE)
}

