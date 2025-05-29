rm(list = ls())
library(harmony)
library(Seurat)
library(data.table)
set.seed(1234)

# load data
setwd("res_spatialglue")

latent <- read.csv("latent_spatialglue.csv", row.names = 1)
meta <- read.csv("meta.csv", row.names = 1)

latent <- as.matrix(latent)
obj <- CreateSeuratObject(counts = t(latent*0), meta.data = meta)
obj@assays$RNA@data <- t(latent*0)
obj@assays$RNA@scale.data <- t(latent*0)

pca.dr <- CreateDimReducObject(
  embeddings = as.matrix(latent),
  key = "PC",
  assay = "RNA"
)
obj[['pca']] <- pca.dr

obj <- RunHarmony(obj, 'slice_id')

# save rds and latent
save.dir <- "res_spatialglue"
write.csv(obj@reductions$harmony@cell.embeddings, file = paste0(save.dir, "/spatialglue_harmony_latent.csv"))



