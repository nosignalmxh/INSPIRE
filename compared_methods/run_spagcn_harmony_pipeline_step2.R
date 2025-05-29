rm(list = ls())
library(harmony)
library(Seurat)
library(data.table)
set.seed(1234)

# load data
setwd("res_spagcn_latent")

latent <- read.csv("res_spagcn_sharednet.csv", row.names = 1)
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
save.dir <- "run_spagcn_harmony"
write.csv(obj@reductions$harmony@cell.embeddings, file = paste0(save.dir, "/spagcn_sharednet_harmony_latent.csv"))



