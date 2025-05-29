rm(list = ls())
library(Seurat)
library(harmony)
set.seed(1234)


# The data with format required by Harmony are available at: 
# https://drive.google.com/file/d/1g763Z7ClovTDn7aQXj4hJs6cxLFjvQAS/view?usp=sharing


# load data
data_path <- "data_for_R"
load(paste0(data_path, "/DLPFC_4slices.RData"))
obj.list <- SplitObject(obj, split.by = "slice_id")


# perform standard preprocessing
for (i in 1:length(obj.list)) {
    obj.list[[i]] <- NormalizeData(obj.list[[i]], verbose = FALSE)
    obj.list[[i]] <- FindVariableFeatures(obj.list[[i]], selection.method = "vst", nfeatures = 2000, verbose = FALSE)
    if (i == 1) {
        features <- obj.list[[i]]@assays$RNA@var.features
    } else {
        features <- intersect(features, obj.list[[i]]@assays$RNA@var.features)
    }
}

# concate data with features
for (i in 1:length(obj.list)) {
    tmp_meta <- obj.list[[i]]@meta.data
    tmp_meta <- tmp_meta[, c("slice_id", "layer", "barcode", "spatial_1", "spatial_2")]
    if (i == 1){
        norm_data <- obj.list[[i]]@assays$RNA@data[features, ]
        meta_data <- tmp_meta
    }else{
        norm_data <- cbind(norm_data, obj.list[[i]]@assays$RNA@data[features, ])
        meta_data <- rbind(meta_data, tmp_meta)
    }
}

psd_counts <- matrix(0, nrow = nrow(norm_data), ncol = ncol(norm_data))
rownames(psd_counts) <- rownames(norm_data)
colnames(psd_counts) <- colnames(norm_data)
obj <- CreateSeuratObject(counts = psd_counts, meta.data = meta_data)
obj@assays$RNA@data <- norm_data
obj@assays$RNA@var.features <- features


# preprocess
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, npcs = 20, verbose = FALSE)


# run harmony
obj <- RunHarmony(obj, "slice_id")


# save rds and latent
save.dir <- "run_harmony"
save(obj, file = paste0(save.dir, "/harmony_obj.RData"))
write.csv(obj@reductions$harmony@cell.embeddings, file = paste0(save.dir, "/harmony_latent.csv"))



