rm(list = ls())
library(Seurat)
set.seed(1234)


# # The data with format required by Seurat are available at: 
# https://drive.google.com/file/d/1g763Z7ClovTDn7aQXj4hJs6cxLFjvQAS/view?usp=sharing


# load data
data_path <- "data_for_R"
load(paste0(data_path, "/DLPFC_4slices.RData"))
obj.list <- SplitObject(obj, split.by = "slice_id")


# perform standard preprocessing
for (i in 1:length(obj.list)) {
    obj.list[[i]] <- NormalizeData(obj.list[[i]], verbose = FALSE)
    obj.list[[i]] <- FindVariableFeatures(obj.list[[i]], selection.method = "vst", nfeatures = 2000, verbose = FALSE)
}


# select features that are repeatedly variable across datasets for integration
features <- SelectIntegrationFeatures(object.list = obj.list)


# identify anchors and perform integration
obj.anchors <- FindIntegrationAnchors(object.list = obj.list, dims = 1:30, anchor.features = features)
obj.integrated <- IntegrateData(anchorset = obj.anchors, dims = 1:30)


# integrated umap
DefaultAssay(obj.integrated) <- "integrated"
obj.integrated <- ScaleData(obj.integrated, verbose = FALSE)
obj.integrated <- RunPCA(obj.integrated, npcs = 30, verbose = FALSE)
obj.integrated <- RunUMAP(obj.integrated, reduction = "pca", dims = 1:30, verbose = FALSE)


# save rds and umap plots
save.dir <- "run_seurat"
save(obj.integrated, file = paste0(save.dir, "/obj_integrated_dlpfc_4slices.RData"))



