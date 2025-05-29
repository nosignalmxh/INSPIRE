rm(list = ls())
library(Seurat)
library(rliger)
set.seed(1234)


# The data with format required by LIGER are available at: 
# https://drive.google.com/file/d/1g763Z7ClovTDn7aQXj4hJs6cxLFjvQAS/view?usp=sharing


# load data
data_path <- "data_for_R"
load(paste0(data_path, "/DLPFC_4slices.RData"))
obj@meta.data$row <- as.vector(obj@meta.data$array_row)
obj@meta.data$col <- as.vector(obj@meta.data$array_col)
obj.list <- SplitObject(obj, split.by = "slice_id")

# perform standard preprocessing
for (i in 1:length(obj.list)) {
    obj.list[[i]] <- FindVariableFeatures(obj.list[[i]], selection.method = "vst", nfeatures = 6000, verbose = FALSE)
    obj.list[[i]] <- obj.list[[i]][obj.list[[i]]@assays$RNA@var.features, ]
}

# run liger
ifnb_liger <- createLiger(list(slice1 = as.matrix(obj.list[[1]][['RNA']]@counts), 
	                           slice2 = as.matrix(obj.list[[2]][['RNA']]@counts),
	                           slice3 = as.matrix(obj.list[[3]][['RNA']]@counts),
	                           slice4 = as.matrix(obj.list[[4]][['RNA']]@counts)))
ifnb_liger <- normalize(ifnb_liger)
ifnb_liger <- selectGenes(ifnb_liger)
ifnb_liger <- scaleNotCenter(ifnb_liger)

ifnb_liger <- optimizeALS(ifnb_liger, k = 20)
ifnb_liger <- quantile_norm(ifnb_liger)


# save results
setwd("run_liger")
save(ifnb_liger, file = "liger_obj.RData")

H.norm <- ifnb_liger@H.norm
write.csv(H.norm, file = "liger_latent.csv")
