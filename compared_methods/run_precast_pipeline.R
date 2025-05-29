rm(list = ls())
library(Seurat)
library(PRECAST)
set.seed(1234)


# The data with format required by PRECAST are available at: 
# https://drive.google.com/file/d/1g763Z7ClovTDn7aQXj4hJs6cxLFjvQAS/view?usp=sharing


# load data
data_path <- "data_for_R"
load(paste0(data_path, "/DLPFC_4slices.RData"))
obj@meta.data$row <- as.vector(obj@meta.data$array_row)
obj@meta.data$col <- as.vector(obj@meta.data$array_col)
obj.list <- SplitObject(obj, split.by = "slice_id")


# run precast
seuList <- obj.list
preobj <- CreatePRECASTObject(seuList = seuList, selectGenesMethod = "HVGs", gene.number = 2000)
PRECASTObj <- AddAdjList(preobj, platform = "Visium")
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = TRUE, coreNum = 1, maxIter = 30, verbose = TRUE)
PRECASTObj <- PRECAST(PRECASTObj, K = 7)
resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")


# save results
setwd("run_precast/res_precast")
save(resList, file = "resList_dlpfc.RData")
save(PRECASTObj, file = "PRECASTObj_dlpfc.RData")
save(seuInt, file = "seuInt_dlpfc.RData")

write.csv(seuInt@meta.data, file = "res_cluster.csv")
write.csv(seuInt@reductions[["PRECAST"]]@cell.embeddings, file = "res_latent.csv")
