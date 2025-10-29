#!/usr/bin/env python
"""Utility script to compute basic quality metrics for Slide-seq V2 x MERFISH union model."""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score


try:
    import scipy
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required for evaluation") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate INSPIRE union experiment")
    parser.add_argument("--integrated_h5ad", required=True, help="Integrated AnnData file from run_difftech_brain_union")
    parser.add_argument("--slice_key", default="slice", help="Observation column with technology labels")
    parser.add_argument("--latent_key", default="latent", help="obsm key containing latent embedding")
    parser.add_argument("--output_dir", default="results_union", help="Directory to store evaluation outputs")
    return parser.parse_args()


def compute_tech_asw(adata: sc.AnnData, slice_key: str, latent_key: str) -> Optional[float]:
    if slice_key not in adata.obs.columns or latent_key not in adata.obsm:
        return None
    labels = adata.obs[slice_key].astype(str).values
    latent = np.asarray(adata.obsm[latent_key])
    if latent.shape[0] != labels.shape[0] or latent.shape[0] < 3:
        return None
    try:
        score = silhouette_score(latent, labels)
    except ValueError:
        return None
    return float(score)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading integrated AnnData...")
    adata = sc.read_h5ad(args.integrated_h5ad)

    metrics = {}
    tech_asw = compute_tech_asw(adata, args.slice_key, args.latent_key)
    metrics["tech_ASW"] = tech_asw

    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
