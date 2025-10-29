#!/usr/bin/env python
"""Run INSPIRE with gene-union dual-decoder configuration on Slide-seq V2 x MERFISH."""

import argparse
import os
import sys
from pathlib import Path

import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from INSPIRE.utils import preprocess, build_graph_LGCN
from INSPIRE.model import Model_LGCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INSPIRE dual-decoder brain integration experiment")
    parser.add_argument("--data_slideseq", required=True, help="Path to the Slide-seq V2 anndata (.h5ad)")
    parser.add_argument("--data_merfish", required=True, help="Path to the MERFISH anndata (.h5ad)")
    parser.add_argument("--output_dir", default="results_union", help="Directory to save model outputs")
    parser.add_argument("--num_hvgs", type=int, default=1000, help="Number of HVGs when not using gene union")
    parser.add_argument("--per_dataset_hvg", type=int, default=1000, help="Number of HVGs per dataset when constructing union")
    parser.add_argument("--min_genes_qc", type=int, default=100, help="Minimum genes per cell for QC")
    parser.add_argument("--min_cells_qc", type=int, default=100, help="Minimum cells per gene for QC")
    parser.add_argument("--spot_size", type=float, default=50.0, help="Spot size for visualization")
    parser.add_argument("--rad_cutoff", type=float, default=100.0, help="Radius for LGCN graph construction")
    parser.add_argument("--k_lgcn", type=int, default=1, help="Number of aggregation steps for node features")
    parser.add_argument("--n_spatial_factors", type=int, default=60, help="Number of spatial factors")
    parser.add_argument("--n_training_steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=2048, help="Training batch size")
    parser.add_argument("--different_platforms", action="store_true", help="Enable platform-specific basis decoder")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--lambda_ae", type=float, default=0.1, help="Coefficient for AE loss")
    parser.add_argument("--lambda_cons", type=float, default=0.1, help="Coefficient for consistency loss")
    parser.add_argument("--lambda_sparse", type=float, default=1e-4, help="Coefficient for sparsity loss")
    parser.add_argument("--warmup_steps", type=int, default=1500, help="Number of AE warmup steps")
    parser.add_argument("--concat_mask_to_input", dest="concat_mask_to_input", action="store_true", help="Concatenate measurement mask to node features")
    parser.add_argument("--no_concat_mask_to_input", dest="concat_mask_to_input", action="store_false")
    parser.add_argument("--use_interpretable_ae", dest="use_interpretable_ae", action="store_true", help="Use interpretable AE head with private loadings")
    parser.add_argument("--use_mlp_ae", dest="use_interpretable_ae", action="store_false", help="Use MLP AE heads instead of interpretable loadings")
    parser.add_argument("--disable_gene_union", dest="use_gene_union", action="store_false", help="Disable gene union workflow")
    parser.set_defaults(concat_mask_to_input=True, use_interpretable_ae=True, use_gene_union=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading input datasets...")
    adata_slideseq = sc.read_h5ad(args.data_slideseq)
    adata_merfish = sc.read_h5ad(args.data_merfish)

    try:
        adata_list, adata_full = preprocess(
            [adata_slideseq, adata_merfish],
            num_hvgs=args.num_hvgs,
            min_genes_qc=args.min_genes_qc,
            min_cells_qc=args.min_cells_qc,
            spot_size=args.spot_size,
            limit_num_genes=not args.use_gene_union,
            use_gene_union=args.use_gene_union,
            per_dataset_hvg=args.per_dataset_hvg,
            concat_mask_to_input=args.concat_mask_to_input,
        )
    except TypeError as err:
        if "use_gene_union" in str(err):
            raise RuntimeError(
                "The installed INSPIRE package is outdated. Please run this script from the updated repository "
                "root or reinstall INSPIRE from source to use the gene-union workflow."
            ) from err
        raise

    print("Constructing LGCN graphs...")
    rad_cutoff_list = [args.rad_cutoff for _ in adata_list]
    adata_list = build_graph_LGCN(
        adata_list,
        rad_cutoff_list=rad_cutoff_list,
        k_lgcn=args.k_lgcn,
        concat_mask_to_input=args.concat_mask_to_input,
    )

    print("Initialising INSPIRE model (LGCN backend)...")
    model = Model_LGCN(
        adata_list,
        n_spatial_factors=args.n_spatial_factors,
        n_training_steps=args.n_training_steps,
        batch_size=args.batch_size,
        different_platforms=args.different_platforms,
        seed=args.seed,
        use_gene_union=args.use_gene_union,
        use_interpretable_ae=args.use_interpretable_ae,
        lambda_ae=args.lambda_ae,
        lambda_cons=args.lambda_cons,
        lambda_sparse=args.lambda_sparse,
        warmup_steps=args.warmup_steps,
    )

    print("Starting training...")
    model.train(adata_list)

    print("Running evaluation/latent export...")
    adata_full, basis_df = model.eval(adata_list, adata_full)

    latent_path = os.path.join(args.output_dir, "adata_integrated.h5ad")
    basis_path = os.path.join(args.output_dir, "basis_loadings.csv")
    print(f"Saving integrated AnnData to {latent_path}")
    adata_full.write(latent_path)
    print(f"Saving shared loadings to {basis_path}")
    basis_df.to_csv(basis_path)


if __name__ == "__main__":
    main()
