import os

from argparse import ArgumentParser

from src.config.retriever import load_yaml
from src.dataset.motifs import (
    EXCLUDED_TRIAD_NAMES,
    build_motif_cache_for_split,
    motif_cache_file,
)


def parse_splits(splits: str):
    return [s.strip() for s in splits.split(",") if s.strip()]


def main(args):
    config_file = f"configs/retriever/{args.dataset}.yaml"
    config = load_yaml(config_file)
    motif_cfg = config.get("motif", {})

    top_k_tokens = args.top_k_tokens if args.top_k_tokens is not None else motif_cfg.get("top_k_tokens", 4)
    backend = args.backend if args.backend is not None else motif_cfg.get("backend", "python")
    orca_path = args.orca_path if args.orca_path is not None else motif_cfg.get("orca_path", "")
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() or 1)
    start_method = args.start_method
    shard_size = args.shard_size
    splits = parse_splits(args.splits)

    print("Motif preprocessing configuration")
    print(f"dataset: {args.dataset}")
    print(f"splits: {splits}")
    print(f"top_k_tokens: {top_k_tokens}")
    print(f"backend: {backend}")
    print(f"num_workers: {num_workers}")
    print(f"start_method: {start_method}")
    print(f"shard_size: {shard_size}")
    print(f"overwrite: {args.overwrite}")
    print(f"excluded_triads: {sorted(EXCLUDED_TRIAD_NAMES)}")

    for split in splits:
        out_file = motif_cache_file(args.dataset, split, top_k=top_k_tokens, backend=backend)
        print(f"\nBuilding motif cache for split={split}")
        print(f"output: {out_file}")
        build_motif_cache_for_split(
            dataset_name=args.dataset,
            split=split,
            top_k=top_k_tokens,
            backend=backend,
            orca_path=orca_path,
            num_workers=num_workers,
            start_method=start_method,
            shard_size=shard_size,
            overwrite=args.overwrite,
        )
    print("\nDone.")


if __name__ == "__main__":
    parser = ArgumentParser("Offline motif cache preprocessing")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["webqsp", "cwq"])
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to preprocess",
    )
    parser.add_argument("--top_k_tokens", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--orca_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--start_method",
        type=str,
        default="spawn",
        choices=["spawn", "forkserver", "fork"],
        help="Multiprocessing start method. Use spawn on HPC to avoid fork deadlocks.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=2000,
        help="Number of samples per shard file for streaming writes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
