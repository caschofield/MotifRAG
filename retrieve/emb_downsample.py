import os
import shutil
import time

import numpy as np
import torch

from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset


RETRIEVE_ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT_KEY_TO_HF = {
    "train": "train",
    "val": "validation",
    "test": "test",
}
SPLIT_SEED_OFFSET = {
    "train": 0,
    "val": 1,
    "test": 2,
}


def parse_splits(splits: str):
    parsed = [s.strip() for s in splits.split(",") if s.strip()]
    valid = set(SPLIT_KEY_TO_HF.keys())
    invalid = [s for s in parsed if s not in valid]
    if len(invalid) > 0:
        raise ValueError(
            f"Invalid split(s): {invalid}. Valid choices are: {sorted(valid)}"
        )
    if len(parsed) == 0:
        raise ValueError("No valid splits were provided.")
    return parsed


def _sample_indices(n: int, factor: int, seed: int):
    if factor < 1:
        raise ValueError(f"--factor must be >= 1, got {factor}")
    sample_size = max(1, n // factor)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=sample_size, replace=False)
    indices = np.sort(indices)
    return indices.astype(int).tolist(), sample_size


def downsample_raw_splits(raw_splits, split_keys, factor: int, seed: int):
    sampled_raw_splits = {}
    split_stats = {}
    for split_key in split_keys:
        raw_set = raw_splits[split_key]
        split_seed = seed + SPLIT_SEED_OFFSET[split_key]
        indices, sample_size = _sample_indices(len(raw_set), factor=factor, seed=split_seed)
        sampled_raw_splits[split_key] = raw_set.select(indices)
        split_stats[split_key] = {
            "seed": split_seed,
            "raw": len(raw_set),
            "sampled": sample_size,
        }
    return sampled_raw_splits, split_stats


def _remove_path(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def backup_or_clear_targets(dataset_root: str, target_paths, overwrite: bool, backup: bool):
    existing = [p for p in target_paths if os.path.exists(p)]
    if len(existing) == 0:
        return None

    if not overwrite:
        raise FileExistsError(
            "Found existing generated outputs:\n"
            + "\n".join(existing)
            + "\nRe-run with --overwrite (and optionally --backup/--no-backup)."
        )

    backup_root = None
    if backup:
        backup_root = os.path.join(
            dataset_root,
            "backups",
            f"emb_downsample_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}",
        )
        os.makedirs(backup_root, exist_ok=True)
        for src in existing:
            dst = os.path.join(backup_root, os.path.basename(src))
            shutil.move(src, dst)
            print(f"Backed up: {src} -> {dst}")
    else:
        for src in existing:
            _remove_path(src)
            print(f"Removed: {src}")

    return backup_root


def get_emb(subset, text_encoder, save_file, split_name):
    emb_dict = {}
    for i in tqdm(range(len(subset)), desc=f"emb:{split_name}"):
        sample_id, q_text, text_entity_list, relation_list = subset[i]
        q_emb, entity_embs, relation_embs = text_encoder(
            q_text, text_entity_list, relation_list
        )
        emb_dict[sample_id] = {
            "q_emb": q_emb,
            "entity_embs": entity_embs,
            "relation_embs": relation_embs,
        }
    torch.save(emb_dict, save_file)


def load_raw_splits(dataset: str, split_keys):
    if dataset == "cwq":
        input_file = os.path.join("rmanluo", "RoG-cwq")
    elif dataset == "webqsp":
        input_file = os.path.join("ml1996", "webqsp")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    raw_splits = {}
    for split_key in split_keys:
        hf_split = SPLIT_KEY_TO_HF[split_key]
        raw_splits[split_key] = load_dataset(input_file, split=hf_split)
        print(f"Loaded raw split={split_key} ({hf_split}) size={len(raw_splits[split_key])}")
    return raw_splits


def main(args):
    split_keys = parse_splits(args.splits)

    config_file = f"configs/emb/gte-large-en-v1.5/{args.dataset}.yaml"
    config = load_yaml(config_file)
    torch.set_num_threads(config["env"]["num_threads"])

    raw_splits = load_raw_splits(args.dataset, split_keys)
    sampled_raw_splits, split_stats = downsample_raw_splits(
        raw_splits=raw_splits,
        split_keys=split_keys,
        factor=args.factor,
        seed=args.seed,
    )

    for split_key in split_keys:
        stats = split_stats[split_key]
        print(
            f"Sampled split={split_key}: {stats['sampled']} / {stats['raw']} "
            f"(seed={stats['seed']})"
        )

    entity_identifiers = []
    with open(config["entity_identifier_file"], "r") as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)

    dataset_root = os.path.join(RETRIEVE_ROOT, "data_files", args.dataset)
    os.makedirs(dataset_root, exist_ok=True)

    processed_dir = os.path.join(dataset_root, "processed")
    emb_root = os.path.join(dataset_root, "emb")
    triple_scores_dir = os.path.join(dataset_root, "triple_scores")
    motif_tokens_dir = os.path.join(dataset_root, "motif_tokens")

    target_paths = [
        processed_dir,
        emb_root,
        triple_scores_dir,
        motif_tokens_dir,
    ]
    backup_root = backup_or_clear_targets(
        dataset_root=dataset_root,
        target_paths=target_paths,
        overwrite=args.overwrite,
        backup=args.backup,
    )

    os.makedirs(processed_dir, exist_ok=True)

    processed_sets = {}
    for split_key in split_keys:
        processed_file = os.path.join(processed_dir, f"{split_key}.pkl")
        if split_key == "test":
            processed_sets[split_key] = EmbInferDataset(
                sampled_raw_splits[split_key],
                entity_identifiers,
                processed_file,
                skip_no_topic=False,
                skip_no_ans=False,
            )
        else:
            processed_sets[split_key] = EmbInferDataset(
                sampled_raw_splits[split_key],
                entity_identifiers,
                processed_file,
            )

    device = torch.device(args.device)
    text_encoder_name = config["text_encoder"]["name"]
    if text_encoder_name == "gte-large-en-v1.5":
        from src.model.text_encoders import GTELargeEN

        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(text_encoder_name)

    emb_save_dir = os.path.join(emb_root, text_encoder_name)
    os.makedirs(emb_save_dir, exist_ok=True)

    for split_key in split_keys:
        emb_file = os.path.join(emb_save_dir, f"{split_key}.pth")
        get_emb(processed_sets[split_key], text_encoder, emb_file, split_key)

    print("\nDownsampling complete.")
    if backup_root is not None:
        print(f"backup_dir: {backup_root}")
    print(f"processed_dir: {processed_dir}")
    print(f"emb_dir: {emb_save_dir}")
    print("split stats:")
    for split_key in split_keys:
        stats = split_stats[split_key]
        print(
            f"  - {split_key}: raw={stats['raw']} sampled={stats['sampled']} "
            f"processed={len(processed_sets[split_key])}"
        )


if __name__ == "__main__":
    parser = ArgumentParser("Downsample + embed preprocessing")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cwq",
        choices=["cwq"],
        help="Dataset name (CWQ-only downsampling path).",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=10,
        help="Downsampling factor. Sample size is floor(len(split)/factor), min 1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split keys from: train,val,test",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing generated outputs.",
    )

    backup_group = parser.add_mutually_exclusive_group()
    backup_group.add_argument(
        "--backup",
        dest="backup",
        action="store_true",
        help="Back up existing generated outputs before overwrite (default).",
    )
    backup_group.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Do not back up; delete existing generated outputs on overwrite.",
    )
    parser.set_defaults(backup=True)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for text encoder embedding inference.",
    )

    args = parser.parse_args()
    main(args)
