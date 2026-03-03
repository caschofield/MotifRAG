import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from networkx.algorithms.triads import TRIAD_NAMES
from tqdm import tqdm

from src.dataset.motifs import load_motif_cache


DISCONNECTED_TRIADS = {"003", "012", "102"}
MOTIF_VOCAB_SIZE = 1 + len(TRIAD_NAMES)  # PAD + 16 directed triads


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_q_emb(entry) -> Optional[np.ndarray]:
    if entry is None:
        return None
    q_emb = entry.get("q_emb")
    if q_emb is None:
        return None
    if isinstance(q_emb, torch.Tensor):
        arr = q_emb.detach().cpu().numpy()
    else:
        arr = np.asarray(q_emb)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _motif_distribution_from_scored_triples(
    sample: Dict,
    top_k: int,
    use_scores: bool,
    excluded_token_ids: set,
) -> Optional[np.ndarray]:
    scored_tokens = sample.get("scored_triple_motif_tokens", [])
    scored_triples = sample.get("scored_triples", [])
    if top_k > 0:
        scored_tokens = scored_tokens[:top_k]
        scored_triples = scored_triples[:top_k]

    motif_mass = np.zeros(MOTIF_VOCAB_SIZE, dtype=np.float64)
    total_mass = 0.0

    for i, token_info in enumerate(scored_tokens):
        triple_weight = 1.0
        if use_scores and i < len(scored_triples) and len(scored_triples[i]) >= 4:
            try:
                s = float(scored_triples[i][3])
                triple_weight = s if np.isfinite(s) and s > 0.0 else 0.0
            except Exception:
                triple_weight = 1.0

        for token_id, wt in zip(token_info.get("ids", []), token_info.get("wts", [])):
            token_id = int(token_id)
            wt = float(wt)
            if token_id <= 0 or token_id >= MOTIF_VOCAB_SIZE:
                continue
            if token_id in excluded_token_ids:
                continue
            if wt <= 0.0 or triple_weight <= 0.0:
                continue
            contrib = triple_weight * wt
            motif_mass[token_id] += contrib
            total_mass += contrib

    if total_mass <= 0.0:
        return None
    motif_mass /= total_mass
    return motif_mass.astype(np.float32)


def _collect_query_pairs(
    retrieval_result: Dict,
    q_emb_lookup: Dict,
    top_k: int,
    use_scores: bool,
    excluded_token_ids: set,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    q_rows = []
    motif_rows = []
    dropped_no_q = 0
    dropped_no_motif = 0

    for sample_id, sample in tqdm(retrieval_result.items(), desc="collect", unit="sample"):
        motif_dist = _motif_distribution_from_scored_triples(
            sample=sample,
            top_k=top_k,
            use_scores=use_scores,
            excluded_token_ids=excluded_token_ids,
        )
        if motif_dist is None:
            dropped_no_motif += 1
            continue

        q_entry = q_emb_lookup.get(sample_id)
        if q_entry is None:
            q_entry = q_emb_lookup.get(str(sample_id))
        q_vec = _extract_q_emb(q_entry)
        if q_vec is None:
            dropped_no_q += 1
            continue

        q_rows.append(q_vec)
        motif_rows.append(motif_dist)

    if len(q_rows) == 0:
        raise ValueError("No valid query-motif pairs were built.")

    stats = {
        "total_samples": int(len(retrieval_result)),
        "num_valid_pairs": int(len(q_rows)),
        "dropped_no_q_emb": int(dropped_no_q),
        "dropped_no_motif_dist": int(dropped_no_motif),
    }
    return (
        np.stack(q_rows, axis=0).astype(np.float32),
        np.stack(motif_rows, axis=0).astype(np.float32),
        stats,
    )


def _accumulate_distribution_from_rows(
    ids_rows: np.ndarray,
    wts_rows: np.ndarray,
    selected_rows: np.ndarray,
    excluded_token_ids: set,
) -> Optional[np.ndarray]:
    motif_mass = np.zeros(MOTIF_VOCAB_SIZE, dtype=np.float64)
    total_mass = 0.0

    for row_idx in selected_rows.tolist():
        row_ids = ids_rows[row_idx]
        row_wts = wts_rows[row_idx]
        for token_id, wt in zip(row_ids, row_wts):
            token_id = int(token_id)
            wt = float(wt)
            if token_id <= 0 or token_id >= MOTIF_VOCAB_SIZE:
                continue
            if token_id in excluded_token_ids:
                continue
            if wt <= 0.0:
                continue
            motif_mass[token_id] += wt
            total_mass += wt

    if total_mass <= 0.0:
        return None
    motif_mass /= total_mass
    return motif_mass.astype(np.float32)


def _infer_dataset_from_q_emb_path(q_emb_file: str) -> Optional[str]:
    parts = os.path.normpath(q_emb_file).split(os.sep)
    for i, part in enumerate(parts):
        if part == "data_files" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _parse_pool_splits(pool_splits_csv: str) -> List[str]:
    splits = [x.strip() for x in pool_splits_csv.split(",") if x.strip()]
    if len(splits) == 0:
        raise ValueError("pool_splits is empty")
    for split in splits:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split in pool_splits: {split}")
    return splits


def _build_global_pool_rows(
    dataset_name: str,
    pool_splits: List[str],
    motif_top_k_tokens: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    ids_chunks = []
    wts_chunks = []
    num_samples = 0

    for split in pool_splits:
        motif_dict = load_motif_cache(
            dataset_name=dataset_name,
            split=split,
            top_k=motif_top_k_tokens,
        )
        for entry in tqdm(motif_dict.values(), desc=f"pool:{split}", unit="sample"):
            ids_rows = np.asarray(entry.get("triple_motif_token_ids", []), dtype=np.int16)
            wts_rows = np.asarray(entry.get("triple_motif_token_wts", []), dtype=np.float32)
            if ids_rows.ndim != 2 or ids_rows.shape[0] == 0:
                continue
            ids_chunks.append(ids_rows)
            wts_chunks.append(wts_rows)
            num_samples += 1

    if len(ids_chunks) == 0:
        raise ValueError("Global random pool is empty. Check motif caches.")

    pool_ids = np.concatenate(ids_chunks, axis=0).astype(np.int16, copy=False)
    pool_wts = np.concatenate(wts_chunks, axis=0).astype(np.float32, copy=False)

    return pool_ids, pool_wts, int(num_samples)


def _build_random_distributions_from_global_pool(
    n_queries: int,
    pool_ids: np.ndarray,
    pool_wts: np.ndarray,
    random_k: int,
    excluded_token_ids: set,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_pool = pool_ids.shape[0]
    if n_pool == 0:
        raise ValueError("Global random pool has zero triples")

    y_rand = []
    for _ in tqdm(range(n_queries), desc="random:global-pool", unit="sample"):
        if random_k > 0 and n_pool > random_k:
            selected = rng.choice(n_pool, size=random_k, replace=False)
        else:
            selected = np.arange(n_pool, dtype=np.int64)
        dist = _accumulate_distribution_from_rows(pool_ids, pool_wts, selected, excluded_token_ids)
        if dist is None:
            raise ValueError("Failed to build random distribution from global pool")
        y_rand.append(dist)

    return np.stack(y_rand, axis=0).astype(np.float32)


def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sq_norm = np.sum(x * x, axis=1, keepdims=True)
    d2 = sq_norm + sq_norm.T - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    np.sqrt(d2, out=d2)
    return d2


def _double_center(d: np.ndarray) -> np.ndarray:
    row_mean = d.mean(axis=1, keepdims=True)
    col_mean = d.mean(axis=0, keepdims=True)
    return d - row_mean - col_mean + d.mean()


def _distance_correlation_from_centered(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    dcov2 = float(np.mean(a * b))
    dvar_x = float(np.mean(a * a))
    dvar_y = float(np.mean(b * b))
    denom = np.sqrt(max(dvar_x * dvar_y, eps))
    dcor2 = max(dcov2, 0.0) / denom
    return float(np.sqrt(max(dcor2, 0.0)))


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same number of rows")
    if x.shape[0] < 4:
        raise ValueError("Need at least 4 paired samples")
    a = _double_center(_pairwise_euclidean(x))
    b = _double_center(_pairwise_euclidean(y))
    return _distance_correlation_from_centered(a, b)


def main(args):
    print(f"Loading retrieval result: {args.retrieval_result}")
    retrieval_result = _torch_load(args.retrieval_result)
    print(f"Loading q_emb file: {args.q_emb_file}")
    q_emb_lookup = _torch_load(args.q_emb_file)

    dataset_name = args.dataset.strip() if args.dataset else ""
    if len(dataset_name) == 0:
        inferred = _infer_dataset_from_q_emb_path(args.q_emb_file)
        if inferred is not None:
            dataset_name = inferred
    if len(dataset_name) == 0:
        raise ValueError("Could not infer dataset; pass --dataset")

    excluded_triads = DISCONNECTED_TRIADS if args.exclude_disconnected else set()
    excluded_token_ids = {TRIAD_NAMES.index(name) + 1 for name in excluded_triads}

    q_mat, y_retrieved, collect_stats = _collect_query_pairs(
        retrieval_result=retrieval_result,
        q_emb_lookup=q_emb_lookup,
        top_k=args.top_k,
        use_scores=(not args.uniform_triple_weight),
        excluded_token_ids=excluded_token_ids,
    )

    pool_splits = _parse_pool_splits(args.pool_splits)
    pool_ids, pool_wts, pool_num_samples = _build_global_pool_rows(
        dataset_name=dataset_name,
        pool_splits=pool_splits,
        motif_top_k_tokens=args.motif_top_k_tokens,
    )
    y_random = _build_random_distributions_from_global_pool(
        n_queries=q_mat.shape[0],
        pool_ids=pool_ids,
        pool_wts=pool_wts,
        random_k=args.top_k,
        excluded_token_ids=excluded_token_ids,
        seed=args.seed,
    )

    observed_dcor = _distance_correlation(q_mat, y_retrieved)
    random_dcor = _distance_correlation(q_mat, y_random)

    print("\n=== Motif Dependence Test (distance correlation) ===")
    print(f"dataset: {dataset_name}")
    print(f"pool splits: {pool_splits}")
    print(f"paired samples used: {q_mat.shape[0]}")
    print(f"pool samples/triples: {pool_num_samples} / {pool_ids.shape[0]}")
    print(
        f"dropped no q/no motif: "
        f"{collect_stats['dropped_no_q_emb']} / {collect_stats['dropped_no_motif_dist']}"
    )
    print(f"q_emb dim: {q_mat.shape[1]} | motif dim: {y_retrieved.shape[1]}")
    print(f"observed dCor: {observed_dcor:.6f}")
    print(f"random-candidate dCor: {random_dcor:.6f}")
    print(f"delta (observed-random): {observed_dcor - random_dcor:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Query-motif dependence test (top-K vs global random pool)")
    parser.add_argument("--retrieval_result", type=str, required=True, help="Path to retrieval_result.pth")
    parser.add_argument("--q_emb_file", type=str, required=True, help="Path to emb/.../{split}.pth containing q_emb by sample id")
    parser.add_argument("--dataset", type=str, default="", choices=["", "webqsp", "cwq"], help="Dataset name for motif cache (auto-infer if omitted)")
    parser.add_argument("--motif_top_k_tokens", type=int, default=4, help="Motif top-k used in motif preprocessing")
    parser.add_argument("--top_k", type=int, default=100, help="K for retrieved triples and random global-pool triples")
    parser.add_argument("--pool_splits", type=str, default="train,val,test", help="Comma-separated splits for global random pool")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--uniform_triple_weight", action="store_true", help="Use uniform triple weight instead of score-weighted motif mass")
    parser.add_argument("--exclude_disconnected", dest="exclude_disconnected", action="store_true", help="Exclude 003/012/102 motifs")
    parser.add_argument("--include_disconnected", dest="exclude_disconnected", action="store_false", help="Include 003/012/102 motifs")
    parser.set_defaults(exclude_disconnected=True)
    args = parser.parse_args()
    main(args)
