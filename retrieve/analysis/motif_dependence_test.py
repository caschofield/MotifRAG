import argparse
import gc
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from networkx.algorithms.triads import TRIAD_NAMES
from tqdm import tqdm


DISCONNECTED_TRIADS = {"003", "012", "102"}
MOTIF_VOCAB_SIZE = 1 + len(TRIAD_NAMES)  # PAD + 16 directed triads


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_q_emb(entry) -> Optional[np.ndarray]:
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


def _sample_motif_distribution(
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
                raw_score = float(scored_triples[i][3])
                triple_weight = raw_score if np.isfinite(raw_score) and raw_score > 0.0 else 0.0
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
    return motif_mass


def _collect_pairs(
    retrieval_result: Dict,
    q_emb_lookup: Dict,
    top_k: int,
    use_scores: bool,
    excluded_token_ids: set,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    q_embs = []
    motif_dists = []

    dropped_no_motif = 0
    dropped_no_q_emb = 0
    total = len(retrieval_result)

    pbar = tqdm(total=total, desc="collect", unit="sample")
    while len(retrieval_result) > 0:
        sample_id, sample = retrieval_result.popitem()
        sid = str(sample_id)

        motif_dist = _sample_motif_distribution(
            sample=sample,
            top_k=top_k,
            use_scores=use_scores,
            excluded_token_ids=excluded_token_ids,
        )
        if motif_dist is None:
            dropped_no_motif += 1
            pbar.update(1)
            continue

        q_entry = q_emb_lookup.get(sample_id)
        if q_entry is None:
            q_entry = q_emb_lookup.get(sid)
        q_vec = _extract_q_emb(q_entry) if q_entry is not None else None
        if q_vec is None:
            dropped_no_q_emb += 1
            pbar.update(1)
            continue

        q_embs.append(q_vec)
        motif_dists.append(motif_dist.astype(np.float32))
        pbar.update(1)
    pbar.close()

    if len(q_embs) == 0:
        raise ValueError("No valid query-motif pairs were built.")

    q_mat = np.stack(q_embs, axis=0).astype(np.float32)
    y_mat = np.stack(motif_dists, axis=0).astype(np.float32)
    drop_stats = {
        "dropped_no_motif_dist": int(dropped_no_motif),
        "dropped_no_q_emb": int(dropped_no_q_emb),
    }
    return q_mat, y_mat, drop_stats


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


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows.")
    if x.shape[0] < 4:
        raise ValueError("Need at least 4 paired samples for a stable dependence estimate.")

    a = _double_center(_pairwise_euclidean(x))
    b = _double_center(_pairwise_euclidean(y))
    return _distance_correlation_from_centered(a, b), a, b


def _query_shuffled_null(a: np.ndarray, b: np.ndarray, num_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    out = np.zeros(num_perm, dtype=np.float64)
    for i in tqdm(range(num_perm), desc="shuffle-null", unit="perm"):
        perm = rng.permutation(n)
        a_perm = a[np.ix_(perm, perm)]
        out[i] = _distance_correlation_from_centered(a_perm, b)
    return out


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading retrieval result: {args.retrieval_result}")
    retrieval_result = _torch_load(args.retrieval_result)
    total_queries = len(retrieval_result)

    print(f"Loading q_emb file: {args.q_emb_file}")
    q_emb_lookup = _torch_load(args.q_emb_file)

    excluded_triads = DISCONNECTED_TRIADS if args.exclude_disconnected else set()
    excluded_token_ids = {TRIAD_NAMES.index(name) + 1 for name in excluded_triads}

    q_mat, y_mat, drop_stats = _collect_pairs(
        retrieval_result=retrieval_result,
        q_emb_lookup=q_emb_lookup,
        top_k=args.top_k,
        use_scores=(not args.uniform_triple_weight),
        excluded_token_ids=excluded_token_ids,
    )
    del retrieval_result
    del q_emb_lookup
    gc.collect()

    print(f"Paired samples used: {q_mat.shape[0]}")
    print(f"q_emb dim: {q_mat.shape[1]} | motif dim: {y_mat.shape[1]}")
    observed_dcor, a, b = _distance_correlation(q_mat, y_mat)

    null_vals = _query_shuffled_null(a=a, b=b, num_perm=args.num_permutations, seed=args.seed)

    null_mean = float(np.mean(null_vals))
    null_std = float(np.std(null_vals))
    ge_count = int(np.sum(null_vals >= observed_dcor))
    p_value = float((1.0 + ge_count) / (1.0 + len(null_vals)))
    z_score = float((observed_dcor - null_mean) / max(null_std, 1e-12))

    stats = {
        "retrieval_result": os.path.abspath(args.retrieval_result),
        "q_emb_file": os.path.abspath(args.q_emb_file),
        "top_k": int(args.top_k),
        "num_queries_total": int(total_queries),
        "num_pairs_used": int(q_mat.shape[0]),
        "num_permutations": int(args.num_permutations),
        "exclude_disconnected": bool(args.exclude_disconnected),
        "uniform_triple_weight": bool(args.uniform_triple_weight),
        "drop_stats": drop_stats,
        "distance_correlation_observed": float(observed_dcor),
        "null_mean": null_mean,
        "null_std": null_std,
        "null_median": float(np.median(null_vals)),
        "null_p95": float(np.percentile(null_vals, 95)),
        "null_p99": float(np.percentile(null_vals, 99)),
        "z_score": z_score,
        "p_value_one_sided": p_value,
        "effect_over_null_mean": float(observed_dcor - null_mean),
    }

    out_file = os.path.join(args.output_dir, args.output_name)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Motif Dependence Test (distance correlation) ===")
    print(f"observed dCor: {observed_dcor:.6f}")
    print(f"null mean/std: {null_mean:.6f} / {null_std:.6f}")
    print(f"z-score       : {z_score:.3f}")
    print(f"p-value (1-sided, shuffled null): {p_value:.6g}")
    print(f"saved         : {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Query-Motif dependence test with query-shuffled null")
    parser.add_argument("--retrieval_result", type=str, required=True, help="Path to retrieval_result.pth")
    parser.add_argument("--q_emb_file", type=str, required=True, help="Path to emb/.../{split}.pth containing q_emb by sample id")
    parser.add_argument("--top_k", type=int, default=100, help="Use top-K retrieved triples per query")
    parser.add_argument("--num_permutations", type=int, default=300, help="Permutation count for shuffled null")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--uniform_triple_weight", action="store_true", help="Use uniform triple weight instead of score-weighted motif mass")
    parser.add_argument(
        "--exclude_disconnected",
        dest="exclude_disconnected",
        action="store_true",
        help="Exclude 003/012/102 motifs",
    )
    parser.add_argument(
        "--include_disconnected",
        dest="exclude_disconnected",
        action="store_false",
        help="Include 003/012/102 motifs",
    )
    parser.add_argument("--output_dir", type=str, default="analysis/figures", help="Output directory")
    parser.add_argument("--output_name", type=str, default="motif_dependence_test.json", help="Output JSON filename")
    parser.set_defaults(exclude_disconnected=True)
    args = parser.parse_args()
    main(args)
