import argparse
import csv
import gc
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

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


def _triad_edge_count(triad_name: str) -> int:
    edge_count_map = {
        "003": 0,
        "012": 1,
        "102": 2,
        "021D": 2,
        "021U": 2,
        "021C": 2,
        "111D": 3,
        "111U": 3,
        "030T": 3,
        "030C": 3,
        "201": 4,
        "120D": 4,
        "120U": 4,
        "120C": 4,
        "210": 5,
        "300": 6,
    }
    if triad_name not in edge_count_map:
        raise ValueError(f"Unknown triad name: {triad_name}")
    return edge_count_map[triad_name]


def _auto_k(num_samples: int) -> int:
    # Practical default: moderate cluster count that scales sublinearly.
    return int(np.clip(round(np.sqrt(max(2, num_samples) / 30.0)), 4, 12))


def _parse_motif_names(name_csv: str) -> List[str]:
    names = [x.strip() for x in name_csv.split(",") if len(x.strip()) > 0]
    if len(names) == 0:
        raise ValueError("No motif names were provided.")
    unknown = [x for x in names if x not in TRIAD_NAMES]
    if len(unknown) > 0:
        raise ValueError(f"Unknown motif names: {unknown}")
    return names


def _resolve_emb_path(dataset: Optional[str], text_encoder: str, split: str, emb_path: Optional[str]) -> str:
    if emb_path is not None:
        return emb_path
    if dataset is None:
        raise ValueError("Either --emb_path or --dataset must be provided.")
    return os.path.join("data_files", dataset, "emb", text_encoder, f"{split}.pth")


def _resolve_key(d: Dict, sample_id: str):
    candidates = [sample_id]
    try:
        i = int(sample_id)
        candidates.extend([i, str(i)])
    except Exception:
        pass
    for k in candidates:
        if k in d:
            return k
    return None


def _to_numpy_q_emb(q_emb) -> np.ndarray:
    if isinstance(q_emb, torch.Tensor):
        arr = q_emb.detach().cpu().float().numpy()
    else:
        arr = np.asarray(q_emb, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32)


def _sample_retrieved_motif_distribution(
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
                if np.isfinite(raw_score) and raw_score > 0.0:
                    triple_weight = raw_score
                else:
                    triple_weight = 0.0
            except Exception:
                triple_weight = 1.0

        ids = token_info.get("ids", [])
        wts = token_info.get("wts", [])
        for token_id, wt in zip(ids, wts):
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


def _collect_motif_distributions(
    retrieval_result: Dict,
    top_k: int,
    use_scores: bool,
    excluded_token_ids: set,
) -> Tuple[List[str], np.ndarray]:
    sample_ids: List[str] = []
    distributions: List[np.ndarray] = []

    total = len(retrieval_result)
    pbar = tqdm(total=total, desc="motif-dist", unit="sample")
    while len(retrieval_result) > 0:
        sample_id, sample = retrieval_result.popitem()
        dist = _sample_retrieved_motif_distribution(
            sample=sample,
            top_k=top_k,
            use_scores=use_scores,
            excluded_token_ids=excluded_token_ids,
        )
        if dist is not None:
            sample_ids.append(str(sample_id))
            distributions.append(dist)
        pbar.update(1)
    pbar.close()

    if len(distributions) == 0:
        raise ValueError("No valid motif distributions were extracted from retrieval_result.")

    return sample_ids, np.stack(distributions, axis=0).astype(np.float64)


def _collect_question_embeddings(sample_ids: Sequence[str], emb_dict: Dict) -> Tuple[List[str], np.ndarray]:
    kept_ids: List[str] = []
    q_emb_list: List[np.ndarray] = []
    pbar = tqdm(sample_ids, desc="q-emb", unit="sample")
    for sample_id in pbar:
        key = _resolve_key(emb_dict, sample_id)
        if key is None:
            continue
        entry = emb_dict.get(key, {})
        if "q_emb" not in entry:
            continue
        q_vec = _to_numpy_q_emb(entry["q_emb"])
        if q_vec.size == 0 or (not np.all(np.isfinite(q_vec))):
            continue
        kept_ids.append(sample_id)
        q_emb_list.append(q_vec)

    if len(q_emb_list) == 0:
        raise ValueError("No question embeddings matched the retrieval samples.")
    x = np.stack(q_emb_list, axis=0).astype(np.float32)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return kept_ids, x


def _build_cluster_motif_stats(
    kept_ids: Sequence[str],
    q_emb: np.ndarray,
    mass_by_sample_id: Dict[str, np.ndarray],
    k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise ImportError("scikit-learn is required. Install it to run k-means clustering.") from e

    if q_emb.shape[0] < 2:
        raise ValueError("Need at least 2 samples for k-means clustering.")

    k = min(int(k), int(q_emb.shape[0]))
    if k < 2:
        raise ValueError("k must be >= 2 after clipping to available samples.")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(1024, max(64, q_emb.shape[0] // 4)),
        n_init=10,
    )
    labels = kmeans.fit_predict(q_emb)

    cluster_mass = np.zeros((k, MOTIF_VOCAB_SIZE), dtype=np.float64)
    cluster_sizes = np.zeros(k, dtype=np.int64)
    for i, sample_id in enumerate(tqdm(kept_ids, desc="aggregate", unit="sample")):
        c = int(labels[i])
        cluster_sizes[c] += 1
        cluster_mass[c] += mass_by_sample_id[sample_id]
    return labels, cluster_mass, cluster_sizes


def _write_heatmap_csv(
    out_file: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    matrix: np.ndarray,
):
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster"] + list(col_labels))
        for row_label, row in zip(row_labels, matrix):
            writer.writerow([row_label] + [f"{float(v):.6f}" for v in row.tolist()])


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading retrieval result: {args.retrieval_result}")
    retrieval_result = _torch_load(args.retrieval_result)
    excluded_triads = set()
    if args.exclude_disconnected:
        excluded_triads = set(DISCONNECTED_TRIADS)
    excluded_token_ids = {TRIAD_NAMES.index(name) + 1 for name in excluded_triads}

    sample_ids, motif_mass = _collect_motif_distributions(
        retrieval_result=retrieval_result,
        top_k=args.top_k,
        use_scores=(not args.uniform_triple_weight),
        excluded_token_ids=excluded_token_ids,
    )
    del retrieval_result
    gc.collect()

    mass_by_sample_id = {sample_id: motif_mass[i] for i, sample_id in enumerate(sample_ids)}

    emb_path = _resolve_emb_path(
        dataset=args.dataset,
        text_encoder=args.text_encoder,
        split=args.split,
        emb_path=args.emb_path,
    )
    print(f"Loading question embeddings: {emb_path}")
    emb_dict = _torch_load(emb_path)
    kept_ids, q_emb = _collect_question_embeddings(sample_ids=sample_ids, emb_dict=emb_dict)
    del emb_dict
    gc.collect()

    aligned_mass = np.stack([mass_by_sample_id[sample_id] for sample_id in kept_ids], axis=0)

    k = int(args.k)
    if k <= 0:
        k = _auto_k(len(kept_ids))
    print(f"Running k-means with k={k} over {len(kept_ids)} questions")
    _, cluster_mass, cluster_sizes = _build_cluster_motif_stats(
        kept_ids=kept_ids,
        q_emb=q_emb,
        mass_by_sample_id=mass_by_sample_id,
        k=k,
        seed=args.seed,
    )

    motif_ids = []
    for motif_id, triad_name in enumerate(TRIAD_NAMES, start=1):
        if triad_name in excluded_triads:
            continue
        motif_ids.append(motif_id)
    motif_ids = sorted(motif_ids, key=lambda m: (_triad_edge_count(TRIAD_NAMES[m - 1]), TRIAD_NAMES[m - 1]))
    motif_names = [TRIAD_NAMES[m - 1] for m in motif_ids]

    eps = 1e-8
    cluster_prob = cluster_mass[:, motif_ids] + eps
    cluster_prob /= np.clip(cluster_prob.sum(axis=1, keepdims=True), eps, None)
    global_prob = aligned_mass[:, motif_ids].sum(axis=0) + eps
    global_prob /= np.clip(np.sum(global_prob), eps, None)
    enrichment = np.log2(cluster_prob / global_prob.reshape(1, -1))

    simple_names = _parse_motif_names(args.simple_motifs)
    simple_ids = [TRIAD_NAMES.index(name) + 1 for name in simple_names if (name not in excluded_triads)]
    simple_col_mask = np.array([m in set(simple_ids) for m in motif_ids], dtype=bool)
    if simple_col_mask.any():
        simple_share = cluster_prob[:, simple_col_mask].sum(axis=1)
    else:
        simple_share = np.zeros(cluster_prob.shape[0], dtype=np.float64)

    row_order = np.argsort(-simple_share)
    enrichment = enrichment[row_order]
    cluster_sizes = cluster_sizes[row_order]
    simple_share = simple_share[row_order]
    ordered_cluster_ids = row_order.tolist()

    row_labels = [
        f"C{int(cluster_id)} (n={int(cluster_sizes[i])}, simple={float(simple_share[i]):.3f})"
        for i, cluster_id in enumerate(ordered_cluster_ids)
    ]

    import matplotlib.pyplot as plt

    max_abs = float(np.percentile(np.abs(enrichment), 95))
    max_abs = max(max_abs, 0.25)

    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.75 * len(motif_ids) + 3.0), max(4.0, 0.6 * len(row_labels) + 2.0))
    )
    im = ax.imshow(enrichment, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(np.arange(len(motif_names)))
    ax.set_xticklabels(motif_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Motif Type (ordered by structural complexity)")
    ax.set_ylabel("Question Cluster (k-means on q_emb)")
    ax.set_title(
        f"Question-Cluster Motif Enrichment (log2)\n"
        f"topK={args.top_k}, k={k}, score_weighted={not args.uniform_triple_weight}"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2(cluster motif prob / global motif prob)")
    fig.tight_layout()

    heatmap_png = os.path.join(args.output_dir, "question_cluster_motif_heatmap.png")
    fig.savefig(heatmap_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    heatmap_csv = os.path.join(args.output_dir, "question_cluster_motif_heatmap.csv")
    _write_heatmap_csv(heatmap_csv, row_labels=row_labels, col_labels=motif_names, matrix=enrichment)

    summary = {
        "retrieval_result": os.path.abspath(args.retrieval_result),
        "emb_path": os.path.abspath(emb_path),
        "num_samples_with_motif_mass": int(len(sample_ids)),
        "num_samples_with_q_emb": int(len(kept_ids)),
        "k": int(k),
        "top_k_triples": int(args.top_k),
        "score_weighted": bool(not args.uniform_triple_weight),
        "exclude_disconnected": bool(args.exclude_disconnected),
        "excluded_triads": sorted(list(excluded_triads)),
        "simple_motifs": simple_names,
        "motif_columns": motif_names,
        "cluster_order": [int(x) for x in ordered_cluster_ids],
        "cluster_sizes": [int(x) for x in cluster_sizes.tolist()],
        "simple_share_by_row": [float(x) for x in simple_share.tolist()],
        "outputs": {
            "heatmap_png": os.path.abspath(heatmap_png),
            "heatmap_csv": os.path.abspath(heatmap_csv),
        },
    }
    summary_file = os.path.join(args.output_dir, "question_cluster_motif_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved heatmap: {heatmap_png}")
    print(f"Saved matrix: {heatmap_csv}")
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Question-cluster motif heatmap")
    parser.add_argument(
        "--retrieval_result",
        type=str,
        required=True,
        help="Path to retrieval_result.pth from inference.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (used to auto-resolve emb path when --emb_path is not provided).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split for question embeddings.",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="gte-large-en-v1.5",
        help="Text encoder name for embedding path resolution.",
    )
    parser.add_argument(
        "--emb_path",
        type=str,
        default=None,
        help="Optional explicit path to precomputed question embeddings .pth.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/motif_projection",
        help="Output directory for heatmap artifacts.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-K retrieved triples to use per question.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Number of k-means clusters. Use 0 for automatic choice.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--simple_motifs",
        type=str,
        default="021D,021U,021C",
        help="Comma-separated triad names treated as simple motifs.",
    )
    parser.add_argument(
        "--uniform_triple_weight",
        action="store_true",
        help="Use uniform per-triple weights instead of retrieval scores.",
    )
    parser.add_argument(
        "--exclude_disconnected",
        action="store_true",
        default=True,
        help="Exclude disconnected triads (003, 012, 102) from analysis (default).",
    )
    parser.add_argument(
        "--include_disconnected",
        action="store_false",
        dest="exclude_disconnected",
        help="Include disconnected triads (003, 012, 102) in analysis.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=260,
        help="PNG resolution.",
    )
    main(parser.parse_args())
