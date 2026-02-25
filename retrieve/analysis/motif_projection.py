import argparse
import csv
import gc
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from networkx.algorithms.triads import TRIAD_NAMES
from tqdm import tqdm


@dataclass
class MotifInstance:
    sample_id: str
    source: str
    rank: int
    motif_id: int
    triple_score: float
    token_mass: float
    token_entropy: float
    vector: np.ndarray


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_motif_weight(state_dict: Dict[str, torch.Tensor]) -> Tuple[str, torch.Tensor]:
    preferred = ["motif_emb.weight", "module.motif_emb.weight"]
    for key in preferred:
        if key in state_dict:
            return key, state_dict[key]

    for key, value in state_dict.items():
        if key.endswith("motif_emb.weight"):
            return key, value

    raise KeyError(
        "Could not find motif embedding table in checkpoint state_dict "
        "(expected a key ending with 'motif_emb.weight')."
    )


def _triad_name(motif_id: int) -> str:
    if motif_id <= 0:
        return "PAD"
    idx = motif_id - 1
    if 0 <= idx < len(TRIAD_NAMES):
        return TRIAD_NAMES[idx]
    return f"motif_{motif_id}"


def _aggregate_from_tokens(
    token_info: Dict,
    motif_table: np.ndarray,
) -> Optional[Tuple[np.ndarray, int, float, float]]:
    ids = token_info.get("ids", [])
    wts = token_info.get("wts", [])
    if len(ids) == 0 or len(wts) == 0:
        return None

    vec = np.zeros(motif_table.shape[1], dtype=np.float32)
    valid_pairs = []
    for token_id, wt in zip(ids, wts):
        token_id = int(token_id)
        wt = float(wt)
        if token_id <= 0 or token_id >= motif_table.shape[0] or wt <= 0:
            continue
        vec += np.float32(wt) * motif_table[token_id]
        valid_pairs.append((token_id, wt))

    if len(valid_pairs) == 0:
        return None

    mass = float(sum(w for _, w in valid_pairs))
    if mass > 0:
        vec /= np.float32(mass)

    dominant = max(valid_pairs, key=lambda x: x[1])[0]
    probs = np.array([w / max(mass, 1e-12) for _, w in valid_pairs], dtype=np.float64)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    return vec, dominant, mass, entropy


def _reservoir_add(
    reservoir: List[MotifInstance],
    item: MotifInstance,
    seen_count: int,
    max_keep: int,
    rng: np.random.Generator,
) -> int:
    if max_keep is None or max_keep <= 0:
        return seen_count + 1

    seen_count += 1
    if len(reservoir) < max_keep:
        reservoir.append(item)
        return seen_count

    j = int(rng.integers(low=0, high=seen_count))
    if j < max_keep:
        reservoir[j] = item
    return seen_count


def _extract_instances_streaming(
    pred_dict: Dict,
    motif_table: np.ndarray,
    max_scored: int,
    max_target: int,
    seed: int,
    max_scored_per_sample: int,
    max_target_per_sample: int,
) -> Tuple[List[MotifInstance], Dict]:
    rng = np.random.default_rng(seed)
    scored_reservoir: List[MotifInstance] = []
    target_reservoir: List[MotifInstance] = []

    seen_scored = 0
    seen_target = 0
    all_counts_scored = Counter()
    all_counts_target = Counter()

    total = len(pred_dict)
    pbar = tqdm(total=total, desc="motif-instances", unit="sample")
    processed = 0

    while len(pred_dict) > 0:
        sample_id, sample = pred_dict.popitem()

        scored_tokens = sample.get("scored_triple_motif_tokens", [])
        scored_triples = sample.get("scored_triples", [])
        if max_scored_per_sample is not None and max_scored_per_sample > 0:
            scored_tokens = scored_tokens[:max_scored_per_sample]
            scored_triples = scored_triples[:max_scored_per_sample]

        for i, token_info in enumerate(scored_tokens):
            agg = _aggregate_from_tokens(token_info, motif_table)
            if agg is None:
                continue
            vec, motif_id, mass, entropy = agg
            all_counts_scored[motif_id] += 1

            triple_score = float("nan")
            if i < len(scored_triples) and len(scored_triples[i]) >= 4:
                triple_score = float(scored_triples[i][3])

            item = MotifInstance(
                sample_id=str(sample_id),
                source="scored",
                rank=i + 1,
                motif_id=motif_id,
                triple_score=triple_score,
                token_mass=mass,
                token_entropy=entropy,
                vector=vec,
            )
            seen_scored = _reservoir_add(scored_reservoir, item, seen_scored, max_scored, rng)

        target_tokens = sample.get("target_relevant_triple_motif_tokens", [])
        if max_target_per_sample is not None and max_target_per_sample > 0:
            target_tokens = target_tokens[:max_target_per_sample]

        for i, token_info in enumerate(target_tokens):
            agg = _aggregate_from_tokens(token_info, motif_table)
            if agg is None:
                continue
            vec, motif_id, mass, entropy = agg
            all_counts_target[motif_id] += 1

            item = MotifInstance(
                sample_id=str(sample_id),
                source="target",
                rank=i + 1,
                motif_id=motif_id,
                triple_score=float("nan"),
                token_mass=mass,
                token_entropy=entropy,
                vector=vec,
            )
            seen_target = _reservoir_add(target_reservoir, item, seen_target, max_target, rng)

        processed += 1
        pbar.update(1)

        if processed % 500 == 0:
            gc.collect()

    pbar.close()

    instances = scored_reservoir + target_reservoir
    instances.sort(key=lambda x: (x.sample_id, x.source, x.rank))

    summary = {
        "num_samples": int(total),
        "seen_scored_instances": int(seen_scored),
        "seen_target_instances": int(seen_target),
        "sampled_scored_instances": int(len(scored_reservoir)),
        "sampled_target_instances": int(len(target_reservoir)),
        "all_counts_scored": {str(k): int(v) for k, v in sorted(all_counts_scored.items())},
        "all_counts_target": {str(k): int(v) for k, v in sorted(all_counts_target.items())},
    }
    return instances, summary


def _run_pca(x: np.ndarray):
    from sklearn.decomposition import PCA
    from sklearn.manifold import trustworthiness

    pca = PCA(n_components=2)
    y = pca.fit_transform(x)
    metrics = {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    }
    n_neighbors = min(10, max(1, (len(x) - 1) // 2))
    if len(x) > 2 and n_neighbors >= 1:
        metrics["trustworthiness"] = float(trustworthiness(x, y, n_neighbors=n_neighbors))
    return y, metrics


def _run_tsne(x: np.ndarray, seed: int, perplexity_hint: float):
    from sklearn.manifold import TSNE, trustworthiness

    if len(x) < 4:
        raise ValueError("t-SNE requires at least 4 points.")

    max_perp = max(2.0, min(50.0, (len(x) - 1) / 3.0))
    perplexity = min(float(perplexity_hint), max_perp)
    perplexity = min(perplexity, float(len(x) - 1))
    if perplexity < 2.0:
        raise ValueError("t-SNE perplexity became too small for the current sample size.")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    y = tsne.fit_transform(x)
    metrics = {"perplexity": float(perplexity)}
    n_neighbors = min(10, max(1, (len(x) - 1) // 2))
    metrics["trustworthiness"] = float(trustworthiness(x, y, n_neighbors=n_neighbors))
    return y, metrics


def _run_umap(x: np.ndarray, seed: int, n_neighbors_hint: int, min_dist: float):
    try:
        import umap.umap_ as umap
    except Exception as e:
        raise ImportError("UMAP is unavailable. Install `umap-learn` to enable UMAP projection.") from e

    from sklearn.manifold import trustworthiness

    if len(x) < 4:
        raise ValueError("UMAP requires at least 4 points.")

    n_neighbors = min(int(n_neighbors_hint), max(2, len(x) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=seed,
    )
    y = reducer.fit_transform(x)

    metrics = {"n_neighbors": int(n_neighbors), "min_dist": float(min_dist)}
    n_neighbors_t = min(10, max(1, (len(x) - 1) // 2))
    metrics["trustworthiness"] = float(trustworthiness(x, y, n_neighbors=n_neighbors_t))
    return y, metrics


def _project_all(
    x: np.ndarray,
    methods: List[str],
    seed: int,
    tsne_perplexity: float,
    umap_neighbors: int,
    umap_min_dist: float,
):
    outputs = {}
    metrics = {}
    for method in tqdm(methods, desc="projections", unit="method"):
        method = method.strip().lower()
        if method == "pca":
            y, met = _run_pca(x)
            outputs[method] = y
            metrics[method] = {"status": "ok", **met}
        elif method == "tsne":
            try:
                y, met = _run_tsne(x, seed=seed, perplexity_hint=tsne_perplexity)
                outputs[method] = y
                metrics[method] = {"status": "ok", **met}
            except Exception as e:
                metrics[method] = {"status": "skipped", "reason": str(e)}
        elif method == "umap":
            try:
                y, met = _run_umap(
                    x,
                    seed=seed,
                    n_neighbors_hint=umap_neighbors,
                    min_dist=umap_min_dist,
                )
                outputs[method] = y
                metrics[method] = {"status": "ok", **met}
            except Exception as e:
                metrics[method] = {"status": "skipped", "reason": str(e)}
        else:
            metrics[method] = {"status": "skipped", "reason": "Unknown method"}
    return outputs, metrics


def _plot_atlas(
    out_file: str,
    projected: Dict[str, np.ndarray],
    proj_metrics: Dict[str, Dict],
    motif_table: np.ndarray,
    instances: List[MotifInstance],
    dpi: int,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = [m for m in projected.keys()]
    if len(methods) == 0:
        raise ValueError("No successful projections to plot.")

    motif_ids = np.arange(1, motif_table.shape[0], dtype=np.int32)
    num_anchor = len(motif_ids)
    num_inst = len(instances)
    all_ids = np.array([x.motif_id for x in instances], dtype=np.int32) if num_inst > 0 else np.array([])
    all_src = np.array([x.source for x in instances], dtype=object) if num_inst > 0 else np.array([])
    all_scores = np.array([x.triple_score for x in instances], dtype=np.float32) if num_inst > 0 else np.array([])

    counts_scored = Counter([x.motif_id for x in instances if x.source == "scored"])
    counts_target = Counter([x.motif_id for x in instances if x.source == "target"])
    counts_total = {m: counts_scored.get(m, 0) + counts_target.get(m, 0) for m in motif_ids.tolist()}
    max_count = max(counts_total.values()) if len(counts_total) > 0 else 0
    if max_count > 0:
        anchor_sizes = np.array(
            [120.0 + 520.0 * np.sqrt(counts_total[m] / max_count) for m in motif_ids.tolist()],
            dtype=np.float32,
        )
    else:
        anchor_sizes = np.full(shape=(len(motif_ids),), fill_value=180.0, dtype=np.float32)

    cmap = plt.get_cmap("tab20", max(20, motif_table.shape[0] + 1))
    fig, axes = plt.subplots(1, len(methods), figsize=(7.0 * len(methods), 6.0), squeeze=False)

    for j, method in enumerate(methods):
        ax = axes[0, j]
        coords = projected[method]
        anchor_xy = coords[:num_anchor]
        inst_xy = coords[num_anchor:]

        if num_inst > 0:
            scored_mask = all_src == "scored"
            target_mask = all_src == "target"

            if scored_mask.any():
                scored_ids = all_ids[scored_mask]
                scored_scores = all_scores[scored_mask]
                sizes = np.full(scored_ids.shape[0], 10.0, dtype=np.float32)
                finite_mask = np.isfinite(scored_scores)
                if finite_mask.any():
                    s = scored_scores[finite_mask]
                    smin, smax = float(np.min(s)), float(np.max(s))
                    if smax > smin:
                        sizes[finite_mask] = 8.0 + 36.0 * ((s - smin) / (smax - smin))
                    else:
                        sizes[finite_mask] = 22.0
                ax.scatter(
                    inst_xy[scored_mask, 0],
                    inst_xy[scored_mask, 1],
                    c=scored_ids,
                    cmap=cmap,
                    s=sizes,
                    alpha=0.18,
                    marker="o",
                    linewidths=0.0,
                )

            if target_mask.any():
                target_ids = all_ids[target_mask]
                ax.scatter(
                    inst_xy[target_mask, 0],
                    inst_xy[target_mask, 1],
                    c=target_ids,
                    cmap=cmap,
                    s=32.0,
                    alpha=0.85,
                    marker="^",
                    linewidths=0.2,
                    edgecolors="black",
                )

        ax.scatter(
            anchor_xy[:, 0],
            anchor_xy[:, 1],
            c=motif_ids,
            cmap=cmap,
            s=anchor_sizes,
            marker="*",
            edgecolors="black",
            linewidths=0.7,
            alpha=0.98,
            zorder=10,
        )

        for i, motif_id in enumerate(motif_ids.tolist()):
            label = f"{motif_id}:{_triad_name(motif_id)}"
            ax.annotate(
                label,
                (anchor_xy[i, 0], anchor_xy[i, 1]),
                fontsize=7,
                xytext=(3, 2),
                textcoords="offset points",
            )

        m = proj_metrics.get(method, {})
        tw = m.get("trustworthiness")
        tw_text = "" if tw is None else f" | trust={tw:.3f}"
        ax.set_title(f"{method.upper()}{tw_text}")
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        ax.grid(True, alpha=0.25)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Retrieved Triple Instance", markerfacecolor="gray", alpha=0.5, markersize=7),
        Line2D([0], [0], marker="^", color="w", label="Target-Relevant Triple Instance", markerfacecolor="gray", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="*", color="w", label="Motif Type Anchor", markerfacecolor="gray", markeredgecolor="black", markersize=10),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=3, frameon=False)
    #fig.suptitle("Motif Embedding Atlas (anchors + retrieved motif instances)", y=0.99, fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_projection_table(
    out_file: str,
    projected: Dict[str, np.ndarray],
    motif_table: np.ndarray,
    instances: List[MotifInstance],
):
    motif_ids = np.arange(1, motif_table.shape[0], dtype=np.int32)
    num_anchor = len(motif_ids)
    header = [
        "method",
        "point_type",
        "sample_id",
        "source",
        "rank",
        "motif_id",
        "triad_name",
        "triple_score",
        "token_mass",
        "token_entropy",
        "x",
        "y",
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for method, coords in tqdm(projected.items(), desc="writing-csv", unit="method"):
            anchor_xy = coords[:num_anchor]
            inst_xy = coords[num_anchor:]

            for i, motif_id in enumerate(motif_ids.tolist()):
                writer.writerow([
                    method,
                    "motif_anchor",
                    "",
                    "anchor",
                    -1,
                    motif_id,
                    _triad_name(motif_id),
                    "",
                    "",
                    "",
                    float(anchor_xy[i, 0]),
                    float(anchor_xy[i, 1]),
                ])

            for i, item in enumerate(instances):
                writer.writerow([
                    method,
                    "motif_instance",
                    item.sample_id,
                    item.source,
                    int(item.rank),
                    int(item.motif_id),
                    _triad_name(item.motif_id),
                    float(item.triple_score),
                    float(item.token_mass),
                    float(item.token_entropy),
                    float(inst_xy[i, 0]),
                    float(inst_xy[i, 1]),
                ])


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = _torch_load(args.checkpoint)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    motif_key, motif_weight = _find_motif_weight(state_dict)
    motif_table = motif_weight.detach().cpu().numpy().astype(np.float32)
    del checkpoint
    gc.collect()

    if motif_table.ndim != 2 or motif_table.shape[0] <= 1:
        raise ValueError(f"Invalid motif embedding table shape: {motif_table.shape}")

    instances: List[MotifInstance] = []
    retrieval_meta = {"status": "missing"}

    if args.retrieval_result is not None and os.path.exists(args.retrieval_result):
        print(f"Loading retrieval result: {args.retrieval_result}")
        pred_dict = _torch_load(args.retrieval_result)

        instances, stream_meta = _extract_instances_streaming(
            pred_dict=pred_dict,
            motif_table=motif_table,
            max_scored=args.max_scored,
            max_target=args.max_target,
            seed=args.seed,
            max_scored_per_sample=args.max_scored_per_sample,
            max_target_per_sample=args.max_target_per_sample,
        )

        del pred_dict
        gc.collect()

        retrieval_meta = {
            "status": "loaded",
            "num_samples": int(stream_meta["num_samples"]),
            "seen_scored_instances": int(stream_meta["seen_scored_instances"]),
            "seen_target_instances": int(stream_meta["seen_target_instances"]),
            "sampled_scored_instances": int(stream_meta["sampled_scored_instances"]),
            "sampled_target_instances": int(stream_meta["sampled_target_instances"]),
            "num_instances_after_sampling": int(len(instances)),
            "all_counts_scored": stream_meta["all_counts_scored"],
            "all_counts_target": stream_meta["all_counts_target"],
        }
    else:
        print("No retrieval_result provided; projecting motif anchors only.")

    motif_vectors = motif_table[1:]
    if len(instances) > 0:
        x_inst = np.stack([x.vector for x in instances], axis=0).astype(np.float32)
        x = np.concatenate([motif_vectors, x_inst], axis=0)
    else:
        x = motif_vectors

    if x.shape[0] < 3:
        raise ValueError("Need at least 3 points for projection.")

    methods = [m.strip() for m in args.methods.split(",") if len(m.strip()) > 0]
    projected, proj_metrics = _project_all(
        x=x,
        methods=methods,
        seed=args.seed,
        tsne_perplexity=args.tsne_perplexity,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    del x
    gc.collect()

    if len(projected) == 0:
        raise RuntimeError(f"All projection methods were skipped. Details: {proj_metrics}")

    atlas_file = os.path.join(args.output_dir, "motif_embedding_atlas.png")
    _plot_atlas(
        out_file=atlas_file,
        projected=projected,
        proj_metrics=proj_metrics,
        motif_table=motif_table,
        instances=instances,
        dpi=args.dpi,
    )

    table_file = os.path.join(args.output_dir, "motif_embedding_projection.csv")
    _save_projection_table(
        out_file=table_file,
        projected=projected,
        motif_table=motif_table,
        instances=instances,
    )

    counts_scored = Counter(int(x.motif_id) for x in instances if x.source == "scored")
    counts_target = Counter(int(x.motif_id) for x in instances if x.source == "target")

    metric_payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "retrieval_result": None if args.retrieval_result is None else os.path.abspath(args.retrieval_result),
        "motif_weight_key": motif_key,
        "motif_table_shape": [int(v) for v in motif_table.shape],
        "retrieval_meta": retrieval_meta,
        "num_motif_instances": int(len(instances)),
        "counts_by_sampled_source": {
            "scored": {str(k): int(v) for k, v in sorted(counts_scored.items())},
            "target": {str(k): int(v) for k, v in sorted(counts_target.items())},
        },
        "projection_metrics": proj_metrics,
        "outputs": {
            "atlas_png": os.path.abspath(atlas_file),
            "projection_csv": os.path.abspath(table_file),
        },
    }

    metrics_file = os.path.join(args.output_dir, "motif_embedding_projection_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metric_payload, f, indent=2)

    print(f"Saved atlas: {atlas_file}")
    print(f"Saved projection table: {table_file}")
    print(f"Saved metrics: {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Motif embedding projection atlas")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., webqsp_xxx/cpt.pth).",
    )
    parser.add_argument(
        "--retrieval_result",
        type=str,
        default=None,
        help="Optional path to retrieval_result.pth for motif instance projections.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/motif_projection",
        help="Output directory for plots and metrics.",
    )
    parser.add_argument("--methods", type=str, default="pca,tsne,umap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_scored", type=int, default=8000)
    parser.add_argument("--max_target", type=int, default=8000)
    parser.add_argument("--max_scored_per_sample", type=int, default=500)
    parser.add_argument("--max_target_per_sample", type=int, default=500)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=260)
    main(parser.parse_args())
