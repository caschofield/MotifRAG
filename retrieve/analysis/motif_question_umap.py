import argparse
import math
import os
from typing import Dict, List, Tuple

from networkx.algorithms.triads import TRIAD_NAMES
import numpy as np
import torch
from tqdm import tqdm


EXCLUDED_TRIADS = {"003", "012", "102"}


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
    raise KeyError("Could not find motif embedding table in checkpoint state_dict.")


def _collect_anchor_embeddings(motif_table: np.ndarray):
    token_ids: List[int] = []
    anchor_names: List[str] = []
    anchor_vecs: List[np.ndarray] = []
    for token_id, triad_name in enumerate(TRIAD_NAMES, start=1):
        if triad_name in EXCLUDED_TRIADS:
            continue
        if token_id >= motif_table.shape[0]:
            continue
        token_ids.append(token_id)
        anchor_names.append(triad_name)
        anchor_vecs.append(motif_table[token_id])

    if len(anchor_vecs) == 0:
        raise ValueError("No motif anchors available to project.")
    return token_ids, anchor_names, np.stack(anchor_vecs, axis=0).astype(np.float32)


def _build_anchor_xy_lookup(token_ids: List[int], anchor_xy: np.ndarray):
    lookup = {}
    for i, token_id in enumerate(token_ids):
        lookup[int(token_id)] = anchor_xy[i].astype(np.float32)
    return lookup


def _token_info_to_xy(token_info: Dict, anchor_xy_lookup: Dict[int, np.ndarray]):
    ids = token_info.get("ids", [])
    wts = token_info.get("wts", [])
    if len(ids) == 0 or len(wts) == 0:
        return None

    xy = np.zeros(2, dtype=np.float32)
    mass = 0.0
    for token_id, wt in zip(ids, wts):
        token_id = int(token_id)
        wt = float(wt)
        if token_id <= 0 or wt <= 0.0:
            continue
        anchor_xy = anchor_xy_lookup.get(token_id)
        if anchor_xy is None:
            continue
        xy += np.float32(wt) * anchor_xy
        mass += wt

    if mass <= 0.0:
        return None
    return xy / np.float32(mass)


def _extract_question_points(sample: Dict, anchor_xy_lookup: Dict[int, np.ndarray], top_k: int):
    scored_tokens = sample.get("scored_triple_motif_tokens", [])
    if top_k > 0:
        scored_tokens = scored_tokens[:top_k]

    points = []
    for token_info in scored_tokens:
        xy = _token_info_to_xy(token_info, anchor_xy_lookup)
        if xy is not None:
            points.append(xy)

    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(points, axis=0).astype(np.float32)


def _choose_questions(
    pred_dict: Dict,
    anchor_xy_lookup: Dict[int, np.ndarray],
    num_questions: int,
    top_k: int,
    min_points: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    sample_ids = list(pred_dict.keys())
    rng.shuffle(sample_ids)

    chosen = []
    for sample_id in tqdm(sample_ids, desc="select-questions", unit="sample"):
        sample = pred_dict[sample_id]
        points = _extract_question_points(sample=sample, anchor_xy_lookup=anchor_xy_lookup, top_k=top_k)
        if points.shape[0] < min_points:
            continue
        chosen.append(
            {
                "sample_id": str(sample_id),
                "question": str(sample.get("question", "")),
                "xy": points,
            }
        )
        if len(chosen) >= num_questions:
            break
    return chosen


def _run_anchor_umap(anchor_vecs: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int):
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "UMAP is not installed. Install it with `pip install umap-learn`."
        ) from e

    n = anchor_vecs.shape[0]
    if n < 3:
        raise ValueError("UMAP needs at least 3 motif embeddings.")
    used_neighbors = int(max(2, min(n_neighbors, n - 1)))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=used_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    anchor_xy = reducer.fit_transform(anchor_vecs)
    return anchor_xy, used_neighbors


def _plot_questions(
    selected: List[Dict],
    out_file: str,
    top_k: int,
    dpi: int,
    anchor_names: List[str],
    anchor_xy: np.ndarray,
):
    import matplotlib.pyplot as plt

    n = len(selected)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 4.8 * rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    last_scatter = None
    for i, item in enumerate(selected):
        ax = axes_flat[i]
        y = item["xy"]
        ranks = np.arange(1, y.shape[0] + 1)
        last_scatter = ax.scatter(
            y[:, 0],
            y[:, 1],
            c=ranks,
            cmap="viridis_r",
            s=28.0,
            alpha=0.85,
            linewidths=0.0,
        )
        q = item["question"].replace("\n", " ").strip()
        if len(q) > 88:
            q = q[:85] + "..."
        ax.set_title(
            f"{item['sample_id']} | n={y.shape[0]}\n{q}",
            fontsize=9,
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.25)

        ax.scatter(
            anchor_xy[:, 0],
            anchor_xy[:, 1],
            marker="*",
            s=180.0,
            c="#f1c40f",
            edgecolors="black",
            linewidths=0.7,
            alpha=0.95,
            zorder=10,
        )
        for j, name in enumerate(anchor_names):
            ax.annotate(
                name,
                (anchor_xy[j, 0], anchor_xy[j, 1]),
                fontsize=7,
                xytext=(3, 2),
                textcoords="offset points",
                zorder=11,
            )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    right_margin = 1.0
    if last_scatter is not None:
        right_margin = 0.90
        cax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
        cbar = fig.colorbar(last_scatter, cax=cax)
        cbar.set_label("Retrieved triple rank (1 = highest score)")

    fig.suptitle(
        f"Per-Question UMAP of Top-{top_k} Retrieved Triple Motif Embeddings "
        f"(fit on motif anchors only)",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=[0.0, 0.0, right_margin, 0.97])
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = _torch_load(args.checkpoint)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    _, motif_weight = _find_motif_weight(state_dict)
    motif_table = motif_weight.detach().cpu().numpy().astype(np.float32)
    del checkpoint

    token_ids, anchor_names, anchor_vecs = _collect_anchor_embeddings(motif_table)
    anchor_xy, _ = _run_anchor_umap(
        anchor_vecs,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )
    anchor_xy_lookup = _build_anchor_xy_lookup(token_ids, anchor_xy)

    print(f"Loading retrieval result: {args.retrieval_result}")
    pred_dict = _torch_load(args.retrieval_result)

    selected = _choose_questions(
        pred_dict=pred_dict,
        anchor_xy_lookup=anchor_xy_lookup,
        num_questions=args.num_questions,
        top_k=args.top_k,
        min_points=args.min_points,
        seed=args.seed,
    )
    if len(selected) == 0:
        raise ValueError("No questions met the minimum valid motif-point count.")

    if len(selected) < args.num_questions:
        print(
            f"Only found {len(selected)} questions with >= {args.min_points} valid motif points "
            f"(requested {args.num_questions})."
        )

    out_png = os.path.join(args.output_dir, "question_topk_motif_umap.png")
    _plot_questions(
        selected=selected,
        out_file=out_png,
        top_k=args.top_k,
        dpi=args.dpi,
        anchor_names=anchor_names,
        anchor_xy=anchor_xy,
    )

    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Per-question UMAP of retrieved triple motif embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--retrieval_result", type=str, required=True, help="Path to retrieval_result.pth")
    parser.add_argument("--output_dir", type=str, default="analysis/motif_projection")
    parser.add_argument("--num_questions", type=int, default=5, help="Number of random questions to plot.")
    parser.add_argument("--top_k", type=int, default=100, help="Top-K retrieved triples per question.")
    parser.add_argument("--min_points", type=int, default=30, help="Minimum valid motif points per question.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_neighbors", type=int, default=8)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--dpi", type=int, default=260)
    main(parser.parse_args())
