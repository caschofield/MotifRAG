import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


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


def _aggregate_from_tokens(token_info: Dict, motif_table: np.ndarray) -> Optional[np.ndarray]:
    ids = token_info.get("ids", [])
    wts = token_info.get("wts", [])
    if len(ids) == 0 or len(wts) == 0:
        return None

    vec = np.zeros(motif_table.shape[1], dtype=np.float32)
    mass = 0.0
    for token_id, wt in zip(ids, wts):
        token_id = int(token_id)
        wt = float(wt)
        if token_id <= 0 or token_id >= motif_table.shape[0] or wt <= 0.0:
            continue
        vec += np.float32(wt) * motif_table[token_id]
        mass += wt
    if mass <= 0.0:
        return None
    return vec / np.float32(mass)


def _extract_question_vectors(
    sample: Dict,
    motif_table: np.ndarray,
    top_k: int,
) -> np.ndarray:
    scored_tokens = sample.get("scored_triple_motif_tokens", [])
    if top_k > 0:
        scored_tokens = scored_tokens[:top_k]

    vectors = []
    for token_info in scored_tokens:
        vec = _aggregate_from_tokens(token_info, motif_table)
        if vec is not None:
            vectors.append(vec)

    if len(vectors) == 0:
        return np.zeros((0, motif_table.shape[1]), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32)


def _choose_questions(
    pred_dict: Dict,
    motif_table: np.ndarray,
    num_questions: int,
    top_k: int,
    min_points: int,
    seed: int,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    sample_ids = list(pred_dict.keys())
    rng.shuffle(sample_ids)

    chosen = []
    for sample_id in tqdm(sample_ids, desc="select-questions", unit="sample"):
        sample = pred_dict[sample_id]
        vecs = _extract_question_vectors(sample=sample, motif_table=motif_table, top_k=top_k)
        if vecs.shape[0] < min_points:
            continue
        chosen.append(
            {
                "sample_id": str(sample_id),
                "question": str(sample.get("question", "")),
                "vectors": vecs,
            }
        )
        if len(chosen) >= num_questions:
            break
    return chosen


def _run_pca(x: np.ndarray):
    from sklearn.decomposition import PCA

    if x.shape[0] < 2:
        raise ValueError("PCA needs at least 2 vectors.")
    pca = PCA(n_components=2)
    y = pca.fit_transform(x)
    return y, pca.explained_variance_ratio_


def _project_selected_with_global_pca(selected: List[Dict]) -> np.ndarray:
    all_vec = np.concatenate([item["vectors"] for item in selected], axis=0)
    all_xy, var_ratio = _run_pca(all_vec)
    start = 0
    for item in selected:
        n = item["vectors"].shape[0]
        item["pca_xy"] = all_xy[start : start + n]
        start += n
    return var_ratio


def _plot_questions(
    selected: List[Dict],
    out_file: str,
    top_k: int,
    dpi: int,
    global_var_ratio: np.ndarray,
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
        y = item["pca_xy"]
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
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes_flat.tolist(), shrink=0.9)
        cbar.set_label("Retrieved triple rank (1 = highest score)")

    fig.suptitle(
        f"Per-Question PCA of Top-{top_k} Retrieved Triple Motif Embeddings "
        f"(global fit, var={float(np.sum(global_var_ratio)):.2f})",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = _torch_load(args.checkpoint)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    motif_key, motif_weight = _find_motif_weight(state_dict)
    motif_table = motif_weight.detach().cpu().numpy().astype(np.float32)
    del checkpoint

    print(f"Loading retrieval result: {args.retrieval_result}")
    pred_dict = _torch_load(args.retrieval_result)

    selected = _choose_questions(
        pred_dict=pred_dict,
        motif_table=motif_table,
        num_questions=args.num_questions,
        top_k=args.top_k,
        min_points=args.min_points,
        seed=args.seed,
    )
    if len(selected) == 0:
        raise ValueError("No questions met the minimum valid motif-vector count.")

    if len(selected) < args.num_questions:
        print(
            f"Only found {len(selected)} questions with >= {args.min_points} valid motif vectors "
            f"(requested {args.num_questions})."
        )

    global_var_ratio = _project_selected_with_global_pca(selected)
    out_png = os.path.join(args.output_dir, "question_topk_motif_pca.png")
    _plot_questions(
        selected=selected,
        out_file=out_png,
        top_k=args.top_k,
        dpi=args.dpi,
        global_var_ratio=global_var_ratio,
    )

    meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "retrieval_result": os.path.abspath(args.retrieval_result),
        "motif_weight_key": motif_key,
        "num_selected_questions": int(len(selected)),
        "top_k": int(args.top_k),
        "min_points": int(args.min_points),
        "global_pca_explained_variance_ratio": [float(v) for v in global_var_ratio.tolist()],
        "selected_questions": [
            {
                "sample_id": item["sample_id"],
                "question": item["question"],
                "num_vectors": int(item["vectors"].shape[0]),
            }
            for item in selected
        ],
        "output_png": os.path.abspath(out_png),
    }
    out_json = os.path.join(args.output_dir, "question_topk_motif_pca.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved plot: {out_png}")
    print(f"Saved metadata: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Per-question PCA of retrieved triple motif embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--retrieval_result", type=str, required=True, help="Path to retrieval_result.pth")
    parser.add_argument("--output_dir", type=str, default="analysis/motif_projection")
    parser.add_argument("--num_questions", type=int, default=5, help="Number of random questions to plot.")
    parser.add_argument("--top_k", type=int, default=100, help="Top-K retrieved triples per question.")
    parser.add_argument("--min_points", type=int, default=30, help="Minimum valid motif vectors per question.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=260)
    main(parser.parse_args())
