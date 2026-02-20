import os
from collections import Counter
from typing import Dict, List, Tuple

import networkx as nx
import torch
from networkx.algorithms.triads import TRIAD_NAMES
from tqdm import tqdm

PAD_TOKEN_ID = 0
MOTIF_VOCAB_SIZE = 1 + len(TRIAD_NAMES)  # +1 for PAD


def _build_mask_to_token() -> List[int]:
    mask_to_token = [PAD_TOKEN_ID for _ in range(64)]
    for mask in range(64):
        g = nx.DiGraph()
        g.add_nodes_from([0, 1, 2])
        bits = [
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (1, 2), (2, 1),
        ]
        for bit_idx, (u, v) in enumerate(bits):
            if (mask >> bit_idx) & 1:
                g.add_edge(u, v)
        census = nx.triadic_census(g)
        triad_name = None
        for name in TRIAD_NAMES:
            if census[name] == 1:
                triad_name = name
                break
        if triad_name is None:
            continue
        mask_to_token[mask] = TRIAD_NAMES.index(triad_name) + 1
    return mask_to_token


MASK_TO_TOKEN = _build_mask_to_token()


def _mask_for_triplet_order(a: int, b: int, c: int, out_adj: List[set]) -> int:
    return (
        (1 if b in out_adj[a] else 0) << 0
        | (1 if a in out_adj[b] else 0) << 1
        | (1 if c in out_adj[a] else 0) << 2
        | (1 if a in out_adj[c] else 0) << 3
        | (1 if c in out_adj[b] else 0) << 4
        | (1 if b in out_adj[c] else 0) << 5
    )


def _counter_to_topk(counter: Counter, top_k: int) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    ids = torch.zeros(top_k, dtype=torch.long)
    wts = torch.zeros(top_k, dtype=torch.float)
    if len(counter) == 0:
        return ids, wts

    ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    total = sum(v for _, v in ranked)
    if total == 0:
        return ids, wts

    for i, (token_id, count) in enumerate(ranked):
        ids[i] = int(token_id)
        wts[i] = float(count) / float(total)
    return ids, wts


def build_motif_tokens_for_sample(sample: Dict, top_k: int = 4, backend: str = "python", orca_path: str = "") -> Dict:
    if backend == "orca":
        # Placeholder hook: we keep ORCA optional and non-blocking.
        if not orca_path or not os.path.exists(orca_path):
            backend = "python"
        else:
            backend = "python"

    h_id_list = sample["h_id_list"]
    t_id_list = sample["t_id_list"]
    num_entities = len(sample["text_entity_list"]) + len(sample["non_text_entity_list"])
    num_triples = len(h_id_list)

    out_adj = [set() for _ in range(num_entities)]
    for h_i, t_i in zip(h_id_list, t_id_list):
        out_adj[h_i].add(t_i)

    triple_counters = [Counter() for _ in range(num_triples)]
    node_counters = [Counter() for _ in range(num_entities)]

    for triple_id, (h_i, t_i) in enumerate(zip(h_id_list, t_id_list)):
        counter = triple_counters[triple_id]
        for c_i in range(num_entities):
            if c_i == h_i or c_i == t_i:
                continue
            mask = _mask_for_triplet_order(h_i, t_i, c_i, out_adj)
            token_id = MASK_TO_TOKEN[mask]
            if token_id == PAD_TOKEN_ID:
                continue
            counter[token_id] += 1

        for token_id, count in counter.items():
            node_counters[h_i][token_id] += count
            node_counters[t_i][token_id] += count

    node_ids = torch.zeros((num_entities, top_k), dtype=torch.long)
    node_wts = torch.zeros((num_entities, top_k), dtype=torch.float)
    for node_id in range(num_entities):
        ids_i, wts_i = _counter_to_topk(node_counters[node_id], top_k)
        node_ids[node_id] = ids_i
        node_wts[node_id] = wts_i

    triple_ids = torch.zeros((num_triples, top_k), dtype=torch.long)
    triple_wts = torch.zeros((num_triples, top_k), dtype=torch.float)
    for triple_id in range(num_triples):
        ids_i, wts_i = _counter_to_topk(triple_counters[triple_id], top_k)
        triple_ids[triple_id] = ids_i
        triple_wts[triple_id] = wts_i

    return {
        "node_motif_token_ids": node_ids,
        "node_motif_token_wts": node_wts,
        "triple_motif_token_ids": triple_ids,
        "triple_motif_token_wts": triple_wts,
    }


def load_or_build_motif_cache(
    dataset_name: str,
    split: str,
    processed_dict_list: List[Dict],
    top_k: int = 4,
    backend: str = "python",
    orca_path: str = "",
) -> Dict:
    save_dir = os.path.join("data_files", dataset_name, "motif_tokens")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{split}_triad3_top{top_k}_{backend}.pth")

    if os.path.exists(save_file):
        return torch.load(save_file)

    motif_dict = {}
    for sample in tqdm(processed_dict_list):
        motif_dict[sample["id"]] = build_motif_tokens_for_sample(
            sample,
            top_k=top_k,
            backend=backend,
            orca_path=orca_path,
        )

    torch.save(motif_dict, save_file)
    return motif_dict
