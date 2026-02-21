import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
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


def motif_cache_file(dataset_name: str, split: str, top_k: int = 4, backend: str = "python") -> str:
    save_dir = os.path.join("data_files", dataset_name, "motif_tokens")
    return os.path.join(save_dir, f"{split}_triad3_top{top_k}_{backend}.pth")


def _count_motifs_for_pair(
    h_i: int,
    t_i: int,
    num_entities: int,
    out_adj: List[set],
    any_adj: List[set],
) -> Counter:
    counter = Counter()
    base_mask = ((1 if t_i in out_adj[h_i] else 0) << 0) | ((1 if h_i in out_adj[t_i] else 0) << 1)

    candidates = any_adj[h_i] | any_adj[t_i]
    if h_i in candidates:
        candidates.remove(h_i)
    if t_i in candidates:
        candidates.remove(t_i)

    disconnected = max(0, (num_entities - 2) - len(candidates))
    if disconnected > 0:
        token_id = MASK_TO_TOKEN[base_mask]
        if token_id != PAD_TOKEN_ID:
            counter[token_id] += disconnected

    for c_i in candidates:
        mask = base_mask
        mask |= (1 if c_i in out_adj[h_i] else 0) << 2
        mask |= (1 if h_i in out_adj[c_i] else 0) << 3
        mask |= (1 if c_i in out_adj[t_i] else 0) << 4
        mask |= (1 if t_i in out_adj[c_i] else 0) << 5
        token_id = MASK_TO_TOKEN[mask]
        if token_id != PAD_TOKEN_ID:
            counter[token_id] += 1

    return counter


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
    in_adj = [set() for _ in range(num_entities)]
    for h_i, t_i in zip(h_id_list, t_id_list):
        out_adj[h_i].add(t_i)
        in_adj[t_i].add(h_i)
    any_adj = [out_adj[i] | in_adj[i] for i in range(num_entities)]

    pair_to_triple_ids = defaultdict(list)
    for triple_id, (h_i, t_i) in enumerate(zip(h_id_list, t_id_list)):
        pair_to_triple_ids[(h_i, t_i)].append(triple_id)

    pair_counter_cache = {}
    triple_counters = [None for _ in range(num_triples)]
    node_counters = [Counter() for _ in range(num_entities)]

    for (h_i, t_i), triple_ids in pair_to_triple_ids.items():
        pair_key = (h_i, t_i)
        if pair_key not in pair_counter_cache:
            pair_counter_cache[pair_key] = _count_motifs_for_pair(
                h_i=h_i,
                t_i=t_i,
                num_entities=num_entities,
                out_adj=out_adj,
                any_adj=any_adj,
            )
        counter = pair_counter_cache[pair_key]
        multiplicity = len(triple_ids)

        for token_id, count in counter.items():
            add_count = count * multiplicity
            node_counters[h_i][token_id] += add_count
            node_counters[t_i][token_id] += add_count
        for triple_id in triple_ids:
            triple_counters[triple_id] = counter

    node_ids = torch.zeros((num_entities, top_k), dtype=torch.long)
    node_wts = torch.zeros((num_entities, top_k), dtype=torch.float)
    for node_id in range(num_entities):
        ids_i, wts_i = _counter_to_topk(node_counters[node_id], top_k)
        node_ids[node_id] = ids_i
        node_wts[node_id] = wts_i

    triple_ids = torch.zeros((num_triples, top_k), dtype=torch.long)
    triple_wts = torch.zeros((num_triples, top_k), dtype=torch.float)
    for triple_id in range(num_triples):
        counter = triple_counters[triple_id]
        if counter is None:
            counter = Counter()
        ids_i, wts_i = _counter_to_topk(counter, top_k)
        triple_ids[triple_id] = ids_i
        triple_wts[triple_id] = wts_i

    return {
        "node_motif_token_ids": node_ids,
        "node_motif_token_wts": node_wts,
        "triple_motif_token_ids": triple_ids,
        "triple_motif_token_wts": triple_wts,
    }


def _build_single(args):
    sample, top_k, backend, orca_path = args
    return sample["id"], build_motif_tokens_for_sample(
        sample=sample,
        top_k=top_k,
        backend=backend,
        orca_path=orca_path,
    )


def build_motif_cache_for_split(
    dataset_name: str,
    split: str,
    top_k: int = 4,
    backend: str = "python",
    orca_path: str = "",
    num_workers: int = 1,
    overwrite: bool = False,
) -> str:
    save_file = motif_cache_file(dataset_name, split, top_k=top_k, backend=backend)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    if (not overwrite) and os.path.exists(save_file):
        return save_file

    processed_file = os.path.join("data_files", dataset_name, "processed", f"{split}.pkl")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(
            f"Missing processed split file: {processed_file}. Run `python emb.py -d {dataset_name}` first."
        )
    with open(processed_file, "rb") as f:
        processed_dict_list = pickle.load(f)

    motif_dict = {}
    if num_workers <= 1:
        for sample in tqdm(processed_dict_list, desc=f"motifs:{split}"):
            sample_id, motif_entry = _build_single((sample, top_k, backend, orca_path))
            motif_dict[sample_id] = motif_entry
    else:
        work_items = ((sample, top_k, backend, orca_path) for sample in processed_dict_list)
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for sample_id, motif_entry in tqdm(
                ex.map(_build_single, work_items, chunksize=32),
                total=len(processed_dict_list),
                desc=f"motifs:{split}",
            ):
                motif_dict[sample_id] = motif_entry
    torch.save(motif_dict, save_file)
    return save_file


def load_motif_cache(dataset_name: str, split: str, top_k: int = 4, backend: str = "python") -> Dict:
    save_file = motif_cache_file(dataset_name, split, top_k=top_k, backend=backend)
    if not os.path.exists(save_file):
        raise FileNotFoundError(save_file)
    return torch.load(save_file)
