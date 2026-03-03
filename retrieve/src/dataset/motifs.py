import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import networkx as nx
from networkx.algorithms.triads import TRIAD_NAMES
from tqdm import tqdm

PAD_TOKEN_ID = 0
EXCLUDED_TRIAD_NAMES = {"003", "012", "102"}
EXCLUDED_TOKEN_IDS = {TRIAD_NAMES.index(name) + 1 for name in EXCLUDED_TRIAD_NAMES}
MOTIF_CACHE_FILTER_TAG = "no003_012_102"
MOTIF_BACKEND_TAG = "python"
INCLUDE_DISCONNECTED_CONTRIBUTIONS = False
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
        if triad_name in EXCLUDED_TRIAD_NAMES:
            # Excluded low-information/disconnected motifs are treated as PAD.
            continue
        mask_to_token[mask] = TRIAD_NAMES.index(triad_name) + 1
    return mask_to_token


MASK_TO_TOKEN = _build_mask_to_token()


def _sanitize_token_lists(token_ids: List[int], token_wts: List[float], top_k: int) -> Tuple[List[int], List[float]]:
    kept = []
    for token_id, wt in zip(token_ids, token_wts):
        token_id = int(token_id)
        wt = float(wt)
        if token_id in EXCLUDED_TOKEN_IDS:
            continue
        if token_id <= PAD_TOKEN_ID or wt <= 0:
            continue
        kept.append((token_id, wt))

    ids = [0 for _ in range(top_k)]
    wts = [0.0 for _ in range(top_k)]
    if len(kept) == 0:
        return ids, wts

    mass = sum(w for _, w in kept)
    if mass <= 0:
        return ids, wts

    for i, (token_id, wt) in enumerate(kept[:top_k]):
        ids[i] = token_id
        wts[i] = float(wt / mass)
    return ids, wts


def _sanitize_motif_entry(entry: Dict[str, Any], top_k: int):
    triple_ids = entry.get("triple_motif_token_ids", [])
    triple_wts = entry.get("triple_motif_token_wts", [])

    for i in range(len(triple_ids)):
        ids_i, wts_i = _sanitize_token_lists(triple_ids[i], triple_wts[i], top_k)
        triple_ids[i] = ids_i
        triple_wts[i] = wts_i


def _sanitize_motif_dict(motif_dict: Dict[str, Dict[str, Any]], top_k: int):
    for sample_id in motif_dict:
        _sanitize_motif_entry(motif_dict[sample_id], top_k)


def _counter_to_topk(counter: Counter, top_k: int) -> Tuple[List[int], List[float]]:
    ids = [0 for _ in range(top_k)]
    wts = [0.0 for _ in range(top_k)]
    if len(counter) == 0:
        return ids, wts

    ranked = [
        (token_id, count)
        for token_id, count in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if token_id not in EXCLUDED_TOKEN_IDS
    ][:top_k]
    total = sum(v for _, v in ranked)
    if total == 0:
        return ids, wts

    for i, (token_id, count) in enumerate(ranked):
        ids[i] = int(token_id)
        wts[i] = float(count) / float(total)
    return ids, wts


def motif_cache_file(dataset_name: str, split: str, top_k: int = 4) -> str:
    save_dir = os.path.join("data_files", dataset_name, "motif_tokens")
    return os.path.join(
        save_dir,
        f"{split}_triad3_top{top_k}_{MOTIF_BACKEND_TAG}_{MOTIF_CACHE_FILTER_TAG}.pth",
    )


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

    if INCLUDE_DISCONNECTED_CONTRIBUTIONS:
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


def build_motif_tokens_for_sample(sample: Dict, top_k: int = 4) -> Dict:
    h_id_list = sample["h_id_list"]
    t_id_list = sample["t_id_list"]
    num_entities = sample.get("num_entities")
    if num_entities is None:
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
        for triple_id in triple_ids:
            triple_counters[triple_id] = counter

    triple_ids = [[0 for _ in range(top_k)] for _ in range(num_triples)]
    triple_wts = [[0.0 for _ in range(top_k)] for _ in range(num_triples)]
    for triple_id in range(num_triples):
        counter = triple_counters[triple_id]
        if counter is None:
            counter = Counter()
        ids_i, wts_i = _counter_to_topk(counter, top_k)
        triple_ids[triple_id] = ids_i
        triple_wts[triple_id] = wts_i

    entry = {
        "triple_motif_token_ids": triple_ids,
        "triple_motif_token_wts": triple_wts,
    }
    _sanitize_motif_entry(entry, top_k)
    return entry


def _minimal_payload(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": sample["id"],
        "h_id_list": sample["h_id_list"],
        "t_id_list": sample["t_id_list"],
        "num_entities": len(sample["text_entity_list"]) + len(sample["non_text_entity_list"]),
    }


def _write_shard(shard_file: str, shard_data: Dict[str, Dict[str, Any]]):
    with open(shard_file, "wb") as f:
        pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_motif_cache_for_split(
    dataset_name: str,
    split: str,
    top_k: int = 4,
    shard_size: int = 2000,
    overwrite: bool = False,
) -> str:
    save_file = motif_cache_file(dataset_name, split, top_k=top_k)
    save_dir = os.path.dirname(save_file)
    os.makedirs(save_dir, exist_ok=True)
    shard_dir = os.path.join(
        save_dir,
        f"{split}_triad3_top{top_k}_{MOTIF_BACKEND_TAG}_{MOTIF_CACHE_FILTER_TAG}.shards",
    )

    if (not overwrite) and os.path.exists(save_file):
        return save_file
    if overwrite and os.path.isdir(shard_dir):
        for fn in os.listdir(shard_dir):
            fp = os.path.join(shard_dir, fn)
            if os.path.isfile(fp):
                os.remove(fp)
    os.makedirs(shard_dir, exist_ok=True)

    processed_file = os.path.join("data_files", dataset_name, "processed", f"{split}.pkl")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(
            f"Missing processed split file: {processed_file}. Run `python emb.py -d {dataset_name}` first."
        )
    with open(processed_file, "rb") as f:
        processed_dict_list = pickle.load(f)
    num_samples = len(processed_dict_list)

    # Minimize resident memory: drop unused fields early and keep only minimal payloads.
    for i in range(num_samples):
        processed_dict_list[i] = _minimal_payload(processed_dict_list[i])

    shard_files = []
    shard_data: Dict[str, Dict[str, Any]] = {}
    shard_idx = 0

    def flush_shard():
        nonlocal shard_data, shard_idx
        if len(shard_data) == 0:
            return
        shard_file = os.path.join(shard_dir, f"part_{shard_idx:05d}.pkl")
        _write_shard(shard_file, shard_data)
        shard_files.append(os.path.basename(shard_file))
        shard_idx += 1
        shard_data = {}

    for i, sample_payload in enumerate(tqdm(processed_dict_list, total=num_samples, desc=f"motifs:{split}")):
        sample_id = sample_payload["id"]
        motif_entry = build_motif_tokens_for_sample(
            sample=sample_payload,
            top_k=top_k,
        )
        shard_data[sample_id] = motif_entry
        processed_dict_list[i] = None
        if len(shard_data) >= shard_size:
            flush_shard()
    flush_shard()
    del processed_dict_list

    manifest = {
        "format": "sharded_v1",
        "dataset_name": dataset_name,
        "split": split,
        "top_k": top_k,
        "backend": MOTIF_BACKEND_TAG,
        "filter_tag": MOTIF_CACHE_FILTER_TAG,
        "excluded_triad_names": sorted(list(EXCLUDED_TRIAD_NAMES)),
        "num_samples": num_samples,
        "shard_size": shard_size,
        "shard_files": shard_files,
    }
    with open(save_file, "wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)
    return save_file


def load_motif_cache(dataset_name: str, split: str, top_k: int = 4) -> Dict:
    save_file = motif_cache_file(dataset_name, split, top_k=top_k)
    if not os.path.exists(save_file):
        raise FileNotFoundError(save_file)

    try:
        with open(save_file, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        # Backward-compatibility for old torch serialized caches.
        import torch
        return torch.load(save_file, weights_only=False)

    if isinstance(payload, dict) and payload.get("format") == "sharded_v1":
        motif_dict = {}
        base_dir = os.path.dirname(save_file)
        filter_tag = payload.get("filter_tag", "")
        shard_suffix = f"_{filter_tag}" if filter_tag else ""
        shard_dir = os.path.join(
            base_dir,
            f"{payload.get('split')}_triad3_top{payload.get('top_k')}_{payload.get('backend', MOTIF_BACKEND_TAG)}{shard_suffix}.shards",
        )
        for shard_file in payload.get("shard_files", []):
            shard_path = shard_file if os.path.isabs(shard_file) else os.path.join(shard_dir, shard_file)
            with open(shard_path, "rb") as f:
                shard = pickle.load(f)
            motif_dict.update(shard)
        _sanitize_motif_dict(motif_dict, top_k)
        return motif_dict

    if isinstance(payload, dict):
        _sanitize_motif_dict(payload, top_k)
        return payload
    raise ValueError(f"Invalid motif cache format in {save_file}")
