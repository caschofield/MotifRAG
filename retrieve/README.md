# Stage 1: Retrieval

## Table of Contents

- [Supported Datasets](#supported-datasets)
- [1-1 Embedding Pre-Computation](#1-1-entity-and-relation-embedding-pre-computation)
    * [Installation](#installation)
    * [Inference (Embedding Computation)](#inference-embedding-computation)
- [1-2 Retriever Development](#1-2-retriever-development)
    * [Installation](#installation-1)
    * [Offline Motif Preprocessing](#offline-motif-preprocessing)
    * [Training](#training)
    * [Inference](#inference)
    * [Evaluation](#evaluation)

## Supported Datasets

We support two built-in multi-hop knowledge graph question answering (KGQA) datasets:

- `webqsp`
- `cwq`

## 1-1: Entity and Relation Embedding Pre-Computation

We first pre-compute and cache entity and relation embeddings for all samples to save time for later training and inference of retrievers.

### Installation

We use `gte-large-en-v1.5` for text encoder, hence the environment name.

```bash
conda create -n gte_large_en_v1-5 python=3.10 -y
conda activate gte_large_en_v1-5
pip install -r requirements/gte_large_en_v1-5.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

### Inference (Embedding Computation)

```bash
python emb.py -d D
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets).

## 1-2: Retriever Development

We now train a retriever, employ it for retrieval (inference), and evaluate the retrieval results.

### Motif-Aware Retrieval (Subgraph RAG 2.0)

The retriever now supports motif-driven tokenization with directed 3-node motifs (triads).

- Motif cache files are saved under `data_files/{dataset}/motif_tokens/`.
- Motif support is configured in `configs/retriever/{dataset}.yaml` via:
  - `motif.enabled`
  - `motif.backend` (`python` by default)
  - `motif.top_k_tokens` (`4` by default)
  - `motif.motif_emb_dim` (`64` by default)
- Optional ORCA integration is scaffolded through `motif.backend=orca` and `motif.orca_path`.
  If unavailable, the code falls back to the Python implementation.
- Motif computation is now **offline only**. `RetrieverDataset` loads cache files and fails fast if they are missing.
- Retrieval outputs keep backward compatibility and append:
  - `scored_triple_motif_tokens`
  - `target_relevant_triple_motif_tokens`
- Evaluation now additionally reports:
  - `motif_recall@k`
  - `motif_precision@k`
  - `motif_f1@k`

### Installation

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
pip install -r requirements/retriever.txt
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Offline Motif Preprocessing

Before training/inference with motif retrieval enabled, generate motif caches:

```bash
python motif_preprocess.py -d D --splits train,val,test --num_workers 32
```

where `D` is one of the supported datasets.

Notes:

- This step is parallelized across samples and uses a neighbor-based motif counter.
- Worker payloads are minimal (`id`, `h_id_list`, `t_id_list`, `num_entities`) to reduce IPC memory.
- Results are stream-written into shard files to avoid large in-memory accumulation.
- On HPC, keep the multiprocessing start method as `spawn` (default) to avoid `fork` deadlocks:
  - `python motif_preprocess.py -d D --num_workers 32 --start_method spawn`
- Tune shard flushing with `--shard_size` if needed:
  - `python motif_preprocess.py -d D --num_workers 32 --shard_size 1000`
- The resulting cache files are saved in `data_files/{dataset}/motif_tokens/`.
- If caches are missing, `train.py` and `inference.py` will fail immediately with a clear message.

### Training

```bash
python train.py -d D
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets).

For logged learning curves, go to the corresponding Wandb interface. 

Once trained, there will be a folder in the current directory of the form `{dataset}_{time}` (e.g., `webqsp_Nov08-01:14:47/`) that stores the trained model checkpoint `cpt.pth`.

### Inference

```bash
python inference.py -p P
```
where `P` is the path to a saved model checkpoint. The predicted retrieval result will be stored in the same folder as the model checkpoint. For example, if `P` is `webqsp_Nov08-01:14:47/cpt.pth`, then the retrieval result will be saved as `webqsp_Nov08-01:14:47/retrieval_result.pth`.

### Evaluation

```bash
python eval.py -d D -p P
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets) and `P` is the path to [inference result](#inference), e.g., `webqsp_Nov08-01:14:47/retrieval_result.pth`.
