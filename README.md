# MotifRAG, Motif-Aware KG-Based RAG

MotifRAG is a motif-aware extension of the SubgraphRAG retrieval pipeline for knowledge-graph question answering. It keeps the same two-stage retrieval and reasoning division, where the reasoning stage is unchanged from SubgraphRAG. In the retrieval stage, motif structural signals are added to improve retrieval accuracy.

## Acknowledgements

This project is built directly on the **SubgraphRAG** architecture, and reuses its core embedding, inference, and evaluation processes. 

- SubgraphRAG paper: [https://arxiv.org/abs/2410.20724](https://arxiv.org/abs/2410.20724)  
- SubgraphRAG code: [https://github.com/Graph-COM/SubgraphRAG](https://github.com/Graph-COM/SubgraphRAG)

We would like to thank Li et al. for the foundation for this work.

## Changes in MotifRAG

Compared to the baseline retriever from SubgraphRAG, MotifRAG adds:
- **Motif counting** to find the top-$k$ motifs that each triple belongs to.
- **Motif representations** of triples using a weighted average of motif embeddings.
- **Multi-channel architecture** used by retriever to provide query-level weighting of different signals (the lexical neighborhood, positional encodings, and motif structural signals).

The code also includes analysis utilities for inspecting the significance of motifs in retrieved triples.