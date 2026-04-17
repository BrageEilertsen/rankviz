"""Shared fixtures for rankviz tests.

Generates structured synthetic embeddings that reproduce the
poison-vs-specialists retrieval scenario.
"""

from __future__ import annotations

import numpy as np
import pytest


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return x / norms


@pytest.fixture()
def synthetic_data():
    """Return a dict of synthetic embeddings for testing.

    Structure:
    - 3 query clusters of 20 queries each (60 total) in 64-D.
    - 300 corpus documents: 100 specialists near each cluster.
    - 1 poison document at the geometric median of the cluster centres.

    Returns dict with keys: queries, corpus, poison, query_labels.
    """
    rng = np.random.default_rng(42)
    dim = 64
    n_per_cluster = 20
    n_corpus_per_cluster = 100

    # Cluster centres — well separated on the unit sphere.
    centres = _l2_normalise(rng.standard_normal((3, dim)))

    # Queries: tight clusters around centres.
    queries = []
    labels = []
    for i, c in enumerate(centres):
        noise = rng.standard_normal((n_per_cluster, dim)) * 0.15
        cluster = _l2_normalise(c + noise)
        queries.append(cluster)
        labels.extend([f"domain_{i}"] * n_per_cluster)

    queries = np.concatenate(queries, axis=0).astype(np.float32)
    query_labels = np.array(labels)

    # Corpus: specialists (each near one cluster).
    corpus = []
    for c in centres:
        noise = rng.standard_normal((n_corpus_per_cluster, dim)) * 0.3
        docs = _l2_normalise(c + noise)
        corpus.append(docs)
    corpus = np.concatenate(corpus, axis=0).astype(np.float32)

    # Poison: near the geometric median of centres (high similarity to all).
    median_direction = _l2_normalise(centres.mean(axis=0, keepdims=True))
    poison = median_direction.astype(np.float32).squeeze()  # (dim,)

    return {
        "queries": queries,
        "corpus": corpus,
        "poison": poison,
        "query_labels": query_labels,
    }
