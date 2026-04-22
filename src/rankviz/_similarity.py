"""Cosine similarity and rank computation engine.

All retrieval-relevant quantities in rankviz flow through this module.
Embeddings go in; similarity matrices and rank arrays come out.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

# Two gigabytes in bytes — threshold for a memory warning.
_TWO_GB = 2 * 1024 ** 3


def _validate_embeddings(
    arr: np.ndarray,
    name: str,
    *,
    expected_dim: int | None = None,
) -> np.ndarray:
    """Validate and coerce an embedding array to float32 2-D.

    Parameters
    ----------
    arr : array-like
        Input embeddings.  Accepted shapes: ``(d,)`` for a single vector,
        ``(n, d)`` for *n* vectors.
    name : str
        Human-readable name for error messages (e.g. ``"queries"``).
    expected_dim : int, optional
        If given, assert the embedding dimensionality matches.

    Returns
    -------
    np.ndarray
        Always ``(n, d)`` float32.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]  # (d,) -> (1, d)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {arr.ndim}-D.")
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(
            f"{name} has dimension {arr.shape[1]}, expected {expected_dim}."
        )
    return arr


def cosine_similarity_matrix(
    queries: np.ndarray,
    documents: np.ndarray,
) -> np.ndarray:
    """Compute the full cosine-similarity matrix between two sets of vectors.

    Parameters
    ----------
    queries : ndarray, shape ``(n_q, d)``
        Query embeddings (assumed L2-normalised).
    documents : ndarray, shape ``(n_d, d)``
        Document embeddings (assumed L2-normalised).

    Returns
    -------
    ndarray, shape ``(n_q, n_d)``
        ``S[i, j] = cos(queries[i], documents[j])``.

    Raises
    ------
    UserWarning
        If the resulting matrix would exceed 2 GB.
    """
    n_q, n_d = queries.shape[0], documents.shape[0]
    nbytes = n_q * n_d * 4  # float32
    if nbytes > _TWO_GB:
        warnings.warn(
            f"Similarity matrix ({n_q} x {n_d}) will use "
            f"~{nbytes / 1024**3:.1f} GB.  A chunked computation path is "
            f"planned for a future release.",
            stacklevel=2,
        )
    return queries @ documents.T


def ranks_from_similarities(
    similarities: np.ndarray,
) -> np.ndarray:
    """Convert a similarity matrix to a rank matrix (1-indexed, lower is better).

    Ties are broken arbitrarily by numpy's argsort stability.

    Parameters
    ----------
    similarities : ndarray, shape ``(n_q, n_d)``

    Returns
    -------
    ndarray, shape ``(n_q, n_d)``
        ``ranks[i, j]`` is the rank of document *j* for query *i*
        (1 = most similar).
    """
    # argsort descending: highest similarity gets rank 1.
    order = np.argsort(-similarities, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(similarities.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, similarities.shape[1] + 1)
    return ranks


def compute_retrieval_data(
    queries: np.ndarray | None = None,
    corpus: np.ndarray | None = None,
    highlight: np.ndarray | None = None,
    *,
    similarities: np.ndarray | None = None,
    highlight_similarities: np.ndarray | None = None,
    exclude_highlight_from_corpus: bool = True,
    highlight_indices: np.ndarray | Sequence[int] | None = None,
) -> dict:
    """Central computation entry point used by all visualisations.

    Supports two calling conventions:

    **Embedding path** (default)::

        compute_retrieval_data(queries=Q, corpus=D, highlight=H)

    **Precomputed path**::

        compute_retrieval_data(
            similarities=S_corpus,
            highlight_similarities=S_highlight,
        )

    Parameters
    ----------
    queries, corpus, highlight : ndarray, optional
        Embedding arrays.  *highlight* may be ``(d,)`` for a single
        document or ``(m, d)`` for multiple.
    similarities : ndarray, shape ``(n_q, n_d)``, optional
        Precomputed corpus similarity matrix.
    highlight_similarities : ndarray, shape ``(n_q,)`` or ``(n_q, m)``, optional
        Precomputed highlight similarity vectors.
    exclude_highlight_from_corpus : bool
        When *highlight_indices* is given and *highlight* vectors live
        inside *corpus*, exclude them from corpus rank computation so
        they don't compete with themselves.
    highlight_indices : array-like of int, optional
        Column indices into *corpus* (or *similarities*) that correspond
        to the highlight documents.  Used for the exclude path.

    Returns
    -------
    dict with keys:

    - ``"corpus_similarities"`` : ``(n_q, n_d')`` — similarities to corpus docs
    - ``"highlight_similarities"`` : ``(n_q, m)`` — similarities to highlight docs
    - ``"corpus_ranks"`` : ``(n_q, n_d')`` — corpus ranks (1-indexed)
    - ``"highlight_ranks"`` : ``(n_q, m)`` — highlight ranks within the full set
    - ``"n_queries"`` : int
    - ``"n_corpus"`` : int (after exclusion)
    - ``"n_highlight"`` : int
    """
    # ---- precomputed path ------------------------------------------------
    if similarities is not None:
        sim_corpus = np.asarray(similarities, dtype=np.float32)
        if highlight_similarities is not None:
            sim_hl = np.asarray(highlight_similarities, dtype=np.float32)
            if sim_hl.ndim == 1:
                sim_hl = sim_hl[:, None]
        else:
            sim_hl = np.empty((sim_corpus.shape[0], 0), dtype=np.float32)

        # Combine for joint ranking then split.
        sim_all = np.concatenate([sim_corpus, sim_hl], axis=1)
        ranks_all = ranks_from_similarities(sim_all)
        n_d = sim_corpus.shape[1]

        return {
            "corpus_similarities": sim_corpus,
            "highlight_similarities": sim_hl,
            "corpus_ranks": ranks_all[:, :n_d],
            "highlight_ranks": ranks_all[:, n_d:],
            "n_queries": sim_corpus.shape[0],
            "n_corpus": n_d,
            "n_highlight": sim_hl.shape[1],
        }

    # ---- embedding path --------------------------------------------------
    if queries is None or corpus is None:
        raise ValueError(
            "Provide either (queries, corpus) or (similarities,)."
        )

    queries = _validate_embeddings(queries, "queries")
    corpus = _validate_embeddings(corpus, "corpus", expected_dim=queries.shape[1])

    # Handle highlight.
    if highlight is not None:
        highlight = _validate_embeddings(
            highlight, "highlight", expected_dim=queries.shape[1],
        )
    else:
        highlight = np.empty((0, queries.shape[1]), dtype=np.float32)

    # Exclude highlight rows from corpus if they are the same vectors.
    if (
        exclude_highlight_from_corpus
        and highlight_indices is not None
        and len(highlight_indices) > 0
    ):
        mask = np.ones(corpus.shape[0], dtype=bool)
        mask[highlight_indices] = False
        corpus = corpus[mask]

    sim_corpus = cosine_similarity_matrix(queries, corpus)
    sim_hl = cosine_similarity_matrix(queries, highlight)

    # Joint ranking: highlights compete against the full corpus.
    sim_all = np.concatenate([sim_corpus, sim_hl], axis=1)
    ranks_all = ranks_from_similarities(sim_all)
    n_d = sim_corpus.shape[1]

    return {
        "corpus_similarities": sim_corpus,
        "highlight_similarities": sim_hl,
        "corpus_ranks": ranks_all[:, :n_d],
        "highlight_ranks": ranks_all[:, n_d:],
        "n_queries": queries.shape[0],
        "n_corpus": n_d,
        "n_highlight": sim_hl.shape[1],
    }
