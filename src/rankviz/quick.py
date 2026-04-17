"""Convenience function for common one-liner usage."""

from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.figure import Figure

from rankviz.rank_carpet import RankCarpet
from rankviz.similarity_waterfall import SimilarityWaterfall
from rankviz.rank_distribution import RankDistribution

_REGISTRY: dict[str, type] = {
    "rank_carpet": RankCarpet,
    "similarity_waterfall": SimilarityWaterfall,
    "rank_distribution": RankDistribution,
}


def quick_plot(
    queries: np.ndarray | None = None,
    corpus: np.ndarray | None = None,
    highlight: np.ndarray | None = None,
    *,
    kind: str = "rank_carpet",
    query_labels: np.ndarray | None = None,
    similarities: np.ndarray | None = None,
    highlight_similarities: np.ndarray | None = None,
    **kwargs,
) -> Figure:
    """One-liner: create, fit, and plot a rankviz visualisation.

    Parameters
    ----------
    queries, corpus, highlight : ndarray, optional
        Embedding arrays (see :meth:`BaseVisualisation.fit`).
    kind : str
        Visualisation type.  One of ``"rank_carpet"``,
        ``"similarity_waterfall"``, ``"rank_distribution"``.
    query_labels : ndarray, optional
        Per-query domain labels.
    similarities, highlight_similarities : ndarray, optional
        Precomputed similarity matrices (alternative to embeddings).
    **kwargs
        Forwarded to the visualisation constructor.

    Returns
    -------
    Figure
    """
    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown kind {kind!r}.  Choose from: {list(_REGISTRY)}"
        )
    cls = _REGISTRY[kind]
    vis = cls(**kwargs)
    vis.fit(
        queries=queries,
        corpus=corpus,
        highlight=highlight,
        query_labels=query_labels,
        similarities=similarities,
        highlight_similarities=highlight_similarities,
    )
    return vis.plot()
