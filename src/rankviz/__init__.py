"""rankviz — retrieval-aware visualisation for dense-retrieval and RAG systems.

Provides visualisation tools that preserve retrieval-relevant quantities
(rank, cosine similarity) rather than ambient geometry, answering
questions that PCA / t-SNE / UMAP cannot.
"""

__version__ = "0.1.0"

from rankviz.rank_carpet import RankCarpet
from rankviz.similarity_waterfall import SimilarityWaterfall
from rankviz.rank_distribution import RankDistribution
from rankviz.quick import quick_plot
from rankviz.core import CORE
from rankviz.plot_landscape import plot_landscape
from rankviz._style import style, apply_style

__all__ = [
    "CORE",
    "plot_landscape",
    "RankCarpet",
    "SimilarityWaterfall",
    "RankDistribution",
    "quick_plot",
    "style",
    "apply_style",
]
