"""Similarity Waterfall — per-query cosine similarity with threshold context.

X-axis: queries (sorted by highlight rank or similarity).
Y-axis: cosine similarity.
Shows the highlight document's similarity as bold markers and the
top-k threshold (similarity of the k-th ranked document) as a
reference line, revealing the margin by which the highlight clears
or misses the retrieval cutoff.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from rankviz._base import BaseVisualisation
from rankviz._style import (
    CORPUS_COLOUR,
    CORPUS_COLOUR_DARK,
    get_highlight_colours,
)


class SimilarityWaterfall(BaseVisualisation):
    """Visualise per-query similarity and retrieval threshold.

    Parameters
    ----------
    k : int
        The top-*k* threshold to display (default ``10``).
    show_threshold : bool
        Draw the top-*k* threshold line (default ``True``).
    shade_margin : bool
        Shade the region between the highlight similarity and the
        threshold (default ``True``).
    highlight_color : str or list of str, optional
        Override colour(s) for highlight documents.
    sort_by : int or None
        Index of the highlight document whose similarity determines
        x-axis ordering.  ``0`` (default) sorts by the first highlight's
        rank; ``None`` preserves input order.
    figsize : tuple of float
        Figure size in inches.
    highlight_labels : list of str, optional
        Display names for highlight documents.
    """

    def __init__(
        self,
        *,
        k: int = 10,
        show_threshold: bool = True,
        shade_margin: bool = True,
        highlight_color: str | list[str] | None = None,
        sort_by: int | None = 0,
        figsize: tuple[float, float] = (10, 4),
        highlight_labels: list[str] | None = None,
    ) -> None:
        super().__init__(figsize=figsize, highlight_labels=highlight_labels)
        self.k = k
        self.show_threshold = show_threshold
        self.shade_margin = shade_margin
        self.highlight_color = highlight_color
        self.sort_by = sort_by

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(self, **kwargs) -> Figure:
        data = self._data
        n_q = data["n_queries"]
        n_hl = data["n_highlight"]
        corpus_sim = data["corpus_similarities"]      # (n_q, n_d)
        highlight_sim = data["highlight_similarities"]  # (n_q, m)
        highlight_ranks = data["highlight_ranks"]       # (n_q, m)

        # --- sort order ---------------------------------------------------
        if self.sort_by is not None and n_hl > 0:
            sort_idx = np.argsort(highlight_ranks[:, self.sort_by])
        else:
            sort_idx = np.arange(n_q)

        corpus_sim = corpus_sim[sort_idx]
        highlight_sim = highlight_sim[sort_idx]
        highlight_ranks = highlight_ranks[sort_idx]

        x = np.arange(n_q)

        # --- top-k threshold: similarity of the k-th corpus document ------
        # Combine corpus + highlight similarities for a faithful threshold.
        all_sim = np.concatenate([corpus_sim, highlight_sim], axis=1)
        # k-th highest similarity per query (0-indexed → index k-1).
        k_eff = min(self.k, all_sim.shape[1])
        # Partition: the k largest values are in the last k positions.
        partitioned = np.partition(all_sim, -k_eff, axis=1)
        threshold = partitioned[:, -k_eff]  # the k-th highest

        # --- figure -------------------------------------------------------
        fig, ax = plt.subplots(figsize=self.figsize)

        # Corpus context: faded band showing min–max similarity range.
        corpus_min = np.min(corpus_sim, axis=1)
        corpus_max = np.max(corpus_sim, axis=1)
        ax.fill_between(
            x, corpus_min, corpus_max,
            color=CORPUS_COLOUR, alpha=0.20, linewidth=0,
            label="corpus range",
        )

        # Threshold line.
        if self.show_threshold:
            ax.plot(
                x, threshold,
                color=CORPUS_COLOUR_DARK, linewidth=0.8, linestyle="--",
                label=f"top-{self.k} threshold", zorder=5,
            )

        # Highlight lines + optional margin shading.
        colours = self._resolve_colours(n_hl)
        for h in range(n_hl):
            sim_h = highlight_sim[:, h]
            ax.plot(
                x, sim_h,
                color=colours[h], linewidth=1.2,
                label=self._highlight_label(h), zorder=10,
            )
            if self.shade_margin and self.show_threshold:
                # Green where above threshold, red where below.
                above = sim_h >= threshold
                ax.fill_between(
                    x, sim_h, threshold,
                    where=above,
                    color=colours[h], alpha=0.12, linewidth=0,
                )
                ax.fill_between(
                    x, sim_h, threshold,
                    where=~above,
                    color="#CC3333", alpha=0.10, linewidth=0,
                )

        # --- formatting ---------------------------------------------------
        ax.set_xlabel("Queries (sorted)")
        ax.set_ylabel("Cosine similarity")
        ax.set_xlim(0, n_q - 1)

        if n_hl > 0 or self.show_threshold:
            ax.legend(loc="lower left", fontsize=7)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_colours(self, n: int) -> list[str]:
        if self.highlight_color is not None:
            if isinstance(self.highlight_color, str):
                return [self.highlight_color] * n
            return list(self.highlight_color) + get_highlight_colours(
                max(0, n - len(self.highlight_color))
            )
        return get_highlight_colours(n)

    def _transform(self) -> np.ndarray:
        """Return highlight similarities ``(n_q, m)``."""
        return self._data["highlight_similarities"]
