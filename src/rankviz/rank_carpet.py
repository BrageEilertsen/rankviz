"""Rank Carpet — per-query rank profile of highlight documents.

X-axis: queries (sorted by the highlight document's rank).
Y-axis: rank (log-scaled, inverted so rank-1 is at the top).
Corpus documents are rendered as percentile bands (default) or
individual lines (``individual_lines=True`` for small corpora).
Highlight documents are bold coloured lines.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from rankviz._base import BaseVisualisation
from rankviz._style import (
    CORPUS_COLOUR,
    CORPUS_COLOUR_DARK,
    get_highlight_colours,
)


class RankCarpet(BaseVisualisation):
    """Visualise how highlight documents rank across a query set.

    Parameters
    ----------
    log_scale : bool
        Use a log-scaled y-axis (default ``True``).
    highlight_color : str or list of str, optional
        Override colour(s) for highlight documents.  Falls back to the
        Paul Tol qualitative palette.
    reference_ranks : list of int
        Horizontal reference lines drawn at these ranks (default ``[10, 100]``).
    individual_lines : bool
        Draw every corpus document as an individual grey line instead of
        percentile bands.  Only practical for corpora under ~200 docs.
    sort_by : int or None
        Index of the highlight document whose rank determines x-axis
        ordering.  ``0`` (default) sorts by the first highlight;
        ``None`` preserves the input query order.
    figsize : tuple of float
        Figure size in inches.
    highlight_labels : list of str, optional
        Display names for highlight documents in the legend.
    """

    def __init__(
        self,
        *,
        log_scale: bool = True,
        highlight_color: str | list[str] | None = None,
        reference_ranks: list[int] | None = None,
        individual_lines: bool = False,
        sort_by: int | None = 0,
        figsize: tuple[float, float] = (10, 4),
        highlight_labels: list[str] | None = None,
    ) -> None:
        super().__init__(figsize=figsize, highlight_labels=highlight_labels)
        self.log_scale = log_scale
        self.highlight_color = highlight_color
        self.reference_ranks = reference_ranks if reference_ranks is not None else [10, 100]
        self.individual_lines = individual_lines
        self.sort_by = sort_by

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(self, **kwargs) -> Figure:
        data = self._data
        n_q = data["n_queries"]
        n_hl = data["n_highlight"]
        corpus_ranks = data["corpus_ranks"]        # (n_q, n_d)
        highlight_ranks = data["highlight_ranks"]   # (n_q, m)

        # --- determine query sort order -----------------------------------
        if self.sort_by is not None and n_hl > 0:
            sort_idx = np.argsort(highlight_ranks[:, self.sort_by])
        else:
            sort_idx = np.arange(n_q)

        # Apply within-domain sorting if labels are provided.
        if self._query_labels is not None and self.sort_by is not None and n_hl > 0:
            sort_idx = _sort_within_domains(
                highlight_ranks[:, self.sort_by],
                self._query_labels,
            )

        corpus_ranks = corpus_ranks[sort_idx]
        highlight_ranks = highlight_ranks[sort_idx]
        labels_sorted = (
            self._query_labels[sort_idx]
            if self._query_labels is not None
            else None
        )

        x = np.arange(n_q)

        # --- figure -------------------------------------------------------
        fig, ax = plt.subplots(figsize=self.figsize)

        # Corpus background.
        if self.individual_lines:
            for j in range(corpus_ranks.shape[1]):
                ax.plot(x, corpus_ranks[:, j], color=CORPUS_COLOUR,
                        linewidth=0.3, alpha=0.4, rasterized=True)
        else:
            _draw_percentile_bands(ax, x, corpus_ranks)

        # Highlight lines.
        colours = self._resolve_colours(n_hl)
        for h in range(n_hl):
            ax.plot(
                x,
                highlight_ranks[:, h],
                color=colours[h],
                linewidth=1.5,
                label=self._highlight_label(h),
                zorder=10,
            )

        # Reference lines.
        for r in self.reference_ranks:
            if r <= corpus_ranks.shape[1]:
                ax.axhline(
                    r, color=CORPUS_COLOUR_DARK, linewidth=0.5,
                    linestyle="--", zorder=5,
                )
                ax.text(
                    n_q + 0.5, r, f"k={r}", fontsize=7,
                    color=CORPUS_COLOUR_DARK, va="center",
                )

        # Domain separators.
        if labels_sorted is not None:
            _draw_domain_separators(ax, labels_sorted)

        # Axes formatting.
        ax.set_xlabel("Queries (sorted)")
        ax.set_ylabel("Rank")
        ax.set_xlim(0, n_q - 1)
        ax.invert_yaxis()

        if self.log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)

        if n_hl > 0:
            ax.legend(loc="upper left", fontsize=7)

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
        return self._data["highlight_ranks"]


# ======================================================================
# Module-private helpers
# ======================================================================

def _draw_percentile_bands(
    ax,
    x: np.ndarray,
    corpus_ranks: np.ndarray,
) -> None:
    """Render corpus ranks as stacked percentile bands."""
    percentiles = [5, 25, 50, 75, 95]
    values = np.percentile(corpus_ranks, percentiles, axis=1)  # (5, n_q)

    # Bands: 5–95, 25–75, and median line.
    ax.fill_between(
        x, values[0], values[4],
        color=CORPUS_COLOUR, alpha=0.25, linewidth=0, label="corpus 5–95 %ile",
    )
    ax.fill_between(
        x, values[1], values[3],
        color=CORPUS_COLOUR, alpha=0.40, linewidth=0, label="corpus 25–75 %ile",
    )
    ax.plot(
        x, values[2],
        color=CORPUS_COLOUR_DARK, linewidth=0.6, linestyle=":",
        label="corpus median",
    )


def _sort_within_domains(
    highlight_ranks_1d: np.ndarray,
    query_labels: np.ndarray,
) -> np.ndarray:
    """Sort queries within each domain by highlight rank, keeping domains grouped."""
    unique_labels = []
    seen = set()
    for lbl in query_labels:
        if lbl not in seen:
            unique_labels.append(lbl)
            seen.add(lbl)

    indices = []
    for lbl in unique_labels:
        mask = query_labels == lbl
        domain_idx = np.where(mask)[0]
        order = np.argsort(highlight_ranks_1d[domain_idx])
        indices.append(domain_idx[order])

    return np.concatenate(indices)


def _draw_domain_separators(ax, labels_sorted: np.ndarray) -> None:
    """Draw thin vertical lines at domain boundaries."""
    prev = labels_sorted[0]
    for i, lbl in enumerate(labels_sorted):
        if lbl != prev:
            ax.axvline(i - 0.5, color="#AAAAAA", linewidth=0.3, zorder=1)
            prev = lbl
