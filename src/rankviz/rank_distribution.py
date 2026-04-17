"""Rank Distribution — aggregate rank profiles of highlight documents.

Shows how each highlight document's rank is distributed across the full
query set.  Supports histogram, KDE, and empirical CDF modes, with
optional faceting by domain (query label).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from rankviz._base import BaseVisualisation
from rankviz._style import get_highlight_colours, CORPUS_COLOUR


class RankDistribution(BaseVisualisation):
    """Visualise the distribution of a highlight document's rank.

    Parameters
    ----------
    mode : ``"histogram"`` | ``"cdf"`` | ``"kde"``
        Display mode (default ``"histogram"``).
    log_scale : bool
        Log-scale the rank axis (default ``True``).
    bins : int or str
        Bin specification passed to :func:`numpy.histogram_bin_edges`
        (default ``"auto"``).
    highlight_color : str or list of str, optional
        Override colour(s) for highlight documents.
    facet_by_label : bool
        When ``True`` and *query_labels* were provided to :meth:`fit`,
        create a faceted figure with one panel per domain.
    reference_ranks : list of int
        Vertical reference lines at these ranks (default ``[10, 100]``).
    figsize : tuple of float
        Figure size in inches.  For faceted plots this is the size of
        each panel.
    highlight_labels : list of str, optional
        Display names for highlight documents.
    """

    def __init__(
        self,
        *,
        mode: Literal["histogram", "cdf", "kde"] = "histogram",
        log_scale: bool = True,
        bins: int | str = "auto",
        highlight_color: str | list[str] | None = None,
        facet_by_label: bool = False,
        reference_ranks: list[int] | None = None,
        figsize: tuple[float, float] = (6, 4),
        highlight_labels: list[str] | None = None,
    ) -> None:
        super().__init__(figsize=figsize, highlight_labels=highlight_labels)
        self.mode = mode
        self.log_scale = log_scale
        self.bins = bins
        self.highlight_color = highlight_color
        self.facet_by_label = facet_by_label
        self.reference_ranks = (
            reference_ranks if reference_ranks is not None else [10, 100]
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(self, **kwargs) -> Figure:
        data = self._data
        n_hl = data["n_highlight"]
        highlight_ranks = data["highlight_ranks"]  # (n_q, m)
        colours = self._resolve_colours(n_hl)

        if self.facet_by_label and self._query_labels is not None:
            return self._plot_faceted(highlight_ranks, colours)
        return self._plot_single(highlight_ranks, colours)

    def _plot_single(
        self,
        highlight_ranks: np.ndarray,
        colours: list[str],
    ) -> Figure:
        """Single-panel distribution."""
        fig, ax = plt.subplots(figsize=self.figsize)
        n_hl = highlight_ranks.shape[1]

        for h in range(n_hl):
            ranks = highlight_ranks[:, h]
            self._draw_distribution(ax, ranks, colours[h], self._highlight_label(h))

        self._format_ax(ax)
        if n_hl > 0:
            ax.legend(fontsize=7)
        fig.tight_layout()
        return fig

    def _plot_faceted(
        self,
        highlight_ranks: np.ndarray,
        colours: list[str],
    ) -> Figure:
        """Faceted figure — one panel per domain."""
        labels = self._query_labels
        unique_labels = _unique_ordered(labels)
        n_facets = len(unique_labels)
        n_hl = highlight_ranks.shape[1]

        ncols = min(4, n_facets)
        nrows = int(np.ceil(n_facets / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(self.figsize[0] * ncols, self.figsize[1] * nrows),
            squeeze=False,
        )

        for idx, lbl in enumerate(unique_labels):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            mask = labels == lbl

            for h in range(n_hl):
                ranks = highlight_ranks[mask, h]
                label = self._highlight_label(h) if idx == 0 else None
                self._draw_distribution(ax, ranks, colours[h], label)

            self._format_ax(ax)
            ax.set_title(str(lbl), fontsize=8, pad=3)

        # Hide unused panels.
        for idx in range(n_facets, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        if n_hl > 0:
            axes[0, 0].legend(fontsize=7)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Distribution drawing
    # ------------------------------------------------------------------

    def _draw_distribution(self, ax, ranks, colour, label):
        """Draw a single distribution on *ax* according to *self.mode*."""
        if self.mode == "histogram":
            self._draw_histogram(ax, ranks, colour, label)
        elif self.mode == "cdf":
            self._draw_cdf(ax, ranks, colour, label)
        elif self.mode == "kde":
            self._draw_kde(ax, ranks, colour, label)
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")

    def _draw_histogram(self, ax, ranks, colour, label):
        if self.log_scale:
            max_rank = max(ranks.max(), 10)
            bin_edges = np.logspace(0, np.log10(max_rank + 1), 30)
        else:
            bin_edges = self.bins
        ax.hist(
            ranks,
            bins=bin_edges,
            color=colour,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )

    def _draw_cdf(self, ax, ranks, colour, label):
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        ax.step(sorted_ranks, cdf, color=colour, linewidth=1.2, label=label, where="post")
        ax.set_ylabel("Cumulative proportion")

    def _draw_kde(self, ax, ranks, colour, label):
        # Simple Gaussian KDE on log-transformed ranks.
        log_ranks = np.log10(ranks.clip(min=1))
        x_grid = np.linspace(log_ranks.min() - 0.5, log_ranks.max() + 0.5, 200)

        # Silverman bandwidth.
        std = log_ranks.std()
        n = len(log_ranks)
        bw = 1.06 * std * n ** (-1 / 5) if std > 0 else 0.5

        kernel = np.exp(
            -0.5 * ((x_grid[None, :] - log_ranks[:, None]) / bw) ** 2
        ) / (bw * np.sqrt(2 * np.pi))
        density = kernel.mean(axis=0)

        ax.fill_between(10 ** x_grid, density, alpha=0.3, color=colour, linewidth=0)
        ax.plot(10 ** x_grid, density, color=colour, linewidth=1.0, label=label)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_ax(self, ax):
        for r in self.reference_ranks:
            ax.axvline(r, color=CORPUS_COLOUR, linewidth=0.5, linestyle="--")
            ax.text(
                r, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
                f"k={r}", fontsize=6, color="#888888", ha="left", va="top",
            )

        ax.set_xlabel("Rank")
        if self.log_scale:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.get_major_formatter().set_scientific(False)

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


def _unique_ordered(arr: np.ndarray) -> list:
    """Unique values preserving first-occurrence order."""
    seen = set()
    out = []
    for v in arr:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out
