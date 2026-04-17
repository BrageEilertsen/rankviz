"""Base class for all rankviz visualisations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from rankviz._similarity import compute_retrieval_data
from rankviz._style import style as _style_ctx


class BaseVisualisation(ABC):
    """Abstract base for every rankviz visualisation.

    Subclasses must implement :meth:`_plot` (the actual drawing logic).
    The public :meth:`fit` / :meth:`plot` / :meth:`transform` interface
    is defined here and follows sklearn conventions.

    Parameters
    ----------
    figsize : tuple of float
        Figure size in inches ``(width, height)``.
    highlight_labels : list of str, optional
        Display names for each highlight document.
    """

    def __init__(
        self,
        *,
        figsize: tuple[float, float] = (8, 4),
        highlight_labels: list[str] | None = None,
    ) -> None:
        self.figsize = figsize
        self.highlight_labels = highlight_labels

        # Populated by fit().
        self._data: dict | None = None
        self._query_labels: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        queries: np.ndarray | None = None,
        corpus: np.ndarray | None = None,
        highlight: np.ndarray | None = None,
        *,
        query_labels: np.ndarray | Sequence[str] | None = None,
        similarities: np.ndarray | None = None,
        highlight_similarities: np.ndarray | None = None,
        exclude_highlight_from_corpus: bool = True,
        highlight_indices: np.ndarray | Sequence[int] | None = None,
    ) -> "BaseVisualisation":
        """Compute ranks and similarities needed for plotting.

        Accepts either raw embeddings or precomputed similarity matrices.
        See :func:`rankviz._similarity.compute_retrieval_data` for full
        parameter documentation.

        Returns *self* for method chaining.
        """
        self._data = compute_retrieval_data(
            queries=queries,
            corpus=corpus,
            highlight=highlight,
            similarities=similarities,
            highlight_similarities=highlight_similarities,
            exclude_highlight_from_corpus=exclude_highlight_from_corpus,
            highlight_indices=highlight_indices,
        )
        if query_labels is not None:
            self._query_labels = np.asarray(query_labels)
        else:
            self._query_labels = None
        return self

    def plot(self, **kwargs) -> Figure:
        """Render the visualisation and return a :class:`~matplotlib.figure.Figure`.

        Must be called after :meth:`fit`.  All keyword arguments are
        forwarded to the subclass's ``_plot`` implementation.
        """
        if self._data is None:
            raise RuntimeError("Call fit() before plot().")
        with _style_ctx():
            fig = self._plot(**kwargs)
        return fig

    def transform(self) -> np.ndarray:
        """Return the underlying numeric data for custom plotting.

        The exact array returned depends on the subclass — typically
        the highlight-rank or highlight-similarity matrix.
        """
        if self._data is None:
            raise RuntimeError("Call fit() before transform().")
        return self._transform()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _plot(self, **kwargs) -> Figure:
        """Subclasses draw here.  Must return a Figure."""

    def _transform(self) -> np.ndarray:
        """Default transform returns highlight ranks."""
        return self._data["highlight_ranks"]

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def _highlight_label(self, idx: int) -> str:
        """Return the display label for highlight document *idx*."""
        if self.highlight_labels and idx < len(self.highlight_labels):
            return self.highlight_labels[idx]
        if self._data["n_highlight"] == 1:
            return "highlight"
        return f"highlight {idx}"
