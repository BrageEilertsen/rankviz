"""Tests for the quick_plot convenience function."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from rankviz import quick_plot


class TestQuickPlot:
    def test_rank_carpet(self, synthetic_data):
        fig = quick_plot(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
            kind="rank_carpet",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_similarity_waterfall(self, synthetic_data):
        fig = quick_plot(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
            kind="similarity_waterfall",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_rank_distribution(self, synthetic_data):
        fig = quick_plot(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
            kind="rank_distribution",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_kind_raises(self, synthetic_data):
        with pytest.raises(ValueError, match="Unknown kind"):
            quick_plot(
                queries=synthetic_data["queries"],
                corpus=synthetic_data["corpus"],
                highlight=synthetic_data["poison"],
                kind="nonexistent",
            )
