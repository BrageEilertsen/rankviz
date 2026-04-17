"""Smoke tests for SimilarityWaterfall."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from rankviz import SimilarityWaterfall


class TestSimilarityWaterfall:
    def test_fit_returns_self(self, synthetic_data):
        sw = SimilarityWaterfall()
        result = sw.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        assert result is sw

    def test_plot_returns_figure(self, synthetic_data):
        sw = SimilarityWaterfall()
        sw.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        fig = sw.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_before_fit_raises(self):
        sw = SimilarityWaterfall()
        with pytest.raises(RuntimeError, match="fit"):
            sw.plot()

    def test_no_threshold(self, synthetic_data):
        sw = SimilarityWaterfall(show_threshold=False, shade_margin=False)
        sw.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        fig = sw.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_transform_shape(self, synthetic_data):
        sw = SimilarityWaterfall()
        sw.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        sims = sw.transform()
        assert sims.shape == (60, 1)

    def test_custom_k(self, synthetic_data):
        sw = SimilarityWaterfall(k=5)
        sw.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        fig = sw.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
