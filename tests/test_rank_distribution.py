"""Smoke tests for RankDistribution."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rankviz import RankDistribution


class TestRankDistribution:
    def test_fit_returns_self(self, synthetic_data):
        rd = RankDistribution()
        result = rd.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        assert result is rd

    def test_histogram_mode(self, synthetic_data):
        rd = RankDistribution(mode="histogram")
        rd.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        fig = rd.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cdf_mode(self, synthetic_data):
        rd = RankDistribution(mode="cdf")
        rd.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        fig = rd.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_kde_mode(self, synthetic_data):
        rd = RankDistribution(mode="kde")
        rd.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        fig = rd.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_faceted(self, synthetic_data):
        rd = RankDistribution(facet_by_label=True)
        rd.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
            query_labels=synthetic_data["query_labels"],
        )
        fig = rd.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_before_fit_raises(self):
        rd = RankDistribution()
        with pytest.raises(RuntimeError, match="fit"):
            rd.plot()

    def test_multiple_highlights(self, synthetic_data):
        highlights = np.stack([
            synthetic_data["poison"],
            synthetic_data["corpus"][0],
        ])
        rd = RankDistribution(
            mode="cdf",
            highlight_labels=["poison", "specialist"],
        )
        rd.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=highlights,
        )
        fig = rd.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_mode_raises(self, synthetic_data):
        rd = RankDistribution(mode="invalid")
        rd.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        with pytest.raises(ValueError, match="Unknown mode"):
            rd.plot()
