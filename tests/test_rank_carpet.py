"""Smoke tests for RankCarpet."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rankviz import RankCarpet


class TestRankCarpet:
    def test_fit_returns_self(self, synthetic_data):
        rc = RankCarpet()
        result = rc.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        assert result is rc

    def test_plot_returns_figure(self, synthetic_data):
        rc = RankCarpet()
        rc.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        fig = rc.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_before_fit_raises(self):
        rc = RankCarpet()
        with pytest.raises(RuntimeError, match="fit"):
            rc.plot()

    def test_individual_lines(self, synthetic_data):
        # Small corpus subset to test individual_lines path.
        rc = RankCarpet(individual_lines=True)
        rc.fit(
            queries=synthetic_data["queries"][:10],
            corpus=synthetic_data["corpus"][:50],
            highlight=synthetic_data["poison"],
        )
        fig = rc.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_query_labels(self, synthetic_data):
        rc = RankCarpet()
        rc.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
            query_labels=synthetic_data["query_labels"],
        )
        fig = rc.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_transform_shape(self, synthetic_data):
        rc = RankCarpet()
        rc.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        ranks = rc.transform()
        assert ranks.shape == (60, 1)

    def test_multiple_highlights(self, synthetic_data):
        highlights = np.stack([
            synthetic_data["poison"],
            synthetic_data["corpus"][0],
        ])
        rc = RankCarpet(highlight_labels=["poison", "corpus doc 0"])
        rc.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=highlights,
        )
        fig = rc.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_log_scale_false(self, synthetic_data):
        rc = RankCarpet(log_scale=False)
        rc.fit(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        fig = rc.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
