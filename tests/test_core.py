"""Tests for the CORE projection algorithm."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rankviz import CORE, plot_landscape


class TestFitShapes:
    def test_fit_3d(self, synthetic_data):
        core = CORE(n_components=3, n_iter=50)
        core.fit(synthetic_data["queries"], synthetic_data["corpus"])
        assert core.query_embedding_.shape == (synthetic_data["queries"].shape[0], 3)
        assert core.corpus_embedding_.shape == (synthetic_data["corpus"].shape[0], 3)

    def test_fit_2d(self, synthetic_data):
        core = CORE(n_components=2, n_iter=50)
        core.fit(synthetic_data["queries"], synthetic_data["corpus"])
        assert core.query_embedding_.shape == (synthetic_data["queries"].shape[0], 2)
        assert core.corpus_embedding_.shape == (synthetic_data["corpus"].shape[0], 2)

    def test_invalid_n_components(self):
        with pytest.raises(ValueError, match="2 or 3"):
            CORE(n_components=4)

    def test_dim_mismatch(self, synthetic_data):
        core = CORE(n_iter=10)
        bad_corpus = synthetic_data["corpus"][:, :32]  # wrong dim
        with pytest.raises(ValueError):
            core.fit(synthetic_data["queries"], bad_corpus)


class TestLossBehaviour:
    def test_loss_decreases(self, synthetic_data):
        core = CORE(n_components=2, n_iter=200)
        core.fit(synthetic_data["queries"], synthetic_data["corpus"])
        # Loss at end should be noticeably lower than at the start.
        loss = core.loss_history_
        assert loss[-1] < loss[0] * 0.9

    def test_loss_history_length(self, synthetic_data):
        core = CORE(n_iter=30)
        core.fit(synthetic_data["queries"], synthetic_data["corpus"])
        assert len(core.loss_history_) == 30


class TestTransform:
    def test_transform_single_vector(self, synthetic_data):
        core = CORE(n_components=3, n_iter=100).fit(
            synthetic_data["queries"], synthetic_data["corpus"],
        )
        out = core.transform(synthetic_data["poison"])
        assert out.shape == (3,)

    def test_transform_batch(self, synthetic_data):
        core = CORE(n_components=2, n_iter=100).fit(
            synthetic_data["queries"], synthetic_data["corpus"],
        )
        batch = synthetic_data["corpus"][:5]
        out = core.transform(batch)
        assert out.shape == (5, 2)

    def test_transform_before_fit_raises(self, synthetic_data):
        core = CORE()
        with pytest.raises(RuntimeError, match="fit"):
            core.transform(synthetic_data["poison"])

    def test_transform_landing_consistent(self, synthetic_data):
        """An embedding that IS a query should project near that query."""
        core = CORE(n_components=3, n_iter=200).fit(
            synthetic_data["queries"], synthetic_data["corpus"],
        )
        q0 = synthetic_data["queries"][0]
        projected = core.transform(q0)
        # Distance to query 0's fitted position should be the smallest among queries.
        dists = np.linalg.norm(core.query_embedding_ - projected, axis=1)
        assert np.argmin(dists) == 0


class TestRankPreservation:
    def test_top_similarity_order_preserved(self, synthetic_data):
        """The original retrieval ranking for each query should be approximately
        preserved in the low-dim projection (top-k agreement)."""
        Q = synthetic_data["queries"]
        D = synthetic_data["corpus"]
        core = CORE(n_components=3, n_iter=300, weight="retrieval").fit(Q, D)

        # Original ranking for a handful of queries.
        sims_hi = Q @ D.T                        # (n_q, n_d)
        top_k_hi = np.argsort(-sims_hi, axis=1)[:, :10]

        # Low-dim ranking (closest documents in projection).
        Q_low = core.query_embedding_
        D_low = core.corpus_embedding_
        # Pairwise distances.
        dists_lo = np.linalg.norm(
            Q_low[:, None, :] - D_low[None, :, :], axis=-1,
        )
        top_k_lo = np.argsort(dists_lo, axis=1)[:, :10]

        # Compute overlap between top-10 sets across queries.
        overlaps = []
        for i in range(Q.shape[0]):
            hi = set(top_k_hi[i].tolist())
            lo = set(top_k_lo[i].tolist())
            overlaps.append(len(hi & lo) / 10)
        mean_overlap = float(np.mean(overlaps))

        # CORE should beat the random baseline (~3 % for n_d=300) and at
        # least match PCA's top-10 overlap (~28 % on this synthetic set).
        # A lossy 3-D projection from 64-D cannot preserve full top-10.
        assert mean_overlap > 0.28, f"mean top-10 overlap was {mean_overlap:.3f}"


class TestWeightSchemes:
    @pytest.mark.parametrize("scheme", ["uniform", "retrieval", "rank"])
    def test_all_schemes_fit(self, synthetic_data, scheme):
        core = CORE(n_components=2, n_iter=50, weight=scheme)
        core.fit(synthetic_data["queries"], synthetic_data["corpus"])
        assert core.query_embedding_.shape[1] == 2


class TestPlotLandscape:
    def test_matplotlib_2d(self, synthetic_data):
        core = CORE(n_components=2, n_iter=30).fit(
            synthetic_data["queries"], synthetic_data["corpus"],
        )
        fig = plot_landscape(
            core,
            highlight=synthetic_data["poison"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_matplotlib_3d(self, synthetic_data):
        core = CORE(n_components=3, n_iter=30).fit(
            synthetic_data["queries"], synthetic_data["corpus"],
        )
        fig = plot_landscape(
            core,
            highlight=synthetic_data["poison"],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_before_fit_raises(self):
        core = CORE()
        with pytest.raises(RuntimeError, match="fitted"):
            plot_landscape(core)

    def test_plotly_3d(self, synthetic_data):
        pytest.importorskip("plotly")
        core = CORE(n_components=3, n_iter=30).fit(
            synthetic_data["queries"], synthetic_data["corpus"],
        )
        fig = plot_landscape(
            core,
            highlight=synthetic_data["poison"],
            backend="plotly",
        )
        # Plotly figure has a .to_html() method.
        assert hasattr(fig, "to_html")

    def test_trajectory_and_target(self, synthetic_data):
        core = CORE(n_components=2, n_iter=30).fit(
            synthetic_data["queries"], synthetic_data["corpus"],
        )
        # Synthesise a small trajectory.
        traj = np.stack([
            synthetic_data["poison"] * 0.5 + synthetic_data["queries"][0] * 0.5,
            synthetic_data["poison"],
        ])
        # Normalise.
        traj = traj / np.linalg.norm(traj, axis=1, keepdims=True)

        fig = plot_landscape(
            core,
            highlight=synthetic_data["poison"],
            trajectory=traj,
            target=synthetic_data["queries"][0],
            backend="matplotlib",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
