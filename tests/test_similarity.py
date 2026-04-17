"""Tests for the similarity and ranking computation engine."""

from __future__ import annotations

import numpy as np
import pytest

from rankviz._similarity import (
    _validate_embeddings,
    cosine_similarity_matrix,
    ranks_from_similarities,
    compute_retrieval_data,
)


class TestValidateEmbeddings:
    def test_1d_promoted_to_2d(self):
        arr = np.ones(10, dtype=np.float32)
        result = _validate_embeddings(arr, "test")
        assert result.shape == (1, 10)
        assert result.dtype == np.float32

    def test_2d_passthrough(self):
        arr = np.ones((5, 10), dtype=np.float64)
        result = _validate_embeddings(arr, "test")
        assert result.shape == (5, 10)
        assert result.dtype == np.float32

    def test_3d_raises(self):
        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            _validate_embeddings(arr, "test")

    def test_dimension_mismatch_raises(self):
        arr = np.ones((5, 10))
        with pytest.raises(ValueError, match="expected 8"):
            _validate_embeddings(arr, "test", expected_dim=8)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sim = cosine_similarity_matrix(v, v)
        np.testing.assert_allclose(sim, [[1.0]], atol=1e-6)

    def test_orthogonal_vectors(self):
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        d = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = cosine_similarity_matrix(q, d)
        np.testing.assert_allclose(sim, [[0.0]], atol=1e-6)

    def test_known_similarity(self):
        # cos(45 degrees) = sqrt(2)/2 ≈ 0.7071
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        d = np.array([[1.0, 1.0]], dtype=np.float32) / np.sqrt(2)
        sim = cosine_similarity_matrix(q, d)
        np.testing.assert_allclose(sim, [[np.sqrt(2) / 2]], atol=1e-5)

    def test_shape(self):
        q = np.random.randn(10, 64).astype(np.float32)
        d = np.random.randn(50, 64).astype(np.float32)
        sim = cosine_similarity_matrix(q, d)
        assert sim.shape == (10, 50)


class TestRanksFromSimilarities:
    def test_simple_ranking(self):
        # Query 0: doc order by similarity is [2, 0, 1]
        sim = np.array([[0.5, 0.1, 0.9]], dtype=np.float32)
        ranks = ranks_from_similarities(sim)
        assert ranks[0, 2] == 1  # highest sim → rank 1
        assert ranks[0, 0] == 2
        assert ranks[0, 1] == 3

    def test_rank_1_indexed(self):
        sim = np.random.randn(5, 20).astype(np.float32)
        ranks = ranks_from_similarities(sim)
        assert ranks.min() == 1
        assert ranks.max() == 20

    def test_each_row_is_permutation(self):
        sim = np.random.randn(8, 15).astype(np.float32)
        ranks = ranks_from_similarities(sim)
        for i in range(8):
            assert set(ranks[i]) == set(range(1, 16))


class TestComputeRetrievalData:
    def test_embedding_path(self, synthetic_data):
        result = compute_retrieval_data(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=synthetic_data["poison"],
        )
        assert result["n_queries"] == 60
        assert result["n_corpus"] == 300
        assert result["n_highlight"] == 1
        assert result["corpus_similarities"].shape == (60, 300)
        assert result["highlight_ranks"].shape == (60, 1)
        # Highlight ranks are 1-indexed and within [1, 301].
        assert result["highlight_ranks"].min() >= 1
        assert result["highlight_ranks"].max() <= 301

    def test_precomputed_path(self):
        rng = np.random.default_rng(0)
        sim_corpus = rng.standard_normal((10, 50)).astype(np.float32)
        sim_hl = rng.standard_normal((10,)).astype(np.float32)

        result = compute_retrieval_data(
            similarities=sim_corpus,
            highlight_similarities=sim_hl,
        )
        assert result["n_queries"] == 10
        assert result["n_corpus"] == 50
        assert result["n_highlight"] == 1

    def test_multiple_highlights(self, synthetic_data):
        two_highlights = np.stack([
            synthetic_data["poison"],
            synthetic_data["corpus"][0],
        ])
        result = compute_retrieval_data(
            queries=synthetic_data["queries"],
            corpus=synthetic_data["corpus"],
            highlight=two_highlights,
        )
        assert result["n_highlight"] == 2
        assert result["highlight_ranks"].shape == (60, 2)

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            compute_retrieval_data()
