"""CORE — Cosine-Ordered Retrieval Embedding.

A dimensionality-reduction algorithm specialised for retrieval.  Unlike
PCA, t-SNE, or UMAP — which optimise for variance, local neighbourhoods,
or manifold topology — CORE preserves the bipartite query-document
cosine structure that actually determines retrieval.

The objective: place queries and documents in low-dim space such that
Euclidean distance approximates 1 - cos(query, document) for every
query-document pair, weighted by retrieval importance.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


class CORE:
    """Cosine-Ordered Retrieval Embedding.

    Projects a bipartite query/document set into 2-D or 3-D coordinates
    that preserve query-document cosine similarities as Euclidean
    distances, weighted so top-of-ranking is preserved more faithfully
    than the tail.

    Parameters
    ----------
    n_components : int
        Output dimensionality (2 or 3).
    weight : ``"uniform"`` | ``"retrieval"`` | ``"rank"``
        Pair-weighting scheme.  ``"retrieval"`` (default) uses
        ``cos^4``, emphasising high-similarity pairs; ``"rank"`` uses
        ``1 / rank`` to explicitly preserve top-k structure;
        ``"uniform"`` gives plain bipartite MDS.
    n_iter : int
        Number of full-batch gradient-descent iterations.
    learning_rate : float
        Initial learning rate (decayed linearly to 10 % of its value).
    init : ``"svd"`` | ``"random"``
        Initialisation strategy.  ``"svd"`` uses the top singular
        vectors of the stacked query/document matrix; ``"random"`` uses
        Gaussian noise.
    random_state : int, optional
        Seed for reproducibility.

    Attributes
    ----------
    query_embedding_ : ndarray, shape ``(n_queries, n_components)``
        Low-dim coordinates for the fitted queries.
    corpus_embedding_ : ndarray, shape ``(n_corpus, n_components)``
        Low-dim coordinates for the fitted corpus.
    loss_history_ : ndarray, shape ``(n_iter,)``
        Training loss at each iteration.
    """

    def __init__(
        self,
        n_components: int = 3,
        *,
        weight: Literal["uniform", "retrieval", "rank"] = "retrieval",
        n_iter: int = 500,
        learning_rate: float = 0.1,
        init: Literal["svd", "random"] = "svd",
        random_state: int | None = 42,
    ) -> None:
        if n_components not in (2, 3):
            raise ValueError("n_components must be 2 or 3.")
        self.n_components = n_components
        self.weight = weight
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.init = init
        self.random_state = random_state

        self.query_embedding_: np.ndarray | None = None
        self.corpus_embedding_: np.ndarray | None = None
        self.loss_history_: np.ndarray | None = None
        self._Q_ref: np.ndarray | None = None
        self._C_ref: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, queries: np.ndarray, corpus: np.ndarray) -> "CORE":
        """Learn the bipartite projection."""
        Q = np.asarray(queries, dtype=np.float32)
        D = np.asarray(corpus, dtype=np.float32)
        if Q.ndim != 2 or D.ndim != 2:
            raise ValueError("queries and corpus must be 2-D (n, d).")
        if Q.shape[1] != D.shape[1]:
            raise ValueError("queries and corpus must share dim.")

        sims = Q @ D.T                     # (n_q, n_d)
        target_dists = 1.0 - sims          # (n_q, n_d)
        weights = _pair_weights(sims, self.weight)

        Q_low, D_low = self._init_embedding(Q, D)

        # Normalisation: gradients are averaged over the bipartite pair
        # set so the step size is independent of n_q and n_d.
        n_q, n_d = Q.shape[0], D.shape[0]
        norm_Q = 1.0 / n_d
        norm_D = 1.0 / n_q

        losses = np.empty(self.n_iter, dtype=np.float32)
        for it in range(self.n_iter):
            lr = self.learning_rate * (1.0 - 0.9 * it / max(1, self.n_iter - 1))

            # Pairwise low-dim differences and distances.
            diff = Q_low[:, None, :] - D_low[None, :, :]       # (n_q, n_d, k)
            dist = np.sqrt((diff ** 2).sum(axis=-1)) + 1e-4    # (n_q, n_d)

            err = dist - target_dists                           # (n_q, n_d)
            losses[it] = float((weights * err ** 2).mean())

            coeff = (weights * err / dist)[..., None]           # (n_q, n_d, 1)
            grad_Q = (coeff * diff).sum(axis=1) * norm_Q        # (n_q, k)
            grad_D = -(coeff * diff).sum(axis=0) * norm_D       # (n_d, k)

            # Gradient clipping guards against pathological early steps.
            _clip_inplace(grad_Q, 1.0)
            _clip_inplace(grad_D, 1.0)

            Q_low -= lr * grad_Q
            D_low -= lr * grad_D

        self.query_embedding_ = Q_low
        self.corpus_embedding_ = D_low
        self.loss_history_ = losses
        self._Q_ref = Q
        self._C_ref = D
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new points into the fitted query landscape.

        Each new point's position is found by a small optimisation
        against the fixed query coordinates — the query landscape does
        not change.  This keeps trajectories and comparisons stable.

        Parameters
        ----------
        X : ndarray, shape ``(d,)`` or ``(n, d)``
            New embedding(s) in the original D-dim space.

        Returns
        -------
        ndarray, shape ``(n_components,)`` or ``(n, n_components)``
        """
        if self.query_embedding_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=np.float32)
        squeeze = False
        if X.ndim == 1:
            X = X[None, :]
            squeeze = True

        n = X.shape[0]
        out = np.empty((n, self.n_components), dtype=np.float32)

        sims_all = self._Q_ref @ X.T               # (n_q, n)
        targets_all = 1.0 - sims_all               # (n_q, n)

        for k in range(n):
            sims = sims_all[:, k]
            target = targets_all[:, k]
            w = _pair_weights(sims[None, :], self.weight)[0]  # (n_q,)

            # Initialise at weighted query centroid.
            y = (w[:, None] * self.query_embedding_).sum(axis=0) / (w.sum() + 1e-8)

            n_q = self.query_embedding_.shape[0]
            for it in range(200):
                lr = 0.1 * (1.0 - 0.9 * it / 199)
                diff = y[None, :] - self.query_embedding_      # (n_q, k)
                dist = np.sqrt((diff ** 2).sum(axis=-1)) + 1e-4
                err = dist - target                            # (n_q,)
                coeff = (w * err / dist)[:, None]              # (n_q, 1)
                grad = (coeff * diff).sum(axis=0) / n_q        # (k,)
                g_norm = np.linalg.norm(grad)
                if g_norm > 1.0:
                    grad = grad / g_norm
                y -= lr * grad

            out[k] = y

        return out[0] if squeeze else out

    def fit_transform(
        self,
        queries: np.ndarray,
        corpus: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: fit, then return ``(Q_low, D_low)``."""
        self.fit(queries, corpus)
        return self.query_embedding_, self.corpus_embedding_

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_embedding(
        self, Q: np.ndarray, D: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initial low-dim placement."""
        n_q = Q.shape[0]
        rng = np.random.default_rng(self.random_state)

        if self.init == "svd":
            stacked = np.concatenate([Q, D], axis=0)
            centred = stacked - stacked.mean(axis=0, keepdims=True)
            # Top-k right singular vectors.
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            low = centred @ Vt[: self.n_components].T
            # Scale so typical pairwise distance sits near 1 (comparable
            # to 1 - cos range).
            scale = np.std(low) + 1e-8
            low = low / scale * 0.5
            return low[:n_q].astype(np.float32), low[n_q:].astype(np.float32)

        elif self.init == "random":
            Q_low = rng.standard_normal((n_q, self.n_components)).astype(np.float32) * 0.1
            D_low = rng.standard_normal((D.shape[0], self.n_components)).astype(np.float32) * 0.1
            return Q_low, D_low

        raise ValueError(f"Unknown init {self.init!r}")


# ----------------------------------------------------------------------
# Module-private weight helpers
# ----------------------------------------------------------------------

def _pair_weights(sims: np.ndarray, scheme: str) -> np.ndarray:
    """Compute pair weights from a similarity matrix (or row)."""
    if scheme == "uniform":
        return np.ones_like(sims, dtype=np.float32)

    if scheme == "retrieval":
        w = np.clip(sims, 0.0, None) ** 4
        mean = w.mean()
        if mean > 0:
            w = w / mean
        return w.astype(np.float32)

    if scheme == "rank":
        # Rank within each query row (1 = most similar).
        if sims.ndim == 1:
            order = np.argsort(-sims)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, sims.shape[0] + 1)
            return (1.0 / ranks).astype(np.float32)
        order = np.argsort(-sims, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(sims.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, sims.shape[1] + 1)
        return (1.0 / ranks).astype(np.float32)

    raise ValueError(f"Unknown weight scheme {scheme!r}")


def _clip_inplace(grad: np.ndarray, max_norm: float) -> None:
    """Clip per-row gradient L2 norm to ``max_norm`` in-place."""
    norms = np.linalg.norm(grad, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + 1e-8))
    grad *= scale
