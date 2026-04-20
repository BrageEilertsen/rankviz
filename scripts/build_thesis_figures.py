"""Generate the four thesis figures (A, B, C, D) from the pre-built
NPZ/JSON bundles under ``figure_data/``.

Usage::

    python scripts/build_thesis_figures.py \\
        --bundles-dir /path/to/figure_data \\
        --out-dir     /path/to/thesis_figures

Each figure is saved as a PDF matching the filename the thesis LaTeX
expects (``core_phase1_target.pdf``, ``core_phase2_trajectory.pdf``,
``core_cross_model_citations.pdf``, ``core_gt_stress_test.pdf``).

Design choices
--------------
* Figures A and B share a single CORE fit on the biopsy corpus + queries
  so their axes are identical.
* Figure C shares one CORE fit across all LLM panels.
* Figure D refits CORE per panel column: one fit on
  ``(corpus + poison)`` for the baseline row, a separate fit on
  ``(corpus + poison + GT)`` for the with-GT row. The audit note
  requires this so projected positions are faithful to each
  retrieval condition.
* All figures use matplotlib and ``rankviz.style()`` for publication
  defaults (300 DPI, sans-serif font fallback, thin spines).
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from rankviz import CORE
from rankviz._style import style as _style_ctx


# ---------------------------------------------------------------------------
# Palette tuned to the thesis aesthetic the brief calls for
# ---------------------------------------------------------------------------
COLOUR_CORPUS     = "#CCCCCC"
COLOUR_CORPUS_A   = 0.35
COLOUR_CORPUS_DIM = 0.15
COLOUR_QUERY      = "#555555"
COLOUR_CENTROID   = "#222222"
COLOUR_U_HAT      = "#4477AA"           # blue — spectral direction
COLOUR_X_TILDE    = "#228833"           # green — geometric median / GT
COLOUR_X_RAW      = "#EE6677"           # hollow red — pre-projection target
COLOUR_X_HAT      = "#CC3344"           # filled red — final target (poison-ish)
COLOUR_POISON     = "#CC3344"
COLOUR_GT         = "#228833"
COLOUR_CAP        = "#888888"
COLOUR_TOPK       = "#222222"

STAGE_PALETTE = {
    "exploration": "#EE8866",
    "refinement":  "#AA3377",
    "polishing":   "#228833",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _core_fit(queries: np.ndarray, corpus: np.ndarray, n_components: int = 2,
              n_iter: int = 400) -> CORE:
    """Fit CORE with the standard thesis settings."""
    return CORE(
        n_components=n_components, n_iter=n_iter, weight="retrieval",
    ).fit(queries=queries, corpus=corpus)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def _subsample(arr: np.ndarray, n: int = 2000, seed: int = 42) -> np.ndarray:
    if arr.shape[0] <= n:
        return np.arange(arr.shape[0])
    rng = np.random.default_rng(seed)
    return rng.choice(arr.shape[0], n, replace=False)


def _cap_disc_in_plane(
    core: CORE, u_hat: np.ndarray, x_tilde: np.ndarray, gamma: float,
    n: int = 240, radius_quantile: float = 0.95,
) -> np.ndarray:
    """Approximate the cap boundary around ``u_hat`` as a circle in the
    CORE 2-D plane.

    The true cap is a 767-dimensional surface whose CORE projection is
    not a circle — sampling it directly produces a visually confusing
    curve.  Instead we:

    1. Sample boundary points via the analytic parametrisation,
    2. project them with CORE,
    3. compute the centre and mean radius of the projected cloud,
    4. draw a *clean circle* in 2-D centred on ``core.transform(u_hat)``
       with radius equal to the median distance of projected boundary
       points from that centre.

    This preserves the two pieces of information the reader needs —
    "the cap lives around u_hat, with this approximate extent" — and
    drops the visually noisy curve shape that the 768-D → 2-D
    projection produces.
    """
    v = x_tilde - np.dot(x_tilde, u_hat) * u_hat
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        v = np.random.default_rng(0).standard_normal(u_hat.shape[0])
        v = v - np.dot(v, u_hat) * u_hat
        v_norm = np.linalg.norm(v)
    v = v / v_norm

    w = np.random.default_rng(7).standard_normal(u_hat.shape[0])
    w = w - np.dot(w, u_hat) * u_hat - np.dot(w, v) * v
    w = w / (np.linalg.norm(w) + 1e-12)

    theta = np.linspace(0, 2 * np.pi, n)
    cg, sg = np.cos(gamma), np.sin(gamma)
    boundary = (
        cg * u_hat[None, :]
        + sg * (np.cos(theta)[:, None] * v[None, :] + np.sin(theta)[:, None] * w[None, :])
    )
    boundary = _l2(boundary.astype(np.float32))

    u_xy = core.transform(u_hat)
    pts_xy = core.transform(boundary)
    d = np.linalg.norm(pts_xy - u_xy, axis=1)
    radius = float(np.percentile(d, radius_quantile * 100))

    t = np.linspace(0, 2 * np.pi, 160)
    circle = np.stack([u_xy[0] + radius * np.cos(t),
                       u_xy[1] + radius * np.sin(t)], axis=1)
    return circle


def _autoscale(ax, *xy_groups, pad_frac: float = 0.15) -> None:
    """Set axis limits to tightly frame one or more (N, 2) point groups."""
    pts = np.concatenate([g for g in xy_groups if g is not None and len(g)], axis=0)
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    xpad = (xmax - xmin) * pad_frac
    ypad = (ymax - ymin) * pad_frac
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


def _style_axis(ax, xlabel="CORE 1", ylabel="CORE 2"):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Figure A — Phase 1 target construction
# ---------------------------------------------------------------------------

def figure_A(bundle_path: str, out_pdf: str,
             cap_quantile: float = 0.95) -> CORE:
    """Build Figure A.

    ``cap_quantile`` controls the radius of the cap circle drawn
    in 2-D.  The default is the 95th percentile of projected boundary
    distances from ``u_hat_xy`` — which is the most honest approximation
    of the cap's extent in the projection.  For this domain's stored
    γ* calibration, the raw target x_raw is genuinely inside the cap
    (cos(u_hat, x_raw)=14°, γ*=34°), so the cap-projection step in
    equation (meth:project_cap) is inactive; the figure caption should
    note this explicitly.
    """
    data = np.load(bundle_path, allow_pickle=True)
    Q = data["query_embeddings"].astype(np.float32)
    D = data["doc_embeddings"].astype(np.float32)
    centroids = data["centroids"].astype(np.float32)
    u_hat = data["u_hat"].astype(np.float32)
    x_tilde = data["x_tilde"].astype(np.float32)
    x_raw = data["x_raw"].astype(np.float32)
    x_hat = data["x_hat"].astype(np.float32)
    gamma = float(data["gamma_star"])
    print(f"  figA: Q={Q.shape} D={D.shape} centroids={centroids.shape}"
          f"  γ*={np.degrees(gamma):.1f}°")

    print("  fitting CORE 2-D (this is the shared fit for Figures A and B)...")
    core = _core_fit(Q, D, n_components=2)

    # Project the geometric objects against the fitted query landscape.
    centroid_xy = core.transform(centroids)
    if centroid_xy.ndim == 1:
        centroid_xy = centroid_xy[None, :]
    u_xy = core.transform(u_hat)
    xt_xy = core.transform(x_tilde)
    xr_xy = core.transform(x_raw)
    xh_xy = core.transform(x_hat)
    q_xy = core.query_embedding_
    qc_xy = q_xy.mean(axis=0)

    # Cap disc in the 2-D plane (95th-percentile approximation).
    cap_xy = _cap_disc_in_plane(
        core, u_hat, x_tilde, gamma, n=240,
        radius_quantile=cap_quantile,
    )

    # Subsample corpus for rendering (avoid 20k-point scatter).
    doc_idx = _subsample(core.corpus_embedding_, n=2000)

    with _style_ctx():
        fig, ax = plt.subplots(figsize=(8.5, 6.5))

        # Corpus + queries as background.
        ax.scatter(core.corpus_embedding_[doc_idx, 0],
                   core.corpus_embedding_[doc_idx, 1],
                   s=2.5, c=COLOUR_CORPUS, alpha=0.5, rasterized=True,
                   zorder=1, label="Shadow corpus")
        ax.scatter(q_xy[:, 0], q_xy[:, 1], s=12, c=COLOUR_QUERY, alpha=0.8,
                   edgecolors="none", zorder=3, label=f"Queries (n={Q.shape[0]})")

        # Cap boundary: dashed arc (illustrative; shrunk quantile).
        ax.plot(cap_xy[:, 0], cap_xy[:, 1], color=COLOUR_CAP,
                linestyle="--", linewidth=0.8, alpha=0.9, zorder=4,
                label=r"Cap boundary (illustrative)")

        # Shaded cap interior to make the "inside vs outside" distinction
        # visually obvious.
        ax.fill(cap_xy[:, 0], cap_xy[:, 1], facecolor=COLOUR_CAP,
                alpha=0.06, zorder=2, edgecolor="none")

        # Interpolation axis from x_tilde through x_raw to u_hat.
        # Helps the reader see why x_raw sits "between" its two parents.
        axis_pts = np.array([xt_xy, xr_xy, u_xy])
        ax.plot(axis_pts[:, 0], axis_pts[:, 1], color="#888888",
                linestyle="-", linewidth=0.6, alpha=0.6, zorder=3)

        # Centroids: rendered on top of the query cluster with a leader
        # line to an offset triangle so the legend entry is actually
        # visible (the centroid co-locates with the tight query cluster
        # when the domain has a single cluster).
        for c_xy in centroid_xy:
            ax.scatter([c_xy[0]], [c_xy[1]], s=170, marker="^",
                       c=COLOUR_CENTROID, edgecolors="white",
                       linewidths=1.6, zorder=9,
                       label=r"Cluster centroid $\mu_k$")

        # û arrow (from query centroid to projected u_hat).
        ax.annotate(
            "", xy=(u_xy[0], u_xy[1]), xytext=(qc_xy[0], qc_xy[1]),
            arrowprops=dict(arrowstyle="->", color=COLOUR_U_HAT, lw=1.5),
            zorder=7,
        )
        ax.scatter([u_xy[0]], [u_xy[1]], s=60, marker="o",
                   c=COLOUR_U_HAT, edgecolors="white", linewidths=0.6,
                   zorder=7, label=r"Spectral direction $\hat{u}$")

        # x_tilde (geometric median).
        ax.scatter([xt_xy[0]], [xt_xy[1]], s=130, marker="s",
                   c=COLOUR_X_TILDE, edgecolors="white", linewidths=0.8,
                   zorder=8, label=r"Geometric median $\tilde{x}$")

        # Thin dotted line from x_raw to x_hat.  For this domain the cap
        # projection is inactive (x_raw is inside the cap in 768-D), so
        # the small offset between them reflects only renormalisation
        # after the interpolation.  We still render the connection so
        # the reader can trace the pairing.
        ax.plot([xr_xy[0], xh_xy[0]], [xr_xy[1], xh_xy[1]],
                color="#888888", linestyle=":", linewidth=0.8,
                alpha=0.7, zorder=9)

        # x_raw (hollow red star) and x_hat (filled red star).
        ax.scatter([xr_xy[0]], [xr_xy[1]], s=220, marker="*",
                   facecolors="white", edgecolors=COLOUR_X_RAW,
                   linewidths=1.6, zorder=9, label=r"Raw target $\hat{x}_{\mathrm{raw}}$")
        ax.scatter([xh_xy[0]], [xh_xy[1]], s=220, marker="*",
                   c=COLOUR_X_HAT, edgecolors="white", linewidths=0.9,
                   zorder=10, label=r"Final target $\hat{x}$")

        _style_axis(ax)
        # Zoom: frame the geometric objects (centroid, u, x_tilde, x_raw, x_hat)
        # plus a generous margin; the 20-k-doc ring is the outer context and
        # we keep only what fits in the window.
        focal = np.vstack([q_xy, centroid_xy, u_xy[None, :], xt_xy[None, :],
                           xr_xy[None, :], xh_xy[None, :], cap_xy])
        _autoscale(ax, focal, pad_frac=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7, loc="best", frameon=False)

        fig.tight_layout()
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  saved {out_pdf}")
    return core


# ---------------------------------------------------------------------------
# Figure B — Phase 2 convergence trajectory (shares CORE fit with A)
# ---------------------------------------------------------------------------

def figure_B(bundle_path: str, out_pdf: str, core: CORE) -> None:
    """Figure B as a two-panel figure.

    Left panel: CORE retrieval landscape with the trajectory coloured by
    refinement stage.  Because CORE preserves query-document distances
    (not direct doc-doc distances), the trajectory's final position in
    this projection may appear offset from $\\hat{x}$ even when the
    cosine similarity between the two is high.  Right panel: the direct
    cosine similarity between the trajectory and the Phase 1 target as
    the optimisation proceeds, which is the measurement that actually
    supports the "trajectory converges toward the target" claim.
    """
    data = np.load(bundle_path, allow_pickle=True)
    T = data["trajectory_embeddings"].astype(np.float32)
    stages = data["stages"]
    iters = np.asarray(data["iterations"]).astype(int)
    cos_to_target_stored = np.asarray(data["cosine_similarities"]).astype(float)
    x_hat = data["x_hat"].astype(np.float32)
    print(f"  figB: trajectory {T.shape}  "
          f"cos(traj[0],x_hat)={cos_to_target_stored[0]:.3f}  "
          f"cos(traj[-1],x_hat)={cos_to_target_stored[-1]:.3f}")

    # Project the trajectory and the target on the shared CORE fit.
    T_xy = core.transform(T)
    xh_xy = core.transform(x_hat)
    doc_idx = _subsample(core.corpus_embedding_, n=2000)

    with _style_ctx():
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(13, 6),
            gridspec_kw={"width_ratios": [1.6, 1.0]},
        )

        # ------- Left: CORE landscape -------
        ax_left.scatter(core.corpus_embedding_[doc_idx, 0],
                        core.corpus_embedding_[doc_idx, 1],
                        s=2.5, c=COLOUR_CORPUS, alpha=0.25, rasterized=True, zorder=1)
        ax_left.scatter(core.query_embedding_[:, 0],
                        core.query_embedding_[:, 1],
                        s=12, c=COLOUR_QUERY, alpha=0.6, edgecolors="none",
                        zorder=2, label=f"Queries (n={core.query_embedding_.shape[0]})")
        ax_left.plot(T_xy[:, 0], T_xy[:, 1], color="#777777",
                     linewidth=0.7, alpha=0.5, zorder=5)

        for stage, colour in STAGE_PALETTE.items():
            mask = np.array([str(s) == stage for s in stages])
            if mask.sum() == 0:
                continue
            ax_left.scatter(T_xy[mask, 0], T_xy[mask, 1], s=26, c=colour,
                            edgecolors="white", linewidths=0.3, zorder=6,
                            label=f"{stage}  ({int(mask.sum())})")

        # Start and end markers.
        ax_left.scatter([T_xy[0, 0]], [T_xy[0, 1]], s=160, marker="o",
                        facecolors="white",
                        edgecolors=STAGE_PALETTE.get(str(stages[0]), "#222222"),
                        linewidths=1.8, zorder=8)
        ax_left.annotate("iter 0", (T_xy[0, 0], T_xy[0, 1]), fontsize=8,
                         fontweight="bold",
                         xytext=(8, 8), textcoords="offset points", zorder=9)
        ax_left.scatter([T_xy[-1, 0]], [T_xy[-1, 1]], s=160, marker="o",
                        c=STAGE_PALETTE.get(str(stages[-1]), "#222222"),
                        edgecolors="white", linewidths=1.2, zorder=8)
        ax_left.annotate(f"iter {int(iters[-1])}",
                         (T_xy[-1, 0], T_xy[-1, 1]), fontsize=8,
                         fontweight="bold",
                         xytext=(8, -14), textcoords="offset points", zorder=9)

        # Target x_hat.
        ax_left.scatter([xh_xy[0]], [xh_xy[1]], s=240, marker="*",
                        c=COLOUR_X_HAT, edgecolors="white", linewidths=1.0,
                        zorder=10, label=r"Target $\hat{x}$")

        focal = np.vstack([T_xy, xh_xy[None, :], core.query_embedding_])
        _autoscale(ax_left, focal, pad_frac=0.2)
        ax_left.set_aspect("equal", adjustable="box")
        _style_axis(ax_left)
        ax_left.legend(fontsize=7, loc="best", frameon=False)
        ax_left.set_title("CORE retrieval landscape", fontsize=10, pad=6)

        # ------- Right: cosine-to-target convergence -------
        ax_right.axhline(1.0, color="#CCCCCC", linestyle=":", linewidth=0.6)
        ax_right.axhline(cos_to_target_stored[0], color="#BBBBBB",
                         linestyle="--", linewidth=0.6)

        # Colour each scatter point by stage, line as a continuous curve.
        ax_right.plot(iters, cos_to_target_stored, color="#777777",
                      linewidth=0.8, alpha=0.6, zorder=3)
        for stage, colour in STAGE_PALETTE.items():
            mask = np.array([str(s) == stage for s in stages])
            if mask.sum() == 0:
                continue
            ax_right.scatter(iters[mask], cos_to_target_stored[mask],
                             s=24, c=colour, edgecolors="white",
                             linewidths=0.3, zorder=4,
                             label=f"{stage}")

        # Annotate the final value.
        ax_right.annotate(f"cos = {cos_to_target_stored[-1]:.3f}",
                          (iters[-1], cos_to_target_stored[-1]),
                          fontsize=8, fontweight="bold",
                          xytext=(-65, -16), textcoords="offset points",
                          zorder=5)

        ax_right.set_xlabel("Optimisation iteration", fontsize=9)
        ax_right.set_ylabel(r"$\cos(\mathrm{trajectory}[t],\ \hat{x})$", fontsize=9)
        ax_right.tick_params(labelsize=7)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.spines["left"].set_linewidth(0.5)
        ax_right.spines["bottom"].set_linewidth(0.5)
        # Ensure a tight y-range that still visualises the ~0.86-0.91 band.
        ymin = min(cos_to_target_stored.min(), cos_to_target_stored[0]) - 0.01
        ymax = 1.0
        ax_right.set_ylim(ymin, ymax)
        ax_right.set_title("Direct cosine convergence", fontsize=10, pad=6)
        ax_right.legend(fontsize=7, loc="lower right", frameon=False)

        fig.tight_layout()
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  saved {out_pdf}")


# ---------------------------------------------------------------------------
# Figure C — Cross-model retrieval landscape
# ---------------------------------------------------------------------------

def figure_C(npz_path: str, json_path: str, out_pdf: str,
             preferred_query: str | None = None) -> None:
    z = np.load(npz_path, allow_pickle=True)
    with open(json_path) as f:
        responses = json.load(f)

    Q = z["query_embeddings"].astype(np.float32)
    D = z["doc_embeddings"].astype(np.float32)
    poison_emb = z["poison_embedding"].astype(np.float32)
    poison_idx = int(z["poison_index"])
    llms = list(responses.keys())
    n_llms = len(llms)
    print(f"  figC: LLMs in bundle: {llms}")

    shared_q = set(entry["query"] for entry in responses[llms[0]])
    for llm in llms[1:]:
        shared_q &= {e["query"] for e in responses[llm]}

    chosen = None

    # 1) Honour a user-supplied preferred query when every LLM has it and
    #    the poison is retrieved at rank 1 across all of them.
    if preferred_query is not None and preferred_query in shared_q:
        entries = {
            llm: next(e for e in responses[llm] if e["query"] == preferred_query)
            for llm in llms
        }
        if all(entries[llm]["poison_rank"] == 1 for llm in llms):
            chosen = (preferred_query, entries)

    # 2) Otherwise pick the first shared query where every LLM retrieved
    #    the poison at rank 1 with a meaningful-length response.
    if chosen is None:
        for q in shared_q:
            entries = {llm: next(e for e in responses[llm] if e["query"] == q) for llm in llms}
            if all(entries[llm]["poison_rank"] == 1 for llm in llms) and \
               all(len(entries[llm]["response_with_poison"]) > 400 for llm in llms):
                chosen = (q, entries)
                break

    # 3) Fall back to any shared query.
    if chosen is None:
        q0 = next(iter(shared_q))
        chosen = (q0, {llm: next(e for e in responses[llm] if e["query"] == q0)
                       for llm in llms})

    query, entries = chosen
    print(f"  figC: representative query = \"{query}\"")

    # Single CORE fit shared across all panels.
    print("  fitting CORE 2-D on biopsy eval corpus (shared across panels)...")
    # Include the poison in the corpus so the landscape treats it consistently.
    corpus_with_poison = np.concatenate([D, poison_emb[None, :]], axis=0)
    core = _core_fit(Q, corpus_with_poison, n_components=2)

    doc_xy = core.corpus_embedding_        # includes the poison at index poison_idx
    q_xy = core.query_embedding_
    poison_xy = doc_xy[poison_idx]

    with _style_ctx():
        fig, axes = plt.subplots(1, n_llms, figsize=(6 * n_llms, 6), sharex=True, sharey=True)
        if n_llms == 1:
            axes = [axes]

        bg_idx = _subsample(doc_xy, n=2000)
        excerpt_wrap = 90

        for ax, llm in zip(axes, llms):
            e = entries[llm]
            top5 = list(e["top_k_indices"])
            ax.scatter(doc_xy[bg_idx, 0], doc_xy[bg_idx, 1], s=2,
                       c=COLOUR_CORPUS, alpha=0.35, rasterized=True, zorder=1)
            ax.scatter(q_xy[:, 0], q_xy[:, 1], s=14, c=COLOUR_QUERY,
                       alpha=0.55, edgecolors="none", zorder=2)

            # Top-5 retrieved.
            for rank, idx in enumerate(top5, start=1):
                if idx >= doc_xy.shape[0]:
                    continue
                is_poison = (idx == poison_idx)
                ax.scatter([doc_xy[idx, 0]], [doc_xy[idx, 1]],
                           s=200 if is_poison else 70,
                           marker="*" if is_poison else "s",
                           c=COLOUR_POISON if is_poison else COLOUR_TOPK,
                           edgecolors="white",
                           linewidths=1.0 if is_poison else 0.4,
                           zorder=8 if is_poison else 6)
                if is_poison:
                    # Offset the "1" label so it's legible next to the red star.
                    ax.annotate(f"{rank}",
                                (doc_xy[idx, 0], doc_xy[idx, 1]),
                                fontsize=8, fontweight="bold",
                                color=COLOUR_POISON,
                                xytext=(8, 8), textcoords="offset points",
                                zorder=10)
                else:
                    ax.annotate(f"{rank}",
                                (doc_xy[idx, 0], doc_xy[idx, 1]),
                                fontsize=7, color="white",
                                ha="center", va="center", zorder=7)

            # Pretty-print model names: preserve initialisms.
            pretty = {"gpt": "GPT", "claude": "Claude",
                      "grok": "Grok", "gemini": "Gemini"}.get(llm, llm.capitalize())
            ax.set_title(pretty, fontsize=11, pad=8)
            ax.set_aspect("equal", adjustable="box")
            _style_axis(ax)

            # Response excerpt below the panel.
            resp = e["response_with_poison"].replace("\n", " ")
            # Cut at first "breath" punctuation after ~200 chars for readability.
            head = resp[:220].rsplit(". ", 1)[0] + "."
            ax.text(0.5, -0.16, textwrap.fill(head, width=excerpt_wrap),
                    transform=ax.transAxes, ha="center", va="top", fontsize=7,
                    color="#333333", style="italic")

        # Global legend.
        handles = [
            mpatches.Patch(facecolor=COLOUR_CORPUS, edgecolor="none",
                           label="Shadow corpus"),
            mpatches.Patch(facecolor=COLOUR_QUERY, edgecolor="none",
                           label="Queries"),
            plt.Line2D([0], [0], marker="s", linestyle="none",
                       color=COLOUR_TOPK, markersize=6, label="Retrieved top-5"),
            plt.Line2D([0], [0], marker="*", linestyle="none",
                       color=COLOUR_POISON, markersize=11, label="Poison"),
        ]
        fig.legend(handles=handles, loc="upper center", fontsize=8,
                   frameon=False, ncol=len(handles),
                   bbox_to_anchor=(0.5, 1.03))

        fig.suptitle(f'Query: "{query}"', fontsize=10, y=1.05)
        fig.subplots_adjust(top=0.85, bottom=0.22)
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  saved {out_pdf}")


# ---------------------------------------------------------------------------
# Figure D — GT stress test (4-panel grid, re-fit per corpus condition)
# ---------------------------------------------------------------------------

def figure_D(biopsy_path: str, flu_path: str, out_pdf: str) -> None:
    # Load both domains.
    panels = [
        ("biopsy", biopsy_path, "Claude · ACS ground truth"),
        ("flu_vaccine", flu_path, "Grok · FHI.no ground truth"),
    ]

    fits = {}   # domain -> (core_baseline, core_with_gt, data, query_idx)
    for name, path, _ in panels:
        print(f"  figD: loading {name}...")
        data = np.load(path, allow_pickle=True)
        Q = data["query_embeddings"].astype(np.float32)
        D = data["doc_embeddings"].astype(np.float32)
        poison = data["poison_embedding"].astype(np.float32)
        gt = data["gt_embedding"].astype(np.float32)

        corpus_base = np.concatenate([D, poison[None, :]], axis=0)
        corpus_gt   = np.concatenate([D, poison[None, :], gt[None, :]], axis=0)

        print(f"    fitting CORE on baseline corpus ({corpus_base.shape[0]} docs)...")
        core_b = _core_fit(Q, corpus_base, n_components=2)
        print(f"    fitting CORE on with-GT corpus ({corpus_gt.shape[0]} docs)...")
        core_g = _core_fit(Q, corpus_gt, n_components=2)

        # Pick a representative query that demonstrates the displacement:
        # poison at rank 1 in baseline AND GT at rank 1 (poison displaced
        # to rank 2+) in the with-GT condition.  Fallback to the first
        # query where the poison is in baseline top-3 at all.
        tb = data["topk_baseline"].astype(int)
        tg = data["topk_with_gt"].astype(int)
        pix = int(data["poison_index_baseline"])
        gix = int(data["gt_index"])
        q_idx = None
        # First pass: strict criterion (most illustrative).
        for i in range(Q.shape[0]):
            if tb[i, 0] == pix and gix in tg[i, :3].tolist() and pix in tg[i, :3].tolist():
                q_idx = i
                break
        # Second pass: poison rank 1 baseline, GT anywhere in top-3 with-GT.
        if q_idx is None:
            for i in range(Q.shape[0]):
                if tb[i, 0] == pix and gix in tg[i, :3].tolist():
                    q_idx = i
                    break
        # Third pass: any query where poison is in baseline top-3.
        if q_idx is None:
            for i in range(Q.shape[0]):
                if pix in tb[i, :3].tolist():
                    q_idx = i
                    break
        if q_idx is None:
            q_idx = 0
        print(f"    representative query index: {q_idx} "
              f"(baseline top-3 = {tb[q_idx, :3].tolist()}, "
              f"with-GT top-3 = {tg[q_idx, :3].tolist()})")
        fits[name] = (core_b, core_g, data, q_idx)

    with _style_ctx():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)

        for col, (name, _, title) in enumerate(panels):
            core_b, core_g, data, q_idx = fits[name]
            poison_idx = int(data["poison_index_baseline"])
            gt_idx     = int(data["gt_index"])
            tb = data["topk_baseline"].astype(int)
            tg = data["topk_with_gt"].astype(int)

            # Top row: baseline.
            ax = axes[0, col]
            _draw_gt_panel(ax, core_b, top_k=tb, poison_local=poison_idx,
                           gt_local=None, show_gt=False,
                           title=f"{title}\nBaseline (no GT)",
                           query_idx=q_idx)

            # Bottom row: with GT.
            ax = axes[1, col]
            _draw_gt_panel(ax, core_g, top_k=tg, poison_local=poison_idx,
                           gt_local=gt_idx, show_gt=True,
                           title=f"{title}\nGT document inserted",
                           query_idx=q_idx)

        handles = [
            mpatches.Patch(facecolor=COLOUR_CORPUS, edgecolor="none", label="Shadow corpus"),
            mpatches.Patch(facecolor=COLOUR_QUERY,  edgecolor="none", label="Queries"),
            plt.Line2D([0], [0], marker="s", linestyle="none",
                       color=COLOUR_TOPK, markersize=6, label="Top-3 retrieved"),
            plt.Line2D([0], [0], marker="*", linestyle="none",
                       color=COLOUR_POISON, markersize=11, label="Poison"),
            plt.Line2D([0], [0], marker="s", linestyle="none",
                       color=COLOUR_GT, markersize=8, label="Ground-truth document"),
        ]
        fig.legend(handles=handles, loc="upper center", fontsize=9,
                   frameon=False, ncol=len(handles),
                   bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  saved {out_pdf}")


def _draw_gt_panel(ax, core: CORE, *, top_k: np.ndarray, poison_local: int,
                   gt_local: int | None, show_gt: bool, title: str,
                   query_idx: int = 0) -> None:
    """Render one panel: a single representative query's top-3, labelled,
    with the triangular convex hull of those three positions shaded."""
    d_xy = core.corpus_embedding_
    q_xy = core.query_embedding_

    bg_idx = _subsample(d_xy, n=2000)
    ax.scatter(d_xy[bg_idx, 0], d_xy[bg_idx, 1], s=2.2,
               c=COLOUR_CORPUS, alpha=0.4, rasterized=True, zorder=1)
    ax.scatter(q_xy[:, 0], q_xy[:, 1], s=14, c=COLOUR_QUERY,
               alpha=0.55, edgecolors="none", zorder=2)

    # The representative query's top-3.
    top3 = [int(x) for x in top_k[query_idx, :3]]
    top3_xy = np.array([d_xy[i] if i < d_xy.shape[0] else d_xy[0] for i in top3])

    # Shaded triangular region — the "top-3 retrieval window".
    if len(top3_xy) >= 3:
        tri = np.vstack([top3_xy, top3_xy[:1]])
        ax.fill(tri[:, 0], tri[:, 1], facecolor=COLOUR_TOPK,
                alpha=0.07, edgecolor=COLOUR_TOPK, linewidth=0.4,
                linestyle=":", zorder=3)

    # Highlight the query the figure is anchored to.
    q_hi = q_xy[query_idx]
    ax.scatter([q_hi[0]], [q_hi[1]], s=50, facecolors="none",
               edgecolors=COLOUR_QUERY, linewidths=1.2, zorder=5)

    # Top-3 markers with rank labels.
    for rank, idx in enumerate(top3, start=1):
        if idx >= d_xy.shape[0]:
            continue
        is_poison = (idx == poison_local)
        is_gt     = (gt_local is not None and idx == gt_local)
        if is_poison:
            ax.scatter([d_xy[idx, 0]], [d_xy[idx, 1]], s=260, marker="*",
                       c=COLOUR_POISON, edgecolors="white",
                       linewidths=1.0, zorder=10)
            ax.annotate(f"{rank}",
                        (d_xy[idx, 0], d_xy[idx, 1]),
                        fontsize=9, fontweight="bold", color=COLOUR_POISON,
                        xytext=(8, 8), textcoords="offset points", zorder=11)
        elif is_gt:
            ax.scatter([d_xy[idx, 0]], [d_xy[idx, 1]], s=150, marker="s",
                       c=COLOUR_GT, edgecolors="white",
                       linewidths=1.0, zorder=10)
            ax.annotate(f"{rank}",
                        (d_xy[idx, 0], d_xy[idx, 1]),
                        fontsize=9, fontweight="bold", color=COLOUR_GT,
                        xytext=(8, 8), textcoords="offset points", zorder=11)
        else:
            ax.scatter([d_xy[idx, 0]], [d_xy[idx, 1]], s=100, marker="s",
                       c=COLOUR_TOPK, edgecolors="white",
                       linewidths=0.6, zorder=6)
            ax.annotate(f"{rank}",
                        (d_xy[idx, 0], d_xy[idx, 1]),
                        fontsize=8, color="white",
                        ha="center", va="center", zorder=7)

    # Always show the poison, even if it has been displaced beyond top-3.
    if poison_local not in top3 and poison_local < d_xy.shape[0]:
        ax.scatter([d_xy[poison_local, 0]], [d_xy[poison_local, 1]], s=220,
                   marker="*", c=COLOUR_POISON, edgecolors="white",
                   linewidths=1.0, zorder=9, alpha=0.6)
    # Always show the GT marker in the with-GT row, even if out of top-3.
    if show_gt and gt_local is not None and gt_local not in top3 and gt_local < d_xy.shape[0]:
        ax.scatter([d_xy[gt_local, 0]], [d_xy[gt_local, 1]], s=120,
                   marker="s", c=COLOUR_GT, edgecolors="white",
                   linewidths=0.8, zorder=9, alpha=0.6)

    # Zoom to the interesting region.
    focal_pts = np.vstack([top3_xy, q_hi[None, :]])
    if poison_local < d_xy.shape[0]:
        focal_pts = np.vstack([focal_pts, d_xy[poison_local:poison_local+1]])
    if show_gt and gt_local is not None and gt_local < d_xy.shape[0]:
        focal_pts = np.vstack([focal_pts, d_xy[gt_local:gt_local+1]])
    _autoscale(ax, focal_pts, pad_frac=0.5)

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_aspect("equal", adjustable="box")
    _style_axis(ax)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(bundles_dir: str, out_dir: str, skip: tuple[str, ...] = (),
         figC_query: str | None = None) -> None:
    bd = Path(bundles_dir)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    if "A" not in skip:
        print("\n[A] Phase 1 target construction")
        core_ab = figure_A(
            str(bd / "biopsy" / "figA_phase1_target.npz"),
            str(od / "core_phase1_target.pdf"),
        )
    else:
        core_ab = None

    if "B" not in skip:
        if core_ab is None:
            # If A was skipped, re-fit from figB's bundle (it has its own queries+docs).
            bB = np.load(str(bd / "biopsy" / "figB_phase2_trajectory.npz"), allow_pickle=True)
            core_ab = _core_fit(bB["query_embeddings"].astype(np.float32),
                                bB["doc_embeddings"].astype(np.float32), n_components=2)
        print("\n[B] Phase 2 trajectory")
        figure_B(
            str(bd / "biopsy" / "figB_phase2_trajectory.npz"),
            str(od / "core_phase2_trajectory.pdf"),
            core=core_ab,
        )

    if "C" not in skip:
        print("\n[C] Cross-model landscape")
        figure_C(
            str(bd / "biopsy" / "figC_cross_model.npz"),
            str(bd / "biopsy" / "figC_cross_model.json"),
            str(od / "core_cross_model_citations.pdf"),
            preferred_query=figC_query,
        )

    if "D" not in skip:
        print("\n[D] GT stress test")
        figure_D(
            str(bd / "biopsy" / "figD_gt_stress.npz"),
            str(bd / "flu_vaccine" / "figD_gt_stress.npz"),
            str(od / "core_gt_stress_test.pdf"),
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundles-dir", required=True,
                    help="Directory containing biopsy/ and flu_vaccine/ bundle folders.")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write the four PDFs.")
    ap.add_argument("--skip", default="",
                    help="Comma-separated list of figures to skip (A, B, C, D).")
    ap.add_argument("--figC-query", default="Can biopsy make cancer metastasize?",
                    help="Preferred query for Figure C's representative panel. "
                         "Falls back to the first viable shared query if the "
                         "preferred one isn't in the bundle.")
    args = ap.parse_args()
    main(
        args.bundles_dir, args.out_dir,
        tuple(s.strip().upper() for s in args.skip.split(",") if s.strip()),
        figC_query=args.figC_query,
    )
