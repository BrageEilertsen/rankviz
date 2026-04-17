"""Plotting utilities for CORE projections.

Supports two backends:

- ``"matplotlib"`` — 2-D or 3-D static figure (for papers, PDFs).
- ``"plotly"`` — 2-D or 3-D interactive figure (requires ``plotly``).

The function is a thin visualiser on top of fitted :class:`CORE`
coordinates; for deeper customisation, access ``core.query_embedding_``
and ``core.corpus_embedding_`` directly.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from rankviz.core import CORE
from rankviz._style import (
    CORPUS_COLOUR,
    CORPUS_COLOUR_DARK,
    get_highlight_colours,
    style as _style_ctx,
)


def plot_landscape(
    core: CORE,
    *,
    highlight: np.ndarray | None = None,
    trajectory: np.ndarray | None = None,
    target: np.ndarray | None = None,
    highlight_labels: Sequence[str] | None = None,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    figsize: tuple[float, float] = (8, 6),
    corpus_sample: int | None = 2000,
    show_centroid: bool = True,
    show_connections: bool = True,
    random_state: int | None = 42,
) -> Any:
    """Render a CORE projection.

    Parameters
    ----------
    core : CORE
        A fitted :class:`CORE` instance.
    highlight : ndarray, shape ``(d,)`` or ``(m, d)``, optional
        Documents to highlight in the plot (e.g. a poisoned document).
    trajectory : ndarray, shape ``(n_steps, d)``, optional
        A sequence of embeddings to render as a connected path (e.g.
        an optimisation trajectory).
    target : ndarray, shape ``(d,)``, optional
        A "target" embedding — rendered as a gold diamond, with a
        dashed line from each highlight.
    highlight_labels : list of str, optional
        Display labels for highlight documents.
    backend : ``"matplotlib"`` | ``"plotly"``
        Rendering backend.  ``matplotlib`` is static; ``plotly`` is
        interactive and requires the optional ``plotly`` dependency.
    figsize : tuple of float
        Matplotlib figure size in inches (ignored for plotly).
    corpus_sample : int, optional
        Randomly subsample the corpus to at most this many points.
        Pass ``None`` to plot every point.
    show_centroid : bool
        Render a hollow marker at the mean query position.
    show_connections : bool
        Draw dashed lines from each highlight to the query centroid
        and to the target (if given).
    random_state : int, optional
        Seed for corpus subsampling.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if core.query_embedding_ is None:
        raise RuntimeError("CORE instance must be fitted before plotting.")

    ctx = _build_context(
        core=core,
        highlight=highlight,
        trajectory=trajectory,
        target=target,
        highlight_labels=highlight_labels,
        corpus_sample=corpus_sample,
        random_state=random_state,
    )

    if backend == "matplotlib":
        return _plot_matplotlib(
            ctx,
            figsize=figsize,
            show_centroid=show_centroid,
            show_connections=show_connections,
        )
    if backend == "plotly":
        return _plot_plotly(
            ctx,
            show_centroid=show_centroid,
            show_connections=show_connections,
        )
    raise ValueError(f"Unknown backend {backend!r}.")


# =====================================================================
# Context assembly
# =====================================================================

def _build_context(
    *,
    core: CORE,
    highlight,
    trajectory,
    target,
    highlight_labels,
    corpus_sample: int | None,
    random_state: int | None,
) -> dict:
    """Project auxiliary points and assemble everything the backends need."""
    Q_low = core.query_embedding_
    D_low = core.corpus_embedding_
    n_components = core.n_components

    # Subsample corpus for rendering.
    if corpus_sample is not None and D_low.shape[0] > corpus_sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(D_low.shape[0], size=corpus_sample, replace=False)
        D_vis = D_low[idx]
    else:
        D_vis = D_low

    # Highlights.
    if highlight is not None:
        hl_low = core.transform(highlight)
        if hl_low.ndim == 1:
            hl_low = hl_low[None, :]
    else:
        hl_low = None

    if highlight_labels is None and hl_low is not None:
        if hl_low.shape[0] == 1:
            highlight_labels = ["highlight"]
        else:
            highlight_labels = [f"highlight {i}" for i in range(hl_low.shape[0])]

    # Trajectory.
    if trajectory is not None:
        traj_arr = np.asarray(trajectory, dtype=np.float32)
        if traj_arr.ndim == 1:
            traj_arr = traj_arr[None, :]
        traj_low = core.transform(traj_arr)
    else:
        traj_low = None

    # Target.
    tgt_low = core.transform(target) if target is not None else None

    return dict(
        n_components=n_components,
        Q_low=Q_low,
        D_low=D_vis,
        hl_low=hl_low,
        highlight_labels=highlight_labels,
        traj_low=traj_low,
        tgt_low=tgt_low,
    )


# =====================================================================
# matplotlib backend
# =====================================================================

def _plot_matplotlib(ctx, *, figsize, show_centroid, show_connections):
    import matplotlib.pyplot as plt

    with _style_ctx():
        fig = plt.figure(figsize=figsize)
        if ctx["n_components"] == 3:
            ax = fig.add_subplot(111, projection="3d")
            _draw_mpl_3d(ax, ctx, show_centroid, show_connections)
        else:
            ax = fig.add_subplot(111)
            _draw_mpl_2d(ax, ctx, show_centroid, show_connections)
        fig.tight_layout()
    return fig


def _draw_mpl_2d(ax, ctx, show_centroid, show_connections):
    Q = ctx["Q_low"]; D = ctx["D_low"]
    ax.scatter(D[:, 0], D[:, 1], s=2, c=CORPUS_COLOUR, alpha=0.35, rasterized=True, zorder=1)
    ax.scatter(Q[:, 0], Q[:, 1], s=12, c="#4477AA", alpha=0.7, edgecolors="none", zorder=3,
               label=f"Queries (n={Q.shape[0]})")

    if show_centroid:
        qc = Q.mean(axis=0)
        ax.scatter([qc[0]], [qc[1]], s=80, facecolor="none", edgecolor="#4477AA",
                   linewidths=1.5, zorder=4, label="Query centroid")

    if ctx["traj_low"] is not None:
        T = ctx["traj_low"]
        ax.plot(T[:, 0], T[:, 1], color="#AA3377", linewidth=1.2, alpha=0.7, zorder=5)
        ax.scatter(T[:, 0], T[:, 1], s=14, c=np.arange(len(T)), cmap="RdYlGn_r",
                   edgecolors="black", linewidths=0.3, zorder=6)

    if ctx["hl_low"] is not None:
        H = ctx["hl_low"]
        colours = get_highlight_colours(H.shape[0])
        for i, (pt, lbl, c) in enumerate(zip(H, ctx["highlight_labels"], colours)):
            ax.scatter([pt[0]], [pt[1]], s=70, c=c, edgecolors="black", linewidths=0.8,
                       zorder=10, label=lbl)
            if show_connections:
                qc = Q.mean(axis=0)
                ax.plot([pt[0], qc[0]], [pt[1], qc[1]], color="#4477AA",
                        linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
                if ctx["tgt_low"] is not None:
                    tg = ctx["tgt_low"]
                    ax.plot([pt[0], tg[0]], [pt[1], tg[1]], color="#CCBB44",
                            linewidth=0.8, linestyle="--", alpha=0.8, zorder=2)

    if ctx["tgt_low"] is not None:
        tg = ctx["tgt_low"]
        ax.scatter([tg[0]], [tg[1]], s=100, marker="D", c="#CCBB44",
                   edgecolors="black", linewidths=0.8, zorder=11, label="Target")

    ax.set_xlabel("CORE 1", fontsize=9)
    ax.set_ylabel("CORE 2", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="best", frameon=False)


def _draw_mpl_3d(ax, ctx, show_centroid, show_connections):
    Q = ctx["Q_low"]; D = ctx["D_low"]
    ax.scatter(D[:, 0], D[:, 1], D[:, 2], s=1.5, c=CORPUS_COLOUR, alpha=0.3, rasterized=True)
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=12, c="#4477AA", alpha=0.7,
               edgecolors="none", label=f"Queries (n={Q.shape[0]})")

    if show_centroid:
        qc = Q.mean(axis=0)
        ax.scatter([qc[0]], [qc[1]], [qc[2]], s=80, facecolor="none",
                   edgecolor="#4477AA", linewidths=1.5, label="Query centroid")

    if ctx["traj_low"] is not None:
        T = ctx["traj_low"]
        ax.plot(T[:, 0], T[:, 1], T[:, 2], color="#AA3377", linewidth=1.2, alpha=0.7)
        ax.scatter(T[:, 0], T[:, 1], T[:, 2], s=14, c=np.arange(len(T)),
                   cmap="RdYlGn_r", edgecolors="black", linewidths=0.3)

    if ctx["hl_low"] is not None:
        H = ctx["hl_low"]
        colours = get_highlight_colours(H.shape[0])
        for i, (pt, lbl, c) in enumerate(zip(H, ctx["highlight_labels"], colours)):
            ax.scatter([pt[0]], [pt[1]], [pt[2]], s=70, c=c,
                       edgecolors="black", linewidths=0.8, label=lbl)
            if show_connections:
                qc = Q.mean(axis=0)
                ax.plot([pt[0], qc[0]], [pt[1], qc[1]], [pt[2], qc[2]],
                        color="#4477AA", linewidth=0.8, linestyle="--", alpha=0.6)
                if ctx["tgt_low"] is not None:
                    tg = ctx["tgt_low"]
                    ax.plot([pt[0], tg[0]], [pt[1], tg[1]], [pt[2], tg[2]],
                            color="#CCBB44", linewidth=0.8, linestyle="--", alpha=0.8)

    if ctx["tgt_low"] is not None:
        tg = ctx["tgt_low"]
        ax.scatter([tg[0]], [tg[1]], [tg[2]], s=100, marker="D", c="#CCBB44",
                   edgecolors="black", linewidths=0.8, label="Target")

    ax.set_xlabel("CORE 1", fontsize=9)
    ax.set_ylabel("CORE 2", fontsize=9)
    ax.set_zlabel("CORE 3", fontsize=9)
    ax.legend(fontsize=7, loc="best", frameon=False)


# =====================================================================
# plotly backend
# =====================================================================

def _plot_plotly(ctx, *, show_centroid, show_connections):
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "The plotly backend requires the optional `plotly` package.  "
            "Install with: pip install plotly"
        ) from e

    Q = ctx["Q_low"]; D = ctx["D_low"]
    fig = go.Figure()

    if ctx["n_components"] == 3:
        _add_corpus_3d(fig, D)
        _add_queries_3d(fig, Q)
        if show_centroid:
            _add_centroid_3d(fig, Q.mean(axis=0))
        if ctx["traj_low"] is not None:
            _add_trajectory_3d(fig, ctx["traj_low"])
        if ctx["hl_low"] is not None:
            _add_highlights_3d(fig, ctx["hl_low"], ctx["highlight_labels"],
                               Q.mean(axis=0), ctx["tgt_low"], show_connections)
        if ctx["tgt_low"] is not None:
            _add_target_3d(fig, ctx["tgt_low"])
        fig.update_layout(
            scene=dict(
                xaxis_title="CORE 1", yaxis_title="CORE 2", zaxis_title="CORE 3",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            width=900, height=700,
            legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        )
    else:
        _add_corpus_2d(fig, D)
        _add_queries_2d(fig, Q)
        if show_centroid:
            _add_centroid_2d(fig, Q.mean(axis=0))
        if ctx["traj_low"] is not None:
            _add_trajectory_2d(fig, ctx["traj_low"])
        if ctx["hl_low"] is not None:
            _add_highlights_2d(fig, ctx["hl_low"], ctx["highlight_labels"],
                               Q.mean(axis=0), ctx["tgt_low"], show_connections)
        if ctx["tgt_low"] is not None:
            _add_target_2d(fig, ctx["tgt_low"])
        fig.update_layout(
            xaxis_title="CORE 1", yaxis_title="CORE 2",
            width=800, height=650,
            legend=dict(x=0.01, y=0.99, font=dict(size=10)),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def _add_corpus_3d(fig, D):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter3d(
        x=D[:, 0], y=D[:, 1], z=D[:, 2], mode="markers",
        marker=dict(size=1.8, color=CORPUS_COLOUR, opacity=0.3),
        name=f"Corpus (n={D.shape[0]})", hoverinfo="skip",
    ))


def _add_queries_3d(fig, Q):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter3d(
        x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], mode="markers",
        marker=dict(size=4, color="#4477AA", opacity=0.75),
        name=f"Queries (n={Q.shape[0]})",
        hovertemplate="Query %{pointNumber}<extra></extra>",
    ))


def _add_centroid_3d(fig, qc):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter3d(
        x=[qc[0]], y=[qc[1]], z=[qc[2]], mode="markers",
        marker=dict(size=10, color="rgba(68,119,170,0.15)",
                    line=dict(width=2, color="#4477AA")),
        name="Query centroid", hoverinfo="skip",
    ))


def _add_trajectory_3d(fig, T):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter3d(
        x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="lines+markers",
        line=dict(color="rgba(170,51,119,0.6)", width=4),
        marker=dict(size=4, color=np.arange(len(T)), colorscale="RdYlGn_r",
                    line=dict(width=0.5, color="black")),
        name="Trajectory",
        hovertemplate="step %{pointNumber}<extra></extra>",
    ))


def _add_highlights_3d(fig, H, labels, qc, tgt_low, connect):
    import plotly.graph_objects as go
    colours = get_highlight_colours(H.shape[0])
    for i, (pt, lbl) in enumerate(zip(H, labels)):
        c = colours[i]
        fig.add_trace(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers+text",
            marker=dict(size=11, color=c, line=dict(width=1.5, color="black")),
            text=[lbl], textposition="top center", textfont=dict(size=12, color=c),
            name=lbl,
        ))
        if connect:
            fig.add_trace(go.Scatter3d(
                x=[pt[0], qc[0]], y=[pt[1], qc[1]], z=[pt[2], qc[2]],
                mode="lines",
                line=dict(color="rgba(68,119,170,0.55)", width=3, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            if tgt_low is not None:
                fig.add_trace(go.Scatter3d(
                    x=[pt[0], tgt_low[0]], y=[pt[1], tgt_low[1]], z=[pt[2], tgt_low[2]],
                    mode="lines",
                    line=dict(color="rgba(204,187,68,0.7)", width=3, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ))


def _add_target_3d(fig, tg):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter3d(
        x=[tg[0]], y=[tg[1]], z=[tg[2]], mode="markers+text",
        marker=dict(size=12, color="#CCBB44", symbol="diamond",
                    line=dict(width=1.5, color="black")),
        text=["Target"], textposition="top center", textfont=dict(size=12, color="#998800"),
        name="Target",
    ))


# --- 2-D plotly helpers ----

def _add_corpus_2d(fig, D):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=D[:, 0], y=D[:, 1], mode="markers",
        marker=dict(size=3, color=CORPUS_COLOUR, opacity=0.4),
        name=f"Corpus (n={D.shape[0]})", hoverinfo="skip",
    ))


def _add_queries_2d(fig, Q):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=Q[:, 0], y=Q[:, 1], mode="markers",
        marker=dict(size=6, color="#4477AA", opacity=0.75),
        name=f"Queries (n={Q.shape[0]})",
        hovertemplate="Query %{pointNumber}<extra></extra>",
    ))


def _add_centroid_2d(fig, qc):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=[qc[0]], y=[qc[1]], mode="markers",
        marker=dict(size=14, color="rgba(68,119,170,0.15)",
                    line=dict(width=2, color="#4477AA")),
        name="Query centroid", hoverinfo="skip",
    ))


def _add_trajectory_2d(fig, T):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=T[:, 0], y=T[:, 1], mode="lines+markers",
        line=dict(color="rgba(170,51,119,0.6)", width=2),
        marker=dict(size=6, color=np.arange(len(T)), colorscale="RdYlGn_r",
                    line=dict(width=0.5, color="black")),
        name="Trajectory",
        hovertemplate="step %{pointNumber}<extra></extra>",
    ))


def _add_highlights_2d(fig, H, labels, qc, tgt_low, connect):
    import plotly.graph_objects as go
    colours = get_highlight_colours(H.shape[0])
    for i, (pt, lbl) in enumerate(zip(H, labels)):
        c = colours[i]
        fig.add_trace(go.Scatter(
            x=[pt[0]], y=[pt[1]], mode="markers+text",
            marker=dict(size=14, color=c, line=dict(width=1.5, color="black")),
            text=[lbl], textposition="top center", textfont=dict(size=12, color=c),
            name=lbl,
        ))
        if connect:
            fig.add_trace(go.Scatter(
                x=[pt[0], qc[0]], y=[pt[1], qc[1]], mode="lines",
                line=dict(color="rgba(68,119,170,0.55)", width=2, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            if tgt_low is not None:
                fig.add_trace(go.Scatter(
                    x=[pt[0], tgt_low[0]], y=[pt[1], tgt_low[1]], mode="lines",
                    line=dict(color="rgba(204,187,68,0.7)", width=2, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ))


def _add_target_2d(fig, tg):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=[tg[0]], y=[tg[1]], mode="markers+text",
        marker=dict(size=15, color="#CCBB44", symbol="diamond",
                    line=dict(width=1.5, color="black")),
        text=["Target"], textposition="top center", textfont=dict(size=12, color="#998800"),
        name="Target",
    ))
