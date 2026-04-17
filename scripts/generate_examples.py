"""Generate the canonical 2-D and 3-D example figures shown in the README.

Run with:
    python scripts/generate_examples.py /path/to/trajectory.npz

The input .npz must contain:
    - query_embeddings:     (n_queries, d)
    - shadow_doc_embeddings: (n_corpus, d)
    - trajectory:            (n_steps, d)    — the optimisation path
    - target:                (d,)
    - cosine_similarities:   (n_steps,)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from rankviz import CORE, plot_landscape

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "examples" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(npz_path: str) -> None:
    data = np.load(npz_path, allow_pickle=True)
    Q = data["query_embeddings"].astype(np.float32)
    D = data["shadow_doc_embeddings"].astype(np.float32)
    traj = data["trajectory"].astype(np.float32)
    target = data["target"].astype(np.float32)
    poison = traj[-1]

    # --- 2-D: matplotlib static (PDF + PNG) and plotly interactive (HTML)
    print("Fitting CORE 2D...")
    core2 = CORE(n_components=2, weight="retrieval", n_iter=400).fit(
        queries=Q, corpus=D,
    )

    print("  rendering matplotlib 2-D ...")
    fig = plot_landscape(
        core2,
        highlight=poison,
        trajectory=traj,
        target=target,
        highlight_labels=["Poison"],
        backend="matplotlib",
        figsize=(7, 6),
        show_retrieved_top_k=10,
    )
    fig.savefig(OUT_DIR / "core_2d.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "core_2d.png", dpi=300, bbox_inches="tight")

    print("  rendering plotly 2-D ...")
    fig = plot_landscape(
        core2,
        highlight=poison,
        trajectory=traj,
        target=target,
        highlight_labels=["Poison"],
        backend="plotly",
        show_retrieved_top_k=10,
    )
    fig.write_html(OUT_DIR / "core_2d.html")

    # --- 3-D: plotly interactive (HTML) and matplotlib static (PDF + PNG)
    print("Fitting CORE 3D...")
    core3 = CORE(n_components=3, weight="retrieval", n_iter=400).fit(
        queries=Q, corpus=D,
    )

    print("  rendering plotly 3-D ...")
    fig = plot_landscape(
        core3,
        highlight=poison,
        trajectory=traj,
        target=target,
        highlight_labels=["Poison"],
        backend="plotly",
        show_retrieved_top_k=10,
    )
    fig.write_html(OUT_DIR / "core_3d.html")

    print("  rendering matplotlib 3-D ...")
    fig = plot_landscape(
        core3,
        highlight=poison,
        trajectory=traj,
        target=target,
        highlight_labels=["Poison"],
        backend="matplotlib",
        figsize=(7, 6),
        show_retrieved_top_k=10,
    )
    fig.savefig(OUT_DIR / "core_3d.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "core_3d.png", dpi=300, bbox_inches="tight")

    print(f"\nWrote example figures to {OUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
