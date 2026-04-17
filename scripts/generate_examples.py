"""Generate the canonical 2-D and 3-D example figures shown in the README.

Two modes:

1. **Planning corpus (default)** — fit CORE on ``shadow_doc_embeddings`` from
   ``trajectory.npz`` (the planning-phase shadow corpus).  Fast but does
   *not* reflect the corpus the production eval runs against::

       python scripts/generate_examples.py \\
           --trajectory /path/to/trajectory.npz

2. **Eval corpus (recommended for thesis figures)** — fit CORE on the
   evaluation retrieval corpus reconstructed by
   ``scripts/build_eval_corpus.py``.  These plots reflect what the attack
   actually sees in production::

       python scripts/generate_examples.py \\
           --trajectory /path/to/trajectory.npz \\
           --corpus     /path/to/eval_corpus.npz

Either input must contain:

- ``query_embeddings`` ``(n_q, 768)``
- ``corpus_embeddings`` **or** ``shadow_doc_embeddings`` ``(n_d, 768)``
- ``trajectory`` ``(n_steps, 768)`` (for the optimisation path; taken from
  the ``--trajectory`` file even when ``--corpus`` is supplied)
- ``target`` ``(768,)``
- Optionally ``poison_embedding`` in ``--corpus``, otherwise the poison is
  taken from ``trajectory[-1]``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from rankviz import CORE, plot_landscape

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "examples" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_inputs(trajectory_npz: str, corpus_npz: str | None):
    """Resolve the five arrays we need regardless of which corpus is used."""
    tdata = np.load(trajectory_npz, allow_pickle=True)

    if corpus_npz is not None:
        cdata = np.load(corpus_npz, allow_pickle=True)
        # Accept either `corpus_embeddings` (build_eval_corpus.py) or the
        # Master-repo naming `doc_embeddings`.
        corpus_key = next(
            (k for k in ("corpus_embeddings", "doc_embeddings") if k in cdata.files),
            None,
        )
        if corpus_key is None:
            raise ValueError(
                f"{corpus_npz} must contain `corpus_embeddings` or `doc_embeddings`."
            )
        if "query_embeddings" not in cdata.files:
            raise ValueError(
                f"{corpus_npz} must contain a `query_embeddings` array. "
                "For eval-style corpora these should be encoded with the "
                "'query: ' prefix."
            )
        Q = cdata["query_embeddings"].astype(np.float32)
        D = cdata[corpus_key].astype(np.float32)
        if "poison_embedding" in cdata.files:
            poison = cdata["poison_embedding"].astype(np.float32)
            if poison.ndim == 2:
                poison = poison.squeeze(0)
        else:
            poison = tdata["trajectory"][-1].astype(np.float32)
    else:
        Q = tdata["query_embeddings"].astype(np.float32)
        D = tdata["shadow_doc_embeddings"].astype(np.float32)
        poison = tdata["trajectory"][-1].astype(np.float32)

    traj = tdata["trajectory"].astype(np.float32)
    target = tdata["target"].astype(np.float32)
    return Q, D, traj, target, poison


def main(trajectory_npz: str, corpus_npz: str | None, out_dir: Path) -> None:
    Q, D, traj, target, poison = _load_inputs(trajectory_npz, corpus_npz)
    print(f"queries={Q.shape}  corpus={D.shape}  trajectory={traj.shape}")

    # Sanity-check the poison's true rank in the chosen corpus, so the plot
    # is interpreted correctly.
    sims_poison = Q @ poison
    sims_corpus = Q @ D.T
    ranks = (sims_corpus > sims_poison[:, None]).sum(axis=1) + 1
    print(f"Poison rank: median={int(np.median(ranks))}  "
          f"min={int(ranks.min())}  max={int(ranks.max())}  "
          f"ASR@1={((ranks<=1).mean()*100):.1f}%  "
          f"ASR@10={((ranks<=10).mean()*100):.1f}%")

    print("Fitting CORE 2D...")
    core2 = CORE(n_components=2, weight="retrieval", n_iter=400).fit(
        queries=Q, corpus=D,
    )

    print("  rendering matplotlib 2-D ...")
    fig = plot_landscape(
        core2, highlight=poison, trajectory=traj, target=target,
        highlight_labels=["Poison"], backend="matplotlib",
        figsize=(7, 6), show_retrieved_top_k=10,
    )
    fig.savefig(out_dir / "core_2d.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "core_2d.png", dpi=300, bbox_inches="tight")

    print("  rendering plotly 2-D ...")
    fig = plot_landscape(
        core2, highlight=poison, trajectory=traj, target=target,
        highlight_labels=["Poison"], backend="plotly",
        show_retrieved_top_k=10,
    )
    fig.write_html(out_dir / "core_2d.html")

    print("Fitting CORE 3D...")
    core3 = CORE(n_components=3, weight="retrieval", n_iter=400).fit(
        queries=Q, corpus=D,
    )

    print("  rendering plotly 3-D ...")
    fig = plot_landscape(
        core3, highlight=poison, trajectory=traj, target=target,
        highlight_labels=["Poison"], backend="plotly",
        show_retrieved_top_k=10,
    )
    fig.write_html(out_dir / "core_3d.html")

    print("  rendering matplotlib 3-D ...")
    fig = plot_landscape(
        core3, highlight=poison, trajectory=traj, target=target,
        highlight_labels=["Poison"], backend="matplotlib",
        figsize=(7, 6), show_retrieved_top_k=10,
    )
    fig.savefig(out_dir / "core_3d.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "core_3d.png", dpi=300, bbox_inches="tight")

    print(f"\nWrote example figures to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate CORE 2-D and 3-D example figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--trajectory",
        required=True,
        help="Path to trajectory.npz from the attack run.",
    )
    ap.add_argument(
        "--corpus",
        default=None,
        help="Optional path to eval_corpus.npz (from build_eval_corpus.py). "
             "If omitted, falls back to shadow_doc_embeddings inside "
             "--trajectory, which is the *planning* corpus.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help=f"Directory to write the figures into. Default: {OUT_DIR}",
    )

    # Back-compat: first positional argument was the trajectory path.
    import sys
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        # Legacy invocation:  generate_examples.py /path/to/trajectory.npz
        args = ap.parse_args(["--trajectory", sys.argv[1], *sys.argv[2:]])
    else:
        args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main(args.trajectory, args.corpus, out)
