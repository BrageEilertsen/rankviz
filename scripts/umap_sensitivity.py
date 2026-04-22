"""UMAP hyperparameter sensitivity sweep.

On two representative domains (C4_chemo, the hardest; C2_flu, among the
easiest), sweep ``n_neighbors × min_dist`` and report top-10 overlap.

The goal is to preempt the reviewer question "did you run UMAP with bad
hyperparameters?".  Either UMAP never recovers (CORE wins robustly) or
we report UMAP's best-case number alongside CORE's.

Supports both corpora:

* Default (planning): reads ``C{N}/trajectory.npz``
  (``query_embeddings`` + ``shadow_doc_embeddings``).
* ``--eval-corpora-dir DIR``: reads ``eval_corpus_C{N}_*.npz`` from DIR
  (``query_embeddings`` + ``doc_embeddings``).
"""
from __future__ import annotations

import argparse, csv, glob, os, re, time, warnings
import numpy as np

warnings.filterwarnings("ignore")


OUT_DEFAULT = os.path.join("examples", "umap_sensitivity.csv")
OUT_EVAL    = os.path.join("examples", "umap_sensitivity_eval.csv")

N_NEIGHBORS = [5, 15, 30, 50]
MIN_DISTS   = [0.0, 0.1, 0.5]
DOMAINS = ["C4_chemo", "C2_flu"]


def top_k_overlap(Q, D, Q_low, D_low, k=10):
    sims = Q @ D.T
    top_hi = np.argsort(-sims, axis=1)[:, :k]
    dists = np.linalg.norm(Q_low[:, None, :] - D_low[None, :, :], axis=-1)
    top_lo = np.argsort(dists, axis=1)[:, :k]
    return float(np.mean([
        len(set(top_hi[i]) & set(top_lo[i])) / k for i in range(Q.shape[0])
    ]))


def _load(dom: str, eval_corpora_dir: str | None, data_root: str | None):
    if eval_corpora_dir is not None:
        # Match "eval_corpus_C4_chemo.npz" or similar.
        matches = sorted(glob.glob(
            os.path.join(eval_corpora_dir, f"eval_corpus_{dom}.npz")
        ))
        if not matches:
            # Also handle stems like C4_chemo matching eval_corpus_C4_*.npz
            matches = sorted(glob.glob(
                os.path.join(eval_corpora_dir, f"eval_corpus_{dom.split('_')[0]}_*.npz")
            ))
        if not matches:
            return None
        data = np.load(matches[0], allow_pickle=True)
        return (
            data["query_embeddings"].astype(np.float32),
            data["doc_embeddings"].astype(np.float32),
        )
    if data_root is None:
        raise SystemExit(
            "Planning mode requires --data-root pointing at a directory of "
            "C{N}_*/trajectory.npz folders. Use --eval-corpora-dir to skip "
            "the planning step."
        )
    npz = os.path.join(data_root, dom, "trajectory.npz")
    if not os.path.exists(npz):
        return None
    data = np.load(npz, allow_pickle=True)
    return (
        data["query_embeddings"].astype(np.float32),
        data["shadow_doc_embeddings"].astype(np.float32),
    )


def main(eval_corpora_dir: str | None, data_root: str | None, out: str):
    from umap import UMAP

    rows = []
    mode = f"eval corpora ({eval_corpora_dir})" if eval_corpora_dir else "planning"
    print(f"[START] UMAP sweep — {mode}", flush=True)

    for dom in DOMAINS:
        pair = _load(dom, eval_corpora_dir, data_root)
        if pair is None:
            print(f"  [SKIP] {dom}", flush=True)
            continue
        Q, D = pair
        stacked = np.concatenate([Q, D], axis=0)
        print(f"\n[{dom}] Q={Q.shape[0]} D={D.shape[0]}", flush=True)

        for nn in N_NEIGHBORS:
            for md in MIN_DISTS:
                t0 = time.time()
                red = UMAP(
                    n_components=3, metric="cosine",
                    n_neighbors=nn, min_dist=md, random_state=42,
                )
                low = red.fit_transform(stacked)
                score = top_k_overlap(Q, D, low[:Q.shape[0]], low[Q.shape[0]:])
                dt = time.time() - t0
                print(f"  n_neighbors={nn:<3d} min_dist={md:.2f}   overlap={score:.3f}   ({dt:.1f}s)", flush=True)
                rows.append({
                    "domain": dom,
                    "n_neighbors": nn,
                    "min_dist": md,
                    "UMAP_3D_overlap": score,
                    "time_s": round(dt, 2),
                })

    keys = ["domain", "n_neighbors", "min_dist", "UMAP_3D_overlap", "time_s"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[SAVED] {out}", flush=True)

    print("\n[SUMMARY — best UMAP config per domain]", flush=True)
    by_dom = {}
    for r in rows:
        by_dom.setdefault(r["domain"], []).append(r)
    for dom, rs in by_dom.items():
        best = max(rs, key=lambda r: r["UMAP_3D_overlap"])
        worst = min(rs, key=lambda r: r["UMAP_3D_overlap"])
        print(f"  {dom}: best={best['UMAP_3D_overlap']:.3f} "
              f"(n={best['n_neighbors']}, md={best['min_dist']})   "
              f"worst={worst['UMAP_3D_overlap']:.3f}", flush=True)
    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--eval-corpora-dir", default=None,
        help="Directory containing eval_corpus_C*.npz files (built by "
             "scripts/build_eval_corpus.py). If unset, use planning "
             "trajectory.npz files (requires --data-root).",
    )
    ap.add_argument(
        "--data-root", default=None,
        help="Root directory of per-domain folders C{N}_*/ each containing "
             "trajectory.npz. Required for planning mode.",
    )
    ap.add_argument(
        "--out", default=None,
        help=f"CSV output path. Defaults (relative to cwd): "
             f"{OUT_DEFAULT} (planning mode) or {OUT_EVAL} (eval-corpora mode).",
    )
    args = ap.parse_args()
    out = args.out or (OUT_EVAL if args.eval_corpora_dir else OUT_DEFAULT)
    main(args.eval_corpora_dir, args.data_root, out)
