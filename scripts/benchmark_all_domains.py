"""Benchmark CORE against PCA / UMAP / t-SNE across all C* domain folders.

Two corpus modes, selected by CLI flag:

* Default (planning corpus): read each domain's ``trajectory.npz`` and use
  ``query_embeddings`` + ``shadow_doc_embeddings``.  This is what the
  poison was *planned* against.

* ``--eval-corpora-dir DIR``: read ``DIR/eval_corpus_C*.npz`` (produced
  by ``scripts/build_eval_corpus.py``) and use ``query_embeddings`` +
  ``doc_embeddings``.  This is the corpus the production retrieval
  pipeline actually scores against — the numbers match the thesis eval.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from rankviz import CORE

OUT_CSV_DEFAULT = os.path.join("examples", "benchmark_results.csv")
OUT_CSV_EVAL    = os.path.join("examples", "benchmark_results_eval.csv")


def top_k_overlap(Q, D, Q_low, D_low, k=10):
    sims_hi = Q @ D.T
    top_hi = np.argsort(-sims_hi, axis=1)[:, :k]
    dists_lo = np.linalg.norm(Q_low[:, None, :] - D_low[None, :, :], axis=-1)
    top_lo = np.argsort(dists_lo, axis=1)[:, :k]
    return float(np.mean([
        len(set(top_hi[i]) & set(top_lo[i])) / k for i in range(Q.shape[0])
    ]))


def pca_project(Q, D, k=3):
    stacked = np.concatenate([Q, D], axis=0)
    centred = stacked - stacked.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    low = centred @ Vt[:k].T
    return low[:Q.shape[0]], low[Q.shape[0]:]


def umap_project(Q, D, k=3):
    from umap import UMAP
    stacked = np.concatenate([Q, D], axis=0)
    red = UMAP(n_components=k, metric="cosine", n_neighbors=15, random_state=42)
    low = red.fit_transform(stacked)
    return low[:Q.shape[0]], low[Q.shape[0]:]


def tsne_project(Q, D, k=3):
    from sklearn.manifold import TSNE
    stacked = np.concatenate([Q, D], axis=0)
    tsne = TSNE(n_components=k, metric="cosine", random_state=42, init="random")
    low = tsne.fit_transform(stacked)
    return low[:Q.shape[0]], low[Q.shape[0]:]


def get_eval_rate(dom, data_root):
    if data_root is None:
        return None
    eval_dir = os.path.join(data_root, dom, "eval")
    if not os.path.isdir(eval_dir):
        return None
    rates = []
    for llm in os.listdir(eval_dir):
        agg = os.path.join(eval_dir, llm, "aggregated_results.json")
        if os.path.exists(agg):
            with open(agg) as f:
                m = json.load(f)["metrics"]
            for key, v in m.items():
                if "poison_retrieved_rate" in key and isinstance(v, dict):
                    rates.append(v["mean"])
    return float(np.mean(rates)) if rates else None


def _resolve_domain_sources(eval_corpora_dir: str | None, data_root: str | None):
    """Yield (domain_label, Q, D, cos_target, eval_rate) triples."""
    if eval_corpora_dir is not None:
        pattern = os.path.join(eval_corpora_dir, "eval_corpus_C*.npz")
        for path in sorted(glob.glob(pattern)):
            m = re.search(r"eval_corpus_(C\d+_[A-Za-z0-9]+)\.npz$", path)
            if not m:
                continue
            dom = m.group(1)
            data = np.load(path, allow_pickle=True)
            Q = data["query_embeddings"].astype(np.float32)
            D = data["doc_embeddings"].astype(np.float32)
            # Pull cos(target) from the companion planning trajectory.npz when
            # data_root is provided and the file is there; otherwise leave blank.
            cos_final = None
            if data_root is not None:
                traj_npz = os.path.join(data_root, dom, "trajectory.npz")
                if os.path.exists(traj_npz):
                    t = np.load(traj_npz, allow_pickle=True)
                    if "cosine_similarities" in t.files:
                        cos_final = float(t["cosine_similarities"][-1])
            yield dom, Q, D, cos_final, get_eval_rate(dom, data_root)
        return

    # Default: read planning-phase trajectory.npz files.
    if data_root is None:
        raise SystemExit(
            "Planning mode requires --data-root pointing at a directory of "
            "C{N}_*/trajectory.npz folders. Use --eval-corpora-dir to skip "
            "the planning step."
        )
    domains = sorted(
        d for d in os.listdir(data_root)
        if d.startswith("C") and "_" in d and os.path.isdir(os.path.join(data_root, d))
    )
    for dom in domains:
        npz_path = os.path.join(data_root, dom, "trajectory.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path, allow_pickle=True)
        Q = data["query_embeddings"].astype(np.float32)
        D = data["shadow_doc_embeddings"].astype(np.float32)
        cos_final = float(data["cosine_similarities"][-1])
        yield dom, Q, D, cos_final, get_eval_rate(dom, data_root)


def main(eval_corpora_dir: str | None, data_root: str | None, out_csv: str):
    print(
        f"[START] corpus source: "
        f"{'eval corpora in ' + eval_corpora_dir if eval_corpora_dir else 'planning (trajectory.npz)'}",
        flush=True,
    )

    results = []
    for dom, Q, D, cos_final, eval_rate in _resolve_domain_sources(eval_corpora_dir, data_root):
        print(f"\n[DOMAIN] {dom}  Q={Q.shape[0]} D={D.shape[0]} "
              f"cos(target)={cos_final} eval_retr={eval_rate}", flush=True)

        row = {
            "domain": dom,
            "n_q": Q.shape[0],
            "n_d": D.shape[0],
            "cos_target": cos_final,
            "eval_retrieval_rate": eval_rate,
        }

        for nc in (2, 3):
            t0 = time.time()
            core = CORE(n_components=nc, n_iter=400, weight="retrieval").fit(Q, D)
            dt = time.time() - t0
            o = top_k_overlap(Q, D, core.query_embedding_, core.corpus_embedding_)
            row[f"CORE_{nc}D"] = o
            print(f"  CORE_{nc}D:    {o:.3f}  ({dt:.1f}s)", flush=True)

        t0 = time.time()
        qp, dp = pca_project(Q, D, 3)
        row["PCA_3D"] = top_k_overlap(Q, D, qp, dp)
        print(f"  PCA_3D:    {row['PCA_3D']:.3f}  ({time.time()-t0:.1f}s)", flush=True)

        try:
            t0 = time.time()
            qp, dp = umap_project(Q, D, 3)
            row["UMAP_3D"] = top_k_overlap(Q, D, qp, dp)
            print(f"  UMAP_3D:   {row['UMAP_3D']:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            row["UMAP_3D"] = None
            print(f"  UMAP_3D:   FAILED ({e})", flush=True)

        try:
            t0 = time.time()
            qp, dp = tsne_project(Q, D, 3)
            row["tSNE_3D"] = top_k_overlap(Q, D, qp, dp)
            print(f"  tSNE_3D:   {row['tSNE_3D']:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            row["tSNE_3D"] = None
            print(f"  tSNE_3D:   FAILED ({e})", flush=True)

        results.append(row)

    # Persist CSV.
    keys = ["domain", "n_q", "n_d", "cos_target", "eval_retrieval_rate",
            "CORE_2D", "CORE_3D", "PCA_3D", "UMAP_3D", "tSNE_3D"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in keys})
    print(f"\n[SAVED] {out_csv}", flush=True)

    # Summary table.
    print("\n[SUMMARY]", flush=True)
    print(f"{'domain':15s} {'retr%':>6s} {'CORE2D':>7s} {'CORE3D':>7s} "
          f"{'PCA3D':>7s} {'UMAP3D':>7s} {'tSNE3D':>7s}", flush=True)
    for r in results:
        er = f"{r['eval_retrieval_rate']*100:5.1f}" if r['eval_retrieval_rate'] is not None else "  -  "
        def f(key): return f"{r[key]:7.3f}" if r.get(key) is not None else "    -  "
        print(f"{r['domain']:15s} {er:>6s} "
              f"{f('CORE_2D')} {f('CORE_3D')} {f('PCA_3D')} "
              f"{f('UMAP_3D')} {f('tSNE_3D')}", flush=True)

    print("-" * 65, flush=True)
    arr = {k: [r[k] for r in results if r.get(k) is not None]
           for k in ["CORE_2D", "CORE_3D", "PCA_3D", "UMAP_3D", "tSNE_3D"]}
    print(f"{'mean':15s} {'':>6s} "
          f"{np.mean(arr['CORE_2D']):7.3f} {np.mean(arr['CORE_3D']):7.3f} "
          f"{np.mean(arr['PCA_3D']):7.3f} {np.mean(arr['UMAP_3D']):7.3f} "
          f"{np.mean(arr['tSNE_3D']):7.3f}", flush=True)

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
             "trajectory.npz. Required for planning mode; optional in "
             "eval-corpora mode (used to look up cos(target) and eval "
             "retrieval rate when present).",
    )
    ap.add_argument(
        "--out", default=None,
        help=f"CSV output path. Defaults (relative to cwd): "
             f"{OUT_CSV_DEFAULT} (planning mode) or {OUT_CSV_EVAL} "
             f"(eval-corpora mode).",
    )
    args = ap.parse_args()
    out = args.out or (OUT_CSV_EVAL if args.eval_corpora_dir else OUT_CSV_DEFAULT)
    main(args.eval_corpora_dir, args.data_root, out)
