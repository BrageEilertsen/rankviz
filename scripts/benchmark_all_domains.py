"""Benchmark CORE against PCA / UMAP / t-SNE across all C* domain folders."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from rankviz import CORE

BASE = "/Users/brageeilertsen/trajectory_data"
OUT_CSV = "/Users/brageeilertsen/trajectory_data/rankviz/benchmark_results.csv"


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


def get_eval_rate(dom):
    eval_dir = os.path.join(BASE, dom, "eval")
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


def main():
    domains = sorted(
        d for d in os.listdir(BASE)
        if d.startswith("C") and "_" in d and os.path.isdir(os.path.join(BASE, d))
    )
    print(f"[START] domains: {domains}", flush=True)

    results = []
    for dom in domains:
        npz_path = os.path.join(BASE, dom, "trajectory.npz")
        if not os.path.exists(npz_path):
            print(f"[SKIP] {dom}: no trajectory.npz", flush=True)
            continue

        data = np.load(npz_path, allow_pickle=True)
        Q = data["query_embeddings"].astype(np.float32)
        D = data["shadow_doc_embeddings"].astype(np.float32)
        cos_final = float(data["cosine_similarities"][-1])
        eval_rate = get_eval_rate(dom)

        print(f"\n[DOMAIN] {dom}  Q={Q.shape[0]} D={D.shape[0]} "
              f"cos(target)={cos_final:.3f} eval_retr={eval_rate}", flush=True)

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
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in keys})
    print(f"\n[SAVED] {OUT_CSV}", flush=True)

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
    main()
