"""Held-out query split experiment.

For each domain:
  * Split queries 80/20 (same random seed across domains and methods).
  * Fit CORE on 80% train queries + full corpus.
  * Project held-out 20% queries via CORE.transform().
  * Score top-10 overlap on the test queries only.
  * Compare against PCA / UMAP / t-SNE fit on train+corpus, test queries
    projected out-of-sample:
       - PCA via explicit linear projection (Vt applied to test).
       - UMAP/t-SNE have no principled OOS: we fit them on the combined
         train+corpus+test matrix (i.e. give them test access for free)
         as an upper bound — this is a handicap match that favours them.

Output: examples/heldout_results.csv
"""
from __future__ import annotations

import csv, json, os, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

from rankviz import CORE

BASE = "/Users/brageeilertsen/trajectory_data"
OUT = os.path.join(BASE, "rankviz", "examples", "heldout_results.csv")

TEST_FRAC = 0.2
SEED = 42


def pca_fit_transform(Q_train, C, Q_test, k=3):
    """PCA on [Q_train; C], then apply the same linear projection to Q_test."""
    stacked = np.concatenate([Q_train, C], axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    centred = stacked - mean
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    basis = Vt[:k].T   # (D, k)
    Q_train_low = (Q_train - mean) @ basis
    C_low = (C - mean) @ basis
    Q_test_low = (Q_test - mean) @ basis
    return Q_train_low, C_low, Q_test_low


def umap_joint(Q_train, C, Q_test, k=3, n_neighbors=15, min_dist=0.1):
    """UMAP on the full matrix — gives UMAP test access (upper bound)."""
    from umap import UMAP
    stacked = np.concatenate([Q_train, C, Q_test], axis=0)
    red = UMAP(n_components=k, metric="cosine", n_neighbors=n_neighbors,
               min_dist=min_dist, random_state=SEED)
    low = red.fit_transform(stacked)
    n_tr, n_c = Q_train.shape[0], C.shape[0]
    return low[:n_tr], low[n_tr:n_tr + n_c], low[n_tr + n_c:]


def tsne_joint(Q_train, C, Q_test, k=3):
    from sklearn.manifold import TSNE
    stacked = np.concatenate([Q_train, C, Q_test], axis=0)
    red = TSNE(n_components=k, metric="cosine", random_state=SEED, init="random")
    low = red.fit_transform(stacked)
    n_tr, n_c = Q_train.shape[0], C.shape[0]
    return low[:n_tr], low[n_tr:n_tr + n_c], low[n_tr + n_c:]


def top_k_test(Q_test, C, Q_test_low, C_low, k=10):
    """Top-k overlap computed on the held-out queries only."""
    sims_hi = Q_test @ C.T                                   # (n_test, n_d)
    top_hi = np.argsort(-sims_hi, axis=1)[:, :k]
    dists = np.linalg.norm(
        Q_test_low[:, None, :] - C_low[None, :, :], axis=-1,
    )
    top_lo = np.argsort(dists, axis=1)[:, :k]
    return float(np.mean([
        len(set(top_hi[i]) & set(top_lo[i])) / k for i in range(Q_test.shape[0])
    ]))


def main():
    domains = sorted(
        d for d in os.listdir(BASE)
        if d.startswith("C") and "_" in d and os.path.isdir(os.path.join(BASE, d))
    )
    rng = np.random.default_rng(SEED)
    results = []

    print(f"[START] held-out split: {len(domains)} domains, test fraction {TEST_FRAC}", flush=True)

    for dom in domains:
        npz = os.path.join(BASE, dom, "trajectory.npz")
        if not os.path.exists(npz):
            continue
        data = np.load(npz, allow_pickle=True)
        Q = data["query_embeddings"].astype(np.float32)
        C = data["shadow_doc_embeddings"].astype(np.float32)

        n = Q.shape[0]
        perm = rng.permutation(n)
        n_test = max(1, int(round(n * TEST_FRAC)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        Q_train, Q_test = Q[train_idx], Q[test_idx]

        row = {"domain": dom, "n_train": len(train_idx), "n_test": len(test_idx)}
        print(f"\n[{dom}] train={len(train_idx)} test={len(test_idx)}", flush=True)

        # CORE 2-D: fit on train, project test via transform().
        for nc in (2, 3):
            t0 = time.time()
            core = CORE(n_components=nc, n_iter=400, weight="retrieval").fit(
                queries=Q_train, corpus=C,
            )
            Q_test_low = core.transform(Q_test)
            score = top_k_test(Q_test, C, Q_test_low, core.corpus_embedding_)
            row[f"CORE_{nc}D"] = score
            print(f"  CORE_{nc}D:  {score:.3f}   ({time.time()-t0:.1f}s, OOS via transform)", flush=True)

        # PCA 3-D: fit on train+corpus, apply same linear map to test.
        t0 = time.time()
        Qtr, Cl, Qte = pca_fit_transform(Q_train, C, Q_test, 3)
        row["PCA_3D"] = top_k_test(Q_test, C, Qte, Cl)
        print(f"  PCA_3D:   {row['PCA_3D']:.3f}   ({time.time()-t0:.1f}s, OOS via linear map)", flush=True)

        # UMAP 3-D: HANDICAP — fit on train+corpus+test (gives UMAP full access).
        try:
            t0 = time.time()
            Qtr, Cl, Qte = umap_joint(Q_train, C, Q_test, 3)
            row["UMAP_3D_handicap"] = top_k_test(Q_test, C, Qte, Cl)
            print(f"  UMAP_3D:  {row['UMAP_3D_handicap']:.3f}   ({time.time()-t0:.1f}s, handicap joint fit)", flush=True)
        except Exception as e:
            row["UMAP_3D_handicap"] = None
            print(f"  UMAP_3D: FAILED ({e})", flush=True)

        # t-SNE 3-D: same handicap.
        try:
            t0 = time.time()
            Qtr, Cl, Qte = tsne_joint(Q_train, C, Q_test, 3)
            row["tSNE_3D_handicap"] = top_k_test(Q_test, C, Qte, Cl)
            print(f"  tSNE_3D:  {row['tSNE_3D_handicap']:.3f}   ({time.time()-t0:.1f}s, handicap joint fit)", flush=True)
        except Exception as e:
            row["tSNE_3D_handicap"] = None
            print(f"  tSNE_3D: FAILED ({e})", flush=True)

        results.append(row)

    keys = ["domain", "n_train", "n_test",
            "CORE_2D", "CORE_3D", "PCA_3D",
            "UMAP_3D_handicap", "tSNE_3D_handicap"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in keys})

    print(f"\n[SAVED] {OUT}", flush=True)
    print("\n[SUMMARY — held-out top-10 overlap]", flush=True)
    arr = {k: [r[k] for r in results if r.get(k) is not None]
           for k in ["CORE_2D", "CORE_3D", "PCA_3D", "UMAP_3D_handicap", "tSNE_3D_handicap"]}
    for k, vals in arr.items():
        print(f"  {k:20s}  mean={np.mean(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}", flush=True)
    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
