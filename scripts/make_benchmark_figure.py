"""Render the benchmark bar chart from a CSV produced by
``scripts/benchmark_all_domains.py``.

Usage::

    python scripts/make_benchmark_figure.py \\
        --csv examples/benchmark_results_eval.csv \\
        --out-png examples/benchmark_figure.png \\
        --out-pdf examples/benchmark_figure.pdf
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["CORE 2D", "CORE 3D", "PCA 3D", "UMAP 3D", "t-SNE 3D"]
CSV_KEYS = {
    "CORE 2D": "CORE_2D",
    "CORE 3D": "CORE_3D",
    "PCA 3D":  "PCA_3D",
    "UMAP 3D": "UMAP_3D",
    "t-SNE 3D":"tSNE_3D",
}
COLOURS  = ["#EE6677", "#AA3377", "#4477AA", "#228833", "#CCBB44"]


def main(csv_path: str, out_png: str, out_pdf: str) -> None:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            parsed = {"domain": r["domain"].replace("_", " ")}
            for label, key in CSV_KEYS.items():
                raw = r.get(key, "")
                parsed[label] = float(raw) if raw not in ("", None) else 0.0
            rows.append(parsed)

    rows.sort(key=lambda d: d["CORE 3D"])
    x = np.arange(len(rows))
    bar_w = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, c) in enumerate(zip(METHODS, COLOURS)):
        vals = [r[m] for r in rows]
        ax.bar(x + (i - 2) * bar_w, vals, bar_w,
               label=m, color=c, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([r["domain"] for r in rows],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Top-10 retrieval overlap", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(labelsize=8, width=0.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf} and {out_png}")

    means = {m: float(np.mean([r[m] for r in rows])) for m in METHODS}
    print("\nMean top-10 overlap:")
    for m, v in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"  {m:10s}  {v:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()
    main(args.csv, args.out_png, args.out_pdf)
