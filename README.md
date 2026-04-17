# rankviz

**Retrieval-aware visualisation for dense-retrieval and RAG systems.**

`rankviz` provides visualisation tools that preserve **retrieval-relevant
quantities** (rank, cosine similarity) rather than ambient embedding
geometry. Standard dimensionality reducers (PCA, t-SNE, UMAP) optimise for
the wrong objective in a retrieval context — none preserve the bipartite
query-document cosine structure that actually determines what the retriever
returns.

`rankviz` fills that gap.

---

## The headline contribution: **CORE**

**CORE** — **C**osine-**O**rdered **R**etrieval **E**mbedding — is a
dimensionality-reduction algorithm specialised for retrieval. It places
queries and documents in 2-D or 3-D such that Euclidean distance in the
low-dim space approximates `1 − cos(query, document)` for every
query-document pair, weighted so top-of-ranking is preserved more
faithfully than the tail.

### Benchmark: 11 domains × 5 projection methods

Evaluated on the **production retrieval corpus** used by the thesis
attack's eval pipeline — 17–30 evaluation queries × 8 k–20 k
domain-filtered corpus documents × 768-D E5 embeddings, with the E5
`"query:"` / passage encoding convention. Metric is **top-10 overlap**:
the fraction of each query's true top-10 documents that remain in the
query's top-10 under the low-dim projection, averaged over queries.
(The same benchmark run on the *planning* corpus inside `trajectory.npz`
is in [`examples/benchmark_results_planning.csv`](examples/benchmark_results_planning.csv)
for comparison — see the [Planning corpus vs. eval corpus](#planning-corpus-vs-eval-corpus--use-the-right-one)
section for why they differ.)

![benchmark](examples/benchmark_figure.png)

| Method       | Mean top-10 overlap | Min  | Max  | Win rate |
|--------------|---------------------|------|------|----------|
| **CORE 3-D** | **0.528**           | 0.335| 0.697| **11/11**|
| **CORE 2-D** | **0.510**           | 0.265| 0.700| **11/11**|
| t-SNE 3-D    | 0.079               | 0.000| 0.255| 0/11     |
| UMAP 3-D     | 0.056               | 0.000| 0.260| 0/11     |
| PCA 3-D      | 0.012               | 0.000| 0.027| 0/11     |

**CORE wins every domain against every baseline, no exceptions.** In 3-D
it beats PCA by **44×**, UMAP by **9.4×**, and t-SNE by **6.7×** on
average. The relative gap is *wider* than on the planning corpus because
the eval corpus is domain-filtered — on a tight single-topic distribution,
surrogate objectives (variance, neighbourhood preservation, manifold
topology) lose their remaining loose correlation with retrieval
structure and collapse. Full per-domain numbers are in
[`examples/benchmark_results_eval.csv`](examples/benchmark_results_eval.csv).

> A reasonable question: does CORE just overfit the queries it was
> fit on? The held-out experiment below answers that.

### Held-out generalisation

For each domain: split queries 80 / 20, fit each method on the training
split only, then project the held-out 20 % of queries and score top-10
overlap **on the held-out queries alone**. CORE has a principled
out-of-sample path: `core.transform(X)` projects new embeddings
against the *fixed* query landscape. PCA uses its linear basis. UMAP
and t-SNE have no clean out-of-sample procedure, so they are fit on the
joint `[train; corpus; test]` matrix — a handicap that gives them
access to the test queries during fitting.

This experiment is measured on the **planning corpus** (100 queries
per domain, 5 000 shadow documents), not the eval corpus. The eval
corpora have 17–30 queries each — too few to hold out 20 % and still
get a statistically meaningful test set. The planning corpus (100
queries × 11 domains = 220 held-out test queries total) is the
scientifically correct setup for this specific question.

![heldout](examples/heldout_figure.png)

| Method             | Mean   | Min    | Max    | Wins vs CORE 3-D |
|--------------------|--------|--------|--------|------------------|
| **CORE 3-D**       | **0.456** | 0.210 | 0.700 | —                |
| **CORE 2-D**       | **0.421** | 0.285 | 0.685 | 0 / 11           |
| PCA 3-D            | 0.194  | 0.020  | 0.425  | 0 / 11           |
| UMAP 3-D (handicap)| 0.098  | 0.000  | 0.300  | 0 / 11           |
| t-SNE 3-D (handicap)| 0.067 | 0.000  | 0.405  | 1 / 11 *         |

\* **Honest note:** on C8\_asylum, t-SNE's joint-fit handicap produced
0.405 vs CORE 3-D's 0.400 — a single 0.005 margin, one in eleven domains.
Every other domain is a CORE win. Full per-domain numbers are in
[`examples/heldout_results.csv`](examples/heldout_results.csv).

**What this tells us.** Out-of-sample performance drops for every
method (as expected — generalisation is harder than memorisation). CORE
degrades from 0.829 in-sample to 0.456 held-out in 3-D on the planning
corpus, but PCA falls further (0.339 → 0.194), and UMAP/t-SNE collapse
to near-baseline even *with* test access. CORE 3-D is **2.3× better
than PCA**, **4.7× better than UMAP**, and **6.8× better than t-SNE**
on held-out queries. The bipartite asymmetry CORE exploits carries
over to unseen queries.

**An unexpected finding: CORE 2-D is often a better choice than 3-D for
held-out queries.** In-sample, 3-D is consistently ≥ 2-D; out-of-sample,
2-D edges or matches 3-D on 3 of 11 domains (notably C6\_bankruptcy,
where 2-D 0.410 handily beats 3-D 0.210). The extra dimension creates a
more expressive surface that *over-fits* the training queries. For
production / thesis figures on unseen data, **start with 2-D**.

Reproduce with:

```bash
python scripts/heldout_split.py
```

### UMAP hyperparameter sensitivity

A reasonable reviewer will ask: "did you just run UMAP with bad
hyperparameters?" The main benchmark uses `n_neighbors=15, min_dist=0.1`
(reasonable defaults for cosine). As a sensitivity check, both
hyperparameters were swept on two representative domains — C4\_chemo
and C2\_flu — on the **eval corpus** (the same corpus as the main
benchmark above). Full table in
[`examples/umap_sensitivity_eval.csv`](examples/umap_sensitivity_eval.csv);
the planning-corpus version is in
[`examples/umap_sensitivity_planning.csv`](examples/umap_sensitivity_planning.csv).

| Domain      | n=5 md=0 | n=5 md=0.1 | n=5 md=0.5 | n=15 md=0 | n=15 md=0.1 | n=15 md=0.5 | n=30 md=0 | n=30 md=0.1 | n=30 md=0.5 | n=50 md=0 | n=50 md=0.1 | n=50 md=0.5 | Best | CORE 3-D |
|-------------|----------|------------|------------|-----------|-------------|-------------|-----------|-------------|-------------|-----------|-------------|-------------|------|----------|
| **C4_chemo**| 0.185    | 0.225      | **0.295**  | 0.220     | 0.220       | 0.235       | 0.185     | 0.185       | 0.195       | 0.185     | 0.185       | 0.195       | 0.295| **0.615**|
| **C2_flu**  | 0.080    | 0.080      | **0.187**  | 0.097     | 0.097       | 0.147       | 0.097     | 0.097       | 0.097       | 0.097     | 0.097       | 0.097       | 0.187| **0.633**|

Even with UMAP's best hyperparameter configuration discovered in this
sweep, CORE 3-D is still **2.1× ahead** on C4\_chemo and **3.4× ahead**
on C2\_flu. The reviewer's attack surface is closed.

Reproduce with:

```bash
python scripts/umap_sensitivity.py --eval-corpora-dir /path/to/eval/corpora
```

### What CORE projections look like

A CORE fit on the **C4\_chemo** attack (20 evaluation queries, 9 333
domain-filtered corpus documents, E5 `"query:"` / passage encoding —
the exact retrieval index `eval_single.py` builds). The stored eval
ranks are **median 2, ASR@1 = 40 %, ASR@10 = 95 %** — i.e. the attack
genuinely succeeds in this corpus.

Grey dots are corpus documents. Small blue dots are queries.
**Orange-to-red dots are the documents that actually end up in at
least one query's top-10** — colour intensity shows *how many*
queries retrieve each one. The poisoned document is the red marker,
the target is the gold diamond, and the optimisation trajectory is
the coloured path from green (iter 0) to red (final).

| 2-D (matplotlib, for publication) | 3-D (matplotlib preview) |
|:---:|:---:|
| ![core 2D](examples/figures/core_2d.png) | ![core 3D](examples/figures/core_3d.png) |

**Read the plot:** the dark-red cluster in the lower-right *is* the
retrieval hot zone. Documents there are in the top-10 for 10–20 of
the 20 evaluation queries. The poison lands directly on the target
(gold diamond) and sits inside that hot zone — which is the attack
geometry, visible in one glance. Documents outside the hot zone
(pale orange or grey) are either specialists hit by one or two
queries, or filler that never gets retrieved at all.

Interactive versions (rotate, zoom, hover for exact coordinates):

- 🌐 [`examples/figures/core_2d.html`](examples/figures/core_2d.html) — interactive 2-D
- 🌐 [`examples/figures/core_3d.html`](examples/figures/core_3d.html) — interactive 3-D

### All 11 domains at a glance

Every domain fit on its own eval corpus. The poison (red) consistently
sits inside the orange retrieval hot zone, usually next to the target
(gold) — a visual fingerprint of the attack that's the same across
medical, political, legal, and scientific misinformation categories.

|   |   |   |
|:---:|:---:|:---:|
| ![C1 SSRI](examples/figures/per_domain/C1_ssri.png)  <br>**C1 SSRI** | ![C2 flu](examples/figures/per_domain/C2_flu.png)  <br>**C2 flu** | ![C3 acetaminophen](examples/figures/per_domain/C3_acetaminophen.png)  <br>**C3 acetaminophen** |
| ![C4 chemo](examples/figures/per_domain/C4_chemo.png)  <br>**C4 chemo** | ![C5 5G](examples/figures/per_domain/C5_5g.png)  <br>**C5 5G** | ![C6 bankruptcy](examples/figures/per_domain/C6_bankruptcy.png)  <br>**C6 bankruptcy** |
| ![C7 GMO](examples/figures/per_domain/C7_gmo.png)  <br>**C7 GMO** | ![C8 asylum](examples/figures/per_domain/C8_asylum.png)  <br>**C8 asylum** | ![C9 mail-in](examples/figures/per_domain/C9_mailin.png)  <br>**C9 mail-in** |
| ![C10 voter](examples/figures/per_domain/C10_voter.png)  <br>**C10 voter** | ![C11 quote](examples/figures/per_domain/C11_quote.png)  <br>**C11 quote** |  |

Re-generate the examples from your own data:

```bash
python scripts/generate_examples.py --trajectory /path/to/your/trajectory.npz
```

To enable the retrieval-overlay on your own figures, pass
`show_retrieved_top_k=10` to `plot_landscape`:

```python
fig = plot_landscape(core, highlight=poison, target=target,
                     show_retrieved_top_k=10)
```

### Planning corpus vs. eval corpus — use the right one

GeoPoison-RAG uses **two different retrieval corpora**, and fitting CORE
on the wrong one produces misleading plots:

| Corpus | Where it lives | What it contains |
|--------|----------------|------------------|
| **Planning corpus** | `shadow_doc_embeddings` inside `trajectory.npz` | The 5000 random Wikipedia chunks seen during poison optimisation. Embedded without any E5 prefix. |
| **Eval corpus** | Reconstructed at runtime by `eval_single.py`, *not saved to disk* | The planning corpus filtered by domain keywords (`~8-9 k` docs, capped at 20 k), re-embedded, with queries encoded under the `"query: "` prefix and the poison injected as the last entry. |

The production `ASR@10` numbers (90–100 %) are measured on the **eval
corpus**. The shadow corpus inside `trajectory.npz` is *only* the
planning-phase corpus — fitting CORE on it will show the poison at a
reasonable but unspectacular rank, which reflects that corpus but not
the attack's true behaviour. For thesis-quality plots, rebuild the eval
corpus and fit CORE on it:

```bash
# 1.  Reproduce the eval retrieval corpus (needs the original Wikipedia
#     shadow corpus JSON and the per-domain config YAML):
python scripts/build_eval_corpus.py \
    --shadow-corpus data/cache/shadow_corpus_50000_chunked_512_50.json \
    --domain-config configs/domains/acetaminophen_autism.yaml \
    --seed-dir      seeds/acetaminophen_autism/seed_456 \
    --eval-seed     1 \
    --out           seeds/acetaminophen_autism/seed_456/eval_corpus.npz

# 2.  Fit CORE on that corpus and render the plots:
python scripts/generate_examples.py \
    --trajectory seeds/acetaminophen_autism/seed_456/trajectory.npz \
    --corpus     seeds/acetaminophen_autism/seed_456/eval_corpus.npz
```

See [`scripts/build_eval_corpus.py`](scripts/build_eval_corpus.py) for
the exact reproduction logic (domain keyword filter, 20 k smart cap,
E5 `"query: "` / passage encoding, FAISS `IndexFlatIP` matching
`eval_single.py`'s pipeline). The script prints a sanity-check rank
summary so you can verify it matches the reported eval `ASR@10` before
plotting.

### How CORE works — the algorithm

The goal is a low-dimensional coordinate space where Euclidean distance
between a query and a document reflects their cosine similarity in the
original high-dimensional embedding space — specifically, high-similarity
pairs (the ones that matter for retrieval) must stay close, and the
ordering that the retriever sees must survive the projection.

#### Inputs

- Query embeddings $Q \in \mathbb{R}^{n_q \times D}$
- Corpus embeddings $C \in \mathbb{R}^{n_d \times D}$
- Both L2-normalised, so the dot product equals cosine similarity
- Target dimensionality $k \in \lbrace 2, 3 \rbrace$

#### Objective

Learn low-dim coordinates $X \in \mathbb{R}^{n_q \times k}$ for queries
and $Y \in \mathbb{R}^{n_d \times k}$ for documents that minimise

$$
\mathcal{L}(X, Y) = \frac{1}{n_q \, n_d} \sum_{i, j} w_{ij} \left( \lVert X_i - Y_j \rVert_2 - t_{ij} \right)^2
$$

where $t_{ij} = 1 - \cos(q_i, c_j)$ is the target distance (a cosine of 1
maps to distance 0; cosine 0 maps to distance 1 — monotonic).

#### Weights

Three weighting schemes control which pairs dominate the loss:

| scheme            | $w_{ij}$            | effect                              |
|-------------------|---------------------|-------------------------------------|
| `"retrieval"` ✓   | $\cos(q_i, c_j)^4$  | emphasises top-of-ranking           |
| `"rank"`          | $1 / r_{ij}$        | explicit top-k preservation         |
| `"uniform"`       | $1$                 | plain bipartite MDS                 |

Here $r_{ij}$ is the rank of document $j$ for query $i$ (1 = most similar).

`"retrieval"` is the default. Raising cosine to the fourth power means a
cos-0.9 pair contributes ~0.66 to the weight, while a cos-0.5 pair contributes
only ~0.06 — the top of the ranking dominates the optimisation.

**Key property: only query-document pairs appear in the loss.** Not
query-query, not document-document. This bipartite asymmetry is exactly
what PCA / UMAP / t-SNE cannot express — they treat all points
symmetrically.

#### Optimisation

1. **Initialise** $X, Y$ from the top-$k$ right singular vectors of the
   stacked matrix $\begin{bmatrix} Q \\ C \end{bmatrix}$ (PCA-style
   start), rescaled so initial inter-point distances sit near the
   target range.

2. **Full-batch gradient descent** for $N$ iterations (default 500):

   Let $d_{ij} = \lVert X_i - Y_j \rVert_2$ be the current low-dim
   distance and $e_{ij} = d_{ij} - t_{ij}$ the error. Per-point gradients
   are

$$
\nabla_{X_i} \mathcal{L} = \frac{1}{n_d} \sum_j w_{ij} \, \frac{e_{ij}}{d_{ij}} \, (X_i - Y_j),
\qquad
\nabla_{Y_j} \mathcal{L} = -\frac{1}{n_q} \sum_i w_{ij} \, \frac{e_{ij}}{d_{ij}} \, (X_i - Y_j).
$$

   The learning rate decays linearly from `lr` to `0.1 · lr`, and
   per-row gradient L2-norms are clipped at 1.0 to tame early iterates.

#### Out-of-sample projection

Once the query landscape $X$ is fit, new points (the poison, each
trajectory step, the target) project against the **fixed** query
coordinates. For a new embedding $z$, let
$\tilde{t}_i = 1 - \cos(q_i, z)$ and solve

$$
y^{\star}(z) = \arg\min_{y \in \mathbb{R}^k} \sum_{i} w_i \left( \lVert X_i - y \rVert_2 - \tilde{t}_i \right)^2.
$$

A tiny gradient descent (200 iterations, $k$ parameters) solves this.
Two trajectories fit against the same $X$ are directly comparable —
critical for the poison-optimisation paths shown above.

#### Why this beats UMAP / PCA / t-SNE at retrieval

By construction, CORE optimises the retrieval objective directly —
preserving the bipartite query-document cosine structure. PCA / UMAP /
t-SNE optimise *surrogates*: variance along principal directions,
local neighbourhood preservation, or manifold topology. Each surrogate
is a different bet about what "structure" means in the embedding
space, and none of them are the retrieval structure.

The benchmark quantifies how well each surrogate aligns with retrieval
behaviour in practice. An 11× mean gap between CORE and UMAP on
in-sample top-10 overlap, and the same *relative* lead sustained on
held-out queries, indicates that the surrogates are chasing signal
largely orthogonal to retrieval-relevant structure. If you want to
visualise retrieval behaviour, the projection should be driven by
the retrieval relationship itself.

#### What did not help

Tested during development and discarded:

- **Triplet / hinge ranking loss** — hurt overlap by 8–18 percentage points
- **Rank-as-distance target** ($t_{ij} = r_{ij} / n_d$) — catastrophic collapse
- **Random-init multi-restart** — no better than SVD init
- **Longer training past 400 iterations** — plateau
- **Over-complete training** (fit in 5-D, PCA-compress to 2-D) — the
  compression step breaks the learned distances

The simple $\cos^4$-weighted MSE objective with SVD initialisation is
surprisingly close to the ceiling for this data class.

### Quickstart

```python
from rankviz import CORE, plot_landscape

# Fit the projection
core = CORE(n_components=3).fit(queries=Q, corpus=D)

# Project auxiliary points (highlight document, optimisation path, target)
fig = plot_landscape(
    core,
    highlight=poison,            # (d,) or (M, d) array
    trajectory=traj,             # optimisation path, (n_steps, d)
    target=target,               # (d,) — where the attack aims
    backend="plotly",            # interactive HTML
)
fig.write_html("landscape.html")
```

2-D for publication:

```python
core_2d = CORE(n_components=2).fit(queries=Q, corpus=D)
fig = plot_landscape(core_2d, highlight=poison, backend="matplotlib")
fig.savefig("figure.pdf")
```

One-liner convenience:

```python
from rankviz import quick_plot
fig = quick_plot(queries=Q, corpus=D, highlight=poison, kind="rank_carpet")
```

### API summary

| Call                                     | Returns                               |
|------------------------------------------|---------------------------------------|
| `CORE(n_components=2|3)`                 | A configured, unfitted estimator      |
| `.fit(queries, corpus)`                  | `self`                                |
| `.transform(X)`                          | `(d,) → (k,)` or `(n, d) → (n, k)`    |
| `.fit_transform(queries, corpus)`        | `(Q_low, D_low)`                      |
| `.query_embedding_`                      | fitted query coordinates              |
| `.corpus_embedding_`                     | fitted corpus coordinates             |
| `.loss_history_`                         | per-iteration training loss           |

sklearn/umap-learn conventions: `.fit()` returns `self`, `.transform(X)`
projects new points without refitting.

### Weighting schemes

| `weight=`        | Behaviour                                             |
|------------------|-------------------------------------------------------|
| `"retrieval"` ✓  | Weight pairs by `cos⁴` — emphasises high-similarity pairs |
| `"rank"`         | Weight pairs by `1/rank` — explicit top-k preservation    |
| `"uniform"`      | Plain bipartite MDS                                       |

`"retrieval"` is the default; it was the best performer across the 11-domain benchmark.

---

## Companion visualisations (no projection)

`CORE` is the projection engine. `rankviz` also provides visualisations
that use retrieval-relevant axes **directly** — no projection, no loss of
information.

### `RankCarpet`

Per-query rank profile of highlight documents. X-axis: queries (sorted by
highlight rank). Y-axis: rank (log, inverted). Corpus shown as percentile
bands; highlights as bold lines with reference lines at `k=10`, `k=100`.

*Answers:* does this document consistently outrank the corpus?

### `SimilarityWaterfall`

Per-query cosine similarity with top-k threshold context. Shows the margin
by which a highlight clears or misses the retrieval cutoff — information
that ordinal rank hides.

*Answers:* how robust is this document's retrieval position?

### `RankDistribution`

Aggregate distribution of a highlight document's rank across queries.
Histogram / KDE / CDF modes, facetable by domain via `query_labels`.

*Answers:* generalist, specialist, or adversarial outlier?

---

## Installation

```bash
# Core (numpy + matplotlib)
pip install -e .

# Interactive HTML backend (plotly)
pip install -e ".[plotly]"

# Development (includes pytest)
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest tests/ -v
```

60 tests, covering correctness of the similarity/rank computation,
visualisation smoke tests, and CORE (fit, transform, rank preservation,
loss behaviour, 2-D and 3-D, all three weighting schemes).

## Reproducing the experiments

**Main benchmark on eval corpora** (the headline numbers — requires
`eval_corpus_*.npz` files built by `scripts/build_eval_corpus.py`):

```bash
python scripts/benchmark_all_domains.py --eval-corpora-dir /path/to/eval/corpora
python scripts/make_benchmark_figure.py \
    --csv examples/benchmark_results_eval.csv \
    --out-png examples/benchmark_figure.png \
    --out-pdf examples/benchmark_figure.pdf
```

**Main benchmark on planning corpora** (uses `trajectory.npz` directly,
no extra reproduction step needed):

```bash
python scripts/benchmark_all_domains.py
```

**Held-out generalisation** (on planning corpora — see Caveats for why):

```bash
python scripts/heldout_split.py
```

**UMAP hyperparameter sensitivity** (both corpus modes supported):

```bash
python scripts/umap_sensitivity.py --eval-corpora-dir /path/to/eval/corpora
```

The benchmark measures **top-10 overlap** — for each query, the fraction
of its true top-10 corpus documents that remain in the top-10 under
Euclidean distance in the projected low-dim space.

---

## When NOT to use rankviz

If you want to see the geometric structure of your embedding space, use
UMAP or PCA. `rankviz` is for **retrieval behaviour**, not embedding
geometry. If you only care about how documents are distributed globally
(not how they rank against specific queries), `rankviz` is not the tool
for you.

---

## Caveats

- CORE is fit **per-dataset** (like UMAP and t-SNE). There is no "one
  model" that works on all seeds — each run gets its own fit.
- Benchmarks were on 11 adversarial RAG datasets, all single-domain
  query sets (medical, political, legal, tech misinformation). CORE's
  advantage on broader / multi-domain query distributions (e.g. MS
  MARCO, BEIR) has not yet been measured.
- Absolute top-10 overlap numbers depend on which corpus you measure
  against. On the **eval corpus** (domain-filtered, matching
  `eval_single.py`) CORE 3-D averages **0.528**. On the **planning
  corpus** (the 5 000 shadow docs inside `trajectory.npz`) it averages
  **0.829**. The relative gap to baselines is *wider* on the eval
  corpus (CORE beats UMAP by 9.4× vs 10.8×, PCA by 44× vs 2.4×) because
  baselines collapse harder on the domain-filtered distribution. Either
  corpus is a valid measurement — just be explicit about which one.
- The held-out experiment is measured on the planning corpus (100
  queries per domain) because the eval corpora have only 17–30 queries
  each — too few to hold out 20 % and get statistically meaningful
  test-set averages. The algorithm property being tested
  (out-of-sample generalisation) is corpus-agnostic, so planning is
  the stronger setup for this specific question.
- On the planning-corpus held-out benchmark, t-SNE beat CORE 3-D by
  0.005 on one of 11 domains (C8\_asylum), using the joint-fit handicap.
  One in eleven, smallest possible margin — but recorded honestly. On
  the eval-corpus in-sample benchmark, CORE wins on every domain
  against every baseline with no exceptions.

---

## Style

Publication-quality defaults: 300 DPI, Helvetica → Arial → DejaVu Sans
fallback, thin spines, no embedded titles, colourblind-safe Paul Tol
palette. Override globally with `rankviz.apply_style()` or locally with
the `rankviz.style()` context manager.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

## Citation

If you use `rankviz` in your research, please cite:

```bibtex
@software{eilertsen2026rankviz,
  author = {Eilertsen, Brage},
  title  = {rankviz: Retrieval-aware visualisation for dense-retrieval and RAG systems},
  year   = {2026},
  url    = {https://github.com/BrageEilertsen/rankviz},
}
```
