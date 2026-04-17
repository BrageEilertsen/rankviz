# Contributing to rankviz

Thanks for your interest. This is an academic research project (a
master's thesis contribution); contributions are welcome but please
scope them clearly.

## Development setup

```bash
git clone https://github.com/BrageEilertsen/rankviz rankviz
cd rankviz
pip install -e ".[dev]"
pytest
```

The `[dev]` extra installs `pytest` and `plotly`. All tests should pass
on a fresh checkout.

## Running tests

```bash
pytest tests/ -v
```

Tests cover:
- Similarity / rank computation correctness
- CORE algorithm (fit, transform, loss decrease, rank preservation)
- Plotting smoke tests (2-D and 3-D, matplotlib and plotly backends)
- All three `weight` schemes

## Style

- **British English** in docs and comments (not en-US)
- **Publication-quality plotting defaults** — 300 DPI, sans-serif,
  thin spines, no embedded titles
- **sklearn conventions** for API shape (`fit`, `transform`, `fit_transform`,
  trailing underscores for learned attributes)
- Only `numpy` and `matplotlib` in core requirements; `plotly` is optional

## Contribution types

Useful:

- Additional benchmark datasets (especially diverse / multi-domain query sets)
- Held-out generalisation tests (fit on a training split, measure on a test split)
- New weighting schemes or loss variants that demonstrably improve top-k
  overlap on the existing benchmark
- Support for sparse retrievers, cross-encoders, and non-cosine metrics
  (API already leaves room — see the `similarities` / `highlight_similarities`
  path in `RankCarpet` / `SimilarityWaterfall` / `RankDistribution`)
- Performance / scalability — chunked computation for very large corpora

Please open an issue to discuss larger changes before sending a PR.

## Commits

Conventional-commits-style prefixes are appreciated (`feat:`, `fix:`,
`docs:`, `test:`, `refactor:`) but not required.
