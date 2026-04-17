"""Reproduce the GeoPoison-RAG eval retrieval corpus for CORE visualisation.

The ``shadow_doc_embeddings`` stored inside ``trajectory.npz`` are from the
*planning* phase.  The production eval pipeline (``eval_single.py``) builds
a different retrieval index: a domain-keyword-filtered subset of the full
Wikipedia shadow corpus, capped at 20 k documents, with the poison
injected at the end.  Fitting CORE on the planning corpus is *not* the
same as fitting on the eval corpus, so ranks and visual positions differ.

This script reproduces the eval-time corpus construction and saves the
embeddings as ``eval_corpus.npz``.  Feed that file to
``scripts/generate_examples.py`` to get CORE plots that reflect what the
actual attack sees.

Usage
-----

::

    python scripts/build_eval_corpus.py \\
        --shadow-corpus data/cache/shadow_corpus_50000_chunked_512_50.json \\
        --domain-config configs/domains/acetaminophen_autism.yaml \\
        --seed-dir seeds/acetaminophen_autism/seed_456 \\
        --eval-seed 1 \\
        --out seeds/acetaminophen_autism/seed_456/eval_corpus.npz

The output ``.npz`` contains:

- ``corpus_embeddings`` ``(n_d, 768)`` — filtered, capped, embedded without
  the ``"query: "`` prefix (E5 passage format).
- ``corpus_texts`` ``(n_d,)`` — the original strings, for debugging.
- ``query_embeddings`` ``(n_q, 768)`` — evaluation queries encoded with the
  ``"query: "`` prefix (E5 query format).
- ``query_texts`` ``(n_q,)`` — the original strings.
- ``poison_embedding`` ``(768,)`` — the poisoned document encoded with
  ``"passage: "`` prefix (same as other corpus docs).
- ``poison_index`` ``int`` — position of the poison in the corpus (always
  ``n_d`` because it's appended last, matching eval_single.py).
- ``filter_keywords`` ``list[str]`` — the keywords used for filtering.
- ``eval_seed`` ``int`` — the seed that controls the subsample when the
  filtered set exceeds the 20 k cap.

All embeddings are L2-normalised (``normalize_embeddings=True``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml


MAX_DOCS = 20_000
MIN_DOCS_FOR_FILTER = 1_000


def load_shadow_corpus(path: str) -> list[str]:
    with open(path) as f:
        docs = json.load(f)
    if not isinstance(docs, list) or not all(isinstance(x, str) for x in docs):
        raise ValueError(
            f"Expected {path} to be a JSON list of strings, got {type(docs)}."
        )
    return docs


def filter_by_keywords(docs: list[str], keywords: list[str]) -> list[str]:
    """Keep only documents matching any domain keyword (case-insensitive)."""
    lowered = [kw.lower() for kw in keywords]
    return [d for d in docs if any(kw in d.lower() for kw in lowered)]


def smart_cap(
    docs: list[str],
    required_keywords: list[str],
    eval_seed: int,
    max_docs: int = MAX_DOCS,
) -> list[str]:
    """If ``len(docs) > max_docs``, keep all required-keyword matches first,
    then fill the remaining slots by random sample."""
    if len(docs) <= max_docs:
        return docs

    lowered_req = [kw.lower() for kw in required_keywords[:5]]
    domain_docs = [d for d in docs if any(kw in d.lower() for kw in lowered_req)]
    other_docs = [d for d in docs if d not in domain_docs]

    remaining = max_docs - len(domain_docs)
    rng = random.Random(eval_seed)
    kept = domain_docs + rng.sample(other_docs, min(remaining, len(other_docs)))
    rng.shuffle(kept)
    return kept


def build_eval_corpus(
    *,
    shadow_corpus_json: str,
    domain_config_yaml: str,
    seed_dir: str,
    eval_seed: int,
    out_path: str,
    model_id: str = "intfloat/e5-base-v2",
    batch_size: int = 32,
) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"[1/6] Loading shadow corpus: {shadow_corpus_json}")
    all_docs = load_shadow_corpus(shadow_corpus_json)
    print(f"      {len(all_docs):,} chunks")

    print(f"[2/6] Loading domain config: {domain_config_yaml}")
    with open(domain_config_yaml) as f:
        cfg = yaml.safe_load(f)
    keywords = cfg.get("filter_keywords", [])
    required = (
        cfg.get("agenda", {}).get("keywords", {}).get("required", [])
    )
    eval_queries = cfg.get("evaluation_queries")
    if not eval_queries:
        raise ValueError(
            f"{domain_config_yaml} has no `evaluation_queries` field."
        )
    print(f"      {len(keywords)} filter keywords, {len(required)} required "
          f"keywords, {len(eval_queries)} evaluation queries")

    print("[3/6] Filtering corpus by keywords...")
    filtered = filter_by_keywords(all_docs, keywords) if keywords else all_docs
    if keywords and len(filtered) < MIN_DOCS_FOR_FILTER:
        print(f"      only {len(filtered)} matched — falling back to full corpus")
        filtered = all_docs
    else:
        print(f"      {len(all_docs):,} → {len(filtered):,} after keyword filter")

    print("[4/6] Applying smart cap...")
    corpus = smart_cap(filtered, required, eval_seed, MAX_DOCS)
    print(f"      {len(filtered):,} → {len(corpus):,} after cap (seed={eval_seed})")

    print(f"[5/6] Loading poison text from: {seed_dir}")
    poison_txt = Path(seed_dir) / "poison.txt"
    if not poison_txt.exists():
        raise FileNotFoundError(f"Missing {poison_txt}")
    poison_text = poison_txt.read_text()

    print(f"[6/6] Embedding with {model_id}...")
    embedder = SentenceTransformer(model_id)

    # Documents: no prefix (E5 passage encoding is the default empty string).
    corpus_texts = np.array(corpus, dtype=object)
    corpus_embs = embedder.encode(
        corpus, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)

    # Queries: prepend "query: " as E5 requires.
    query_texts = np.array(eval_queries, dtype=object)
    query_embs = embedder.encode(
        [f"query: {q}" for q in eval_queries],
        batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Poison: same encoding as corpus docs (no prefix).
    poison_emb = embedder.encode(
        [poison_text], convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)[0]

    poison_index = corpus_embs.shape[0]  # appended at the end

    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        corpus_embeddings=corpus_embs,
        corpus_texts=corpus_texts,
        query_embeddings=query_embs,
        query_texts=query_texts,
        poison_embedding=poison_emb,
        poison_index=poison_index,
        filter_keywords=np.array(keywords, dtype=object),
        eval_seed=eval_seed,
    )
    print(f"\n[SAVED] {out_path}")

    # Sanity check: where does the poison rank per query?
    sims = query_embs @ np.concatenate([corpus_embs, poison_emb[None, :]], axis=0).T
    ranks = (sims > sims[:, poison_index][:, None]).sum(axis=1) + 1
    asr_at_10 = (ranks <= 10).mean() * 100
    asr_at_1  = (ranks <= 1).mean()  * 100
    print(f"\nSanity check:")
    print(f"  poison_index = {poison_index}")
    print(f"  median rank = {int(np.median(ranks))}")
    print(f"  ASR@1  = {asr_at_1:.1f}%")
    print(f"  ASR@10 = {asr_at_10:.1f}%")
    print(f"  (These should match eval_single.py's reported numbers.)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce the GeoPoison-RAG eval retrieval corpus."
    )
    ap.add_argument(
        "--shadow-corpus", required=True,
        help="Path to shadow_corpus_50000_chunked_512_50.json "
             "(the Wikipedia 20220301 shadow corpus, 512-token chunks).",
    )
    ap.add_argument(
        "--domain-config", required=True,
        help="Path to configs/domains/<domain>.yaml — must contain "
             "`filter_keywords`, `evaluation_queries`, and `agenda.keywords.required`.",
    )
    ap.add_argument(
        "--seed-dir", required=True,
        help="Path to the specific seed output directory (contains poison.txt).",
    )
    ap.add_argument(
        "--eval-seed", type=int, default=1,
        help="Evaluation random seed (for reproducible smart-cap sampling). Default: 1.",
    )
    ap.add_argument(
        "--out", required=True,
        help="Path to write eval_corpus.npz.",
    )
    ap.add_argument(
        "--model", default="intfloat/e5-base-v2",
        help="Sentence-Transformers model id. Default: intfloat/e5-base-v2.",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    build_eval_corpus(
        shadow_corpus_json=args.shadow_corpus,
        domain_config_yaml=args.domain_config,
        seed_dir=args.seed_dir,
        eval_seed=args.eval_seed,
        out_path=args.out,
        model_id=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
