# doc-retrieval-eval

A hybrid document-retrieval pipeline with an evaluation harness, built to answer one question:
**when you fuse multiple retrieval routes, how do you know it actually helped?**

Runs with `numpy` only. No API keys, no downloads, no external corpus.

```
python -m docret.cli eval
```

## What it does

1. **Builds a synthetic corpus** of documents with nested sections and planted, checkable facts.
2. **Chunks section-aware** — every chunk carries its heading path, so a retrieved unit can be cited
   precisely rather than "somewhere in the document".
3. **Runs two retrieval routes** — Okapi BM25 and TF-IDF cosine — independently.
4. **Fuses them** with weighted Reciprocal Rank Fusion.
5. **Scores all three** against ground-truth labels (`recall@k`, `MRR`, `nDCG@k`).

The point is step 5. Anyone can add a second retriever; the work is proving it earns its complexity.

## Why the corpus is synthetic

Real document corpora are usually confidential. This one is generated from a fixed seed, so results
are reproducible and the labels are exact rather than hand-annotated. The approach is
deliberately borrowed from how you would evaluate a real system: plant facts you can grade,
then measure whether retrieval finds them.

## Design notes

**Section-aware chunking.** Fixed-window chunking splits a table across two chunks and loses the
heading that gives it meaning. Here each chunk is a complete section with its full ancestor path
(`Document > Data handling > Retention`), so a hit is always attributable.

**Reciprocal Rank Fusion, not score blending.** BM25 scores and cosine similarities live on
different scales and are not comparable. RRF sidesteps that by combining *ranks*:

```
score(d) = sum over routes of  weight / (k + rank(d))
```

with `k = 60`. This is robust when one route produces a wild outlier score.

**Evaluation over vibes.** Three metrics because each answers a different question: recall@k asks
"was it found at all", MRR asks "how high did it land", nDCG@k asks "was the ranking sensible".

## Results

Reproduce with `python -m docret.cli eval`. 40 documents, 250 section chunks, 150 labelled queries,
k=5:

| route | recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| BM25 | 1.000 | 0.886 | 0.915 |
| TF-IDF cosine | 1.000 | 0.886 | 0.915 |
| structural (heading path only) | 1.000 | 0.611 | 0.710 |
| **fused (1.0 / 1.0 / 0.6)** | **1.000** | **0.886** | **0.915** |

**Fusion did not beat the best single route on this corpus. That is the finding, and it is worth
more than a fabricated win.**

What the sweep actually showed:

| structure weight | MRR |
|---|---|
| 0 (body routes only) | 0.886 |
| 0.6 | 0.886 |
| 1.0 | 0.879 |
| 1.5 | 0.798 |

Three conclusions, none of which I could have written without measuring:

1. **RRF buys robustness, not free gains.** Adding a route that is 28 points weaker on its own cost
   nothing at a sensible weight. That tolerance is the actual argument for rank fusion.
2. **Weight is a decision, not a detail.** At 1.5 the weak route stops being a hedge and starts
   dominating — 0.886 collapses to 0.798. Fusion can make you worse.
3. **This corpus is too easy at the top.** Recall@5 is saturated at 1.000 for every route, so the
   ranking metrics are doing all the work. A harder corpus would separate the routes further; that
   is the next thing to build, not a bigger model.

The honest summary: on a synthetic corpus with exact labels, a second lexical route is close to
redundant. The harness is the reusable part — point it at a real corpus and it will tell you whether
your extra route earns its complexity.

## Structure

```
docret/
  corpus.py      synthetic documents, nested sections, labelled queries
  chunking.py    section-aware chunking -> chunks with heading paths
  retrieval.py   BM25, TF-IDF cosine, weighted reciprocal rank fusion
  evaluation.py  recall@k, MRR, nDCG@k
  cli.py         command line entry point
tests/
  test_core.py   chunking, ranking behaviour, fusion and metric correctness
```

## Running the tests

```
python -m pytest tests/ -q
```
