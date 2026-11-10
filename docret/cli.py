"""Command line entry point: python -m docret.cli eval"""
from __future__ import annotations

import argparse

from .chunking import chunk_documents
from .corpus import build
from .evaluation import evaluate
from .retrieval import BM25, TfidfCosine, ranks_from_scores, reciprocal_rank_fusion, tokenize


def structural_scores(path_tokens: list[list[str]]):
    """A route that looks only at where a chunk sits, not what it says.

    Heading paths carry different information from body text: a chunk under
    'Retention' is about retention even if the word never appears in the body.
    Keeping it a separate route is what gives fusion something to disagree with.
    """
    return BM25(path_tokens)


def run(k: int = 5, seed: int = 7, n_docs: int = 30, verbose: bool = False) -> dict:
    docs, labels = build(seed=seed, n_docs=n_docs)
    chunks = chunk_documents(docs)
    if verbose:
        print(f"corpus: {len(docs)} documents, {len(chunks)} chunks, {len(labels)} labelled queries")

    tokenized = [tokenize(c.text + " " + c.display_path) for c in chunks]
    path_only = [tokenize(c.display_path) for c in chunks]
    bm25, tfidf, struct = BM25(tokenized), TfidfCosine(tokenized), structural_scores(path_only)

    by_id = {c.chunk_id: i for i, c in enumerate(chunks)}
    golds, bm25_orders, tfidf_orders, struct_orders, fused_orders = {}, {}, {}, {}, {}

    for q in labels:
        gold_id = f"{q['doc_id']}::{'/'.join(q['path'])}"
        if gold_id not in by_id:
            continue
        gold = by_id[gold_id]
        golds[q["query_id"]] = gold
        o_bm = ranks_from_scores(bm25.scores(q["query"]))
        o_tf = ranks_from_scores(tfidf.scores(q["query"]))
        o_st = ranks_from_scores(struct.scores(q["query"]))
        bm25_orders[q["query_id"]] = o_bm
        tfidf_orders[q["query_id"]] = o_tf
        struct_orders[q["query_id"]] = o_st
        fused = reciprocal_rank_fusion([o_bm, o_tf, o_st], weights=[1.0, 1.0, 0.6])
        fused_orders[q["query_id"]] = ranks_from_scores(fused)

    return {
        "bm25": evaluate(bm25_orders, golds, k),
        "tfidf": evaluate(tfidf_orders, golds, k),
        "structure": evaluate(struct_orders, golds, k),
        "fused": evaluate(fused_orders, golds, k),
        "meta": {"documents": len(docs), "chunks": len(chunks), "queries": len(golds), "k": k},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid retrieval with an evaluation harness")
    ap.add_argument("command", choices=["eval"], nargs="?", default="eval")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--docs", type=int, default=30)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    res = run(k=args.k, seed=args.seed, n_docs=args.docs, verbose=args.verbose)
    m = res["meta"]
    print(f"\n{m['documents']} documents | {m['chunks']} chunks | {m['queries']} graded queries | k={m['k']}\n")
    print(f"{'route':<10}{'recall@k':>10}{'mrr':>10}{'ndcg@k':>10}")
    print("-" * 40)
    for name in ("bm25", "tfidf", "structure", "fused"):
        r = res[name]
        print(f"{name:<10}{r['recall@k']:>10.3f}{r['mrr']:>10.3f}{r['ndcg@k']:>10.3f}")


if __name__ == "__main__":
    main()
