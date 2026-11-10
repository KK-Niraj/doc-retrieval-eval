import math

from docret.chunking import chunk_documents
from docret.corpus import build
from docret.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank
from docret.retrieval import BM25, TfidfCosine, ranks_from_scores, reciprocal_rank_fusion, tokenize


def test_chunks_carry_full_heading_path():
    docs, _ = build(seed=1, n_docs=2)
    by_id = {d.doc_id: d for d in docs}
    chunks = chunk_documents(docs)
    assert chunks, "expected chunks"
    for c in chunks:
        assert len(c.path) >= 2, c
        assert c.path[0] == by_id[c.doc_id].title
    deep = [c for c in chunks if len(c.path) == 3]
    assert deep, "expected nested sections to survive chunking"


def test_bm25_prefers_exact_term():
    docs = [tokenize("the retention period is 90 days"),
            tokenize("the encryption standard is 256 bits"),
            tokenize("governance board approval is required")]
    r = BM25(docs).scores("retention period")
    assert int(r.argmax()) == 0


def test_rrf_rewards_agreement_between_routes():
    a = [0, 1, 2]
    b = [0, 2, 1]
    fused = reciprocal_rank_fusion([a, b])
    assert int(fused.argmax()) == 0
    both_second = fused[1] + fused[2]
    assert both_second < 2 * fused[0]


def test_rrf_weights_shift_the_winner():
    a, b = [0, 1], [1, 0]
    assert int(reciprocal_rank_fusion([a, b], weights=[1.0, 0.0]).argmax()) == 0
    assert int(reciprocal_rank_fusion([a, b], weights=[0.0, 1.0]).argmax()) == 1


def test_metrics_are_computed_correctly():
    ranked = [5, 3, 9]
    assert recall_at_k(ranked, 3, k=2) == 1.0
    assert recall_at_k(ranked, 9, k=2) == 0.0
    assert reciprocal_rank(ranked, 3) == 0.5
    assert reciprocal_rank(ranked, 99) == 0.0
    assert ndcg_at_k(ranked, 5, k=3) == 1.0
    assert math.isclose(ndcg_at_k(ranked, 3, k=3), 1 / math.log2(3))


def test_ranks_are_deterministic_on_ties():
    import numpy as np
    s = np.array([1.0, 1.0, 1.0])
    assert ranks_from_scores(s) == [0, 1, 2]
