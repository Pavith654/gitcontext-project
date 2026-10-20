"""Phase 3: hybrid search (vector + BM25) with RRF and optional reranking."""
import re

from rank_bm25 import BM25Okapi

from .ingest import client, model

_bm25_cache = {}
_reranker = None


def _tok(t):
    return re.findall(r"\w+", t.lower())


def _bm25(name, col):
    if name not in _bm25_cache:
        data = col.get()
        _bm25_cache[name] = (BM25Okapi([_tok(d) for d in data["documents"]]), data)
    return _bm25_cache[name]


def clear_cache(name):
    _bm25_cache.pop(name, None)


def hybrid_search(name, question, k=6, rerank=False, pool=20):
    col = client.get_collection(name)
    vec = col.query(query_embeddings=model.encode([question]).tolist(), n_results=pool)
    vec_ids = vec["ids"][0]

    bm, data = _bm25(name, col)
    scores = bm.get_scores(_tok(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool]
    kw_ids = [data["ids"][i] for i in top if scores[i] > 0]

    rrf = {}  # reciprocal rank fusion
    for ranking in (vec_ids, kw_ids):
        for rank, cid in enumerate(ranking):
            rrf[cid] = rrf.get(cid, 0) + 1 / (60 + rank)
    best = sorted(rrf, key=rrf.get, reverse=True)[:pool]

    lookup = {data["ids"][i]: i for i in range(len(data["ids"]))}
    results = [{"id": cid, "text": data["documents"][lookup[cid]],
                "meta": data["metadatas"][lookup[cid]]} for cid in best]

    if rerank:
        global _reranker
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        s = _reranker.predict([(question, r["text"]) for r in results])
        results = [r for _, r in sorted(zip(s, results), key=lambda x: x[0], reverse=True)]
    return results[:k]
