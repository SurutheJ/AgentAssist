# search.py
# ─────────────────────────────────────────────────────
# KBSeek AI — Hybrid Search + RAG Answer
# What this file does:
#   1. Runs SEMANTIC search (vector similarity via ChromaDB)
#   2. Runs KEYWORD search (BM25 exact matching)
#   3. Blends both rankings using Reciprocal Rank Fusion (RRF)
#   4. Sends top results to LLM for RAG answer
#
# WHY HYBRID:
#   Semantic = great for intent ("can't log in")
#   Keyword  = great for exact terms ("error 0x80070005")
#   Hybrid   = best of both worlds
# ─────────────────────────────────────────────────────

import os
import chromadb
import ollama
from rank_bm25 import BM25Okapi  # BM25 keyword search library

# ── 1. CONNECT TO CHROMADB ───────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
client = chromadb.PersistentClient(path=os.path.join(_BASE_DIR, "kb_store"))
collection = client.get_or_create_collection(
    name="kb_articles",
    metadata={"hnsw:space": "cosine"}
)

# ── 2. LOAD AND CACHE ALL CHUNKS ─────────────────────
# BM25 needs all chunks in memory. Loading from ChromaDB and
# building the index on every query adds 1-2 seconds each time.
# We load once at startup and cache — subsequent queries are instant.

_cache = None

def get_index():
    """Loads chunks from ChromaDB and builds BM25 index once.
    Returns (all_docs, all_metas, all_ids, bm25, id_to_idx).
    id_to_idx is a dict for O(1) chunk lookups instead of O(n) list scans.
    """
    global _cache
    if _cache is None:
        results   = collection.get(include=["documents", "metadatas"])
        all_docs  = results["documents"]
        all_metas = results["metadatas"]
        all_ids   = results["ids"]
        tokenised = [doc.lower().split() for doc in all_docs]
        bm25      = BM25Okapi(tokenised)
        id_to_idx = {chunk_id: i for i, chunk_id in enumerate(all_ids)}
        _cache    = (all_docs, all_metas, all_ids, bm25, id_to_idx)
    return _cache

# ── 4. SEMANTIC SEARCH ───────────────────────────────
# Same as before — embed query, find nearest vectors.
# AI Jargon: dense retrieval

def semantic_search(query, top_k=10):
    """
    We fetch top 10 here (not 3) because we're going to
    re-rank with hybrid fusion, so we need a wider candidate pool.
    AI Jargon: candidate retrieval / first-stage retrieval
    """
    response = ollama.embed(model="nomic-embed-text", input=query)
    query_embedding = response["embeddings"][0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    
    return results

# ── 5. KEYWORD SEARCH (BM25) ─────────────────────────
# AI Jargon: sparse retrieval / lexical matching

def keyword_search(query, documents, bm25_index, top_k=10):
    """
    BM25 scores every document against the query keywords.
    Higher score = more keyword overlap.
    Returns indices of top_k documents sorted by score.
    """
    tokenised_query = query.lower().split()
    scores = bm25_index.get_scores(tokenised_query)
    
    # Get indices sorted by score (highest first)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]
    
    return ranked_indices, scores

# ── 6. RECIPROCAL RANK FUSION (RRF) ──────────────────
# This is the blending step — combines semantic and keyword rankings.
#
# HOW RRF WORKS:
# Each result gets a score based on its RANK position, not raw score.
# Formula: 1 / (rank + 60)   ← the 60 is a smoothing constant
# A result ranked 1st gets: 1/(1+60) = 0.016
# A result ranked 2nd gets: 1/(2+60) = 0.016
# If a chunk ranks high in BOTH methods, it gets double the score.
# AI Jargon: rank fusion / ensemble retrieval

def reciprocal_rank_fusion(semantic_results, keyword_indices, 
                            all_ids, k=60):
    """
    semantic_results: ChromaDB query results
    keyword_indices:  BM25 ranked indices into all_chunks
    all_ids:          all chunk IDs from ChromaDB
    k:                smoothing constant (60 is standard)
    """
    rrf_scores = {}
    
    # Score from semantic search
    semantic_ids = semantic_results["ids"][0]
    for rank, chunk_id in enumerate(semantic_ids):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + \
                                1 / (rank + 1 + k)
    
    # Score from keyword search
    for rank, idx in enumerate(keyword_indices):
        if idx < len(all_ids):
            chunk_id = all_ids[idx]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + \
                                    1 / (rank + 1 + k)
    
    # Sort by combined RRF score (highest = best match)
    sorted_ids = sorted(rrf_scores.items(), 
                       key=lambda x: x[1], 
                       reverse=True)
    
    return sorted_ids

# ── 7. HYBRID SEARCH ─────────────────────────────────
# Orchestrates semantic + keyword + fusion

def hybrid_search(query, top_k=3):
    """
    Full hybrid search pipeline:
    1. Semantic search → top 20 candidates
    2. Keyword search  → top 20 candidates
    3. RRF fusion      → blended ranking
    4. Return top_k    → for RAG
    """
    # Use cached index — no rebuild on repeated queries
    all_docs, all_metas, all_ids, bm25, id_to_idx = get_index()

    # Run both searches with a wider candidate pool so deduplication
    # never leaves us short of 3 distinct articles
    semantic_results = semantic_search(query, top_k=20)
    keyword_indices, _ = keyword_search(query, all_docs, bm25, top_k=20)

    # Fuse rankings
    fused_ranking = reciprocal_rank_fusion(
        semantic_results, keyword_indices, all_ids
    )

    # Get top_k results — one per article URL, O(1) dict lookup
    top_results = []
    seen_urls   = set()

    for chunk_id, rrf_score in fused_ranking:
        if len(top_results) >= top_k:
            break
        idx = id_to_idx.get(chunk_id)
        if idx is None:
            continue
        url = all_metas[idx].get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        top_results.append({
            "document": all_docs[idx],
            "metadata": all_metas[idx],
            "rrf_score": round(rrf_score, 4),
            "id": chunk_id
        })

    return top_results

# ── 8. GENERATE ANSWER WITH RAG ──────────────────────

def generate_answer(query, top_results):
    
    context_parts = []
    sources = []
    
    for i, result in enumerate(top_results):
        context_parts.append(
            f"Article {i+1}: {result['metadata']['title']}\n"
            f"{result['document']}"
        )
        sources.append({
            "title": result["metadata"]["title"],
            "url": result["metadata"]["url"],
            "rrf_score": result["rrf_score"]
        })
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""You are a helpful IT service desk assistant.
A colleague is on a live call with a customer and needs a quick answer.

Use ONLY the information in the articles below to answer the question.
Be concise — the agent is on a live call and needs a fast answer.
If the articles contain partial information, use what is there.
Only say you couldn't find it if the articles are completely unrelated.
Never guess facts not in the articles. Cite which article helped.

ARTICLES:
{context}

AGENT'S QUESTION: {query}

ANSWER:"""

    stream = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    def token_stream():
        for chunk in stream:
            yield chunk["message"]["content"]

    return token_stream(), sources

# ── 9. MAIN FUNCTION ─────────────────────────────────

def ask_kbseek(query):
    print(f"\n🔍 Query: {query}\n")
    
    # Hybrid search
    top_results = hybrid_search(query, top_k=3)
    
    # Generate answer
    answer, sources = generate_answer(query, top_results)
    
    print("💡 ANSWER:")
    print(answer)
    print("\n📚 SOURCES (hybrid ranked):")
    for i, source in enumerate(sources):
        print(f"  {i+1}. {source['title']}")
        print(f"     RRF Score: {source['rrf_score']}")
        print(f"     URL: {source['url']}")
    
    return answer, sources

# ── 10. TEST WITH SAME QUERIES AS BEFORE ─────────────
# We use identical queries so we can compare
# semantic-only vs hybrid results side by side.
# AI Jargon: controlled evaluation / A-B comparison

if __name__ == "__main__":
    
    test_queries = [
        "caller forgot their password and cant log in",
        "user doesn't know what their NetID is",
        "getting a message that password has expired",
        "how do i set up recovery options for my account",
        "student is graduating and wants to keep their email"
    ]
    
    for query in test_queries:
        ask_kbseek(query)
        print("\n" + "="*60 + "\n")