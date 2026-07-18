# AgentAssist

**An AI answer engine for IT service desk agents — built end-to-end as a product case study in retrieval quality, latency budgets, and scope discipline.**

An agent on a live call types the caller's problem in plain English ("caller forgot their password and can't log in"). AgentAssist returns a synthesized answer from the knowledge base plus 20 ranked source articles — ranked results in ~0.6s, answer streaming in immediately after.

![AgentAssist UI](https://raw.githubusercontent.com/SurutheJ/AgentAssist/main/assets/demo.png)

---

## The problem

Service desk agents lose time on every call to the same failure mode: the KB has the answer, but keyword search can't find it because callers describe *symptoms* ("I'm locked out") while articles are written in *IT vocabulary* ("NetID credential reset"). Agents fall back on asking a senior colleague or escalating — both expensive.

The product bet: if the tool can bridge natural-language symptoms to KB articles reliably, and do it *within the rhythm of a live call*, agents will trust it over tribal knowledge. That framing drove every decision below — this was scoped as a retrieval-quality problem with a hard latency budget, not a chatbot problem.

**Corpus:** 5,071 articles scraped from the University of Illinois public KB (answers.uillinois.edu) as a realistic stand-in for an enterprise service desk KB.

---

## Product decisions and tradeoffs

These are the calls I made, what I considered, and what I knowingly gave up. This section is the point of the project.

### 1. Hybrid retrieval, not semantic-only

**Options:** pure vector search · pure BM25 keyword search · hybrid with rank fusion.

Pure semantic search demoed well on conversational queries but whiffed on exact IT tokens — error codes, product names, "NetID" — the exact terms agents type under pressure. Pure BM25 had the inverse failure: no match between "locked out" and "credential reset." I shipped both, blended with Reciprocal Rank Fusion (RRF), which rewards documents that rank well in *either* system without needing to calibrate their incomparable raw scores.

**Cost accepted:** two retrieval paths to maintain and an in-memory BM25 index (built once at startup, cached — a per-query rebuild added 1–2s and was unacceptable against the latency budget).

### 2. Articles-first UX: don't make the agent wait on the LLM

**Options:** show nothing until the full answer is ready · stream everything at once · results-first, answer second.

The LLM is the slowest component. Blocking the whole screen on generation would make the tool feel slower than the old search box — a trust-killer on a live call. So the UI renders the 20 ranked articles at ~0.6s (search only), and the LLM answer streams into a separate panel afterward. An experienced agent often recognizes the right article from the title alone and never needs the answer; the synthesis is a bonus, not a gate.

**Cost accepted:** a two-panel layout that's busier than a clean chat interface. Chosen deliberately: the ranked list is the *trust surface* (agents can verify sources), and the answer is the *speed surface*.

### 3. Fully local stack (Ollama + ChromaDB), not cloud APIs

**Options:** OpenAI/Anthropic APIs · cloud vector DB · fully local open-source.

Service desk queries contain caller PII, and enterprise IT is exactly the buyer who will veto "sends data to a third-party API" in the first security review. Running llama3.1:8b and nomic-embed-text locally via Ollama makes the privacy story trivially defensible and the demo cost $0. 

**Cost accepted:** an 8B local model is measurably weaker at synthesis than frontier APIs, and answer generation is slower on modest hardware. For a prototype whose core value is *retrieval*, that was the right trade — the LLM layer is swappable in one function (`generate_answer`), so the decision is reversible if the deployment context allows cloud.

### 4. Two-phase pipeline: crawl once, iterate offline forever

**Options:** crawl and embed in one pass · fetch pages live at query time · decouple scrape from ingest.

`scrape.py` hits the KB server exactly once and saves raw JSON to disk; `ingest.py` chunks, embeds, and indexes from disk with zero network calls. This meant I could re-run ingestion with different chunk sizes and embedding models repeatedly while iterating on retrieval quality — without re-crawling 5,000 pages or burdening a public university server. Both phases checkpoint and resume, because a 5,000-article crawl *will* get interrupted.

**Explicitly rejected:** live fetching at query time. Crawling is an offline freshness problem, not a query-time problem — putting network I/O in the query path would blow the sub-second budget for zero relevance gain. (Freshness belongs in a scheduled incremental re-crawl; see roadmap.)

**Cost accepted:** the index goes stale between crawls. For an IT KB where articles change weekly, not hourly, that's acceptable.

### 5. Respectful scraping over fast scraping

The crawler is deliberately slow: fully sequential, 2–5s randomized delays, retry backoff, and a hard stop after 10 consecutive 403s. A parallel crawler would have finished in a fraction of the time — and risked hammering a public university service. Being a good citizen of someone else's infrastructure was a constraint, not an afterthought.

### 6. Top-3 chunks to the LLM, top-20 to the human

More context to the LLM sounds better but isn't free: it slows generation, and an 8B model gets *less* accurate as marginally-relevant context dilutes the prompt. Three chunks keeps generation fast and grounded; the other 17 results go where extra recall is actually useful — in front of a human who can scan titles in seconds. The prompt is defensive by design: answer only from provided articles, cite which one, never guess.

---

## Architecture

```
Caller's problem (natural language query)
        │
        ▼
┌─────────────────────────────────────────┐
│              Hybrid Search              │
│                                         │
│  ┌─────────────────┐  ┌──────────────┐  │
│  │  Semantic Search│  │ BM25 Keyword │  │
│  │  (nomic-embed-  │  │    Search    │  │
│  │   text + Chroma)│  │  (in-memory) │  │
│  └────────┬────────┘  └──────┬───────┘  │
│           └────────┬─────────┘          │
│                    ▼                    │
│         Reciprocal Rank Fusion          │
│              (RRF blending)             │
└────────────────────┬────────────────────┘
                     │
           Top 20 ranked chunks
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   LLM Answer (top 3)    Ranked Article Panel
   llama3.1:8b via        (all 20, scrollable,
   Ollama (streaming)      shown immediately)
```

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Vector store | ChromaDB (local persistent) |
| Embeddings | nomic-embed-text via Ollama |
| LLM | llama3.1:8b via Ollama |
| Keyword search | BM25Okapi (rank-bm25) |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| Scraping | BeautifulSoup4 + requests |
| Language | Python 3.11+ |

---

## Known limitations (and what I'd fix next, in order)

Being honest about where the current version falls short matters more than the feature list:

1. **No retrieval eval harness yet.** Relevance is currently judged by eyeballing a fixed set of test queries (`search.py` ships with them). The single highest-leverage next step is a golden set of ~50 real agent queries with labeled correct articles, measuring recall@5 / recall@20 — so every retrieval change ships with a number, not a vibe.
2. **Naive BM25 tokenization.** Queries are tokenized with `lower().split()` — no punctuation stripping or stemming, so "login?" doesn't match "login". Cheap fix, likely a real recall win; it ships behind the eval harness so the gain is measurable.
3. **Single-shot, not conversational.** Every query is independent. The chat evolution (session memory + follow-up query rewriting) is the top roadmap item below.
4. **Titles aren't embedded with chunks.** Prepending the article title to each chunk before embedding is a known cheap retrieval improvement.
5. **No reranker.** A cross-encoder rerank of the top ~50 candidates would trade ~200ms for meaningfully better top-3 precision — worth it, since top-3 feeds the LLM.

---

## Roadmap: from search tool to agent chatbot

- **Chat mode** — multi-turn conversation with session memory; follow-up questions get rewritten into standalone queries (using chat history) before hitting retrieval, so "what about for students?" retrieves correctly.
- **Retrieval eval harness** — golden query set, recall@k tracking, before/after on every change.
- **Feedback loop** — thumbs up/down on answers, which doubles as free labeled eval data.
- **Scheduled incremental re-crawl** — freshness via a nightly/weekly background job diffing the sitemap, never via query-time fetching.
- **Workflow embedding** — surface inside the ticketing tool or Slack, where agents already live, instead of another tab.

---

## Running locally

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Setup

```bash
git clone https://github.com/SurutheJ/AgentAssist.git
cd AgentAssist
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Build the knowledge base

```bash
# Step 1: Scrape articles (hits the KB website, saves JSON files)
python scrape.py

# Step 2: Embed and index (reads JSON files, zero server calls)
python ingest.py
```

### Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Project structure

```
AgentAssist/
├── app.py          # Streamlit UI
├── search.py       # Hybrid search + RAG answer generation
├── ingest.py       # Chunking, embedding, ChromaDB ingestion
├── scrape.py       # KB sitemap scraping pipeline
├── requirements.txt
└── README.md
```
