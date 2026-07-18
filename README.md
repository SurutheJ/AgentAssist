# AgentAssist

An AI-powered knowledge base search tool built for IT service desk agents. AgentAssist lets agents describe a caller's problem in plain English and instantly surfaces the most relevant KB articles — with an LLM-generated answer and a ranked list of related links.

Built as a local-first RAG (Retrieval-Augmented Generation) prototype using open-source models, with no cloud dependencies.

---

## Demo

> Agent types: *"caller forgot their password and can't log in"*

AgentAssist returns a direct answer synthesized from the top KB articles, alongside 20 ranked related articles the agent can reference — all in under 1 second for search results, with the LLM answer streaming in progressively.

![AgentAssist UI](https://raw.githubusercontent.com/SurutheJ/AgentAssist/main/assets/demo.png)

---

## Features

- **Hybrid Search** — combines semantic vector search (dense) with BM25 keyword search (sparse), fused via Reciprocal Rank Fusion (RRF) for better relevance than either method alone
- **Streaming LLM answers** — answer tokens stream progressively into the UI so agents don't wait for the full response
- **Ranked article panel** — 20 related articles ranked by RRF score appear immediately (~0.6s) while the LLM answer generates
- **5,000+ article KB** — scraped and embedded 5,071 articles from the University of Illinois KB (answers.uillinois.edu)
- **Resume-capable pipeline** — scraping and ingestion are both checkpoint-based; re-running resumes from where it left off
- **Fully local** — no data leaves your machine; runs on Ollama (llama3.1:8b + nomic-embed-text) and ChromaDB

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

---

## Tech Stack

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

## How It Works

### Pipeline (one-time setup)

```
scrape.py  →  scraped/*.json  →  ingest.py  →  ChromaDB (kb_store/)
```

1. **scrape.py** — downloads the KB sitemap, scrapes each article, saves `{url, title, text}` JSON files. Anti-blocking measures: random delays, rotating User-Agents, session cookies, retry backoff. Resume-capable via checkpoint.

2. **ingest.py** — reads the JSON files, cleans text (strips nav/footer noise, keeps Keywords section), chunks into 500-word overlapping segments, embeds with nomic-embed-text, stores in ChromaDB. Resume-capable via checkpoint.

### Query (real-time)

```
query → embed → ChromaDB top-20 + BM25 top-20 → RRF → top-3 to LLM + top-20 to UI
```

1. Query is embedded with nomic-embed-text
2. ChromaDB returns top-20 semantically similar chunks
3. BM25 (cached in memory) returns top-20 keyword-matched chunks
4. RRF blends both rankings into a single score
5. Top-20 unique articles render immediately in the right panel (~0.6s)
6. Top-3 chunks are sent to llama3.1:8b to generate a direct answer (streaming)

---

## Running Locally

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

## Project Structure

```
AgentAssist/
├── app.py          # Streamlit UI
├── search.py       # Hybrid search + RAG answer generation
├── ingest.py       # Chunking, embedding, ChromaDB ingestion
├── scrape.py       # KB sitemap scraping pipeline
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

- **Two-phase pipeline**: `scrape.py` hits the server once and saves raw JSON; `ingest.py` reads from disk with zero server calls. Re-embedding with a different model or chunk size only requires re-running `ingest.py`.
- **Hybrid search over pure semantic**: BM25 catches exact IT terminology (error codes, product names, NetIDs) that semantic search can miss.
- **Articles-first UX**: ranked links appear at ~0.6s (after search); LLM answer streams in separately so agents aren't blocked waiting for generation.
- **BM25 cached at module load**: index is built once when the app starts and reused for every query — no per-query rebuild overhead.
