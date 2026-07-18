# ingest.py
# KBSeek AI — Phase 2: scraped/ folder → ChromaDB
# ──────────────────────────────────────────────────
# Reads pre-scraped JSON files from the scraped/ folder.
# Makes ZERO server calls — safe to re-run at any time.
#
# Supports resume: ingest_checkpoint.json tracks which files
# are already embedded. Re-running skips completed articles.
# Run again after scrape.py adds more articles — only new
# ones will be processed.
#
# Change CHUNK_SIZE or EMBED_MODEL and set CLEAR_BEFORE_INGEST = True
# to rebuild the index cleanly with the new settings.
#
# Run with:  python ingest.py
# ──────────────────────────────────────────────────

import os
import json
import hashlib
import chromadb
import ollama

# ── CONFIGURATION ──────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SCRAPED_DIR   = os.path.join(BASE_DIR, "scraped")
KB_STORE      = os.path.join(BASE_DIR, "kb_store")
CHECKPOINT    = os.path.join(BASE_DIR, "ingest_checkpoint.json")

CHUNK_SIZE    = 500   # words per chunk — lower = more precise, higher = more context
CHUNK_OVERLAP = 50    # words repeated between consecutive chunks
EMBED_MODEL   = "nomic-embed-text"   # Ollama embedding model

# Set True when changing CHUNK_SIZE or EMBED_MODEL.
# Clears the existing ChromaDB index and checkpoint before rebuilding.
# Leave False for a normal run (adds new articles, skips already-done ones).
CLEAR_BEFORE_INGEST = False

# ── CHROMADB SETUP ─────────────────────────────────────

client = chromadb.PersistentClient(path=KB_STORE)

if CLEAR_BEFORE_INGEST:
    try:
        client.delete_collection("kb_articles")
        print("🗑️  Cleared existing ChromaDB collection.")
    except Exception:
        pass
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print("🗑️  Cleared ingest checkpoint.\n")

collection = client.get_or_create_collection(
    name="kb_articles",
    metadata={"hnsw:space": "cosine"}
)

# ── HELPERS ────────────────────────────────────────────

def clean_text(text):
    """Strips navigation noise and footer metadata from scraped article text.
    Removes the 'Skip to the main content' nav link at the top, and cuts
    everything from 'Keywords:' downward (Doc ID, Owned by, dates, etc.).
    """
    # Remove top nav line
    text = text.replace("Skip to the main content", "").strip()

    # Cut footer boilerplate — keep Keywords and their values but drop everything
    # below them (Doc ID, Owned by, dates, buttons, etc.)
    for marker in ["Suggest keywords", "Doc ID:", "Owned by:"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
            break

    return text

def chunk_text(text):
    """Splits text into overlapping word-count chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start : start + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_text(text):
    """Returns the embedding vector for a text string.
    Truncates to 6000 characters to stay within nomic-embed-text's context window.
    """
    text = text[:6000]
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]

# ── MAIN ───────────────────────────────────────────────

def ingest():
    if not os.path.exists(SCRAPED_DIR):
        print(f"❌ '{SCRAPED_DIR}/' folder not found.")
        print(f"   Run  python scrape.py  first.\n")
        exit(1)

    all_files = sorted(f for f in os.listdir(SCRAPED_DIR) if f.endswith(".json"))

    if not all_files:
        print(f"❌ No scraped articles found in '{SCRAPED_DIR}/'.")
        print(f"   Run  python scrape.py  first.\n")
        exit(1)

    # Load checkpoint — skip files already embedded
    try:
        with open(CHECKPOINT) as f:
            done = set(json.load(f))
        print(f"♻️  Checkpoint found — {len(done)} articles already ingested, resuming.\n")
    except FileNotFoundError:
        done = set()

    remaining = [f for f in all_files if f not in done]

    print(f"🚀 KBSeek AI — Ingestion pipeline")
    print(f"   Total articles:  {len(all_files)}")
    print(f"   Already done:    {len(done)}")
    print(f"   To process:      {len(remaining)}")
    print(f"   Chunk size:      {CHUNK_SIZE} words  (overlap: {CHUNK_OVERLAP})")
    print(f"   Model:           {EMBED_MODEL}\n")

    total_chunks = 0

    for i, filename in enumerate(remaining):
        filepath = os.path.join(SCRAPED_DIR, filename)

        with open(filepath, encoding="utf-8") as f:
            article = json.load(f)

        url   = article["url"]
        title = article["title"]
        text  = clean_text(article["text"])

        chunks = chunk_text(text)
        print(f"  [{i+1:>5}/{len(remaining)}]  {title[:55]:<55}  ({len(chunks)} chunks)", flush=True)

        for j, chunk in enumerate(chunks):
            chunk_id  = hashlib.md5(f"{url}-{CHUNK_SIZE}-{j}".encode()).hexdigest()
            embedding = embed_text(chunk)
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"url": url, "title": title, "chunk_index": j}]
            )
            total_chunks += 1

        # Save checkpoint after each article
        done.add(filename)
        with open(CHECKPOINT, "w") as f:
            json.dump(list(done), f)

    print(f"\n✅ Ingestion complete — {total_chunks} new chunks indexed across {len(remaining)} articles")
    print(f"   Total articles in KB: {len(all_files)}")
    print(f"📦 Vector store: {KB_STORE}")
    print(f"\nNext step:  streamlit run app.py\n")


if __name__ == "__main__":
    ingest()
