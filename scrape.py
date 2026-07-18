# scrape.py
# KBSeek AI — Phase 1: Sitemap → Scrape → Save to disk
# ──────────────────────────────────────────────────────
# What this file does:
#   1. Downloads the sitemap (or reuses cached sitemap.xml)
#   2. Parses all URLs and shows a namespace breakdown
#   3. Scrapes each URL one at a time with human-like delays
#   4. Saves each article as a JSON file in the scraped/ folder
#   5. Skips already-scraped articles on resume (no re-downloading)
#   6. Stops early if 10 consecutive 403s — IP is likely blocked
#   7. Saves crawled_urls.txt and blocked_urls.txt
#
# Anti-blocking measures built in:
#   - 1 worker (fully sequential — no parallel requests)
#   - Random delay between requests (looks human, not robotic)
#   - Rotates User-Agent across 4 real browser strings
#   - Full browser headers (Accept, Accept-Language, Connection, etc.)
#   - Session with cookies (carries session state like a real browser)
#   - Referer header set to previous page (simulates natural browsing)
#   - Retries on connection errors with exponential backoff
#   - Stops immediately on 10 consecutive 403s to avoid escalation
#
# After this runs, execute:  python ingest.py
# ──────────────────────────────────────────────────────

import os
import re
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from collections import Counter

# ── CONFIGURATION ──────────────────────────────────────

# All paths are relative to this script's location, not the shell's
# working directory. This ensures files are always saved in the project
# folder regardless of where the script is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SITEMAP_URL  = "https://answers.uillinois.edu/sitemap.xml"
SITEMAP_FILE = os.path.join(BASE_DIR, "sitemap.xml")
SCRAPED_DIR  = os.path.join(BASE_DIR, "scraped")
CRAWLED_FILE = os.path.join(BASE_DIR, "crawled_urls.txt")
BLOCKED_FILE = os.path.join(BASE_DIR, "blocked_urls.txt")

MIN_DELAY       = 2    # minimum seconds to wait between requests
MAX_DELAY       = 5    # maximum seconds — actual wait is random in this range
MAX_CONSECUTIVE = 10   # stop if this many 403s in a row
MAX_RETRIES     = 3    # retries on connection errors (not 403s)

# Four real browser User-Agent strings — rotated randomly per request
# so every request doesn't look identical to the server
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]

# Base headers sent with every request — looks like a real browser visit
BASE_HEADERS = {
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Cache-Control":   "max-age=0",
    "DNT":             "1",   # Do Not Track — some sites treat this as a trust signal
}

# ── HELPERS ────────────────────────────────────────────

def separator(title=""):
    if title:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
    else:
        print(f"{'─' * 60}")

def get_namespace(url):
    m = re.match(r"https://answers\.uillinois\.edu/([^/]+)/", url)
    return m.group(1) if m else "(root)"

def url_to_filename(url):
    """Converts a URL to a readable filename.
    e.g. https://answers.uillinois.edu/illinois/49273  →  illinois_49273.json
    """
    path = url.replace("https://answers.uillinois.edu/", "")
    safe = path.strip("/").replace("/", "_")
    return f"{safe}.json"

def random_delay():
    """Waits a random amount of time between MIN_DELAY and MAX_DELAY seconds.
    Random delays look more human than a fixed interval.
    """
    wait = random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(wait)

# ── STEP 1: SITEMAP ────────────────────────────────────

separator("STEP 1 — Sitemap")

if os.path.exists(SITEMAP_FILE):
    print(f"Using cached {SITEMAP_FILE}  (delete it to re-download)")
    with open(SITEMAP_FILE, encoding="utf-8") as f:
        sitemap_text = f.read()
else:
    print(f"Downloading sitemap from {SITEMAP_URL} ...")
    r = requests.get(SITEMAP_URL, headers={**BASE_HEADERS, "User-Agent": USER_AGENTS[0]}, timeout=30)
    if r.status_code != 200:
        print(f"❌ Failed to download sitemap (HTTP {r.status_code}). Exiting.")
        exit(1)
    sitemap_text = r.text
    with open(SITEMAP_FILE, "w", encoding="utf-8") as f:
        f.write(sitemap_text)
    print(f"✅ Saved to {SITEMAP_FILE}  ({len(sitemap_text)/1024:.0f} KB)")

# ── STEP 2: PARSE URLs ─────────────────────────────────

separator("STEP 2 — Parse sitemap URLs")

ns_xml   = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
root     = ET.fromstring(sitemap_text)
all_urls = [loc.text.strip() for loc in root.findall("sm:url/sm:loc", ns_xml)]

print(f"Total URLs in sitemap: {len(all_urls)}")

ns_total = Counter(get_namespace(u) for u in all_urls)
print(f"\n  {'Namespace':<30}  {'URLs':>6}")
print(f"  {'─'*30}  {'─'*6}")
for ns, count in sorted(ns_total.items(), key=lambda x: -x[1]):
    print(f"  /{ns:<29}  {count:>6}")
print(f"\n  {'TOTAL':<30}  {len(all_urls):>6}")

# Deduplicate — preserve sitemap order, keep first occurrence
seen        = set()
unique_urls = []
for u in all_urls:
    if u not in seen:
        seen.add(u)
        unique_urls.append(u)

dupes = len(all_urls) - len(unique_urls)
print(f"\nDuplicates removed: {dupes}  →  {len(unique_urls)} unique URLs to scrape")

# ── STEP 3: SCRAPE ─────────────────────────────────────

separator("STEP 3 — Scrape articles")

os.makedirs(SCRAPED_DIR, exist_ok=True)

# Resume: each saved file in scraped/ is its own checkpoint
already_done = sum(
    1 for u in unique_urls
    if os.path.exists(os.path.join(SCRAPED_DIR, url_to_filename(u)))
)
remaining = [
    u for u in unique_urls
    if not os.path.exists(os.path.join(SCRAPED_DIR, url_to_filename(u)))
]

if already_done:
    print(f"Resuming — {already_done} articles already scraped, {len(remaining)} to go.\n")
else:
    print(f"Starting fresh — {len(remaining)} URLs to scrape.\n")

est_minutes = len(remaining) * ((MIN_DELAY + MAX_DELAY) / 2) / 60
print(f"Delay between requests: {MIN_DELAY}–{MAX_DELAY}s (random)")
print(f"Estimated time:         ~{est_minutes:.0f} minutes\n")

# Use a session — carries cookies across requests like a real browser would
session = requests.Session()
prev_url = "https://answers.uillinois.edu/"   # starting referer

scraped_count   = 0
blocked         = {}
consecutive_403 = 0
stopped_early   = False

for i, url in enumerate(remaining):
    # Rotate User-Agent and update Referer on every request
    session.headers.update({
        **BASE_HEADERS,
        "User-Agent": random.choice(USER_AGENTS),
        "Referer":    prev_url,
    })

    print(f"  [{i+1:>5}/{len(remaining)}] {url}", flush=True)

    # Retry loop — handles transient connection errors, NOT 403s
    status = None
    response = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=15, allow_redirects=True)
            status   = response.status_code
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                backoff = 5 * (2 ** attempt)   # 5s, 10s, 20s
                print(f"           ⚠️  Connection error ({e}) — retrying in {backoff}s")
                time.sleep(backoff)
            else:
                print(f"           ❌ Failed after {MAX_RETRIES} attempts — skipping")
                blocked[url] = 0

    if status is None:
        random_delay()
        continue

    # Handle 403 — potential IP block
    if status == 403:
        print(f"           🚫 403 Forbidden")
        blocked[url] = 403
        consecutive_403 += 1
        if consecutive_403 >= MAX_CONSECUTIVE:
            print(f"\n🛑 Stopped — {MAX_CONSECUTIVE} consecutive 403 Forbidden responses.")
            print(f"   Your IP may be blocked. Wait and re-run — scraped articles are safe.")
            print(f"   Progress: {already_done + scraped_count} articles scraped so far.\n")
            stopped_early = True
            break
        random_delay()
        continue

    # Handle other non-200 responses
    if status != 200:
        print(f"           ⚠️  HTTP {status} — skipping")
        blocked[url] = status
        consecutive_403 = 0
        random_delay()
        continue

    # Success — reset consecutive counter and update referer
    consecutive_403 = 0
    prev_url = url

    # Parse the page
    soup      = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.find("title")
    title     = title_tag.get_text(strip=True) if title_tag else "Untitled"

    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    content = soup.find("div", {"id": "answer-content"}) or soup.find("body")

    if content:
        # Replace each <img> with its alt text so image descriptions
        # are included in the extracted text. Images with no alt text are removed.
        for img in content.find_all("img"):
            alt = img.get("alt", "").strip()
            if alt:
                img.replace_with(f"[Image: {alt}]")
            else:
                img.decompose()

    text = content.get_text(separator="\n", strip=True) if content else ""

    if not text.strip():
        print(f"           ⚠️  Empty content — skipping")
        blocked[url] = -2
        random_delay()
        continue

    # Save to scraped/ folder
    filepath = os.path.join(SCRAPED_DIR, url_to_filename(url))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"url": url, "title": title, "text": text}, f, ensure_ascii=False)

    print(f"           ✅ {title[:60]}  ({len(text.split())} words)")
    scraped_count += 1
    random_delay()

# ── STEP 4: SAVE OUTPUT FILES ──────────────────────────

separator("STEP 4 — Save output files")

# crawled_urls.txt — every URL that has a file in scraped/
crawled = sorted(
    u for u in unique_urls
    if os.path.exists(os.path.join(SCRAPED_DIR, url_to_filename(u)))
)
with open(CRAWLED_FILE, "w") as f:
    for u in crawled:
        f.write(u + "\n")
print(f"crawled_urls.txt  →  {len(crawled)} successfully scraped URLs")

# blocked_urls.txt — every URL not in scraped/
all_blocked = {u: blocked.get(u, -1) for u in unique_urls if u not in set(crawled)}
with open(BLOCKED_FILE, "w") as f:
    f.write(f"{'CODE':<6}  {'NAMESPACE':<30}  URL\n")
    f.write("─" * 90 + "\n")
    for u, code in sorted(all_blocked.items(), key=lambda x: (get_namespace(x[0]), x[0])):
        ns = get_namespace(u)
        f.write(f"{str(code):<6}  /{ns:<29}  {u}\n")
print(f"blocked_urls.txt  →  {len(all_blocked)} blocked/skipped URLs")

# ── STEP 5: SUMMARY ────────────────────────────────────

separator("STEP 5 — Summary")

ns_scraped = Counter(get_namespace(u) for u in crawled)
ns_blocked = Counter(get_namespace(u) for u in all_blocked)

print(f"  {'Namespace':<30}  {'Total':>6}  {'Scraped':>7}  {'Blocked':>7}  {'%':>6}")
print(f"  {'─'*30}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}")
for ns in sorted(ns_total, key=lambda x: -ns_total[x]):
    tot = ns_total[ns]
    sc  = ns_scraped.get(ns, 0)
    bl  = ns_blocked.get(ns, 0)
    pct = sc / tot * 100 if tot else 0
    print(f"  /{ns:<29}  {tot:>6}  {sc:>7}  {bl:>7}  {pct:>5.1f}%")

separator()
if stopped_early:
    print(f"  ⚠️  INCOMPLETE — stopped early due to IP block.")
    print(f"  scraped/ folder   →  {len(crawled)} articles saved so far")
    print(f"  Re-run this script when unblocked — it will resume automatically.")
else:
    print(f"  ✅ All done!")
    print(f"  scraped/ folder   →  {len(crawled)} articles ready for ingestion")
    print(f"  crawled_urls.txt  →  {len(crawled)} URLs")
    print(f"  blocked_urls.txt  →  {len(all_blocked)} URLs")
    print(f"\n  Next step:  python ingest.py")
separator()
