'''
search_helper.py
Production-hardened Google CSE helper with caching, scraping, and embeddings.
Dependencies:
    pip install requests httpx[http2] trafilatura sentence-transformers tenacity
'''

from __future__ import annotations
import asyncio
import time
import json
import sqlite3
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import List, Dict
import requests
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from trafilatura import extract
from sentence_transformers import SentenceTransformer

# ─── Logging Configuration ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).with_name("search_cache.sqlite3")
CACHE_TTL_SECS = 24 * 3600  # 1 day
CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
BLOCKED_DOMAINS = {"linkedin.com"}

QUERY_TEMPLATES: List[str] = [
    "{name} recrutement OR 'recrute' OR 'offres d’emploi' OR 'jobs'",
    "{name} 'nouveau bureau' OR 'ouvre un bureau' OR 'implantation' OR 'locaux'",
    "{name} événement OR salon OR séminaire OR conférence OR webinaire",
    "{name} 'rebranding' OR 'nouvelle identité' OR 'nouvelle charte graphique'",
    "{name} 'levée de fonds' OR 'tour de table' OR investissement OR financement",
    "{name} actualités OR annonce OR nouveauté OR 'dernier communiqué'",
    "{name} chiffre d’affaires OR secteur OR clients OR 'cse'",
    "{name} site:linkedin.com/in intext:\"CSE\" OR \"Office manager\" OR \"Happiness\" OR \"Communication\" OR \"People\" OR \"Talent\" OR \"RH\"",
]

# ─── SQLite Cache Abstraction (Thread-safe) ───────────────────────────────
class SQLiteCache:
    def __init__(self, path: Path, table: str):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.table = table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                key TEXT PRIMARY KEY,
                js  TEXT,
                ts  REAL
            )
        """)

    def get(self, key: str):
        with self.lock:
            row = self.conn.execute(
                f"SELECT js, ts FROM {self.table} WHERE key=?", (key,)
            ).fetchone()
        if row and time.time() - row[1] < CACHE_TTL_SECS:
            return json.loads(row[0])
        return None

    def set(self, key: str, obj):
        with self.lock:
            self.conn.execute(
                f"INSERT OR REPLACE INTO {self.table} (key, js, ts) VALUES (?, ?, ?)",
                (key, json.dumps(obj), time.time())
            )
            self.conn.commit()

# Instantiate caches
search_cache = SQLiteCache(CACHE_PATH, "search_cache")
scrape_cache = SQLiteCache(CACHE_PATH, "scrape_cache")
embed_cache  = SQLiteCache(CACHE_PATH, "embed_cache")

# ─── HTTP Session & Async Client (Connection reuse) ───────────────────────
session = requests.Session()
session.headers.update(HEADERS)
async_client: httpx.AsyncClient | None = None

def get_async_client() -> httpx.AsyncClient:
    global async_client
    if async_client is None:
        async_client = httpx.AsyncClient(http2=True, headers=HEADERS, timeout=7)
    return async_client

# ─── Domain Helper ────────────────────────────────────────────────────────

def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.lower().lstrip("www.")

# ─── Embedding Utilities (Preloaded) ────────────────────────────────────
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> list[float]:
    return _model.encode(text, convert_to_numpy=False).tolist()

# ─── CSE Wrapper w/ Retries ───────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _cse_call(params: Dict) -> List[Dict]:
    """Call Google CSE with retries and log errors."""
    resp = session.get(CSE_ENDPOINT, params=params, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("items", [])
    try:
        error = resp.json().get("error", {}).get("message", resp.text)
    except Exception:
        error = resp.text
    logger.error(f"CSE {resp.status_code}: %s", error)
    resp.raise_for_status()
    return []  # unreachable if raise_for_status() fails

# ─── Google CSE Query Runner ────────────────────────────────────────────────
def run_queries(
    company_name: str,
    api_key: str,
    cx: str,
    num_results: int = 10,
    recent_years: int | None = None,
) -> List[Dict]:
    """Run templated queries, cache results, and dedupe URLs."""
    queries = [tpl.format(name=company_name) for tpl in QUERY_TEMPLATES]
    final_items: List[Dict] = []
    seen_urls = set()

    for q in queries:
        hits = search_cache.get(q)
        if hits is None:
            params = {"key": api_key, "cx": cx, "q": q, "num": num_results, "gl": "fr"}
            if recent_years:
                year = datetime.now(timezone.utc).year
                years = ",".join(f"y{y}" for y in range(year, year - recent_years, -1))
                params["dateRestrict"] = years
            try:
                hits = _cse_call(params)
            except RetryError as e:
                logger.error("CSE call failed after retries: %s", e)
                hits = []
            search_cache.set(q, hits)

        for it in hits:
            url = it.get("link")
            if url and url not in seen_urls:
                seen_urls.add(url)
                final_items.append({
                    "title":   it.get("title", ""),
                    "snippet": it.get("snippet", ""),
                    "link":    url,
                })
    return final_items

# ─── Scraping + Embedding ─────────────────────────────────────────────────
async def _fetch(url: str, snippet: str) -> tuple[str, str]:
    cached = scrape_cache.get(url)
    if cached is not None:
        return url, cached

    if _domain(url) in BLOCKED_DOMAINS:
        scrape_cache.set(url, snippet)
        return url, snippet

    client = get_async_client()
    try:
        r = await client.get(url)
        if r.status_code in (403, 999) or r.status_code >= 300:
            text = snippet
        else:
            text = extract(r.text, include_comments=False, favor_recall=True) or snippet
    except Exception as e:
        logger.warning("Scrape failed %s: %s", url, e)
        text = snippet

    scrape_cache.set(url, text)
    return url, text

async def scrape_urls(pairs: List[tuple[str, str]]) -> Dict[str, str]:
    results = await asyncio.gather(*(_fetch(u, s) for u, s in pairs))
    out: Dict[str, str] = {}
    for url, text in results:
        out[url] = text
        # cache embedding by hash key avoidance
        key = str(hash(text))
        if embed_cache.get(key) is None:
            vec = embed_text(text)
            embed_cache.set(key, vec)
    return out

# ─── Public API ─────────────────────────────────────────────────────────────
def google_cse_search(
    company_name: str,
    api_key: str,
    cx: str,
    num_results: int = 10,
    recent_years: int | None = None,
    structured: bool = False,
) -> str | List[Dict]:
    # 1. Query Google CSE
    items = run_queries(company_name, api_key, cx, num_results, recent_years)

    # 2. Scrape content and cache embeddings
    pairs = [(it["link"], it.get("snippet", "")) for it in items]
    article_map = asyncio.run(scrape_urls(pairs))

    for it in items:
        it["content"] = article_map.get(it["link"])

    # 3. Return structured or blob
    if structured:
        return items

    return "\n\n".join(
        f"🔹 {it['title']}\n{it['snippet']}\n🔗 {it['link']}" for it in items
    )


#test
# import os
# google_key = os.getenv('google_key')
# google_cse_id = os.getenv('google_cse_id')


# results = google_cse_search(
#     "Agicap",
#     api_key=google_key,
#     cx=google_cse_id,
#     structured=True
# )

# # write to disk
# with open("agicap_results.txt", "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)

# print("Saved → agicap_results.txt")