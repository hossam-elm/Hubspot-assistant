'''
search_helper.py
Production-hardened Google CSE helper with unified caching in .cache/reports.db,
scraping, and simplified results aggregation.
Dependencies:
    pip install requests httpx[http2] trafilatura tenacity
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

# ─── Logging Configuration ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  
CACHE_DIR    = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB     = CACHE_DIR / "reports.db"
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
    "{name} ('nouveau bureau 2025' OR 'nouveau bureau 2026') OR ('ouvre un bureau 2025' OR 'ouvre un bureau 2026') OR ('implantation 2025' OR 'implantation 2026') OR ('locaux 2025' OR 'locaux 2026')",
    "{name} 'anniversaire' OR 'fondation' OR 'célèbre' OR 'event'",
    "{name} événement OR salon OR séminaire OR conférence OR webinaire'",
    "{name} ('événement 2025' OR 'événement 2026') OR ('salon 2025' OR 'salon 2026') OR ('séminaire 2025' OR 'séminaire 2026') OR ('conférence 2025' OR 'conférence 2026') OR ('webinaire 2025' OR 'webinaire 2026')",
    "{name} 'rebranding' OR 'nouvelle identité' OR 'nouvelle charte graphique'",
    "{name} ('rebranding 2025' OR 'rebranding 2026') OR ('nouvelle identité 2025' OR 'nouvelle identité 2026') OR ('nouvelle charte graphique 2025' OR 'nouvelle charte graphique 2026')",
    "{name} 'levée de fonds' OR 'tour de table' OR investissement OR financement'",
    "{name} actualités OR annonce OR nouveauté OR 'dernier communiqué'",
    "{name} chiffre d’affaires OR secteur OR clients OR 'cse'",
    "{name} site:linkedin.com/in intext:\"CSE\" OR \"Office manager\" OR \"Happiness\" OR \"Communication\" OR \"People\" OR \"Talent\" OR \"RH\"",
]

# ─── SQLite Cache Abstraction (Thread-safe, WAL mode) ────────────────────
class SQLiteCache:
    def __init__(self, path: Path, table: str):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
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
search_cache = SQLiteCache(CACHE_DB, "search_cache")
scrape_cache = SQLiteCache(CACHE_DB, "scrape_cache")

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

# ─── CSE Wrapper w/ Retries ───────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _cse_call(params: Dict) -> List[Dict]:
    resp = session.get(CSE_ENDPOINT, params=params, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("items", [])
    try:
        error = resp.json().get("error", {}).get("message", resp.text)
    except Exception:
        error = resp.text
    logger.error(f"CSE {resp.status_code}: %s", error)
    resp.raise_for_status()
    return []

# ─── Google CSE Query Runner ───────────────────────────────────────────────
def run_queries(
    company_name: str,
    api_key: str,
    cx: str,
    num_results: int = 10,
    recent_years: int | None = None,
) -> List[Dict]:
    queries = [tpl.format(name=company_name) for tpl in QUERY_TEMPLATES]
    final_items: List[Dict] = []
    seen_urls = set()

    for q in queries:
        cache_key = f"{company_name}│{q}"
        hits = search_cache.get(cache_key)
        if hits is None:
            params = {"key": api_key, "cx": cx, "q": q, "num": num_results, "gl": "fr"}
            if recent_years:
                params["dateRestrict"] = f"y{recent_years}"
            try:
                hits = _cse_call(params)
            except RetryError as e:
                logger.error("CSE call failed after retries: %s", e)
                hits = []
            search_cache.set(cache_key, hits)

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

# ─── Scraping Logic ───────────────────────────────────────────────────────
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
    return {url: text for url, text in results}

# ─── Public API ─────────────────────────────────────────────────────────────
def google_cse_search(
    company_name: str,
    api_key: str,
    cx: str,
    num_results: int = 10,
    recent_years: int | None = None,
    structured: bool = False,
) -> str | List[Dict]:
    items = run_queries(company_name, api_key, cx, num_results, recent_years)
    pairs = [(it["link"], it.get("snippet", "")) for it in items]
    article_map = asyncio.run(scrape_urls(pairs))

    for it in items:
        it["content"] = article_map.get(it["link"])

    needle = company_name.lower()
    filtered = []
    for it in items:
        link    = (it.get("link")    or "").lower()
        title   = (it.get("title")   or "").lower()
        snippet = (it.get("snippet") or "").lower()
        content = (it.get("content") or "").lower()

        if "linkedin.com" in link:
            # Keep LinkedIn only if company_name appears somewhere
            if needle in link or needle in title or needle in snippet or needle in content:
                filtered.append(it)
        else:
            if needle in title or needle in snippet or needle in link or needle in content:
                filtered.append(it)

    items = filtered

    if structured:
        return items

    return "\n\n".join(
        f"🔹 {it['title']}\n{it['snippet']}\n🔗 {it['link']}" for it in items
    )
