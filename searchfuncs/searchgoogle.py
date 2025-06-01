'''
search_helper.py
Production-hardened Google CSE helper with unified caching in .cache/reports.db,
scraping, and simplified results aggregation.
Dependencies:
    pip install requests httpx[http2] trafilatura tenacity
'''

from __future__ import annotations
import asyncio
import random
from typing import List, Dict, Tuple
from pathlib import Path
from urllib.parse import urlparse

import requests
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from trafilatura import extract

from utils.sqlite_cache import SQLiteCache
from utils.setup_log import logger

# ─── Constants ───────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CACHE_DIR      = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB       = CACHE_DIR / "reports.db"
CACHE_TTL_SECS = 24 * 3600  # 1 day
CSE_ENDPOINT   = "https://www.googleapis.com/customsearch/v1"

# A small pool of realistic User-Agent strings to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.5 Safari/605.1.15",

    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/114.0.0.0 Safari/537.36",

    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
]

BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
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

search_cache = SQLiteCache(CACHE_DB, "search_cache")
scrape_cache = SQLiteCache(CACHE_DB, "scrape_cache")

session = requests.Session()
# We will set headers dynamically per request, so do not update session.headers here

async_client: httpx.AsyncClient | None = None

def get_async_client() -> httpx.AsyncClient:
    global async_client
    if async_client is None:
        async_client = httpx.AsyncClient(http2=True, timeout=7)
    return async_client

def get_headers() -> Dict[str, str]:
    """
    Build a realistic header dictionary, rotating User-Agent each call.
    """
    headers = BASE_HEADERS.copy()
    headers["User-Agent"] = random.choice(USER_AGENTS)
    return headers

def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.lower().lstrip("www.")

def _is_blocked_domain(url: str) -> bool:
    host = urlparse(url).hostname or ""
    host = host.lower().lstrip("www.")
    return any(bd in host for bd in BLOCKED_DOMAINS)

# ─── CSE Wrapper w/ Retries ───────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _cse_call(params: Dict) -> List[Dict]:
    # Rotate headers for each CSE request
    session.headers.clear()
    session.headers.update(get_headers())

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

# ─── Modified _fetch ────────────────────────────────────────────────────────
async def _fetch(url: str, snippet: str) -> Tuple[str, str, str]:
    """
    Attempt to scrape `url`. Return (url, text, category), where category ∈ {
      "blocked", "exception", "non200", "trafilatura", "snippet"
    }.

    - If domain is blocked (linkedin.com), cache + return snippet ("blocked").
    - If GET throws an exception, return snippet ("exception").
    - If HTTP 200:
        • If Trafilatura succeeds → ("trafilatura")
        • Else → ("snippet")
    - If HTTP != 200 → return snippet ("non200").
    """
    # 1) Check cache
    cached = scrape_cache.get(url)
    if cached is not None:
        return url, cached, "cached"

    # 2) If domain is blocked, cache and return snippet (category="blocked")
    if _is_blocked_domain(url):
        scrape_cache.set(url, snippet)
        return url, snippet, "blocked"

    # 3) Actual HTTP request
    client = get_async_client()
    client.headers.clear()
    client.headers.update(get_headers())

    try:
        r = await client.get(url)
    except Exception:
        # Network exception → cache snippet and category="exception"
        scrape_cache.set(url, snippet)
        return url, snippet, "exception"

    # 4) If 200 OK, attempt extraction
    if r.status_code == 200:
        extracted = extract(r.text, include_comments=False, favor_recall=True)
        if extracted:
            scrape_cache.set(url, extracted)
            return url, extracted, "trafilatura"
        else:
            scrape_cache.set(url, snippet)
            return url, snippet, "snippet"

    # 5) Non-200 response → cache snippet, category="non200"
    scrape_cache.set(url, snippet)
    return url, snippet, "non200"

# ─── Modified scrape_urls ─────────────────────────────────────────────────
async def scrape_urls(pairs: List[tuple[str, str]]) -> Dict[str, str]:
    """
    Run _fetch for each (url, snippet). Tally categories and log a summary with counts.
    """
    results = await asyncio.gather(*(_fetch(u, s) for u, s in pairs))

    article_map: Dict[str, str] = {}
    counts = {
        "cached": 0,
        "blocked": 0,
        "exception": 0,
        "non200": 0,
        "trafilatura": 0,
        "snippet": 0,
    }

    for url, text, category in results:
        article_map[url] = text
        if category in counts:
            counts[category] += 1

    # Log summary counts (only if any nonzero)
    summary_parts = [f"{cat}={cnt}" for cat, cnt in counts.items() if cnt > 0]
    if summary_parts:
        logger.warning("Scrape summary: " + ", ".join(summary_parts))

    return article_map

def google_cse_search(
    company_name: str,
    api_key: str,
    cx: str,
    num_results: int = 10,
    recent_years: int | None = None,
    structured: bool = False,
) -> str | List[Dict]:
    """
    Perform Google CSE queries, scrape returned URLs for full text, filter by any word in company_name,
    then return either a structured list of dicts or a formatted string.
    """
    logger.debug(f"[CSE SEARCH] Starting search for '{company_name}' "
                 f"(num_results={num_results}, recent_years={recent_years}, structured={structured})")

    # 1) Run the templated queries and gather raw hits
    items = run_queries(company_name, api_key, cx, num_results, recent_years)
    logger.debug(f"[CSE SEARCH] run_queries returned {len(items)} raw items")

    # 2) Build (url, snippet) pairs and scrape each URL for full content
    pairs = [(it["link"], it.get("snippet", "")) for it in items]
    logger.debug(f"[CSE SEARCH] Created {len(pairs)} (url, snippet) pairs for scraping")

    try:
        article_map = asyncio.run(scrape_urls(pairs))
    except Exception as e:
        logger.error(f"[CSE SEARCH] Error during scraping: {e}", exc_info=True)
        article_map = {}

    logger.debug(f"[CSE SEARCH] scrape_urls returned content for {len(article_map)} URLs")

    # 3) Attach scraped content (or None if scraping failed) to each item
    for it in items:
        url = it.get("link")
        it["content"] = article_map.get(url)

    # 4) Filter items based on any word in company_name appearing anywhere
    words = [w.lower() for w in company_name.split() if w.strip()]
    logger.debug(f"[CSE SEARCH] Filtering by keywords: {words}")

    filtered: List[Dict] = []
    for it in items:
        link    = (it.get("link")    or "").lower()
        title   = (it.get("title")   or "").lower()
        snippet = (it.get("snippet") or "").lower()
        content = (it.get("content") or "").lower()

        # Keep if any keyword appears in any field
        if any(word in link or word in title or word in snippet or word in content for word in words):
            filtered.append(it)

    logger.debug(f"[CSE SEARCH] Filtered down to {len(filtered)} items after matching any keyword")

    items = filtered  # override with filtered results

    # 5) Return structured results or formatted string
    if structured:
        logger.debug(f"[CSE SEARCH] Returning structured list of {len(items)} items")
        return items

    formatted = [
        f"🔹 {it['title'].strip()}\n{it.get('snippet','').strip()}\n🔗 {it['link'].strip()}"
        for it in items
    ]
    result_str = "\n\n".join(formatted)
    logger.debug(f"[CSE SEARCH] Returning formatted string with {len(items)} entries")
    return result_str
