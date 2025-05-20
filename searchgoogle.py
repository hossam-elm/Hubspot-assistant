"""
search_helper.py
----------------
Google CSE helper with built-in scraping and logging in the CLI test.

Dependencies
------------
pip install requests httpx[http2] trafilatura
"""

from __future__ import annotations

import asyncio, httpx, requests, os, datetime, textwrap, time
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import List, Dict   
from trafilatura import extract
# ─── cache setup ────────────────────────────────────────────────────────── #
import sqlite3, json, time, pathlib

CACHE_PATH     = pathlib.Path(__file__).with_name("search_cache.sqlite3")
CACHE_TTL_SECS = 24 * 3600            # 1 day

def _db():
    """Return a singleton SQLite connection."""
    conn = getattr(_db, "_conn", None)
    if conn is None:
        conn = sqlite3.connect(CACHE_PATH)
        conn.execute("""CREATE TABLE IF NOT EXISTS search_cache(
                            key TEXT PRIMARY KEY,
                            js  TEXT,
                            ts  REAL )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS scrape_cache(
                            key TEXT PRIMARY KEY,
                            js  TEXT,
                            ts  REAL )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS embed_cache(
                    key TEXT PRIMARY KEY,   -- the raw text we embed
                    vec BLOB,               -- JSON-encoded float list
                    ts  REAL )""")
        _db._conn = conn
    return conn

def _cache_get(table: str, key: str):
    """Return cached object or None if not found / expired."""
    row = _db().execute(
        f"SELECT js, ts FROM {table} WHERE key=?",
        (key,)
    ).fetchone()
    if row and time.time() - row[1] < CACHE_TTL_SECS:
        return json.loads(row[0])
    return None

def _cache_set(table: str, key: str, obj):
    _db().execute(
        f"INSERT OR REPLACE INTO {table} (key, js, ts) VALUES (?, ?, ?)",
        (key, json.dumps(obj), time.time())
    )
    _db().commit()

def _embed_cache_get(text: str):
    row = _db().execute("SELECT vec, ts FROM embed_cache WHERE key=?", (text,)).fetchone()
    if row and time.time() - row[1] < CACHE_TTL_SECS:
        return json.loads(row[0])           # list[float]
    return None

def _embed_cache_set(text: str, vec: list[float]):
    _db().execute("INSERT OR REPLACE INTO embed_cache VALUES (?,?,?)",
                  (text, json.dumps(vec), time.time()))
    _db().commit()
# ──────────────────────────────────────────────────────────────────────────── #
# Config
# ──────────────────────────────────────────────────────────────────────────── #

QUERY_TEMPLATES: List[str] = [
    "{name} intext:actualité",
    "{name} intext:collaborateurs",
    "{name} intext:anniversaire",
    "{name} intext:bureau",
    "{name} intext:effectif",
    "{name} intext:calendrier",
    "{name} intext:clients",
    "{name} intext:événements",
    "{name} intext:site officiel",
    'site:linkedin.com "{name}" intext:event',
    'site:linkedin.com "{name}" intext:cse',
    'site:linkedin.com "{name}" intext:rh',
    'site:linkedin.com "{name}" intext:marketing',
    'site:linkedin.com "{name}" intext:communication',
    'site:linkedin.com "{name}" intext:office',
    'site:linkedin.com "{name}" intext:events',
    'site:linkedin.com "{name}" intext:talent',
]

CSE_ENDPOINT   = "https://www.googleapis.com/customsearch/v1"
HEADERS        = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,"
              "application/xml;q=0.9,*/*;q=0.8",
}
BLOCKED_DOMAINS = {"linkedin.com"}          # 999 or 403 → fallback to snippet

# ──────────────────────────────────────────────────────────────────────────── #
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────── #

def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.lower().lstrip("www.")

def _cse_call(params: Dict) -> List[Dict]:
    """CSE wrapper; surfaces JSON error messages."""
    r = requests.get(CSE_ENDPOINT, params=params, timeout=15)
    if r.status_code == 200:
        return r.json().get("items", [])
    try:
        reason = r.json()["error"]["message"]
    except Exception:
        reason = r.text[:200]
    print(f"[ERROR] CSE {r.status_code}: {reason}")
    return []

# ────────────────────────────────────────────────────────────────────────────
# 1.  _run_all_queries  (search → CSE → SQLite cache)
# ────────────────────────────────────────────────────────────────────────────

def _run_all_queries(
    company_name: str,
    api_key: str,
    cx: str,
    num_results: int,
    recent_years: int | None,
) -> list[dict]:
    """
    ① Builds the 18 template queries
    ② Checks the SQLite cache (search_cache) for each
    ③ Fires only the *missing* ones concurrently via httpx.AsyncClient
    ④ Deduplicates URLs and returns list[dict] exactly like before
    """

    # ── 1️⃣  prep queries & cache lookups
    queries     = [tpl.format(name=company_name) for tpl in QUERY_TEMPLATES]
    cached_hits = {}             # query → CSE items list
    pending     = []             # (query, params) tuples to fetch live

    # optional dateRestrict (y2025,y2024,…)
    date_param = None
    if recent_years:
        this_year  = datetime.now(timezone.utc).year
        years      = ",".join(f"y{y}" for y in range(this_year,
                                                     this_year - recent_years,
                                                     -1))
        date_param = years

    for q in queries:
        hit = _cache_get("search_cache", q)
        if hit is not None:
            cached_hits[q] = hit
            print(f"[CACHE] hit → {q} ({len(hit)} items)")
            continue

        params = {
            "key": api_key,
            "cx":  cx,
            "q":   q,
            "num": num_results,
            "gl":  "fr",
        }
        if date_param:
            params["dateRestrict"] = date_param
        pending.append((q, params))

    # ── 2️⃣  concurrent fetch of pending queries
    async def _fetch_many(pairs):
        if not pairs:
            return {}
        async with httpx.AsyncClient(http2=True, timeout=12) as client:
            tasks = [client.get(CSE_ENDPOINT, params=p) for _, p in pairs]
            resps = await asyncio.gather(*tasks, return_exceptions=True)
        # map query → items (empty list on error)
        out = {}
        for (q, _), r in zip(pairs, resps):
            if isinstance(r, Exception):
                print(f"[ERROR] CSE fetch failed {q} → {r}")
                out[q] = []
                continue
            if r.status_code == 200:
                out[q] = r.json().get("items", [])
            else:
                try:
                    reason = r.json()["error"]["message"]
                except Exception:
                    reason = r.text[:200]
                print(f"[ERROR] {q} → {r.status_code} {reason}")
                out[q] = []
        return out

    live_hits = asyncio.run(_fetch_many(pending))

    # ── 3️⃣  write fresh results to cache
    for q, items in live_hits.items():
        _cache_set("search_cache", q, items)

    # ── 4️⃣  merge cached + live, deduplicate URLs
    seen, final_items = set(), []
    for q in queries:                                       # keep template order
        for it in (cached_hits.get(q) or live_hits.get(q, [])):
            url = it.get("link")
            if not url or url in seen:
                continue
            seen.add(url)
            final_items.append({
                "title":   it.get("title", ""),
                "snippet": it.get("snippet", ""),
                "link":    url,
                "query":   q,
            })

    print(f"[INFO] total unique links: {len(final_items)}")
    return final_items
# ────────────────────────────────────────────────────────────────────────────
# 2.  _fetch  (scrape → article text → SQLite cache)
# ────────────────────────────────────────────────────────────────────────────
async def _fetch(
    url: str,
    snippet: str,
    client: httpx.AsyncClient,
) -> str | None:
    """
    Download `url`, return cleaned article text.

    • If the URL’s domain is in BLOCKED_DOMAINS, or the response status is
      999 / 403, fall back to the Google snippet.
    • Results are cached in `scrape_cache` for CACHE_TTL_SECS seconds.
    """

    # 1️⃣  check scrape cache
    cached = _cache_get("scrape_cache", url)
    if cached is not None:
        return cached

    # 2️⃣  skip domains known to block bots
    if _domain(url).endswith(tuple(BLOCKED_DOMAINS)):
        return snippet or None

    # 3️⃣  live fetch
    try:
        r = await client.get(url, timeout=7, headers=HEADERS)   # already in code
        if r.status_code in (999, 403) or r.status_code >= 300:   # ← add check
            return snippet or None
        else:
            r.raise_for_status()
            text  = extract(r.text, include_comments=False, favor_recall=True)
            final = text or snippet or None
    except Exception as exc:
        print(f"[SCRAPE-ERR] {url} → {exc}")
        final = snippet or None

    # 4️⃣  store in cache (even if it’s just the snippet, so we don’t retry)
    if final is not None:
        _cache_set("scrape_cache", url, final)

    return final


async def _scrape_many(pairs: List[tuple[str, str]]) -> Dict[str, str]:
    """pairs = [(url, snippet)] → {url: article_text_or_snippet}"""
    async with httpx.AsyncClient(http2=True, headers=HEADERS) as client:
        texts = await asyncio.gather(*(_fetch(u, s, client) for u, s in pairs))
    return {u: t for (u, _), t in zip(pairs, texts) if t}

# ──────────────────────────────────────────────────────────────────────────── #
# Public API
# ──────────────────────────────────────────────────────────────────────────── #

def google_cse_search(
    company_name: str,
    api_key: str,
    cx: str,
    num_results: int = 10,
    recent_years: int | None = None,
    structured: bool = False,
) -> str | list[dict]:
    """
    Back-compat signature — returns ONE string exactly like the old helper.
    """
    # 1️⃣ search & scrape (existing helper calls — unchanged)
    search_items = _run_all_queries(
        company_name, api_key, cx, num_results, recent_years
    )
    pairs       = [(it["link"], it["snippet"]) for it in search_items]
    article_map = asyncio.run(_scrape_many(pairs))

    # add full text for future use
    for it in search_items:
        it["content"] = article_map.get(it["link"])  # may be snippet

    # 2️⃣ choose return format
    if structured:
        return search_items                     # ← clean list[dict]

    blob_parts = [
        f"🔹 {it['title']}\n{it['snippet']}\n🔗 {it['link']}"
        for it in search_items
    ]
    return "\n\n".join(blob_parts)              # ← original behaviour


# ──────────────────────────────────────────────────────────────────────────── #
# CLI sanity-test (+ write log file)
# ──────────────────────────────────────────────────────────────────────────── #

# ───────────────── CLI sanity-test (at bottom of search_helper.py) ──────────
# if __name__ == "__main__":
#     from datetime import datetime, timezone          # ← NEW

#     COMPANY = "Agicap"
#     api_key = os.environ["google_key"]
#     cx_id   = os.environ["google_cse_id"]

#     blob = google_cse_search(
#         company_name=COMPANY,
#         api_key=api_key,
#         cx=cx_id,
#         num_results=5,
#         recent_years=1,
#     )

#     # print a short preview
#     import textwrap
#     print("\n──── snippets blob (truncated) ────\n")
#     print(textwrap.shorten(blob, 1500, placeholder=" …"))

#     # save the full blob
#     os.makedirs("search_logs", exist_ok=True)
#     stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")  # ← CHANGED
#     path  = f"search_logs/{COMPANY}_{stamp}.txt"

#     with open(path, "w", encoding="utf-8") as fh:
#         fh.write(blob)

#     print(f"\n[INFO] Full results saved → {path}")
