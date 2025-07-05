import os
import re
import json
import time
import sqlite3
import datetime
import logging
import asyncio

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from httpx import AsyncClient, ReadTimeout, HTTPStatusError
from trafilatura import extract as trafilatura_extract

from searchfuncs.searchgoogle import google_cse_search
from utils.setup_log import logger
from utils.sqlite_cache import SQLiteCache

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

# Embedding model and tokenization
EMBED_MODEL = "text-embedding-ada-002"
ENC = tiktoken.encoding_for_model(EMBED_MODEL)
MAX_EMBED_TOKENS = 8190

# Chunking and batching
MAX_CHUNK_WORDS = 200
BATCH_SIZE = 50

# Similarity thresholds
DEFAULT_SIM_THRESHOLD = 0.85
THRESHOLD_STEP = 0.02
MIN_SIM_THRESHOLD = 0.2
PER_QUERY_THRESHOLDS: Dict[str, float] = {
    "événement": 0.75,
    "event": 0.75,
    "salon": 0.75,
    "secteur ou chiffre d'affaires (CA)": 0.75,
    "rebranding": 0.75,
    "nombre de postes ouverts": 0.75,
    "nombre de salariés": 0.75,
    "salariés": 0.75,
    "webinar": 0.75,
    "employés": 0.75,
    "collaborateurs": 0.75,
    "cse": 0.75,
    "nombre d'emploés": 0.75,
}
MAX_MATCHES_PER_QUERY = 6
MAX_TOTAL_MATCHES = 100

# Semantic queries
SEMANTIC_QUERIES = [
    "date ou référence temporelle",
    "données numériques ou statistiques",
    "fusion acquisition achats",
    "anniversaire",
    "nom d'un événement",
    "informations CSE",
    "événement",
    "event",
    "salon",
    "secteur ou chiffre d'affaires (CA)",
    "rebranding identité visuelle",
    "nombre de postes ouverts",
    "nombre de salariés",
    "bureaux ou siège social",
    "salariés",
    "webinar",
    "employés",
    "collaborateurs",
    "cse",
    "rse",
    "nombre d'emploés",
    "bénéfices",
    "turnover",
    "certification ISO eco-vadis BCORP",
    "taille de l'entreprise",
    "valuation",
]

# Caching and database path
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
DB_PATH = CACHE_DIR / "reports.db"
QUERY_EMB_PATH = CACHE_DIR / "query_embs.json"

# Initialize OpenAI client
client = OpenAI()

# Instantiate caches (unused in refactored code, but kept for compatibility)
search_cache = SQLiteCache(DB_PATH, "search_cache")
scrape_cache = SQLiteCache(DB_PATH, "scrape_cache")

# ─── LOGGING CONFIGURATION ──────────────────────────────────────────────────

# Logger is already configured in utils/setup_log.py

# ─── DATABASE INITIALIZATION ────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """
    Ensure that the SQLite database and required tables exist.
    Tables:
      - chunks: stores raw chunks, LinkedIn flags, embedding JSON, and last_updated timestamp
      - chunk_labels: normalized storage of matched semantic labels per chunk
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create chunks table
    c.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        company TEXT NOT NULL,
        chunk_id INTEGER NOT NULL,
        link TEXT,
        title TEXT,
        chunk TEXT,
        is_linkedin INTEGER NOT NULL DEFAULT 0,
        embedding TEXT,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(company, chunk_id)
    );
    """)

    # Create chunk_labels table
    c.execute("""
    CREATE TABLE IF NOT EXISTS chunk_labels (
        company TEXT NOT NULL,
        chunk_id INTEGER NOT NULL,
        label TEXT NOT NULL,
        PRIMARY KEY(company, chunk_id, label),
        FOREIGN KEY(company, chunk_id) REFERENCES chunks(company, chunk_id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    return conn

# ─── UTILITY FUNCTIONS ───────────────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    """
    Replace any sequence of whitespace characters with a single space, and trim.
    """
    return re.sub(r"\s+", " ", text or "").strip()

def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> List[str]:
    """
    Break a large text into smaller chunks, each with up to max_words words,
    preserving sentence boundaries when possible.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        words = (current + " " + sent).split()
        if current and len(words) > max_words:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}" if current else sent
    if current:
        chunks.append(current.strip())
    return chunks

def embed_clip_or_split(text: str, max_tokens: int = MAX_EMBED_TOKENS) -> List[str]:
    """
    If `text` fits within max_tokens after encoding, return [text].
    Otherwise, split its tokenized form into multiple sub-chunks of size <= max_tokens
    and decode each sub-chunk back to string.
    """
    text = text.strip()
    if not text:
        return []
    token_ids = ENC.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]
    sub_texts: List[str] = []
    for i in range(0, len(token_ids), max_tokens):
        segment = ENC.decode(token_ids[i : i + max_tokens])
        sub_texts.append(segment)
    return sub_texts

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Request embeddings from OpenAI for the provided list of texts.
    - Splits texts further if they exceed max token length.
    - Embeds all resulting sub-chunks, in batches of size BATCH_SIZE.
    - Retries up to MAX_RETRIES on read timeout or HTTP errors.
    Returns a list of embeddings (one per sub-chunk).
    """
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2  # base seconds

    if not texts:
        raise ValueError("No texts provided for embedding.")

    # 1) Split/clip each input text into "safe" sub-chunks
    safe_texts: List[str] = []
    for txt in texts:
        sub_chunks = embed_clip_or_split(txt)
        if not sub_chunks:
            logger.warning("Skipped empty or whitespace-only text")
            continue
        safe_texts.extend(sub_chunks)

    if not safe_texts:
        return []

    logger.debug(f"Embedding {len(safe_texts)} total chunks")

    # 2) Request embeddings in batches
    all_embs: List[List[float]] = []
    for i in range(0, len(safe_texts), BATCH_SIZE):
        batch = safe_texts[i : i + BATCH_SIZE]
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    timeout=30
                )
                if not resp.data:
                    logger.warning(f"Received empty embeddings for batch {i}")
                all_embs.extend(d.embedding for d in resp.data)
                break
            except (ReadTimeout, HTTPStatusError) as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    logger.error(f"Embedding failed after {MAX_RETRIES} attempts: {e}")
                    raise
                wait_time = RETRY_BACKOFF ** attempt
                logger.warning(f"Embedding retry {attempt}/{MAX_RETRIES} after {wait_time}s: {e}")
                time.sleep(wait_time)

    logger.debug(f"Retrieved {len(all_embs)} embeddings")
    return all_embs

def load_embedding(
    cursor: sqlite3.Cursor, company: str, chunk_id: int
) -> Optional[List[float]]:
    """
    Load an embedding JSON from the database and return as a Python list.
    Returns None if no embedding exists or JSON is invalid.
    """
    cursor.execute(
        "SELECT embedding FROM chunks WHERE company = ? AND chunk_id = ?",
        (company, chunk_id),
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        logger.warning(f"Invalid embedding JSON for {company}, chunk {chunk_id}")
        return None

def store_embedding(
    cursor: sqlite3.Cursor,
    company: str,
    chunk_id: int,
    embedding: List[float],
) -> None:
    """
    Serialize embedding as JSON and update the chunks table.
    Also updates last_updated timestamp to CURRENT_TIMESTAMP.
    """
    emb_json = json.dumps(embedding)
    cursor.execute(
        """
        UPDATE chunks
        SET embedding = ?, last_updated = CURRENT_TIMESTAMP
        WHERE company = ? AND chunk_id = ?
        """,
        (emb_json, company, chunk_id),
    )

def clear_old_data(cursor: sqlite3.Cursor, company: str) -> None:
    """
    Delete all rows for a given company from chunks and chunk_labels.
    """
    cursor.execute("DELETE FROM chunk_labels WHERE company = ?", (company,))
    cursor.execute("DELETE FROM chunks WHERE company = ?", (company,))

def parse_linkedin_title(title: str) -> Dict[str, str]:
    """
    Parse a LinkedIn title string of the form:
      - "Name - Company"
      - "Name - Job - Company"
      - "Name - Senior Engineer - R&D - Corp" (or any number of middle segments)
    Returns a dict with keys: name, job, company.
    """
    parts = [p.strip() for p in title.split(" - ")]
    if len(parts) == 1:
        return {"name": parts[0], "job": "", "company": ""}
    if len(parts) == 2:
        name, company = parts
        return {"name": name, "job": "", "company": company}
    # len(parts) >= 3: first is name, last is company, middle parts form job
    name = parts[0]
    company = parts[-1]
    job = " - ".join(parts[1:-1])
    return {"name": name, "job": job, "company": company}


# ─── ASYNC FETCH FOR SCRAPING ─────────────────────────────────────────────────

def is_valid_url(u: str) -> bool:
    parts = urlparse(u)
    return parts.scheme in ("http", "https") and bool(parts.netloc)

async def async_robust_fetch(
    client: AsyncClient,
    url: str,
    max_retries: int = 3,
    backoff_factor: int = 2,
    timeout: float = 10.0
) -> Optional[str]:
    """
    Attempt to GET `url` with up to `max_retries` on ReadTimeout/HTTPStatusError.
    Returns the response text if successful, or None if we give up.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            resp = await client.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except (ReadTimeout, HTTPStatusError):
            attempt += 1
            if attempt >= max_retries:
                return None
            wait = backoff_factor ** attempt
            await asyncio.sleep(wait)
        except Exception:
            return None
    return None

async def fetch_all(
    urls: List[str],
    max_concurrency: int = 20
) -> List[Tuple[str, Optional[str]]]:
    """
    Fetch each URL in `urls` concurrently (up to max_concurrency at a time).
    Returns a list of (url, response_text_or_None).
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_fetch(client: AsyncClient, url: str) -> Tuple[str, Optional[str]]:
        async with semaphore:
            if not is_valid_url(url):
                return url, None
            text = await async_robust_fetch(client, url)
            return url, text

    async with AsyncClient() as client:
        tasks = [sem_fetch(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

# ─── CACHE/LOAD SEMANTIC QUERY EMBEDDINGS ────────────────────────────────────

def _load_or_create_query_embeddings() -> List[List[float]]:
    """
    Load precomputed semantic-query embeddings from JSON if available and valid.
    Otherwise, compute them once and save to disk.
    """
    if QUERY_EMB_PATH.exists():
        try:
            with open(QUERY_EMB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if len(data) == len(SEMANTIC_QUERIES):
                logger.info(f"Loaded {len(data)} semantic-query embeddings from disk")
                return data
            else:
                logger.warning(f"query_embs.json contains {len(data)} entries, expected {len(SEMANTIC_QUERIES)}")
        except Exception as e:
            logger.warning(f"Failed to read {QUERY_EMB_PATH}: {e}")

    # Compute afresh
    try:
        embs = get_embeddings(SEMANTIC_QUERIES)
        if len(embs) == len(SEMANTIC_QUERIES):
            with open(QUERY_EMB_PATH, "w", encoding="utf-8") as f:
                json.dump(embs, f)
            logger.info(f"Computed and saved {len(embs)} semantic-query embeddings")
            return embs
        else:
            logger.error(f"Expected {len(SEMANTIC_QUERIES)} query embeddings, got {len(embs)}")
            return embs
    except Exception as e:
        logger.error(f"Could not compute semantic-query embeddings: {e}")
        return []

# Precompute or load semantic-query embeddings once at import
QUERY_EMBS: List[List[float]] = _load_or_create_query_embeddings()

# ─── CACHE CHECK FUNCTIONS ───────────────────────────────────────────────────

def check_cache_and_return_if_fresh(
    cursor: sqlite3.Cursor, company: str, refresh_days: int
) -> Optional[Dict[str, List[Dict]]]:
    """
    Check whether cached data for `company` is fresh (within refresh_days) and has at least one label.
    If fresh, load all semantic items and LinkedIn profiles and return them.
    Otherwise, return None to indicate a recrawl is needed.
    """
    cursor.execute(
        "SELECT MAX(last_updated) FROM chunks WHERE company = ?", (company,)
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        return None

    last_updated = datetime.datetime.fromisoformat(row[0])
    age_days = (datetime.datetime.now() - last_updated).days
    if age_days >= refresh_days:
        logger.info(f"Cached data for '{company}' is {age_days} days old (older than {refresh_days} days)")
        return None

    # Ensure at least one semantic label exists
    cursor.execute(
        "SELECT COUNT(*) FROM chunk_labels WHERE company = ?", (company,)
    )
    match_count = cursor.fetchone()[0]
    if match_count == 0:
        logger.info(f"No semantic labels found for '{company}', recrawl needed")
        return None

    # Load semantic items
    cursor.execute(
        """
        SELECT c.chunk_id, c.link, c.title, c.chunk, GROUP_CONCAT(cl.label)
        FROM chunks c
        JOIN chunk_labels cl ON c.company = cl.company AND c.chunk_id = cl.chunk_id
        WHERE c.company = ?
        GROUP BY c.chunk_id, c.link, c.title, c.chunk
        """,
        (company,),
    )
    semantic_items: List[Dict] = []
    for chunk_id, link, title, chunk_text_, labels_csv in cursor.fetchall():
        labels = labels_csv.split(",") if labels_csv else []
        semantic_items.append({
            "chunk_id": chunk_id,
            "link": link,
            "title": title,
            "chunk": chunk_text_,
            "matched_queries": labels,
        })

    # Load LinkedIn profiles (chunks flagged as LinkedIn)
    cursor.execute(
        "SELECT chunk_id, link, title, chunk FROM chunks WHERE company = ? AND is_linkedin = 1",
        (company,)
    )
    linkedin_profiles: List[Dict] = []
    for chunk_id, link, title, chunk_text_ in cursor.fetchall():
        parsed = parse_linkedin_title(title)
        linkedin_profiles.append({
            "chunk_id": chunk_id,
            "link": link,
            "title": title,
            "chunk": chunk_text_,
            **parsed
        })

    logger.info(f"Returning cached report for '{company}' (updated {last_updated.strftime('%Y-%m-%d %H:%M')})")
    return {
        "semantic_items": semantic_items,
        "linkedin_profiles": linkedin_profiles,
    }

# ─── RAW SCRAPE AND DB INSERTION ─────────────────────────────────────────────

def fetch_and_store_raw_chunks(
    cursor: sqlite3.Cursor, company: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform a Google CSE search for `company`, clear old rows for that company,
    and insert new raw chunks into `chunks` table. Returns two lists:
      - all_chunks: list of dicts for non-LinkedIn content
      - linkedin_chunks: list of dicts for LinkedIn segments (parsed separately)
    Each dict has keys: chunk_id, link, title, chunk, is_linkedin (0 or 1)
    """
    # 1) Run Google CSE search
    google_key = os.getenv("GOOGLE_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not google_key or not google_cse_id:
        raise RuntimeError("Missing GOOGLE_KEY or GOOGLE_CSE_ID environment variables.")

    raw_results = google_cse_search(
        company,
        api_key=google_key,
        cx=google_cse_id,
        structured=True
    )
    results = raw_results if isinstance(raw_results, (list, tuple)) else json.loads(raw_results)

    # 2) Clear old rows for this company
    clear_old_data(cursor, company)

    all_chunks: List[Dict] = []
    linkedin_chunks: List[Dict] = []
    chunk_idx = 0

    # 3) Collect URLs from CSE results, but skip LinkedIn in the fetch list
    urls_to_visit: List[str] = []
    snippets: Dict[str, str] = {}  # fallback snippet per URL

    for entry in results:
        link = normalize_whitespace(entry.get("link", ""))
        title = normalize_whitespace(entry.get("title", ""))
        snippet = normalize_whitespace(entry.get("content") or entry.get("snippet", ""))

        if not link:
            continue

        # If this is a LinkedIn profile, insert a chunk with is_linkedin=1 and skip fetch
        if "linkedin.com/in" in link:
            chunk_idx += 1
            cursor.execute(
                """
                INSERT OR IGNORE INTO chunks
                  (company, chunk_id, link, title, chunk, is_linkedin, embedding)
                VALUES (?, ?, ?, ?, ?, 1, NULL)
                """,
                (company, chunk_idx, link, title, snippet)
            )
            parsed = parse_linkedin_title(title)
            linkedin_chunks.append({
                "chunk_id": chunk_idx,
                "link": link,
                "title": title,
                "chunk": snippet,
                **parsed
            })
            continue

        # Otherwise, schedule this URL for async fetch
        urls_to_visit.append(link)
        snippets[link] = snippet

    # 4) Fetch all non-LinkedIn pages asynchronously (max 20 concurrent)
    fetch_results = asyncio.run(fetch_all(urls_to_visit, max_concurrency=20))

    # 5) For each fetched (url, html_text), extract with Trafilatura or fallback to snippet
    for url, html in fetch_results:
        title = ""  # We don’t have a reliable <title> tag; optional
        link = url

        if html:
            extracted = trafilatura_extract(html, url)
            content = normalize_whitespace(extracted or "")
        else:
            content = snippets.get(url, "")

        if not content:
            continue

        # Split into chunks and insert each with is_linkedin=0
        segments = chunk_text(content)
        for seg in segments:
            chunk_idx += 1
            cursor.execute(
                """
                INSERT OR IGNORE INTO chunks
                  (company, chunk_id, link, title, chunk, is_linkedin, embedding)
                VALUES (?, ?, ?, ?, ?, 0, NULL)
                """,
                (company, chunk_idx, link, title, seg)
            )
            all_chunks.append({
                "chunk_id": chunk_idx,
                "link": link,
                "title": title,
                "chunk": seg,
                "is_linkedin": 0
            })

    return all_chunks, linkedin_chunks


# ─── EMBEDDING MANAGEMENT ────────────────────────────────────────────────────

def ensure_embeddings(
    cursor: sqlite3.Cursor, company: str, all_chunks: List[Dict]
) -> Tuple[List[List[float]], Dict[int, int]]:
    """
    For each chunk in all_chunks, reuse an existing embedding if it’s already in the DB.
    Otherwise, collect those chunk texts up to a combined 100 000-token budget (after splitting),
    embed each sub-chunk separately, average sub-chunk embeddings per chunk_id, and store.
    Any chunks beyond that budget are deleted (so no chunk remains un-embedded).
    Returns:
      - all_embs: list of all embeddings (existing + newly computed), in order of chunk_id_to_idx
      - chunk_id_to_idx: dict mapping chunk_id -> index in all_embs
    """
    chunk_id_to_idx: Dict[int, int] = {}
    all_embs: List[List[float]] = []
    next_idx = 0

    # 1) Reuse cached embeddings where available
    new_items: List[Dict] = []
    for item in all_chunks:
        cid = item["chunk_id"]
        existing_emb = load_embedding(cursor, company, cid)
        if existing_emb is not None:
            chunk_id_to_idx[cid] = next_idx
            all_embs.append(existing_emb)
            next_idx += 1
        else:
            new_items.append(item)

    # If there are no “new” chunks, delete any leftover NULLs and return
    if not new_items:
        cursor.execute(
            "DELETE FROM chunks WHERE company = ? AND embedding IS NULL",
            (company,)
        )
        return all_embs, chunk_id_to_idx

    # 2) Build a list of new chunk texts (with their chunk_ids),
    #    but only up to 100 000 tokens total (counting sub-chunks).
    MAX_TOTAL_TOKENS = 100_000
    limited_items: List[Dict] = []
    total_tokens = 0

    for item in new_items:
        text = item["chunk"]
        sub_chunks = embed_clip_or_split(text)
        if not sub_chunks:
            continue  # skip empty or whitespace‐only

        # Count tokens across all sub-chunks
        chunk_token_count = sum(len(ENC.encode(sub)) for sub in sub_chunks)

        if total_tokens + chunk_token_count > MAX_TOTAL_TOKENS:
            logger.info(
                f"Reached 100k token budget: current={total_tokens}, "
                f"next_chunk={chunk_token_count}. Skipping remaining {len(new_items) - len(limited_items)} chunks"
            )
            break

        limited_items.append(item)
        total_tokens += chunk_token_count

    # If none fit under 100k tokens, delete all un-embedded rows and return
    if not limited_items:
        logger.warning("No new chunks fit under the 100k token limit; deleting all un-embedded rows")
        cursor.execute(
            "DELETE FROM chunks WHERE company = ? AND embedding IS NULL",
            (company,)
        )
        return all_embs, chunk_id_to_idx

    logger.debug(f"Preparing to embed {len(limited_items)} new chunks; total token-estimate = {total_tokens}")

    # 3) Collect all sub-chunks for those limited_items, remembering mapping to chunk_id
    texts_to_embed: List[str] = []
    chunk_ids_for_embedding: List[int] = []
    for item in limited_items:
        cid = item["chunk_id"]
        for sub in embed_clip_or_split(item["chunk"]):
            texts_to_embed.append(sub)
            chunk_ids_for_embedding.append(cid)

    # 4) Send all sub-chunks to get_embeddings at once
    try:
        sub_embs = get_embeddings(texts_to_embed)
    except Exception as e:
        logger.error(f"Embedding failed for limited batch: {e}")
        cursor.execute(
            "DELETE FROM chunks WHERE company = ? AND embedding IS NULL",
            (company,)
        )
        return all_embs, chunk_id_to_idx

    if len(sub_embs) != len(texts_to_embed):
        logger.warning(
            f"Expected {len(texts_to_embed)} sub-embeddings, but got {len(sub_embs)} back"
        )

    # 5) Average sub-chunk embeddings per chunk_id
    emb_sums: Dict[int, List[float]] = {}
    emb_counts: Dict[int, int] = {}

    for cid, emb in zip(chunk_ids_for_embedding, sub_embs):
        if cid not in emb_sums:
            emb_sums[cid] = emb.copy()
            emb_counts[cid] = 1
        else:
            # elementwise sum
            for i in range(len(emb_sums[cid])):
                emb_sums[cid][i] += emb[i]
            emb_counts[cid] += 1

    averaged_embs: Dict[int, List[float]] = {}
    for cid, summed in emb_sums.items():
        count = emb_counts[cid]
        averaged_embs[cid] = [x / count for x in summed]

    # 6) Store averaged embeddings for each chunk_id and update mappings
    for cid, emb in averaged_embs.items():
        store_embedding(cursor, company, cid, emb)
        chunk_id_to_idx[cid] = next_idx
        all_embs.append(emb)
        next_idx += 1

    # 7) Delete leftover rows with NULL embedding
    cursor.execute(
        "DELETE FROM chunks WHERE company = ? AND embedding IS NULL",
        (company,)
    )

    return all_embs, chunk_id_to_idx

# ─── SEMANTIC MATCHING ───────────────────────────────────────────────────────

def compute_semantic_matches(
    cursor: sqlite3.Cursor,
    company: str,
    all_embs: List[List[float]],
    chunk_id_to_idx: Dict[int, int]
) -> List[Tuple[float, str, int]]:
    """
    Given all embeddings for a company's chunks and a mapping chunk_id->index,
    compute cosine similarities against precomputed semantic-query embeddings.
    Return a list of tuples (score, query, chunk_id) representing top matches,
    respecting per-query and global caps.
    """
    if not all_embs:
        return []

    query_embs = QUERY_EMBS
    if len(query_embs) != len(SEMANTIC_QUERIES):
        logger.error(
            f"Precomputed query embedding count mismatch: "
            f"expected {len(SEMANTIC_QUERIES)}, got {len(query_embs)}"
        )
        return []

    seen_pairs = set()
    top_matches_by_query: Dict[str, List[Tuple[float, str, int]]] = {}
    all_candidates: List[Tuple[float, str, int]] = []

    # Convert to matrix for cosine_similarity
    sims_matrix = cosine_similarity(all_embs, query_embs)

    # For each query, gather top matches
    for q_idx, q in enumerate(SEMANTIC_QUERIES):
        thr = PER_QUERY_THRESHOLDS.get(q, DEFAULT_SIM_THRESHOLD)
        sims = sims_matrix[:, q_idx]
        matches: List[Tuple[float, str, int]] = []
        current_thr = thr

        logger.debug(f"Query '{q}': starting threshold {current_thr}")

        while True:
            candidates = [
                (sims[chunk_idx], q, cid)
                for cid, chunk_idx in chunk_id_to_idx.items()
                if sims[chunk_idx] >= current_thr and (q, cid) not in seen_pairs
            ]
            candidates.sort(reverse=True)
            if len(candidates) >= MAX_MATCHES_PER_QUERY or current_thr <= MIN_SIM_THRESHOLD:
                matches = candidates[:MAX_MATCHES_PER_QUERY]
                break
            current_thr -= THRESHOLD_STEP

        logger.debug(
            f"Query '{q}': found {len(matches)} matches at threshold {current_thr}"
        )

        top_matches_by_query[q] = matches
        seen_pairs.update((q, cid) for _, q, cid in matches)

        # Collect extras above MIN_SIM_THRESHOLD for global fill
        extras = [
            (sims[chunk_idx], q, cid)
            for cid, chunk_idx in chunk_id_to_idx.items()
            if sims[chunk_idx] >= MIN_SIM_THRESHOLD and (q, cid) not in seen_pairs
        ]
        all_candidates.extend(extras)

    # Combine top matches
    final_matches: List[Tuple[float, str, int]] = []
    for matches in top_matches_by_query.values():
        final_matches.extend(matches)

    # Fill up to MAX_TOTAL_MATCHES if needed
    if len(final_matches) < MAX_TOTAL_MATCHES:
        needed = MAX_TOTAL_MATCHES - len(final_matches)
        all_candidates.sort(reverse=True, key=lambda x: x[0])
        for score, q, cid in all_candidates:
            if (q, cid) not in seen_pairs:
                final_matches.append((score, q, cid))
                seen_pairs.add((q, cid))
                if len(final_matches) >= MAX_TOTAL_MATCHES:
                    break

    logger.info(f"Total semantic matches: {len(final_matches)}")
    return final_matches

def write_and_read_matches(
    cursor: sqlite3.Cursor, company: str, matches: List[Tuple[float, str, int]]
) -> List[Dict]:
    """
    Given a list of (score, query, chunk_id), write new labels into chunk_labels,
    update chunks.last_updated timestamp, and then read back all semantic items to return.
    """
    # Insert or ignore each label
    for _, q, cid in matches:
        cursor.execute(
            """
            INSERT OR IGNORE INTO chunk_labels (company, chunk_id, label)
            VALUES (?, ?, ?)
            """,
            (company, cid, q),
        )
        # Update last_updated on chunk as well
        cursor.execute(
            """
            UPDATE chunks
            SET last_updated = CURRENT_TIMESTAMP
            WHERE company = ? AND chunk_id = ?
            """,
            (company, cid),
        )

    # Read all semantic items
    cursor.execute(
        """
        SELECT c.chunk_id, c.link, c.title, c.chunk, GROUP_CONCAT(cl.label)
        FROM chunks c
        JOIN chunk_labels cl
          ON c.company = cl.company AND c.chunk_id = cl.chunk_id
        WHERE c.company = ?
        GROUP BY c.chunk_id, c.link, c.title, c.chunk
        """,
        (company,),
    )
    semantic_items: List[Dict] = []
    for chunk_id, link, title, chunk_text_, labels_csv in cursor.fetchall():
        labels = labels_csv.split(",") if labels_csv else []
        semantic_items.append({
            "chunk_id": chunk_id,
            "link": link,
            "title": title,
            "chunk": chunk_text_,
            "matched_queries": labels,
        })
    return semantic_items

# ─── MAIN REPORT FUNCTION ────────────────────────────────────────────────────

def report(company_name: str, use_cache: bool = True, refresh_days: int = 7) -> Dict[str, List[Dict]]:
    """
    Generate a structured report for `company_name` by:
      1. Checking cache: if fresh and has labels, returns cached data.
      2. Otherwise:
         a. Perform Google CSE search, fetch pages asynchronously, chunk and store raw data.
         b. Ensure embeddings exist for each chunk.
         c. Compute semantic matches & store new labels.
         d. Return semantic items and LinkedIn profiles.
    """
    company = company_name.strip().lower()
    conn = init_db()
    cursor = conn.cursor()

    # 1) Cache check
    if use_cache:
        cached = check_cache_and_return_if_fresh(cursor, company, refresh_days)
        if cached is not None:
            conn.commit()
            conn.close()
            return cached

    # 2a) Fetch & store raw chunks (with async fetching of HTML + Trafilatura extraction)
    all_chunks, linkedin_profiles = fetch_and_store_raw_chunks(cursor, company)
    conn.commit()

    # 2b) Ensure embeddings (up to 100k tokens, average sub-chunks, delete leftovers)
    all_embs, chunk_id_to_idx = ensure_embeddings(cursor, company, all_chunks)
    conn.commit()

    if not all_embs:
        logger.warning(f"No embeddings generated for '{company}'. Returning LinkedIn only")
        conn.commit()
        conn.close()
        return {
            "semantic_items": [],
            "linkedin_profiles": linkedin_profiles
        }

    # 2c) Compute semantic matches
    matches = compute_semantic_matches(cursor, company, all_embs, chunk_id_to_idx)

    # 2d) Write labels and read semantic_items
    semantic_items = write_and_read_matches(cursor, company, matches)
    conn.commit()
    conn.close()

    return {
        "semantic_items": semantic_items,
        "linkedin_profiles": linkedin_profiles
    }
