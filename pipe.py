import os
import re
import json
import time
import sqlite3
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from httpx import ReadTimeout, HTTPStatusError

import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
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

# Initialize OpenAI client
client = OpenAI()

# Instantiate caches (unused in refactored code, but kept for compatibility)
search_cache = SQLiteCache(DB_PATH, "search_cache")
scrape_cache = SQLiteCache(DB_PATH, "scrape_cache")


# ─── LOGGING CONFIGURATION ──────────────────────────────────────────────────

# Route all prints to logger
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.DEBUG)


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
    - Embeds chunks in order, but once the running total of tokens exceeds 100 000,
      no further chunks are sent for embedding.
    - Retries up to MAX_RETRIES on read timeout or HTTP errors.
    Returns a list of embeddings (one per chunk actually sent to the API).
    """
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2  # base seconds
    MAX_TOTAL_TOKENS = 100_000

    if not texts:
        raise ValueError("No texts provided for embedding.")

    # 1) Split/clip each input text into "safe" sub-chunks
    safe_texts: List[str] = []
    for txt in texts:
        sub_chunks = embed_clip_or_split(txt)
        if not sub_chunks:
            logger.warning("[WARN] Skipped empty or whitespace-only text.")
            continue
        safe_texts.extend(sub_chunks)

    if not safe_texts:
        return []

    # 2) Build a list of sub-chunks to embed, stopping when token-sum > MAX_TOTAL_TOKENS
    limited_texts: List[str] = []
    total_tokens = 0

    for chunk in safe_texts:
        chunk_tokens_len = len(ENC.encode(chunk))
        # If adding this chunk would exceed 100k, stop here
        if total_tokens + chunk_tokens_len > MAX_TOTAL_TOKENS:
            logger.info(
                f"[INFO] Reached token budget ({total_tokens} + {chunk_tokens_len} > {MAX_TOTAL_TOKENS}); "
                "stopping further embeddings."
            )
            break

        limited_texts.append(chunk)
        total_tokens += chunk_tokens_len

    if not limited_texts:
        raise RuntimeError("No sub-chunks could be embedded within the 100 000-token limit.")

    logger.debug(f"[DEBUG] Will embed {len(limited_texts)} chunks; total tokens = {total_tokens}")

    # 3) Request embeddings in batches
    all_embs: List[List[float]] = []
    for i in range(0, len(limited_texts), BATCH_SIZE):
        batch = limited_texts[i : i + BATCH_SIZE]
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    timeout=30
                )
                if not resp.data:
                    logger.warning(f"[WARN] Received empty embeddings for batch idx {i}.")
                all_embs.extend(d.embedding for d in resp.data)
                break
            except (ReadTimeout, HTTPStatusError) as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    logger.error(f"[ERROR] Embedding failed after {MAX_RETRIES} attempts: {e}")
                    raise
                wait_time = RETRY_BACKOFF ** attempt
                logger.warning(f"[WARN] Embedding retry {attempt}/{MAX_RETRIES} after {wait_time}s: {e}")
                time.sleep(wait_time)

    logger.debug(f"[DEBUG] Retrieved {len(all_embs)} embeddings")
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
        logger.warning(f"[WARN] Invalid embedding JSON for ({company}, {chunk_id})")
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
    Parse a LinkedIn title string of the form "Name - Title - Company" (2 or 3 parts).
    Returns a dict with keys: name, job, company.
    """
    parts = [p.strip() for p in title.split(" - ")]
    if len(parts) == 2:
        name, company = parts
        return {"name": name, "job": "", "company": company}
    if len(parts) == 3:
        name, job, company = parts
        return {"name": name, "job": job, "company": company}
    return {"name": title, "job": "", "company": ""}


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
        logger.info(f"[CACHE] Cached data for '{company}' is older than {refresh_days} days ({age_days} days).")
        return None

    # Ensure at least one semantic label exists
    cursor.execute(
        "SELECT COUNT(*) FROM chunk_labels WHERE company = ?", (company,)
    )
    match_count = cursor.fetchone()[0]
    if match_count == 0:
        logger.info(f"[CACHE] No semantic labels found for '{company}', recrawl needed.")
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

    logger.info(f"[CACHE] Returning cached report for '{company}' (last updated {last_updated}).")
    return {
        "semantic_items": semantic_items,
        "linkedin_profiles": linkedin_profiles,
    }


# ─── RAW SCRAPE AND DB INSERTION ─────────────────────────────────────────────

def fetch_and_store_raw_chunks(
    cursor: sqlite3.Cursor, company: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform a Google CSE search for `company`, delete old cached rows, and insert new raw chunks.
    Returns two lists:
      - all_chunks: list of dicts for non-LinkedIn content
      - linkedin_chunks: list of dicts for LinkedIn segments to be returned separately
    Each dict has keys: chunk_id, link, title, chunk, is_linkedin (0 or 1)
    """
    # Ensure environment variables are set
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

    # Clear old rows
    clear_old_data(cursor, company)

    all_chunks: List[Dict] = []
    linkedin_chunks: List[Dict] = []
    chunk_idx = 0

    for entry in results:
        link = normalize_whitespace(entry.get("link", ""))
        title = normalize_whitespace(entry.get("title", ""))
        content = normalize_whitespace(entry.get("content") or entry.get("snippet", ""))

        if not content:
            continue

        segments = chunk_text(content)
        for seg in segments:
            chunk_idx += 1
            is_li = 1 if "linkedin.com/in" in link else 0

            cursor.execute(
                """
                INSERT OR IGNORE INTO chunks
                  (company, chunk_id, link, title, chunk, is_linkedin, embedding)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (company, chunk_idx, link, title, seg, is_li)
            )
            item = {
                "chunk_id": chunk_idx,
                "link": link,
                "title": title,
                "chunk": seg,
                "is_linkedin": is_li,
            }
            if is_li:
                parsed = parse_linkedin_title(title)
                linkedin_chunks.append({**parsed, **item})
            else:
                all_chunks.append(item)

    return all_chunks, linkedin_chunks


# ─── EMBEDDING MANAGEMENT ────────────────────────────────────────────────────

def ensure_embeddings(
    cursor: sqlite3.Cursor, company: str, all_chunks: List[Dict]
) -> Tuple[List[List[float]], Dict[int, int]]:
    """
    For each chunk in all_chunks, attempt to reuse an existing embedding from the DB.
    If none exists, collect those chunk texts, request embeddings, and store them.
    Returns:
      - all_embs: list of all embeddings (existing + newly computed), in order of chunk_id_to_idx mapping
      - chunk_id_to_idx: dict mapping chunk_id -> index in all_embs
    """
    chunk_id_to_idx: Dict[int, int] = {}
    all_embs: List[List[float]] = []
    next_idx = 0

    # Reuse cached embeddings where available
    new_items: List[Dict] = []
    new_texts: List[str] = []

    for item in all_chunks:
        cid = item["chunk_id"]
        existing_emb = load_embedding(cursor, company, cid)
        if existing_emb:
            chunk_id_to_idx[cid] = next_idx
            all_embs.append(existing_emb)
            next_idx += 1
        else:
            new_items.append(item)
            new_texts.append(item["chunk"])

    # Compute embeddings for chunks without one
    if new_items:
        new_embs = []
        attempts = 0
        while attempts < 3:
            try:
                logger.debug(f"[EMBED] Requesting embeddings for {len(new_texts)} chunks (attempt {attempts+1}).")
                new_embs = get_embeddings(new_texts)
                if len(new_embs) == len(new_items):
                    break
                logger.warning(f"[EMBED] Mismatch: expected {len(new_items)}, got {len(new_embs)}.")
            except Exception as e:
                logger.error(f"[EMBED] Error while embedding: {e}")
            attempts += 1
            time.sleep(2 ** attempts)

        if len(new_embs) != len(new_items):
            logger.error(
                f"[EMBED] After retries, embedding count mismatch: "
                f"expected {len(new_items)}, got {len(new_embs)}."
            )

        # Store new embeddings
        for item, emb in zip(new_items, new_embs):
            cid = item["chunk_id"]
            store_embedding(cursor, company, cid, emb)
            chunk_id_to_idx[cid] = next_idx
            all_embs.append(emb)
            next_idx += 1

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
    compute cosine similarities against each semantic query embedding.
    Return a list of tuples (score, query, chunk_id) representing top matches,
    respecting per-query and global caps.
    """
    if not all_embs:
        return []

    # Pre-embed queries
    try:
        query_embs = get_embeddings(SEMANTIC_QUERIES)
    except Exception as e:
        logger.error(f"[SEMANTIC] Failed to embed semantic queries: {e}")
        return []

    if len(query_embs) != len(SEMANTIC_QUERIES):
        logger.error(
            f"[SEMANTIC] Query embedding count mismatch: "
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

        logger.debug(f"[SEMANTIC] Query '{q}': starting threshold {current_thr}")

        while True:
            # Collect candidates above current_thr that haven’t been seen for this (q, chunk_id)
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
            f"[SEMANTIC] Query '{q}': found {len(matches)} matches at threshold {current_thr}"
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

    logger.info(f"[SEMANTIC] Total semantic matches: {len(final_matches)}")
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
         a. Perform Google CSE search, chunk and store raw data.
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

    # 2a) Fetch & store raw chunks
    all_chunks, linkedin_profiles = fetch_and_store_raw_chunks(cursor, company)
    conn.commit()

    # 2b) Ensure embeddings
    all_embs, chunk_id_to_idx = ensure_embeddings(cursor, company, all_chunks)
    conn.commit()

    if not all_embs:
        logger.warning(f"[REPORT] No embeddings generated for '{company}'. Returning LinkedIn only.")
        # Load any LinkedIn profiles already stored
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
