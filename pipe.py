import json
import re
from pathlib import Path
from typing import List, Dict
import sqlite3
import os
import datetime, time
import tiktoken
from openai import OpenAI
from httpx import ReadTimeout, HTTPStatusError
from searchfuncs.searchgoogle import google_cse_search
from sklearn.metrics.pairwise import cosine_similarity
from utils.setup_log import logger
from pathlib import Path
from utils.sqlite_cache import SQLiteCache
# ─── CONFIG ────────────────────────────────────────────────────────────────
MAX_CHUNK_WORDS = 200
SIM_THRESHOLD   = 0.85
BATCH_SIZE      = 50
EMBED_MODEL     = "text-embedding-ada-002"

# Embedding token splitter config
ENC = tiktoken.encoding_for_model(EMBED_MODEL)
MAX_EMBED_TOKENS = 8190

SEMANTIC_QUERIES = [
    "date ou référence temporelle",
    "données numériques ou statistiques",
    "contact ou le nom d'une personne",
    "nom d'un événement",
    "informations CSE",
    "événement",
    "event",
    "salon",
    "secteur ou chiffre d'affaires (CA)",
    "rebranding",
    "nombre de postes ouverts",
    "nombre de salariés",
    "bureaux ou siège social",
    "salariés",
    'webinar',
    "employés",
    "collaborateurs",
    "cse",
    "rse"
    ]

# ─── DIRECTORIES ───────────────────────────────────────────────────────────
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
DB_PATH   = CACHE_DIR / "reports.db"

client = OpenAI()
# instantiate the two cache tables
search_cache = SQLiteCache(DB_PATH, "search_cache")
scrape_cache = SQLiteCache(DB_PATH, "scrape_cache")

# ─── UTILS ─────────────────────────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
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


# Split long texts into embedding-safe pieces:
def embed_clip_or_split(text: str, max_tokens: int = MAX_EMBED_TOKENS) -> List[str]:
    token_ids = ENC.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]
    return [ENC.decode(token_ids[i:i+max_tokens]) for i in range(0, len(token_ids), max_tokens)]

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, multiplied by attempt count

def get_embeddings(texts: List[str]) -> List[List[float]]:
    safe_texts: List[str] = []
    for txt in texts:
        safe_texts.extend(embed_clip_or_split(txt))

    all_embs: List[List[float]] = []

    for i in range(0, len(safe_texts), BATCH_SIZE):
        batch = safe_texts[i:i+BATCH_SIZE]
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                # Add a timeout (e.g., 30 seconds)
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    timeout=30
                )
                all_embs.extend(d.embedding for d in resp.data)
                break  # success, exit retry loop
            except (ReadTimeout, HTTPStatusError) as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    raise  # re-raise after max attempts
                wait_time = RETRY_BACKOFF ** attempt
                print(f"Embedding request timeout/error, retry {attempt}/{MAX_RETRIES} after {wait_time}s...")
                time.sleep(wait_time)
    return all_embs



def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        company TEXT,
        chunk_id INTEGER,
        link TEXT,
        title TEXT,
        chunk TEXT,
        is_linkedin INTEGER,
        matched_queries TEXT,
        embedding TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(company, chunk_id)
    )""")
    conn.commit()
    return conn


def parse_linkedin_title(title: str) -> Dict[str, str]:
    parts = [p.strip() for p in title.split(" - ")]
    if len(parts) == 2:
        name, company = parts
        return {"name": name, "job": "", "company": company}
    if len(parts) == 3:
        name, job, company = parts
        return {"name": name, "job": job, "company": company}
    return {"name": title, "job": "", "company": ""}


def report(company_name: str, use_cache: bool = True, refresh_days: int = 7) -> Dict[str, List[Dict]]:
    company_name = company_name.lower()

    conn = init_db()
    c = conn.cursor()

    # ─── CACHING CHECK ────────────────────────────
    if use_cache:
        c.execute("SELECT MAX(timestamp) FROM chunks WHERE company=?", (company_name,))
        row = c.fetchone()
        if row and row[0]:
            last_updated = datetime.datetime.fromisoformat(row[0])
            if (datetime.datetime.now() - last_updated).days < refresh_days:
                logger.info(f"Using cached data for '{company_name}' (last updated {last_updated})")
                c.execute("""
                    SELECT chunk_id, link, title, chunk, matched_queries, is_linkedin 
                    FROM chunks WHERE company=?""", (company_name,))
                semantic_items = []
                linkedin_profiles = []

                for row in c.fetchall():
                    chunk_id, link, title, chunk, matched, is_li = row
                    item = {"chunk_id": chunk_id, "link": link, "title": title, "chunk": chunk}
                    if is_li:
                        parsed = parse_linkedin_title(title)
                        linkedin_profiles.append({**parsed, **item})
                    if matched:
                        item['matched_queries'] = matched.split(",")
                        semantic_items.append(item)

                conn.close()
                return {
                    'semantic_items': semantic_items,
                    'linkedin_profiles': linkedin_profiles
                }

    # ─── RECRAWL ───────────────────────────────────
    raw = google_cse_search(
        company_name,
        api_key=os.getenv('google_key'),
        cx=os.getenv('google_cse_id'),
        structured=True
    )
    results = raw if isinstance(raw, (list, tuple)) else json.loads(raw)

    c.execute("DELETE FROM chunks WHERE company = ?", (company_name,))
    conn.commit()

    all_chunks = []
    linkedin_profile_chunks = []
    idx = 0

    for entry in results:
        link    = normalize_whitespace(entry.get('link',''))
        title   = normalize_whitespace(entry.get('title',''))
        content = normalize_whitespace(entry.get('content') or entry['snippet'])
        for segment in chunk_text(content):
            idx += 1
            is_li      = 'linkedin.com/in' in link
            is_event   = any(k in segment.lower() for k in ("événement","event"))
            is_money   = any(k in segment.lower() for k in ("euros","€","dollars","$"))
            is_webinar = "webinar" in segment.lower()

            c.execute(
                "INSERT OR IGNORE INTO chunks"
                " (company,chunk_id,link,title,chunk,is_linkedin,matched_queries,embedding)"
                " VALUES (?,?,?,?,?,?,?,NULL)",
                (company_name, idx, link, title, segment, int(is_li), '')
            )
            item = {"chunk_id": idx, "link": link, "title": title, "chunk": segment}
            if is_li:
                # save raw LinkedIn segment for separate parsing
                parsed = parse_linkedin_title(title)
                linkedin_profile_chunks.append({**parsed, **item})
            elif is_event or is_webinar or is_money:
                all_chunks.append(item)
            else:
                # optionally skip storing non-relevant
                all_chunks.append(item)
    conn.commit()

    # ─── EMBEDDING & SEMANTIC FILTER (per-query, batched, early-exit) ───────────────
    # 1) Gather or compute all embeddings & build chunk_id → row-index map
    all_embs: List[List[float]] = []
    chunk_id_to_idx: Dict[int, int] = {}
    next_idx = 0

    # Reuse cached embeddings
    for item in all_chunks:
        cid = item["chunk_id"]
        c.execute(
            "SELECT embedding FROM chunks WHERE company=? AND chunk_id=?",
            (company_name, cid)
        )
        row = c.fetchone()
        if row and row[0]:
            emb = json.loads(row[0])
            all_embs.append(emb)
            chunk_id_to_idx[cid] = next_idx
            next_idx += 1

    # Embed & store any new chunks
    new_items = [it for it in all_chunks if it["chunk_id"] not in chunk_id_to_idx]
    if new_items:
        texts   = [it["chunk"] for it in new_items]
        new_embs = get_embeddings(texts)
        for item, emb in zip(new_items, new_embs):
            cid = item["chunk_id"]
            c.execute(
                "UPDATE chunks SET embedding=? WHERE company=? AND chunk_id=?",
                (json.dumps(emb), company_name, cid)
            )
            all_embs.append(emb)
            chunk_id_to_idx[cid] = next_idx
            next_idx += 1
        conn.commit()

    # 2) Pre-embed all queries once
    query_embs = get_embeddings(SEMANTIC_QUERIES)
    # Optionally define per-query thresholds here
    per_query_threshold: Dict[str, float] = {
        "événement": 0.75,
        "event":     0.75,
        "salon":     0.80,
        # others default to SIM_THRESHOLD
    }

    # Clear any old labels
    c.execute("UPDATE chunks SET matched_queries=NULL WHERE company=?", (company_name,))
    conn.commit()

    # 3) Score each query, batch updates, and early-exit when enough matches
    total_matches = 0
    max_matches   = 100
    for q, qemb in zip(SEMANTIC_QUERIES, query_embs):
        thr = per_query_threshold.get(q, SIM_THRESHOLD)
        sims = cosine_similarity(all_embs, [qemb])[:, 0]

        updates = []
        for cid, idx in chunk_id_to_idx.items():
            if sims[idx] >= thr:
                # fetch existing labels only once
                c.execute(
                    "SELECT matched_queries FROM chunks WHERE company=? AND chunk_id=?",
                    (company_name, cid)
                )
                existing = c.fetchone()[0] or ""
                labels = existing.split(",") if existing else []
                if q not in labels:
                    labels.append(q)
                    updates.append((",".join(labels), company_name, cid))
                    total_matches += 1

        if updates:
            c.executemany(
                "UPDATE chunks SET matched_queries=? WHERE company=? AND chunk_id=?",
                updates
            )
            conn.commit()

        # early exit if we have enough
        if total_matches >= max_matches:
            break

    # 4) Rebuild semantic_items from the database
    report_semantic = []
    c.execute(
        "SELECT chunk_id, link, title, chunk, matched_queries "
        "FROM chunks WHERE company=? AND matched_queries IS NOT NULL",
        (company_name,)
    )
    for chunk_id, link, title, chunk, matched in c.fetchall():
        report_semantic.append({
            "chunk_id": chunk_id,
            "link": link,
            "title": title,
            "chunk": chunk,
            "matched_queries": matched.split(",")
        })

    conn.close()

    logger.info(
        "Report ready: %d semantic items, %d linkedin segments",
        len(report_semantic), len(linkedin_profile_chunks)
    )

    return {
        "semantic_items": report_semantic,
        "linkedin_profiles": linkedin_profile_chunks
    }
