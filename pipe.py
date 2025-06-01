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
    'webinar',
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
    text = text.strip()
    if not text:
        return []  # skip empty inputs
    token_ids = ENC.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]
    return [ENC.decode(token_ids[i:i+max_tokens]) for i in range(0, len(token_ids), max_tokens)]


MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, multiplied by attempt count

def get_embeddings(texts: List[str]) -> List[List[float]]:
    print(f"[DEBUG] Received {len(texts)} original texts")
    
    safe_texts: List[str] = []
    for txt in texts:
        chunks = embed_clip_or_split(txt)
        if not chunks:
            print(f"[WARN] Skipped input (empty or too short): '{txt[:60]}'")
        safe_texts.extend(chunks)

    print(f"[DEBUG] Total safe texts to embed: {len(safe_texts)}")
    assert len(safe_texts) > 0, "No valid chunks produced; embedding skipped"

    all_embs: List[List[float]] = []

    for i in range(0, len(safe_texts), BATCH_SIZE):
        batch = safe_texts[i:i+BATCH_SIZE]
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    timeout=30
                )
                if not resp.data:
                    print(f"[WARN] Empty embeddings returned for batch starting with: {batch[0][:60]}")
                all_embs.extend(d.embedding for d in resp.data)
                break
            except (ReadTimeout, HTTPStatusError) as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    raise
                wait_time = RETRY_BACKOFF ** attempt
                print(f"[ERROR] Embedding failed (retry {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(wait_time)

    print(f"[DEBUG] Total embeddings returned: {len(all_embs)}")
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
    logger.info(f"[DEBUG] report() called with company_name={company_name}")
    company_name = company_name.lower()

    conn = init_db()
    c = conn.cursor()

    # ─── CACHING CHECK ────────────────────────────
    if use_cache:
        c.execute("SELECT MAX(timestamp) FROM chunks WHERE company=?", (company_name,))
        row = c.fetchone()
        if row and row[0]:
            last_updated = datetime.datetime.fromisoformat(row[0])
            recent_enough = (datetime.datetime.now() - last_updated).days < refresh_days

            if recent_enough:
                # Check for at least one matched query (semantic match)
                c.execute("""
                    SELECT COUNT(*) FROM chunks 
                    WHERE company=? AND matched_queries IS NOT NULL
                """, (company_name,))
                match_count = c.fetchone()[0]

                if match_count > 0:
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
                else:
                    logger.warning(f"No semantic matches in cache for '{company_name}' — recrawling.")

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


    # ─── EMBEDDING & SEMANTIC FILTER ───────────────────────────────────────────
    # 1) Gather or compute all embeddings & build chunk_id → row-index map
    all_embs: List[List[float]] = []
    chunk_id_to_idx: Dict[int, int] = {}
    next_idx = 0
    has_valid_embeddings = False

    # Reuse cached embeddings
    for item in all_chunks:
        cid = item["chunk_id"]
        c.execute(
            "SELECT embedding FROM chunks WHERE company=? AND chunk_id=?",
            (company_name, cid)
        )
        row = c.fetchone()
        if row and row[0]:
            try:
                emb = json.loads(row[0])
                if emb:  # Verify it's not empty
                    all_embs.append(emb)
                    chunk_id_to_idx[cid] = next_idx
                    next_idx += 1
                    has_valid_embeddings = True
                else:
                    # Empty embedding, clear from DB
                    logger.info(f"Clearing empty embedding for chunk {cid}")
                    c.execute(
                        "UPDATE chunks SET embedding=NULL WHERE company=? AND chunk_id=?",
                        (company_name, cid)
                    )
                    conn.commit()
            except json.JSONDecodeError:
                logger.warning(f"Invalid embedding JSON for chunk {cid}, clearing embedding")
                c.execute(
                    "UPDATE chunks SET embedding=NULL WHERE company=? AND chunk_id=?",
                    (company_name, cid)
                )
                conn.commit()

    # Embed & store any new chunks
    MAX_EMBED_RETRIES = 3
    new_items = [it for it in all_chunks if it["chunk_id"] not in chunk_id_to_idx]

    if new_items:
        texts = [it["chunk"] for it in new_items]
        new_embs = []
        attempt = 0

        while attempt < MAX_EMBED_RETRIES:
            print("[DEBUG] About to call get_embeddings()")
            new_embs = get_embeddings(texts)
            print("[DEBUG] Returned from get_embeddings()")
            if len(new_embs) == len(new_items):
                break  # Success, embeddings count matches chunks
            attempt += 1
            logger.warning(f"Embedding retry {attempt}/{MAX_EMBED_RETRIES} for {len(new_items)} chunks")
            time.sleep(2 ** attempt)  # Exponential backoff before retry

        if len(new_embs) != len(new_items):
            logger.error(
                f"Embedding count mismatch after retries: expected {len(new_items)} "
                f"but got {len(new_embs)}"
            )

        for item, emb in zip(new_items, new_embs):
            if emb:  # Only store non-empty embeddings
                cid = item["chunk_id"]
                c.execute(
                    "UPDATE chunks SET embedding=? WHERE company=? AND chunk_id=?",
                    (json.dumps(emb), company_name, cid)
                )
                all_embs.append(emb)
                chunk_id_to_idx[cid] = next_idx
                next_idx += 1
                has_valid_embeddings = True

        conn.commit()

    # Final verification
    if not has_valid_embeddings:
        logger.warning(f"No valid embeddings found for company '{company_name}'")
        return {
            'semantic_items': [],
            'linkedin_profiles': linkedin_profile_chunks
        }

    logger.debug(f"Total embeddings collected: {len(all_embs)}")
    logger.debug(f"Chunk ID mapping: {chunk_id_to_idx}")

    # 2) Pre-embed all queries once
    query_embs = get_embeddings(SEMANTIC_QUERIES)
    if len(query_embs) != len(SEMANTIC_QUERIES):
        raise RuntimeError(
            f"[ERROR] Expected {len(SEMANTIC_QUERIES)} query embeddings, "
            f"but got {len(query_embs)} — possible embedding failure."
        )
    # Optionally define per-query thresholds here
    per_query_threshold: Dict[str, float] = {
        "événement": 0.75,
        "event": 0.75,
        "salon": 0.75,
        "secteur ou chiffre d'affaires (CA)": 0.75,
        "rebranding": 0.75,
        "nombre de postes ouverts": 0.75,
        "nombre de salariés": 0.75,
        "salariés": 0.75,
        'webinar': 0.75,
        "employés": 0.75,
        "collaborateurs": 0.75,
        "cse": 0.75,
        "nombre d'emploés":0.75,
        # others default to SIM_THRESHOLD
    }

    # Clear any old labels
    c.execute("UPDATE chunks SET matched_queries=NULL WHERE company=?", (company_name,))
    conn.commit()

    # 3) Score each query, batch updates, and early-exit when enough matches
    MAX_MATCHES_PER_QUERY = 6
    MAX_TOTAL_MATCHES = 100
    MIN_SIM_THRESHOLD = 0.2  # Don't lower below this

    top_matches_by_query = {}
    all_candidates = []
    seen_pairs = set()
    # At the start of the semantic matching section (before the loop)
    if not all_embs:
        logger.warning(f"No embeddings found for company '{company_name}' - skipping semantic matching")
        return {
            'semantic_items': [],
            'linkedin_profiles': linkedin_profile_chunks
        }

    if len(query_embs) != len(SEMANTIC_QUERIES):
        logger.error(
            f"Embedding count mismatch: expected {len(SEMANTIC_QUERIES)} "
            f"but got {len(query_embs)} - skipping semantic matching"
        )
        return {
            'semantic_items': [],
            'linkedin_profiles': linkedin_profile_chunks
        }
    logger.info("Starting semantic match selection for %d queries", len(SEMANTIC_QUERIES))

    for q, qemb in zip(SEMANTIC_QUERIES, query_embs):
        thr = per_query_threshold.get(q, SIM_THRESHOLD)
        sims = cosine_similarity(all_embs, [qemb])[:, 0]
        
        matches = []
        step = 0.02
        current_thr = thr
        attempts = 0

        logger.info("Query '%s': initial threshold = %.2f", q, current_thr)

        while len(matches) < MAX_MATCHES_PER_QUERY and current_thr >= MIN_SIM_THRESHOLD:
            last_thr_used = current_thr
            matches = [
                (sims[idx], q, cid)
                for cid, idx in chunk_id_to_idx.items()
                if sims[idx] >= current_thr and (q, cid) not in seen_pairs
            ]
            matches.sort(reverse=True)
            if len(matches) >= MAX_MATCHES_PER_QUERY:
                matches = matches[:MAX_MATCHES_PER_QUERY]
                break
            current_thr -= step
            attempts += 1

        logger.info("Query '%s': found %d matches after %d attempts (final threshold = %.2f)", 
                    q, len(matches), attempts, last_thr_used)

        top_matches_by_query[q] = matches
        seen_pairs.update((q, cid) for _, q, cid in matches)

        # Collect remaining candidates above minimum threshold
        extras = [
            (sims[idx], q, cid)
            for cid, idx in chunk_id_to_idx.items()
            if sims[idx] >= MIN_SIM_THRESHOLD and (q, cid) not in seen_pairs
        ]
        all_candidates.extend(extras)

    # Combine matches from all queries
    final_matches = []
    for matches in top_matches_by_query.values():
        final_matches.extend(matches)

    logger.info("Initial total matches from top queries: %d", len(final_matches))

    # Global fill if under 100
    if len(final_matches) < MAX_TOTAL_MATCHES:
        remaining_needed = MAX_TOTAL_MATCHES - len(final_matches)
        logger.info("Filling remaining %d matches from global leftovers", remaining_needed)

        all_candidates.sort(reverse=True)
        for score, q, cid in all_candidates:
            if (q, cid) not in seen_pairs:
                final_matches.append((score, q, cid))
                seen_pairs.add((q, cid))
                if len(final_matches) >= MAX_TOTAL_MATCHES:
                    break

    logger.info("Final match count: %d", len(final_matches))

    # DB updates
    updates_by_chunk = {}
    for _, q, cid in final_matches:
        c.execute(
            "SELECT matched_queries FROM chunks WHERE company=? AND chunk_id=?",
            (company_name, cid)
        )
        existing = c.fetchone()[0] or ""
        labels = existing.split(",") if existing else []

        if q not in labels:
            labels.append(q)
            updates_by_chunk[cid] = ",".join(labels)

    updates = [(labels, company_name, cid) for cid, labels in updates_by_chunk.items()]
    if updates:
        logger.info("Writing %d chunk updates to database", len(updates))
        c.executemany(
            "UPDATE chunks SET matched_queries=? WHERE company=? AND chunk_id=?",
            updates
        )
        conn.commit()
    else:
        logger.info("No new updates to write.")



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
