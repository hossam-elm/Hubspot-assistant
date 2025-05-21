import json
import re
import logging
from pathlib import Path
from typing import List, Dict
import sqlite3
import os
import datetime
import tiktoken
from openai import OpenAI
from searchfuncs.searchgoogle import google_cse_search
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIG ────────────────────────────────────────────────────────────────
MAX_CHUNK_WORDS = 200
SIM_THRESHOLD   = 0.8
BATCH_SIZE      = 1000
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
    "informations sur agicap",
    "événement ou webinar",
    "secteur ou chiffre d'affaires (CA)",
    "rebranding",
    "nombre de postes ouverts",
    "nombre de salariés",
    "bureaux ou siège social",
]

# ─── DIRECTORIES ───────────────────────────────────────────────────────────
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
DB_PATH   = CACHE_DIR / "reports.db"
LOG_DIR   = CACHE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

client = OpenAI()

# ─── UTILS ─────────────────────────────────────────────────────────────────
def setup_logger(company: str) -> logging.Logger:
    logger = logging.getLogger(company)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(LOG_DIR / f"{company}_{ts}.log", encoding="utf-8")
        fmt = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

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

def embed_clip_or_split(text: str, max_tokens: int = MAX_EMBED_TOKENS) -> List[str]:
    token_ids = ENC.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]
    return [ENC.decode(token_ids[i:i+max_tokens]) for i in range(0, len(token_ids), max_tokens)]

def get_embeddings(texts: List[str]) -> List[List[float]]:
    safe_texts: List[str] = []
    for txt in texts:
        safe_texts.extend(embed_clip_or_split(txt))
    all_embs: List[List[float]] = []
    for i in range(0, len(safe_texts), BATCH_SIZE):
        batch = safe_texts[i:i+BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embs.extend(d.embedding for d in resp.data)
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
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(company, chunk_id)
    )""")
    conn.commit()
    return conn

# ─── REPORT FUNCTION ───────────────────────────────────────────────────────

def report(company_name: str) -> List[Dict]:
    logger = setup_logger(company_name)
    conn = init_db()
    c = conn.cursor()

    # 1) Fetch (or load from google_cse_search's cache) raw hits:
    logger.info("Fetching raw results for %s via google_cse_search()", company_name)
    raw = google_cse_search(
        company_name,
        api_key=os.getenv('google_key'),
        cx=os.getenv('google_cse_id'),
        structured=True
    )
    results = raw if isinstance(raw, (list, tuple)) else json.loads(raw)

    # 2) Load existing chunks for this company:
    c.execute("SELECT chunk_id, link, title, chunk, is_linkedin, matched_queries "
              "FROM chunks WHERE company = ?", (company_name,))
    saved = c.fetchall()

    if saved:
        logger.info("Loaded %d cached chunks for %s", len(saved), company_name)
        all_chunks = [
            dict(chunk_id=r[0], link=r[1], title=r[2], chunk=r[3])
            for r in saved if not r[4]
        ]
        linkedin_chunks = [
            dict(chunk_id=r[0], link=r[1], title=r[2], chunk=r[3], matched_queries=r[5].split(','))
            for r in saved if r[4]
        ]

    else:
        # 3) Chunk and persist every snippet into `chunks` table:
        logger.info("No cached chunks: chunking %d results for %s", len(results), company_name)
        all_chunks = []
        linkedin_chunks = []
        idx = 0
        for entry in results:
            link    = normalize_whitespace(entry.get('link',''))
            title   = normalize_whitespace(entry.get('title',''))
            content = normalize_whitespace(entry.get('content') or entry['snippet'])
            for segment in chunk_text(content):
                idx += 1
                is_li     = 'linkedin.com/in' in link
                is_event  = any(k in segment.lower() for k in ("événement", "event"))
                is_money  = any(k in segment.lower() for k in ("euros", "€", "dollars", "$"))
                is_webinar= "webinar" in segment.lower()
                c.execute("""
                  INSERT OR IGNORE INTO chunks
                    (company, chunk_id, link, title, chunk, is_linkedin, matched_queries)
                  VALUES (?,?,?,?,?,?,?)
                """, (company_name, idx, link, title, segment, int(is_li), ''))
                item = dict(chunk_id=idx, link=link, title=title, chunk=segment)
                if is_li or is_event or is_money or is_webinar:
                    linkedin_chunks.append({**item, 'matched_queries': []})
                else:
                    all_chunks.append(item)
        conn.commit()
        logger.info("Inserted %d new chunks for %s", idx, company_name)

    # 4) Compute embeddings & semantic matches on non‐LinkedIn chunks
    texts     = [c['chunk'] for c in all_chunks]
    if texts:
        chunk_embs = get_embeddings(texts)
        query_embs = get_embeddings(SEMANTIC_QUERIES)
        sims       = cosine_similarity(chunk_embs, query_embs)

        report_items = []
        seen_ids     = set()
        for (item, scores) in zip(all_chunks, sims):
            matched = [q for q,s in zip(SEMANTIC_QUERIES, scores) if s >= SIM_THRESHOLD]
            if matched:
                report_items.append({**item, 'matched_queries': matched})
                seen_ids.add(item['chunk_id'])
                c.execute("""
                  UPDATE chunks
                     SET matched_queries=?
                   WHERE company=? AND chunk_id=?
                """, (','.join(matched), company_name, item['chunk_id']))
        # include any LinkedIn chunks not already in report
        for li in linkedin_chunks:
            if li['chunk_id'] not in seen_ids:
                report_items.append(li)
        conn.commit()

    else:
        report_items = linkedin_chunks  # only LinkedIn or pre‐tagged chunks

    conn.close()
    logger.info("Report ready: %d items for %s", len(report_items), company_name)
    return report_items
