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


# Split long texts into embedding-safe pieces:
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


def parse_linkedin_title(title: str) -> Dict[str, str]:
    parts = [p.strip() for p in title.split(" - ")]
    if len(parts) == 2:
        name, company = parts
        return {"name": name, "job": "", "company": company}
    if len(parts) == 3:
        name, job, company = parts
        return {"name": name, "job": job, "company": company}
    return {"name": title, "job": "", "company": ""}


def report(company_name: str) -> Dict[str, List[Dict]]:
    company_name = company_name.lower()
    logger = setup_logger(company_name)
    conn = init_db()
    c = conn.cursor()

    # 1) Fetch raw results
    raw = google_cse_search(
        company_name,
        api_key=os.getenv('google_key'),
        cx=os.getenv('google_cse_id'),
        structured=True
    )
    results = raw if isinstance(raw, (list, tuple)) else json.loads(raw)

    # 2) Clear & re-chunk all current results
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
                " (company,chunk_id,link,title,chunk,is_linkedin,matched_queries)"
                " VALUES (?,?,?,?,?,?,?)",
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

    # 3) Embed & filter semantic queries on non-linkedin
    text_list = [c['chunk'] for c in all_chunks]
    report_semantic = []
    if text_list:
        chunk_embs = get_embeddings(text_list)
        query_embs = get_embeddings(SEMANTIC_QUERIES)
        sims       = cosine_similarity(chunk_embs, query_embs)
        for item, scores in zip(all_chunks, sims):
            matched = [q for q,s in zip(SEMANTIC_QUERIES,scores) if s >= SIM_THRESHOLD]
            if matched:
                report_semantic.append({**item, 'matched_queries': matched})
                c.execute(
                    "UPDATE chunks SET matched_queries=? WHERE company=? AND chunk_id=?",
                    (','.join(matched), company_name, item['chunk_id'])
                )
        conn.commit()

    conn.close()
    logger.info("Report ready: %d semantic, %d linkedin segments",
                len(report_semantic), len(linkedin_profile_chunks))

    return {
        'semantic_items': report_semantic,
        'linkedin_profiles': linkedin_profile_chunks
    }

