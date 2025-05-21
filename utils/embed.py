"""
embedding_filter.py
-------------------
Drop-in helper that

1. keeps a lightweight SQLite cache of OpenAI embeddings (`embed_cache`)
2. computes cosine similarity between the user query and every candidate
3. returns the top-k most-similar items (adds a `sim` score to each dict)

Dependencies
------------
pip install openai numpy
"""

from __future__ import annotations

import json, sqlite3, time, pathlib
from typing import List, Dict
import numpy as np
from openai import OpenAI

# ── OpenAI client (reuse one instance across your project) ─────────────── #
client = OpenAI()                            # assumes OPENAI_API_KEY env var
EMBED_MODEL        = "text-embedding-3-small"
CACHE_TTL_SECS     = 24 * 3600               # 1 day
CACHE_PATH         = pathlib.Path(__file__).with_name("search_cache.sqlite3")

# ── tiny cache layer (shares the same file as search/scrape caches) ────── #
def _db() -> sqlite3.Connection:
    conn = getattr(_db, "_conn", None)
    if conn is None:
        conn = sqlite3.connect(CACHE_PATH)
        conn.execute("""CREATE TABLE IF NOT EXISTS embed_cache(
                            key TEXT PRIMARY KEY,
                            vec TEXT,
                            ts  REAL)""")
        _db._conn = conn
    return conn

def _embed_cache_get(text: str) -> List[float] | None:
    row = _db().execute(
        "SELECT vec, ts FROM embed_cache WHERE key=?", (text,)
    ).fetchone()
    if row and time.time() - row[1] < CACHE_TTL_SECS:
        return json.loads(row[0])
    return None

def _embed_cache_set(text: str, vec: List[float]) -> None:
    _db().execute(
        "INSERT OR REPLACE INTO embed_cache VALUES (?,?,?)",
        (text, json.dumps(vec), time.time())
    )
    _db().commit()

# ── similarity helper ─────────────────────────────────────────────────── #
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _get_vecs(texts: List[str]) -> List[List[float]]:
    """
    Retrieve embeddings from cache when available; otherwise call OpenAI
    and persist the new vectors.
    """
    out, to_embed = [], []
    needed_idxs   = []

    for idx, txt in enumerate(texts):
        cached = _embed_cache_get(txt)
        if cached is not None:
            out.append(cached)
        else:
            out.append(None)
            to_embed.append(txt)
            needed_idxs.append(idx)

    if to_embed:
        rsp = client.embeddings.create(model=EMBED_MODEL, input=to_embed)
        for pos, data in enumerate(rsp.data):
            vec = data.embedding
            out[needed_idxs[pos]] = vec
            _embed_cache_set(to_embed[pos], vec)

    return out

# ── public API ─────────────────────────────────────────────────────────── #
def filter_by_embedding(
    items: List[Dict],
    user_question: str,
    top_k: int = 20,
    similarity_floor: float = 0.30,
) -> List[Dict]:
    """
    Trim `items` (list[dict] from google_cse_search structured=True) to the
    K most-relevant entries, using embedding cosine similarity.
    """

    # 1️⃣  embed the user query
    q_vec = np.array(_get_vecs([user_question])[0], dtype=np.float32)

    # 2️⃣  embed every candidate (title + snippet)
    texts     = [f"{it['title']} {it['snippet']}" for it in items]
    doc_vecs  = [np.array(v, dtype=np.float32) for v in _get_vecs(texts)]

    # 3️⃣  score & sort
    for it, vec in zip(items, doc_vecs):
        it["sim"] = _cosine(q_vec, vec)

    ranked = sorted(items, key=lambda d: d["sim"], reverse=True)
    ranked = [it for it in ranked if it["sim"] >= similarity_floor]

    return ranked[:top_k]
