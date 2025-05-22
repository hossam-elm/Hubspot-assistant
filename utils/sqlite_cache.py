# utils/sqlite_cache.py
import sqlite3
import threading
import time
import json
from pathlib import Path

CACHE_TTL_SECS = 24 * 3600  # 1 day

class SQLiteCache:
    def __init__(self, path: Path, table: str):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        # enable WAL mode for concurrent readers/writers
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.table = table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                key TEXT PRIMARY KEY,
                js  TEXT,
                ts  REAL
            )
        """)
        self.conn.commit()

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