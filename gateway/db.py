from __future__ import annotations

import os
import time
from typing import Any, List

import psycopg2
from psycopg2.extras import RealDictCursor


DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "agronomy")
DB_USER = os.getenv("DB_USER", "agronomy")
DB_PASSWORD = os.getenv("DB_PASSWORD", "agronomy")


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def wait_for_db(max_attempts: int = 30, sleep_seconds: int = 2) -> None:
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            conn = get_connection()
            conn.close()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Database is not reachable: {last_error}")


def init_db() -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS farmer_notes (
                        id SERIAL PRIMARY KEY,
                        note TEXT NOT NULL,
                        tag TEXT DEFAULT 'general',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (note, tag)
                    )
                    '''
                )
    finally:
        conn.close()


def save_note(note: str, tag: str = "general") -> str:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM farmer_notes WHERE note = %s AND tag = %s LIMIT 1",
                    (note, tag),
                )
                if cur.fetchone():
                    return f"Note already exists under tag='{tag}'."
                cur.execute(
                    "INSERT INTO farmer_notes(note, tag) VALUES(%s, %s)",
                    (note, tag),
                )
                return f"Saved note under tag='{tag}'."
    finally:
        conn.close()


def get_notes(limit: int = 50) -> List[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, note, tag, created_at FROM farmer_notes ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        conn.close()


def get_schema() -> str:
    return (
        "farmer_notes(id SERIAL PRIMARY KEY, note TEXT NOT NULL, tag TEXT DEFAULT 'general', "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(note, tag))"
    )
