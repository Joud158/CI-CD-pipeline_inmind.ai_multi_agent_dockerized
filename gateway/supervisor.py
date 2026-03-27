from __future__ import annotations

import os
import re
from typing import Any, Tuple

import requests

from db import get_notes, get_schema, save_note


RAG_URL = os.getenv("RAG_URL", "http://rag:8001")


BLOCKED_USER_PATTERNS = [
    r"ignore previous instructions",
    r"reveal your system prompt",
    r"developer message",
    r"system prompt",
    r"rm -rf",
    r"drop table",
    r"delete from",
]


BLOCKED_SQL_PATTERNS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bALTER\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bPRAGMA\b",
    r"\bVACUUM\b",
    r"\bTRIGGER\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
]


def validate_user_message(text: str) -> str | None:
    lowered = text.lower()
    for pattern in BLOCKED_USER_PATTERNS:
        if re.search(pattern, lowered):
            return f"Blocked by guardrail: the message matched '{pattern}'."
    return None


def validate_sql(sql_query: str) -> str | None:
    cleaned = sql_query.strip()
    if not cleaned:
        return "Blocked: empty SQL query."
    if ";" in cleaned[:-1]:
        return "Blocked: only one SQL statement is allowed."
    upper = cleaned.upper()
    for pattern in BLOCKED_SQL_PATTERNS:
        if re.search(pattern, upper):
            return f"Blocked SQL pattern: {pattern}"
    if not upper.startswith("SELECT"):
        return "Blocked: the SQL executor is read-only and only allows SELECT."
    if "FARMER_NOTES" not in upper:
        return "Blocked: query only farmer_notes."
    return None


def extract_note_and_tag(user_message: str) -> Tuple[str, str]:
    text = user_message.strip()
    lowered = text.lower()
    tag = "general"
    if "disease" in lowered or "mildew" in lowered or "aphid" in lowered:
        tag = "diseases"
    elif "soil" in lowered or "fertilizer" in lowered or "nitrogen" in lowered:
        tag = "soil"
    elif "water" in lowered or "irrigation" in lowered:
        tag = "irrigation"

    prefixes = [
        "save a note that",
        "save that",
        "save note that",
        "remember that",
        "store that",
        "store a note that",
    ]
    cleaned = text
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = text[len(prefix):].strip()
            break
    cleaned = cleaned.rstrip(".")
    return cleaned, tag


def is_save_request(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(
        phrase in lowered
        for phrase in [
            "save a note",
            "save that",
            "remember that",
            "store that",
            "store a note",
        ]
    )


def is_sql_or_list_request(user_message: str) -> bool:
    lowered = user_message.lower()
    return (
        "use sql" in lowered
        or "show all saved notes" in lowered
        or "list saved notes" in lowered
        or "show saved notes" in lowered
    )


def call_rag_specialist(query: str) -> dict[str, Any]:
    response = requests.post(
        f"{RAG_URL}/ask",
        json={"question": query, "top_k": 3},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "route": "rag_specialist",
        "answer": payload["answer"],
        "retrieved": payload["retrieved"],
    }


def execute_safe_sql(sql_query: str) -> dict[str, Any]:
    block_reason = validate_sql(sql_query)
    if block_reason:
        return {"route": "execute_safe_sql", "answer": block_reason, "rows": []}
    rows = get_notes(limit=50)
    return {
        "route": "execute_safe_sql",
        "answer": str(rows),
        "rows": rows,
        "schema": get_schema(),
    }


def handle_message(user_message: str) -> dict[str, Any]:
    block_reason = validate_user_message(user_message)
    if block_reason:
        return {"route": "guardrail", "answer": block_reason}

    if is_save_request(user_message):
        note, tag = extract_note_and_tag(user_message)
        result = save_note(note, tag)
        return {"route": "save_note", "answer": result, "tag": tag, "note": note}

    if is_sql_or_list_request(user_message):
        sql = "SELECT id, note, tag, created_at FROM farmer_notes ORDER BY id DESC"
        return execute_safe_sql(sql)

    rag_result = call_rag_specialist(user_message)
    return rag_result
