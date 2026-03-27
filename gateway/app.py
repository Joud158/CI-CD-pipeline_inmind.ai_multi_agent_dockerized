from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from db import get_notes, get_schema, init_db, wait_for_db
from supervisor import handle_message


app = FastAPI(title="Agent Gateway", version="1.0.0")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message for the multi-agent lab")


@app.on_event("startup")
def startup() -> None:
    wait_for_db()
    init_db()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "gateway",
        "public_entrypoint": True,
        "private_dependencies": ["rag", "db"],
    }


@app.get("/schema")
def schema() -> dict:
    return {"schema": get_schema()}


@app.get("/notes")
def notes() -> dict:
    return {"notes": get_notes(limit=50)}


@app.post("/chat")
def chat(payload: ChatRequest) -> dict:
    try:
        return handle_message(payload.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
