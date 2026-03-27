from __future__ import annotations

import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from rag_engine import SimpleAgronomyRAG


app = FastAPI(title="RAG Service", version="1.0.0")
rag = SimpleAgronomyRAG(data_dir=os.getenv("DATA_DIR", "data")).build()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "rag", "sources": rag.sources()}


@app.get("/sources")
def sources() -> dict:
    return {"sources": rag.sources()}


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    result = rag.answer(payload.question, return_metadata=True)
    if payload.top_k:
        result["retrieved"] = rag.retrieve(payload.question, payload.top_k)
    return result
