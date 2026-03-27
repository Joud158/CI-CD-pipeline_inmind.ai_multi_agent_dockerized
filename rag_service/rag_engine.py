from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "but", "by", "do", "for",
    "from", "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "should",
    "the", "to", "what", "when", "with", "you", "your"
}


@dataclass
class Doc:
    text: str
    source: str


class SimpleAgronomyRAG:
    def __init__(self, data_dir: str = "data", chunk_size: int = 420, chunk_overlap: int = 60, top_k: int = 3) -> None:
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.docs: List[Doc] = []
        self.chunk_texts: List[str] = []
        self.metas: List[Dict[str, str]] = []
        self.doc_names: List[str] = []

    def load_txt_docs(self) -> List[Doc]:
        docs: List[Doc] = []
        for p in sorted(self.data_dir.glob("*.txt")):
            docs.append(Doc(text=p.read_text(encoding="utf-8"), source=p.name))
        if not docs:
            raise FileNotFoundError(f"No .txt files found in {self.data_dir}")
        self.doc_names = [d.source for d in docs]
        return docs

    def chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def build(self) -> "SimpleAgronomyRAG":
        self.docs = self.load_txt_docs()
        for doc in self.docs:
            for i, chunk in enumerate(self.chunk_text(doc.text)):
                self.chunk_texts.append(chunk)
                self.metas.append({"source": doc.source, "chunk_id": f"{doc.source}::chunk{i}"})
        return self

    def tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if t not in STOPWORDS]

    def score_chunk(self, query: str, chunk: str, source: str) -> float:
        query_tokens = self.tokenize(query)
        chunk_tokens = self.tokenize(chunk)
        if not query_tokens or not chunk_tokens:
            return 0.0
        qset = set(query_tokens)
        cset = set(chunk_tokens)
        overlap = len(qset & cset)
        score = overlap / math.sqrt(max(len(qset), 1) * max(len(cset), 1))

        lowered_query = query.lower()
        lowered_chunk = chunk.lower()

        boosts = {
            "powdery mildew": 2.0,
            "zucchini": 1.4,
            "irrigation": 1.2,
            "soil": 1.1,
            "pricing": 1.1,
            "water": 1.0,
        }
        for phrase, boost in boosts.items():
            if phrase in lowered_query and phrase in lowered_chunk:
                score += boost

        if "save" in lowered_query or "note" in lowered_query:
            if source == "05_farm_business_pricing.txt":
                score += 0.3
        return score

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        actual_top_k = top_k or self.top_k
        ranked: List[Tuple[float, int]] = []
        for idx, chunk in enumerate(self.chunk_texts):
            source = self.metas[idx]["source"]
            score = self.score_chunk(query, chunk, source)
            ranked.append((score, idx))
        ranked.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for score, idx in ranked[:actual_top_k]:
            if score <= 0:
                continue
            results.append({
                "score": round(score, 4),
                "chunk_id": self.metas[idx]["chunk_id"],
                "source": self.metas[idx]["source"],
                "text": self.chunk_texts[idx],
            })
        return results

    def fallback_answer_from_context(self, retrieved: List[Dict[str, Any]]) -> str:
        if not retrieved:
            return "I don't know based on the provided documents."

        lines: List[str] = []
        seen: set[str] = set()
        for item in retrieved:
            chunk_id = item["chunk_id"]
            for raw_line in item["text"].splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("-") or line.startswith("Decision note"):
                    candidate = f"{line} [{chunk_id}]"
                    if candidate not in seen:
                        lines.append(candidate)
                        seen.add(candidate)
                if len(lines) >= 5:
                    break
            if len(lines) >= 5:
                break

        if not lines:
            first = retrieved[0]
            return f"Relevant context found in [{first['chunk_id']}], but no concise action bullets were available."

        return "\n".join(lines)

    def answer(self, question: str, return_metadata: bool = False) -> Dict[str, Any] | str:
        retrieved = self.retrieve(question, self.top_k)
        answer = self.fallback_answer_from_context(retrieved)
        payload = {
            "question": question,
            "retrieved": retrieved,
            "answer": answer,
        }
        return payload if return_metadata else answer

    def sources(self) -> List[str]:
        return list(self.doc_names)
