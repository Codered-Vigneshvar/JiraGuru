"""Per-project lightweight vector index using JSON storage and Gemini embeddings."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai
from fastapi import HTTPException, status
from docx import Document
from pypdf import PdfReader

from ..config import settings
from ..models import Project, Ticket
from ..storage import ensure_project_doc_dir, get_project_by_id

# ----------------------- Embeddings helpers -----------------------


def _configure_gemini() -> None:
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GEMINI_API_KEY not configured.",
        )
    genai.configure(api_key=settings.gemini_api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using Gemini embeddings."""
    _configure_gemini()
    model_name = "models/text-embedding-004"
    vectors: List[List[float]] = []
    for text in texts:
        resp = genai.embed_content(model=model_name, content=text or "")
        vectors.append(resp["embedding"])
    return vectors


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b + 1e-8)


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def _read_doc_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return path.read_bytes().decode(errors="ignore")
    if suffix == ".docx":
        try:
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_bytes().decode(errors="ignore")


def _ticket_text(ticket: Ticket) -> str:
    parts = [
        f"Key: {ticket.key}",
        f"Title: {ticket.title}",
        f"Description: {ticket.description}",
        f"Status: {ticket.status}",
        f"Epic: {ticket.epic_key or ''}",
    ]
    if ticket.acceptance_criteria:
        parts.append("Acceptance Criteria: " + "; ".join(ticket.acceptance_criteria))
    if ticket.blockers:
        parts.append(f"Blockers: {ticket.blockers}")
    if ticket.dependencies:
        parts.append("Dependencies: " + ", ".join(ticket.dependencies))
    return "\n".join(parts)


# ----------------------- JSON index helpers -----------------------


class _IndexStore:
    def __init__(self, project_id: str):
        self.project_id = str(project_id)
        self.index_path = settings.data_dir / "vector_index"
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_path / f"{self.project_id}.json"

    def load(self) -> Dict:
        if not self.index_file.exists():
            return {"items": []}
        try:
            return json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            return {"items": []}

    def save(self, data: Dict) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ----------------------- ProjectIndex -----------------------


class ProjectIndex:
    """Lightweight JSON-backed vector index per project."""

    def __init__(self, project_id: str):
        self.project_id = str(project_id)
        self.store = _IndexStore(self.project_id)

    # --------------------- helpers ---------------------
    def _load_project(self) -> Project:
        return Project.model_validate(get_project_by_id(self.project_id))

    # --------------------- build / ensure ---------------------
    def ensure_built_full(self) -> None:
        data = self.store.load()
        if data.get("items"):
            return
        project = self._load_project()
        # Index docs
        for doc in project.documents:
            try:
                self.upsert_doc(doc.id)
            except Exception:
                continue
        # Index tickets
        for ticket in project.tickets:
            try:
                self.upsert_ticket(ticket.key)
            except Exception:
                continue

    # --------------------- doc indexing ---------------------
    def upsert_doc(self, doc_id: str) -> None:
        project = self._load_project()
        doc_meta = next((d for d in project.documents if d.id == doc_id), None)
        if not doc_meta:
            raise HTTPException(status_code=404, detail="Document not found for indexing.")
        doc_dir = ensure_project_doc_dir(self.project_id)
        path = doc_dir / doc_meta.filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Document file missing.")
        text = _read_doc_text(path)
        chunks = _chunk_text(text)
        if not chunks:
            return
        embeddings = embed_texts(chunks)
        data = self.store.load()
        items = [itm for itm in data.get("items", []) if not (itm.get("type") == "doc" and itm.get("doc_id") == doc_id)]
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            items.append(
                {
                    "id": f"doc:{doc_id}:chunk_{i}",
                    "type": "doc",
                    "doc_id": doc_id,
                    "ticket_key": None,
                    "metadata": {
                        "filename": doc_meta.filename,
                        "chunk_id": i,
                    },
                    "embedding": emb,
                    "text": chunk,
                }
            )
        data["items"] = items
        self.store.save(data)

    # --------------------- ticket indexing ---------------------
    def upsert_ticket(self, ticket_key: str) -> None:
        project = self._load_project()
        ticket = next((t for t in project.tickets if t.key == ticket_key), None)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found for indexing.")
        text = _ticket_text(ticket)
        embedding = embed_texts([text])[0]
        data = self.store.load()
        items = [itm for itm in data.get("items", []) if not (itm.get("type") == "ticket" and itm.get("ticket_key") == ticket_key)]
        items.append(
            {
                "id": f"ticket:{ticket.key}",
                "type": "ticket",
                "doc_id": None,
                "ticket_key": ticket.key,
                "metadata": {
                    "ticket_key": ticket.key,
                    "status": ticket.status,
                    "epic_key": ticket.epic_key,
                    "title": ticket.title,
                },
                "embedding": embedding,
                "text": text,
            }
        )
        data["items"] = items
        self.store.save(data)

    # --------------------- delete helpers ---------------------
    def delete_doc(self, doc_id: str) -> None:
        data = self.store.load()
        data["items"] = [itm for itm in data.get("items", []) if not (itm.get("type") == "doc" and itm.get("doc_id") == doc_id)]
        self.store.save(data)

    def delete_ticket(self, ticket_key: str) -> None:
        data = self.store.load()
        data["items"] = [itm for itm in data.get("items", []) if not (itm.get("type") == "ticket" and itm.get("ticket_key") == ticket_key)]
        self.store.save(data)

    # --------------------- search ---------------------
    def search_tickets(self, query_text: str, k: int = 20) -> List[Dict]:
        if not query_text:
            return []
        data = self.store.load()
        items = [itm for itm in data.get("items", []) if itm.get("type") == "ticket"]
        if not items:
            return []
        query_vec = embed_texts([query_text])[0]
        scored: List[Dict] = []
        for itm in items:
            emb = itm.get("embedding") or []
            score = _cosine(query_vec, emb)
            scored.append(
                {
                    "content": itm.get("text", ""),
                    "metadata": itm.get("metadata", {}),
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
