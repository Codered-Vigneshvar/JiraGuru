"""Requirement document upload and extraction endpoints."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from docx import Document
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pypdf import PdfReader

from ..config import settings
from ..models import RequirementFile
from ..storage import ensure_project_dirs, load_json, save_json
from .sprints import assert_sprint_exists

router = APIRouter(
    prefix="/projects/{project_id}/sprints/{sprint_id}/requirements",
    tags=["requirements"],
)


@router.post("", response_model=RequirementFile, status_code=status.HTTP_201_CREATED)
async def upload_requirement(
    project_id: int, sprint_id: int, file: UploadFile = File(...)
) -> RequirementFile:
    """Upload a requirement document, extract text, and store chunks."""
    assert_sprint_exists(project_id, sprint_id)
    text = await _extract_text(file)
    if not text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty document.")

    req_dir = ensure_project_dirs(settings.data_dir, project_id, sprint_id)
    base_name = Path(file.filename or "requirement").stem
    raw_path = req_dir / f"{base_name}.txt"
    raw_path.write_text(text, encoding="utf-8")

    chunks = _chunk_text(text)
    chunks_file = req_dir / "chunks.json"
    chunk_store = load_json(chunks_file, default=[])
    chunk_store.append({"file": raw_path.name, "chunks": chunks})
    save_json(chunks_file, chunk_store)

    return RequirementFile(
        project_id=project_id,
        sprint_id=sprint_id,
        file_name=raw_path.name,
        text=text,
        chunks=chunks,
    )


@router.get("/chunks", response_model=List[dict])
def list_chunks(project_id: int, sprint_id: int) -> List[dict]:
    """Return stored chunks for a sprint's requirements."""
    assert_sprint_exists(project_id, sprint_id)
    req_dir = ensure_project_dirs(settings.data_dir, project_id, sprint_id)
    chunks_file = req_dir / "chunks.json"
    return load_json(chunks_file, default=[])


async def _extract_text(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    content = await file.read()

    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if suffix == ".docx":
        document = Document(io.BytesIO(content))
        return "\n".join(p.text for p in document.paragraphs)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported file type. Use PDF, DOCX, or TXT.",
    )


def _chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    """Chunk text into overlapping segments for later processing."""
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]
