"""Gemini-powered epic and story generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import google.generativeai as genai
from fastapi import HTTPException, status

from ..config import settings
from ..models import Project, Ticket
from ..storage import (
    ensure_project_doc_dir,
    generate_new_epic_key,
    generate_new_ticket_key,
)

# Base path of the backend folder (backend/app/ai -> ../../)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "ai.log"


def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("jiraguru.ai")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


LOGGER = _setup_logger()


def log_ai_event(message: str, level: int = logging.INFO) -> None:
    """Write an AI event to the shared log file."""
    if level == logging.ERROR:
        LOGGER.error(message)
    elif level == logging.WARNING:
        LOGGER.warning(message)
    else:
        LOGGER.info(message)


def _load_project_documents(project: Project) -> str:
    """Read all project documents as text to feed the model."""
    doc_dir = ensure_project_doc_dir(project.id)
    parts: List[str] = []
    for doc in project.documents:
        file_path = doc_dir / doc.filename
        if not file_path.exists():
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            content = file_path.read_bytes().decode("utf-8", errors="ignore")
        parts.append(f"# Document: {doc.original_name}\n{content}")
    return "\n\n".join(parts) or "No document content available."


def _configure_gemini():
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GEMINI_API_KEY not configured.",
        )
    genai.configure(api_key=settings.gemini_api_key)


def _build_prompt(project: Project, content: str, plan: str | None = None) -> str:
    plan_section = f"\nExisting Requirement Plan (must be reflected in tickets):\n{plan}\n" if plan else ""
    schema = """
{
  "epics": [
    {
      "id": "PROJ-EP01",
      "title": "string",
      "description": "string"
    }
  ],
  "stories": [
    {
      "id": "PROJ-001",
      "epic_id": "PROJ-EP01",
      "title": "string",
      "description": "string",
      "acceptance_criteria": ["string"]
    }
  ]
}
""".strip()
    return f"""
You are a product manager creating execution-ready tickets. Stay 100% within the project documents and requirements plan—do NOT invent features that are not explicitly mentioned or clearly implied. If something is unclear, add a TODO note instead of guessing. Only generate epics and stories that are directly supported by the provided sources.

Project Code: {project.code}
Project Title: {project.title}
Project Description: {project.description}
{plan_section}

Documents:
{content}

Requirements for output:
- Return STRICT JSON ONLY (no markdown, no code fences) with this schema:
{schema}
- Use the project code "{project.code}" when building ids (e.g. {project.code}-EP01, {project.code}-001).
- Create as many epics and stories as needed; every EPIC must have multiple STORIES (at least 2) that it is split into.
- Every story must reference a valid epic_id (tag each story to its EPIC).
- Descriptions must be specific, technical, and traceable to the documents/plan (APIs, data flow, validation, edge cases, dependencies). If a detail is missing, insert a TODO note rather than inventing.
- acceptance_criteria must be a list of short, testable statements tied to functionality.
"""


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence) :]
            if cleaned.endswith("```"):
                cleaned = cleaned[: -len("```")]
            break
    return cleaned.strip()


def _parse_gemini_json(text: str) -> dict | None:
    if not text:
        return None

    variants = []
    cleaned = _strip_code_fences(text)
    variants.append(cleaned)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        variants.append(_strip_code_fences(snippet))

    for candidate in list(variants):
        # minor fallback for single-quoted JSON-like responses
        if "'" in candidate and '"' not in candidate:
            variants.append(candidate.replace("'", '"'))

    last_error: Exception | None = None
    for candidate in variants:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            last_error = exc
            continue

    log_ai_event(f"Gemini JSON parse failed: {last_error} | text={text[:500]}", level=logging.ERROR)
    return None


def validate_story_payload(payload: dict) -> Tuple[bool, List[str]]:
    """Validate the strict schema for epics and stories."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["Payload must be a JSON object."]

    epics = payload.get("epics")
    stories = payload.get("stories")
    if not isinstance(epics, list):
        errors.append("Field 'epics' must be a list.")
        epics = []
    if not isinstance(stories, list):
        errors.append("Field 'stories' must be a list.")
        stories = []

    epic_ids: set[str] = set()
    for idx, epic in enumerate(epics):
        if not isinstance(epic, dict):
            errors.append(f"epics[{idx}] must be an object.")
            continue
        for field in ("id", "title", "description"):
            value = epic.get(field)
            if not value or not isinstance(value, str):
                errors.append(f"epics[{idx}].{field} missing or invalid.")
        if isinstance(epic.get("id"), str):
            epic_ids.add(epic["id"])

    story_epic_map: Dict[str, int] = {}
    for idx, story in enumerate(stories):
        if not isinstance(story, dict):
            errors.append(f"stories[{idx}] must be an object.")
            continue
        for field in ("id", "epic_id", "title", "description"):
            value = story.get(field)
            if not value or not isinstance(value, str):
                errors.append(f"stories[{idx}].{field} missing or invalid.")
        ac = story.get("acceptance_criteria")
        if not isinstance(ac, list) or not ac:
            errors.append(f"stories[{idx}].acceptance_criteria must be a non-empty list.")
        elif any(not isinstance(item, str) or not item.strip() for item in ac):
            errors.append(f"stories[{idx}].acceptance_criteria items must be non-empty strings.")
        if epic_ids and isinstance(story.get("epic_id"), str) and story["epic_id"] not in epic_ids:
            errors.append(f"stories[{idx}].epic_id does not match any epic.")
        else:
            story_epic_map[story.get("epic_id")] = story_epic_map.get(story.get("epic_id"), 0) + 1

    if epic_ids:
        for eid in epic_ids:
            if story_epic_map.get(eid, 0) == 0:
                errors.append(f"No stories provided for epic {eid}. Every epic must have at least one story.")

    return len(errors) == 0, errors


def _generate_with_gemini(prompt: str) -> str:
    """Call Gemini with fallback models and return raw text."""
    # Try configured model first, then a short fallback list.
    model_candidates = []
    if settings.gemini_model:
        model_candidates.append(settings.gemini_model)
    model_candidates.extend(
        [
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-latest",
        ]
    )
    last_exc: Exception | None = None
    for name in model_candidates:
        try:
            model = genai.GenerativeModel(name)
            response = model.generate_content(prompt)
            text = response.text or ""
            log_ai_event(f"[Gemini Response] model={name} | {text}")
            return text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log_ai_event(f"[Gemini Error] model={name} | {exc}", level=logging.ERROR)
            continue
    raise HTTPException(status_code=502, detail=f"Gemini generation failed: {last_exc}") from last_exc


def generate_story_payload(project: Project, plan: str | None = None) -> Tuple[dict | None, str | None]:
    """Generate and validate epics + stories JSON from Gemini."""
    _configure_gemini()
    content = _load_project_documents(project)
    prompt = _build_prompt(project, content, plan)
    log_ai_event(f"[Gemini Request] project={project.id} | {prompt}")

    try:
        raw_text = _generate_with_gemini(prompt)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log_ai_event(f"[Gemini Error] project={project.id} | {exc}", level=logging.ERROR)
        return None, "Gemini generation failed."

    payload = _parse_gemini_json(raw_text)
    if payload is None:
        return None, "Invalid AI response"

    is_valid, errors = validate_story_payload(payload)
    if not is_valid:
        joined = "; ".join(errors)
        log_ai_event(f"[Validation Error] project={project.id} | {joined}", level=logging.ERROR)
        return None, f"Gemini JSON invalid: {errors[0]}"

    clean_epics = [
        {
            "id": str(epic.get("id", "")).strip(),
            "title": str(epic.get("title", "")).strip(),
            "description": str(epic.get("description", "")).strip(),
        }
        for epic in payload.get("epics", [])
    ]
    clean_stories = []
    for story in payload.get("stories", []):
        ac = [item.strip() for item in story.get("acceptance_criteria", []) if isinstance(item, str) and item.strip()]
        clean_stories.append(
            {
                "id": str(story.get("id", "")).strip(),
                "epic_id": str(story.get("epic_id", "")).strip(),
                "title": str(story.get("title", "")).strip(),
                "description": str(story.get("description", "")).strip(),
                "acceptance_criteria": ac,
            }
        )

    return {"epics": clean_epics, "stories": clean_stories}, None


def generate_epics_and_stories_for_project(project: Project, plan: str | None = None) -> List[Ticket]:
    """Generate epics and stories using Gemini and return Ticket objects."""
    payload, error = generate_story_payload(project, plan)
    if error or not payload:
        raise HTTPException(status_code=502, detail=error or "Gemini response invalid.")

    epics_raw = payload.get("epics", [])
    stories_raw = payload.get("stories", [])
    tickets: List[Ticket] = []

    project_dict = project.model_dump()
    existing_ids = [
        int(t.get("id", "t_0").split("_")[1])
        for t in project_dict.get("tickets", [])
        if str(t.get("id", "")).startswith("t_")
    ]
    next_id_num = max(existing_ids or [0]) + 1

    existing_keys = [t.get("key", "") for t in project_dict.get("tickets", [])]
    for idx, epic in enumerate(epics_raw, start=1):
        epic_key = epic.get("id") or generate_new_epic_key(project_dict, idx)
        epic_ticket = Ticket(
            id=f"t_{next_id_num}",
            key=epic_key,
            project_id=project.id,
            type="EPIC",
            title=epic.get("title", f"Epic {idx}"),
            description=epic.get("description", ""),
            status="TODO",
            assignee_username=None,
            epic_key=None,
            dependencies=[],
            blockers=None,
            linked_document_ids=[],
            priority="MEDIUM",
        )
        tickets.append(epic_ticket)
        next_id_num += 1

    for story in stories_raw:
        ac = story.get("acceptance_criteria") or []
        ticket = Ticket(
            id=f"t_{next_id_num}",
            key=story.get("id", generate_new_ticket_key(project_dict)),
            project_id=project.id,
            type="STORY",
            title=story.get("title", "Story"),
            description=story.get("description", ""),
            status="TODO",
            assignee_username=None,
            epic_key=story.get("epic_id"),
            dependencies=[],
            blockers=None,
            linked_document_ids=[],
            acceptance_criteria=[item for item in ac if isinstance(item, str) and item.strip()],
            priority="MEDIUM",
        )
        tickets.append(ticket)
        next_id_num += 1

    return tickets
