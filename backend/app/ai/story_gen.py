"""Gemini-powered epic and story generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import google.generativeai as genai
from fastapi import HTTPException, status

from ..config import settings
from ..models import Project, Ticket
from ..storage import (
    generate_new_epic_key,
    generate_new_ticket_id,
    generate_new_ticket_key,
    ensure_project_doc_dir,
)


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
    plan_section = f"\nExisting Requirement Plan:\n{plan}\n" if plan else ""
    return f"""
You are a product analyst. Based on the project documents below, create a concise backlog of EPICs with user STORIES.

Project Code: {project.code}
Project Title: {project.title}
Project Description: {project.description}
{plan_section}

Documents:
{content}

Return STRICT JSON with this schema:
{{
  "epics": [
    {{
      "title": "string",
      "description": "string",
      "stories": [
        {{
          "title": "string",
          "description": "string",
          "acceptance_criteria": ["string", ...]
        }}
      ]
    }}
  ]
}}

- Limit to 3-5 epics, each with 2-5 stories.
- Keep descriptions actionable and clear.
"""


def generate_epics_and_stories_for_project(project: Project, plan: str | None = None) -> List[Ticket]:
    """Generate epics and stories using Gemini and return Ticket objects."""
    _configure_gemini()
    content = _load_project_documents(project)
    prompt = _build_prompt(project, content, plan)

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
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            response = None
            continue
    if response is None:
        raise HTTPException(status_code=502, detail=f"Gemini generation failed: {last_exc}") from last_exc

    text = response.text or ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to recover JSON from code fences
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=500, detail="Gemini response could not be parsed as JSON") from exc
        else:
            raise HTTPException(status_code=500, detail="Gemini response could not be parsed as JSON")

    epics_raw = data.get("epics", [])
    tickets: List[Ticket] = []

    project_dict = project.model_dump()
    # Seed counters based on existing tickets
    existing_ids = [
        int(t.get("id", "t_0").split("_")[1])
        for t in project_dict.get("tickets", [])
        if str(t.get("id", "")).startswith("t_")
    ]
    next_id_num = max(existing_ids or [0]) + 1

    existing_keys = [t.get("key", "") for t in project_dict.get("tickets", [])]
    existing_key_nums = []
    for key in existing_keys:
        if key.startswith(f"{project.code}-"):
            suffix = key.split("-")[-1]
            try:
                existing_key_nums.append(int(suffix.replace("EP", "")))
            except ValueError:
                continue
    next_key_num = max(existing_key_nums or [0]) + 1

    epic_counter = 1
    for epic in epics_raw:
        epic_key = generate_new_epic_key(project_dict, epic_counter)
        epic_ticket = Ticket(
            id=f"t_{next_id_num}",
            key=epic_key,
            project_id=project.id,
            type="EPIC",
            title=epic.get("title", f"Epic {epic_counter}"),
            description=epic.get("description", ""),
            status="TODO",
            assignee_username=None,
            epic_key=None,
            dependencies=[],
            blockers=None,
            linked_document_ids=[],
        )
        tickets.append(epic_ticket)
        next_id_num += 1

        stories = epic.get("stories", []) or []
        for story in stories:
            ac = story.get("acceptance_criteria") or []
            if not ac:
                # Provide a minimal default acceptance criteria set
                ac = [
                    "All key details are present and accurate.",
                    "Output is clear, concise, and testable.",
                ]
            story_ticket = Ticket(
                id=f"t_{next_id_num}",
                key=f"{project.code}-{next_key_num:03d}",
                project_id=project.id,
                type="STORY",
                title=story.get("title", "Story"),
                description=story.get("description", ""),
                status="TODO",
                assignee_username=None,
                epic_key=epic_key,
                dependencies=[],
                blockers=None,
                linked_document_ids=[],
                acceptance_criteria=ac,
            )
            tickets.append(story_ticket)
            next_id_num += 1
            next_key_num += 1
        epic_counter += 1

    return tickets
