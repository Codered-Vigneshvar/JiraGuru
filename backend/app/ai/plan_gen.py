"""Generate plain-text functional requirements plan using Gemini."""

from __future__ import annotations

import google.generativeai as genai
from fastapi import HTTPException

from ..config import settings
from ..models import Project
from .story_gen import _configure_gemini, _load_project_documents


def generate_plan_for_project(project: Project, requirements: str | None = None) -> str:
    """Generate a detailed plan as plain text (no markdown)."""
    _configure_gemini()
    docs = _load_project_documents(project)
    req_section = f"\nUser Provided Requirements:\n{requirements}\n" if requirements else ""
    prompt = f"""
You are a product analyst. Based on the project info and documents, write a detailed functional requirements and delivery plan.

Project Code: {project.code}
Project Title: {project.title}
Project Description: {project.description}
{req_section}

Documents:
{docs}

Return CLEAN PLAINTEXT ONLY. No markdown symbols, no headings markers, no tables. Use simple labels and hyphen bullets. Avoid numbering prefixes. Sections:
Objectives:
- ...
Key Features:
- ...
Functional Requirements:
- ...
Non-Functional Requirements:
- ...
Risks & Mitigations:
- ...
Phased Plan:
- ...
"""
    model_name = settings.gemini_model or "models/gemini-2.5-flash"
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Gemini plan generation failed: {exc}") from exc
    return response.text or ""
