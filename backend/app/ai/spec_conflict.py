"""LLM-driven spec conflict detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
from fastapi import HTTPException, status

from .story_gen import log_ai_event
from ..config import settings
from ..storage import ensure_project_doc_dir, get_project_by_id
from ..models import Project


def _configure_llm() -> None:
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GEMINI_API_KEY not configured for conflict detection.",
        )
    genai.configure(api_key=settings.gemini_api_key)


def _read_specs(project_id: str) -> List[Tuple[str, str]]:
    """Return list of (path,label,content) for specs/references."""
    specs: List[Tuple[str, str]] = []
    proj_dict = get_project_by_id(project_id)
    plan = proj_dict.get("requirements_plan") or ""
    if plan:
        specs.append((f"/api/projects/{project_id}/requirements/plan", "requirements_plan", plan))
    # Project description
    desc = proj_dict.get("description", "")
    specs.append((f"/projects/{project_id}/view", "project_description", desc))
    # documents (text best effort)
    doc_dir = ensure_project_doc_dir(project_id)
    for doc in proj_dict.get("documents", []):
        path = doc_dir / doc.get("filename", "")
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""
        specs.append((f"/projects/{project_id}/documents/{doc.get('id')}", doc.get("original_name", ""), content[:4000]))
    return specs


def _ticket_text(ticket: Dict[str, Any]) -> str:
    ac = ticket.get("acceptance_criteria") or ticket.get("acceptanceCriteria") or []
    if isinstance(ac, list):
        ac_text = "; ".join([str(x) for x in ac if x])
    else:
        ac_text = str(ac)
    return "\n".join(
        [
            f"Key: {ticket.get('key') or ticket.get('id')}",
            f"Title: {ticket.get('title','')}",
            f"Description: {ticket.get('description','')}",
            f"Status: {ticket.get('status','')}",
            f"Epic: {ticket.get('epic_key') or ticket.get('epic') or ''}",
            f"Acceptance Criteria: {ac_text}",
            f"Dependencies: {', '.join(ticket.get('dependencies', []) or [])}",
            "----",
        ]
    )


def _epic_text(epic: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Key: {epic.get('key')}",
            f"Title: {epic.get('title','')}",
            f"Description: {epic.get('description','')}",
            "----",
        ]
    )


def _build_prompt(project: Project, pending_change: Dict[str, Any], mode: str = "pending-change") -> str:
    specs = _read_specs(project.id)
    specs_block = "\n\n".join(
        [f"Spec: {label}\nPath: {path}\nContent:\n{content}" for path, label, content in specs]
    )
    epics_block = "\n".join([_epic_text(t.model_dump()) for t in project.tickets if t.type == "EPIC"])
    tickets_block = "\n".join([_ticket_text(t.model_dump()) for t in project.tickets if t.type != "EPIC"])
    change_block = json.dumps(pending_change, ensure_ascii=False, indent=2)

    schema = """
{
  "has_conflict": false,
  "conflicts": [
    {
      "id": "conflict-1",
      "severity": "critical",
      "description": "Human-readable explanation of the conflict.",
      "file_1": {
        "id": "file-or-ticket-or-epic-id-1",
        "type": "ticket|epic|spec",
        "path": "path/or/key/identifier/used/in-UI",
        "section_label": "optional label",
        "section_start_marker": "marker or line range",
        "section_end_marker": "marker or line range"
      },
      "file_2": {
        "id": "file-or-ticket-or-epic-id-2",
        "type": "ticket|epic|spec",
        "path": "path/or/key/identifier/used/in-UI",
        "section_label": "optional label",
        "section_start_marker": "marker or line range",
        "section_end_marker": "marker or line range"
      }
    }
  ]
}
""".strip()
    goal = (
        "This is a project-wide alignment check. Find conflicts within the current project content."
        if mode == "project-alignment"
        else "This is a pending-change check. Find conflicts introduced or revealed by the pending change."
    )
    return f"""
You are checking for conflicts/contradictions between project specs, epics, and tickets. {goal} Only flag conflicts that are explicit contradictions in the provided text. Do NOT hallucinate. If no conflict exists, you MUST return has_conflict=false and conflicts=[].

Project Code: {project.code}
Project Title: {project.title}

PROJECT SPECS:
{specs_block}

EPICS:
{epics_block}

TICKETS:
{tickets_block}

PENDING CHANGE (to be applied or empty for alignment-only):
{change_block}

Respond ONLY with strict JSON following this schema:
{schema}

Rules:
- Only mark a conflict if there is a clear contradiction.
- Do not omit any clear conflicts.
- If no conflicts, return has_conflict=false and conflicts: [].
"""


def _parse_response(text: str) -> Dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def detect_spec_conflicts(project_id: str, pending_change: Dict[str, Any], change_type: str) -> Dict[str, Any]:
    """Run LLM conflict detection; returns dict with has_conflict/conflicts."""
    project = Project.model_validate(get_project_by_id(project_id))
    prompt = _build_prompt(project, {"change_type": change_type, "payload": pending_change}, mode="pending-change")
    log_ai_event(f"[SpecConflict] project={project_id} type={change_type} prompt_len={len(prompt)}")
    _configure_llm()
    models = []
    if settings.gemini_model:
        models.append(settings.gemini_model)
    models.extend(["models/gemini-2.5-flash", "models/gemini-2.5-flash-latest"])
    last_exc: Exception | None = None
    for name in models:
        try:
            model = genai.GenerativeModel(name)
            response = model.generate_content(prompt, request_options={"timeout": 40})
            text = response.text or ""
            parsed = _parse_response(text)
            if not parsed:
                continue
            has_conflict = bool(parsed.get("has_conflict"))
            conflicts = parsed.get("conflicts") or []
            return {"has_conflict": has_conflict, "conflicts": conflicts}
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log_ai_event(f"[SpecConflict Error] {name} | {exc}", level=40)
            continue
    # Fallback: treat as no conflicts but log
    log_ai_event(f"[SpecConflict Fallback] project={project_id} type={change_type} last_exc={last_exc}", level=30)
    return {"has_conflict": False, "conflicts": []}


def check_project_alignment(project_id: str) -> Dict[str, Any]:
    """Project-wide alignment check (no pending change)."""
    project = Project.model_validate(get_project_by_id(project_id))
    # If nothing changed and cache exists, reuse last alignment result
    cache_path = settings.data_dir / "alignment_cache" / f"{project_id}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = _build_prompt(project, {"change_type": "project_alignment", "payload": {}}, mode="project-alignment")
    log_ai_event(f"[SpecAlignment] project={project_id} prompt_len={len(prompt)}")
    _configure_llm()
    models = []
    if settings.gemini_model:
        models.append(settings.gemini_model)
    models.extend(["models/gemini-2.5-flash", "models/gemini-2.5-flash-latest"])
    last_exc: Exception | None = None
    for name in models:
        try:
            model = genai.GenerativeModel(name)
            response = model.generate_content(prompt, request_options={"timeout": 50})
            text = response.text or ""
            parsed = _parse_response(text)
            if not parsed:
                continue
            has_conflict = bool(parsed.get("has_conflict"))
            conflicts = parsed.get("conflicts") or []
            result = {"has_conflict": has_conflict, "conflicts": conflicts}
            cache_path.write_text(json.dumps(result, ensure_ascii=False))
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log_ai_event(f"[SpecAlignment Error] {name} | {exc}", level=40)
            continue
    # Fallback to cached result if available
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            log_ai_event(f"[SpecAlignment Fallback Cached] project={project_id}", level=30)
            return cached
        except Exception:
            pass
    log_ai_event(f"[SpecAlignment Fallback] project={project_id} last_exc={last_exc}", level=30)
    return {"has_conflict": False, "conflicts": []}
