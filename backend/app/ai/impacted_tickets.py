"""LLM pipeline for impacted ticket generation with RAG filtering."""

from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

import google.generativeai as genai
from fastapi import HTTPException, status
from langgraph.graph import StateGraph, END

from ..config import settings
from ..models import Project
from ..storage import get_project_by_id
from .project_index import ProjectIndex
from .story_gen import _strip_code_fences, log_ai_event


class ImpactedTicketResult(TypedDict, total=False):
    updatedTickets: List[Dict[str, Any]]
    newTickets: List[Dict[str, Any]]
    changeSummary: str
    usedTickets: List[Dict[str, Any]]


class ImpactTicketsState(TypedDict, total=False):
    project_id: str
    old_spec: str
    new_spec: str
    change_summary: str
    tickets_payload: str
    focus_tickets: List[Dict[str, Any]]
    result: Dict[str, Any]
    delta_text: str
    epics_payload: str


def _configure_llm() -> None:
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GEMINI_API_KEY not configured.",
        )
    genai.configure(api_key=settings.gemini_api_key)


def _summarize_change(old_spec: str, new_spec: str) -> str:
    """Summarize what changed between old and new specs."""
    try:
        _configure_llm()
        model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
        prompt = f"""
You are a senior analyst. Summarize only what changed between the two specs.
Output: one short paragraph, plain text, no lists/markdown.

Old spec:
{old_spec[:4000]}

New spec:
{new_spec[:4000]}
"""
        resp = model.generate_content(prompt)
        return resp.text or ""
    except Exception as exc:  # noqa: BLE001
        log_ai_event(f"[Impact Change Summary Error] {exc}", level=20)
        return ""


def _retrieve_candidates(project_id: str, query: str, k: int = 15) -> List[Dict[str, Any]]:
    """Deprecated: similarity search is no longer used for inclusion."""
    return []


def _diff_lines(old_spec: str, new_spec: str, limit: int = 60) -> str:
    """Produce a brief line-level diff summary."""
    import difflib

    old_lines = (old_spec or "").splitlines()
    new_lines = (new_spec or "").splitlines()
    diff = difflib.ndiff(old_lines, new_lines)
    changes: List[str] = []
    for line in diff:
        if line.startswith("+ "):
            changes.append(f"ADDED: {line[2:]}")
        elif line.startswith("- "):
            changes.append(f"REMOVED: {line[2:]}")
        if len(changes) >= limit:
            break
    return "\n".join(changes)


def _ticket_block(tickets: List[Dict[str, Any]]) -> str:
    """Render tickets into human-readable blocks (no embeddings)."""
    parts: List[str] = []
    for t in tickets:
        ac = t.get("acceptanceCriteria") or t.get("acceptance_criteria") or ""
        if isinstance(ac, list):
            ac_text = "; ".join([str(x) for x in ac if x])
        else:
            ac_text = str(ac)
        parts.append(
            "\n".join(
                [
                    f"Key: {t.get('id') or t.get('key')}",
                    f"Title: {t.get('title','')}",
                    f"Description: {t.get('description','')}",
                    f"Status: {t.get('status','')}",
                    f"Epic: {t.get('epic') or t.get('epic_key') or ''}",
                    f"Acceptance Criteria: {ac_text}",
                    "----",
                ]
            )
        )
    return "\n".join(parts)


def _build_prompt(
    project: Project,
    old_spec: str,
    new_spec: str,
    tickets_block: str,
    change_summary: str,
    delta_text: str,
    epics_block: str,
) -> str:
    """Construct the JSON-oriented prompt for impact analysis."""
    schema = """
{
  "updatedTickets": [
    {
      "id": "existing-ticket-id (use ticket key)",
      "type": "epic|story",
      "changeReason": "why this ticket is impacted",
      "before": {
        "title": "existing title",
        "description": "existing description",
        "acceptanceCriteria": "existing AC text"
      },
      "suggested": {
        "title": "new title",
        "description": "new description",
        "acceptanceCriteria": "new AC text"
      }
    }
  ],
  "newTickets": [
    {
      "type": "epic|story",
      "title": "suggested title",
      "description": "suggested description",
      "acceptanceCriteria": "suggested AC text",
      "changeReason": "which new/changed requirement this covers"
    }
  ]
}
""".strip()
    return f"""
You are an assistant responsible for detecting changes in project specifications, identifying affected tickets, and generating updated ticket versions with full traceability. Follow all instructions precisely. Always return STRICT JSON only, matching the schema provided.

Project Code: {project.code}

OLD PROJECT DATA:
{old_spec[:3500]}

NEW PROJECT DATA:
{new_spec[:3500]}

CHANGES (summary + diff, do not ignore):
{change_summary or "N/A"}
{delta_text or "Line-level diffs included; treat specs as changed."}

EXISTING EPICS (use these where possible):
{epics_block or "None"}

TICKETS:
{tickets_block}

Return ONLY JSON in this exact schema (no markdown, no extra text):
{schema}

Rules:
- updatedTickets must reference existing ticket keys in the id field.
- For new stories, prefer an existing epic from the list; only create a new epic if none fit.
- If creating a new epic, include it in newTickets (type: epic) and set its key; set story epic_id/epic fields to that new epic key.
- Include all tickets that the changes impact; if none, return empty lists.
- Do NOT omit tickets because of similarity ranking; consider all provided tickets.
- Do NOT invent unrelated work.
- Keep changeReason short and specific.
- acceptanceCriteria fields are plain text (later parsed into a list).
"""


def _parse_json_response(raw_text: str) -> Dict[str, Any] | None:
    """Try to parse Gemini response into dict."""
    candidates = []
    cleaned = _strip_code_fences(raw_text)
    candidates.append(cleaned)
    if "{" in raw_text and "}" in raw_text:
        inner = raw_text[raw_text.find("{"): raw_text.rfind("}") + 1]
        candidates.append(_strip_code_fences(inner))
    candidates.append(cleaned.replace("'", '"'))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _build_context(state: ImpactTicketsState) -> ImpactTicketsState:
    project_id = state["project_id"]
    project = Project.model_validate(get_project_by_id(project_id))
    old_spec = state["old_spec"]
    new_spec = state["new_spec"]

    change_summary = _summarize_change(old_spec, new_spec)
    delta_text = _diff_lines(old_spec, new_spec)
    # Use all project tickets (no similarity filter)
    tickets_payload = []
    for t in project.tickets or []:
        tickets_payload.append(
            {
                "id": t.key,
                "type": t.type.lower(),
                "title": t.title,
                "description": t.description,
                "acceptanceCriteria": "; ".join(t.acceptance_criteria or []),
                "status": t.status,
                "epic": t.epic_key,
                "blockers": t.blockers,
                "dependencies": t.dependencies,
            }
        )

    epics_payload = [
        {
            "key": t.key,
            "title": t.title,
            "description": t.description,
        }
        for t in project.tickets
        if getattr(t, "type", "") == "EPIC"
    ]

    state["change_summary"] = change_summary
    state["tickets_payload"] = json.dumps(tickets_payload, ensure_ascii=False)
    state["focus_tickets"] = tickets_payload
    state["delta_text"] = delta_text
    state["epics_payload"] = json.dumps(epics_payload, ensure_ascii=False)
    log_ai_event(
        f"[ImpactedTickets Context] project={project_id} | old_len={len(old_spec)} new_len={len(new_spec)} "
        f"delta_len={len(delta_text)} summary_len={len(change_summary)} epics={len(epics_payload)} tickets={len(tickets_payload)}"
    )
    return state


def _call_llm(state: ImpactTicketsState) -> ImpactTicketsState:
    project = Project.model_validate(get_project_by_id(state["project_id"]))
    all_tickets: List[Dict[str, Any]] = json.loads(state.get("tickets_payload", "[]"))
    epics: List[Dict[str, Any]] = json.loads(state.get("epics_payload", "[]"))
    epics_block = _ticket_block(
        [
            {
                "id": e.get("key"),
                "title": e.get("title"),
                "description": e.get("description"),
                "status": "",
                "epic": "",
                "acceptance_criteria": "",
            }
            for e in epics
        ]
    )
    # Chunk tickets to reduce prompt size if needed (smaller batches to avoid timeouts)
    chunk_size = 4
    combined_updated: List[Dict[str, Any]] = []
    combined_new: List[Dict[str, Any]] = []
    for i in range(0, len(all_tickets) or 1, max(chunk_size, 1)):
        batch = all_tickets[i : i + chunk_size]
        tickets_block = _ticket_block(batch)
        prompt = _build_prompt(
            project,
            state["old_spec"],
            state["new_spec"],
            tickets_block,
            state.get("change_summary", ""),
            state.get("delta_text", ""),
            epics_block,
        )
        log_ai_event(f"[ImpactedTickets Request] project={project.id} batch={i//chunk_size} | {prompt[:1000]}")

        _configure_llm()
        models_to_try = []
        if settings.gemini_model:
            models_to_try.append(settings.gemini_model)
        models_to_try.extend(
            [
                "models/gemini-2.5-flash",
                "models/gemini-2.5-flash-latest",
            ]
        )
        response_text: str | None = None
        last_exc: Exception | None = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={"timeout": 50})
                response_text = response.text or ""
                log_ai_event(f"[ImpactedTickets Response] batch={i//chunk_size} model={model_name} | {response_text[:1000]}")
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log_ai_event(f"[ImpactedTickets Error] batch={i//chunk_size} model={model_name} | {exc}", level=40)
                continue
        if response_text is None:
            raise HTTPException(status_code=502, detail=f"LLM unavailable for impact analysis: {last_exc}")

        payload = _parse_json_response(response_text)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="Invalid AI response for impacted tickets.")
        combined_updated.extend(payload.get("updatedTickets") or [])
        combined_new.extend(payload.get("newTickets") or [])

    state["result"] = {"updatedTickets": combined_updated, "newTickets": combined_new}
    return state


builder = StateGraph(ImpactTicketsState)
builder.add_node("context", _build_context)
builder.add_node("llm", _call_llm)
builder.set_entry_point("context")
builder.add_edge("context", "llm")
builder.add_edge("llm", END)

impacted_tickets_graph = builder.compile()


def generate_impacted_tickets(project_id: str, old_spec: str, new_spec: str) -> ImpactedTicketResult:
    """Generate impacted + new tickets using Gemini with RAG-filtered context."""
    final_state = impacted_tickets_graph.invoke(
        {
            "project_id": project_id,
            "old_spec": old_spec,
            "new_spec": new_spec,
        }
    )
    payload = final_state.get("result") or {}
    updated_tickets = payload.get("updatedTickets") or []
    new_tickets = payload.get("newTickets") or []
    if not isinstance(updated_tickets, list) or not isinstance(new_tickets, list):
        raise HTTPException(status_code=502, detail="AI response missing required fields.")

    return ImpactedTicketResult(
        updatedTickets=updated_tickets,
        newTickets=new_tickets,
        changeSummary=final_state.get("change_summary", ""),
        usedTickets=final_state.get("focus_tickets", []),
    )
