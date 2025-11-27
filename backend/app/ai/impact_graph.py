"""LangGraph pipeline for document-to-ticket impact analysis."""

from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

import google.generativeai as genai
from fastapi import HTTPException, status
from langgraph.graph import END, StateGraph

from ..config import settings
from ..storage import ensure_project_doc_dir, get_project_by_id
from .project_index import ProjectIndex
from .story_gen import log_ai_event


class ImpactState(TypedDict, total=False):
    project_id: str
    doc_id: str
    doc_text: str
    candidates: List[Dict[str, Any]]
    scored_tickets: List[Dict[str, Any]]
    report: str


def _configure_gemini() -> None:
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GEMINI_API_KEY not configured.",
        )
    genai.configure(api_key=settings.gemini_api_key)


def _load_doc_text(project_id: str, doc_id: str) -> str:
    project = get_project_by_id(project_id)
    doc_meta = next((d for d in project.get("documents", []) if d.get("id") == doc_id), None)
    if not doc_meta:
        raise HTTPException(status_code=404, detail="Document not found.")
    doc_dir = ensure_project_doc_dir(project_id)
    path = doc_dir / doc_meta.get("filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document file missing.")
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_bytes().decode(errors="ignore")


# ------------------------------ graph node functions
def load_doc_and_context(state: ImpactState) -> ImpactState:
    project_id = state["project_id"]
    doc_id = state["doc_id"]
    text = _load_doc_text(project_id, doc_id)
    index = ProjectIndex(project_id)
    index.ensure_built_full()
    project = get_project_by_id(project_id)
    proj_desc = project.get("description", "")
    proj_title = project.get("title", "")
    state["project_title"] = proj_title
    state["project_description"] = proj_desc
    state["doc_text"] = text
    return state


def retrieve_candidate_tickets(state: ImpactState) -> ImpactState:
    project_id = state["project_id"]
    text = state.get("doc_text") or ""
    query = f"{state.get('project_title','')} {state.get('project_description','')}\n{text}"
    index = ProjectIndex(project_id)
    hits = index.search_tickets(query, k=30)
    candidates: List[Dict[str, Any]] = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        candidates.append(
            {
                "ticket_key": meta.get("ticket_key"),
                "title": meta.get("title"),
                "status": meta.get("status"),
                "epic_key": meta.get("epic_key"),
                "raw_score": hit.get("score"),
                "content": hit.get("content"),
            }
        )
    state["candidates"] = candidates
    return state


def _classify_impact(doc_text: str, candidate: Dict[str, Any]) -> Dict[str, str]:
    _configure_gemini()
    model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
    prompt = f"""
You are an expert project analyst. Assess the impact of the provided document update on the ticket.
Return ONLY a strict JSON object with fields: impact_level (HIGH|MEDIUM|LOW|NONE) and reason. No other text.
Base impact on how the document changes affect the ticket's scope, dependencies, or acceptance criteria.
Always consider project title, project description, and retrieved ticket context below.

Document (truncated):
{doc_text[:3000]}

Ticket:
Key: {candidate.get("ticket_key")}
Title: {candidate.get("title")}
Status: {candidate.get("status")}
Epic: {candidate.get("epic_key")}
Content:
{candidate.get("content")}
"""
    response = model.generate_content(prompt)
    raw = response.text or ""
    log_ai_event(f"[Impact Score] {candidate.get('ticket_key')} | {raw}")
    variants = [raw, raw.replace("'", '"')]
    # Try to extract JSON object from within text
    if "{" in raw and "}" in raw:
        inner = raw[raw.find("{"): raw.rfind("}") + 1]
        variants.append(inner)
        variants.append(inner.replace("'", '"'))
    for variant in variants:
        try:
            data = json.loads(variant)
            level = str(data.get("impact_level", "")).upper()
            if level not in {"HIGH", "MEDIUM", "LOW", "NONE"}:
                continue
            reason = data.get("reason", "") or ""
            return {"impact_level": level, "reason": reason}
        except json.JSONDecodeError:
            continue
    return {"impact_level": "LOW", "reason": "AI response unreadable; treating as low impact for review."}


def ai_score_impact(state: ImpactState) -> ImpactState:
    doc_text = state.get("doc_text", "")
    candidates = state.get("candidates") or []
    scored: List[Dict[str, Any]] = []
    for cand in candidates:
        res = _classify_impact(doc_text, cand)
        impact_level = res["impact_level"]
        reason = res.get("reason", "")
        record = {**cand, "impact_level": impact_level, "reason": reason}
        scored.append(record)
    # If all were NONE, keep top few as LOW to avoid empty result
    if scored and all(item["impact_level"] == "NONE" for item in scored):
        for item in scored[:5]:
            item["impact_level"] = "LOW"
            item["reason"] = item.get("reason") or "Low/uncertain impact (model returned NONE)."
    else:
        scored = [s for s in scored if s["impact_level"] != "NONE"]
    priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    scored.sort(key=lambda x: (-priority.get(x.get("impact_level", ""), 0), x.get("raw_score", 0)))
    state["scored_tickets"] = scored
    return state


def generate_impact_report(state: ImpactState) -> ImpactState:
    _configure_gemini()
    model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
    tickets = state.get("scored_tickets") or []
    doc_text = state.get("doc_text", "")[:4000]
    project_title = state.get("project_title", "")
    project_desc = state.get("project_description", "")
    summary_payload = json.dumps(
        [
            {
                "ticket_key": t.get("ticket_key"),
                "title": t.get("title"),
                "impact_level": t.get("impact_level"),
                "reason": t.get("reason"),
                "status": t.get("status"),
                "epic_key": t.get("epic_key"),
            }
            for t in tickets
        ],
        ensure_ascii=False,
    )
    prompt = f"""
You are an expert project analyst. Create a concise impact report explaining the product outcome after recent changes, what changed, and how it impacts the overall project.
Return plain text only: no markdown headings (#), no bold/italics (**), no bullets or lists. Separate sections with blank lines.

Project: {project_title}
Project Description: {project_desc}

Recent context (truncated):
{doc_text}

Ticket impacts (JSON):
{summary_payload}

Write the report with these sections:
Overview: Describe the overall product direction after the changes and how the project is affected.
Key Changes: Explain the main changes reflected in the impacted tickets.
Impact on Project: Describe how these changes affect delivery, scope, or users.
Next Steps: Plain-sentence actions to keep momentum.
"""
    response = model.generate_content(prompt)
    report_text = response.text or ""
    log_ai_event(f"[Impact Report] generated {len(tickets)} tickets")
    state["report"] = report_text
    return state


# ------------------------------ Graph wiring
builder = StateGraph(ImpactState)
builder.add_node("load_doc_and_context", load_doc_and_context)
builder.add_node("retrieve_candidates", retrieve_candidate_tickets)
builder.add_node("ai_score_impact", ai_score_impact)
builder.add_node("generate_report", generate_impact_report)
builder.set_entry_point("load_doc_and_context")
builder.add_edge("load_doc_and_context", "retrieve_candidates")
builder.add_edge("retrieve_candidates", "ai_score_impact")
builder.add_edge("ai_score_impact", "generate_report")
builder.add_edge("generate_report", END)

impact_graph = builder.compile()
