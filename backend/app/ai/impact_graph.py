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
    project_title: str
    project_description: str
    delta_text: str
    old_doc_text: str
    all_tickets: List[Dict[str, Any]]
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
    text = state.get("doc_text")
    if text is None:
        text = _load_doc_text(project_id, doc_id)
    index = ProjectIndex(project_id)
    index.ensure_built_full()
    project = get_project_by_id(project_id)
    proj_desc = project.get("description", "")
    proj_title = project.get("title", "")
    state["project_title"] = proj_title
    state["project_description"] = proj_desc
    state["doc_text"] = text
    state["all_tickets"] = project.get("tickets", [])
    return state


def retrieve_candidate_tickets(state: ImpactState) -> ImpactState:
    project_id = state["project_id"]
    text = state.get("doc_text") or ""
    delta = state.get("delta_text") or ""
    focus = delta if delta.strip() else text
    query = f"{state.get('project_title','')} {state.get('project_description','')}\n{focus}"
    index = ProjectIndex(project_id)
    hits = index.search_tickets(query, k=15)
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
    request_options = {"timeout": 20}
    prompt = f"""
You are an expert project analyst. Decide if the document CHANGES materially impact this ticket. If not, return NONE.
Rules:
- Only return HIGH/MEDIUM/LOW when the change alters the ticket's scope, acceptance criteria, dependencies, blockers, or delivery risk.
- If the change is general context or unrelated, return NONE.
- Be concise and specific in the reason; cite the change that drives the impact.

Return ONLY strict JSON: {{"impact_level": "HIGH|MEDIUM|LOW|NONE", "reason": "<short sentence>"}}. No other text.

Change (truncated):
{doc_text[:3000]}

Ticket:
Key: {candidate.get("ticket_key")}
Title: {candidate.get("title")}
Status: {candidate.get("status")}
Epic: {candidate.get("epic_key")}
Content:
{candidate.get("content")}
"""
    try:
        response = model.generate_content(prompt, request_options=request_options)
        raw = response.text or ""
    except Exception as exc:
        log_ai_event(f"[Impact Score Error] {candidate.get('ticket_key')}: {exc}")
        return {"impact_level": "NONE", "reason": "LLM unavailable; defaulting to no impact."}
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
    doc_text = state.get("delta_text") or state.get("doc_text", "")
    candidates = state.get("candidates") or []
    all_tickets = state.get("all_tickets") or []
    ticket_lookup = {t.get("key"): t for t in all_tickets if isinstance(t, dict) and t.get("key")}
    scored: List[Dict[str, Any]] = []
    for cand in candidates:
        res = _classify_impact(doc_text, cand)
        impact_level = res["impact_level"]
        reason = res.get("reason", "")
        record = {**cand, "impact_level": impact_level, "reason": reason}
        scored.append(record)

    # Filter out weak matches before dependency expansion
    min_score = 0.35
    scored = [
        s
        for s in scored
        if s.get("impact_level") in {"HIGH", "MEDIUM"}
        or (s.get("impact_level") == "LOW" and (s.get("raw_score") or 0) >= min_score)
    ]

    # Pull in dependency stories for impacted tickets (one hop) with inherited impact
    scored_by_key = {s.get("ticket_key"): s for s in scored if s.get("ticket_key")}
    for ticket_key, ticket in list(scored_by_key.items()):
        deps = (ticket_lookup.get(ticket_key, {}).get("dependencies") or [])
        for dep_key in deps:
            if dep_key in scored_by_key:
                continue
            dep_ticket = ticket_lookup.get(dep_key)
            if not dep_ticket:
                continue
            dep_record = {
                "ticket_key": dep_ticket.get("key"),
                "title": dep_ticket.get("title"),
                "status": dep_ticket.get("status"),
                "epic_key": dep_ticket.get("epic_key"),
                "raw_score": 0,
                "content": dep_ticket.get("description"),
                "impact_level": ticket.get("impact_level", "LOW"),
                "reason": (
                    f"Dependency of impacted ticket {ticket_key} ({ticket.get('title')}); "
                    f"keep this aligned with the upstream change."
                ),
                "dependency_of": ticket_key,
            }
            scored.append(dep_record)
            scored_by_key[dep_key] = dep_record

    # If all were NONE, keep top few as LOW to avoid empty result
    if scored and all(item["impact_level"] == "NONE" for item in scored):
        for item in scored[:5]:
            item["impact_level"] = "LOW"
            item["reason"] = item.get("reason") or "Low/uncertain impact (model returned NONE)."
    else:
        scored = [s for s in scored if s["impact_level"] != "NONE"]
    priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    scored.sort(key=lambda x: (-priority.get(x.get("impact_level", ""), 0), x.get("raw_score", 0)))
    # Cap the list to avoid returning everything; keep strongest matches
    max_results = 10
    scored = scored[:max_results]
    state["scored_tickets"] = scored
    return state


def generate_impact_report(state: ImpactState) -> ImpactState:
    _configure_gemini()
    model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
    request_options = {"timeout": 20}
    tickets = state.get("scored_tickets") or []
    doc_text = (state.get("delta_text") or state.get("doc_text", ""))[:4000]
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
    try:
        response = model.generate_content(prompt, request_options=request_options)
        report_text = response.text or ""
    except Exception as exc:
        log_ai_event(f"[Impact Report Error] {exc}")
        report_text = "LLM unavailable while generating report. Please retry."
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
