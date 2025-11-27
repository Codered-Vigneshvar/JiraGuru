"""Project health, impact, and advisory endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import google.generativeai as genai
from fastapi import APIRouter, Header, HTTPException, Query, status

from ..models import Project, Ticket
from ..storage import ensure_project_doc_dir, get_project_by_id, load_users, load_ticket_comments
from ..ai.impact_graph import impact_graph, ImpactState
from ..ai.project_index import ProjectIndex
from ..config import settings

router = APIRouter(prefix="/projects", tags=["reports"])


def _get_user(username: str | None) -> Dict | None:
    if not username:
        return None
    for user in load_users():
        if user.get("username") == username:
            return user
    return None


def _require_user(x_user: str | None) -> Dict:
    user = _get_user(x_user)
    if not user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unknown user.")
    return user


def _assert_member(project: Project, user: Dict) -> None:
    if user.get("is_owner"):
        return
    if user["username"] not in {project.owner_username, *project.member_usernames}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of this project")


# ---------------- Impact cache helpers ----------------


def _cache_path(project_id: str, doc_id: str) -> Path:
    cache_dir = settings.data_dir / "impact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{project_id}_{doc_id}.json"


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


def _load_all_docs_text(project_id: str) -> str:
    project = get_project_by_id(project_id)
    parts: List[str] = []
    for doc in project.get("documents", []):
        doc_dir = ensure_project_doc_dir(project_id)
        path = doc_dir / doc.get("filename", "")
        if not path.exists():
            continue
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            try:
                parts.append(path.read_bytes().decode(errors="ignore"))
            except Exception:
                continue
    return "\n\n".join(parts)


def _serialize_tickets(tickets: List[Ticket]) -> str:
    parts: List[str] = []
    for t in tickets:
        parts.append(
            "\n".join(
                [
                    f"Key: {t.key}",
                    f"Title: {t.title}",
                    f"Description: {t.description}",
                    f"Status: {t.status}",
                    f"Priority: {getattr(t, 'priority', '')}",
                    f"Epic: {t.epic_key or ''}",
                    f"AC: {'; '.join(t.acceptance_criteria or [])}",
                    f"Blockers: {t.blockers or ''}",
                ]
            )
        )
    return "\n\n".join(parts)


def _state_string(project: Project, doc_text: str, tickets: List[Ticket]) -> str:
    plan_text = project.model_dump().get("requirements_plan", "") or ""
    project_context = f"{project.title}\n{project.description}\n{plan_text}"
    return project_context + "\n\n" + doc_text + "\n\n" + _serialize_tickets(tickets)


def _state_hash(state_string: str) -> str:
    return hashlib.sha256(state_string.encode("utf-8", errors="ignore")).hexdigest()


def _change_summary(old: str, new: str, old_desc: str, new_desc: str) -> str:
    if not settings.gemini_api_key:
        return "Changes detected."
    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
        prompt = f"""
You are a technical analyst summarizing only the differences between two versions of a project requirement.
Output one short paragraph of plain text only. No markdown, no lists, no bullets, no bold. Mention only what changed.

Old project description:
{old_desc}

New project description:
{new_desc}

Old requirement text:
{old[:3000]}

New requirement text:
{new[:3000]}
"""
        response = model.generate_content(prompt)
        return response.text or "Changes detected."
    except Exception:
        return "Changes detected."


@router.get("/{project_id}/health_report")
def health_report(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> dict:
    """Return aggregate health metrics for a project."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    _assert_member(project, user)

    tickets = project.tickets or []
    total = len(tickets)

    counts_by_status = {"TODO": 0, "IN_PROGRESS": 0, "DONE": 0}
    counts_by_priority = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    unassigned: List[str] = []
    blocked: List[str] = []
    story_points_values: List[int] = []

    for t in tickets:
        status_value = t.status or "TODO"
        counts_by_status[status_value] = counts_by_status.get(status_value, 0) + 1

        priority_value = getattr(t, "priority", None) or "MEDIUM"
        if priority_value not in counts_by_priority:
            counts_by_priority[priority_value] = 0
        counts_by_priority[priority_value] += 1

        if not (t.assignee_username or "").strip():
            unassigned.append(t.key)
        if t.blockers and str(t.blockers).strip():
            blocked.append(t.key)

        sp = getattr(t, "story_points", None)
        if isinstance(sp, int):
            story_points_values.append(sp)

    completion_percentage = 0.0
    if total:
        completion_percentage = (counts_by_status.get("DONE", 0) / total) * 100

    average_story_points = 0.0
    if story_points_values:
        average_story_points = sum(story_points_values) / len(story_points_values)

    return {
        "project_id": project.id,
        "project_code": project.code,
        "total_tickets": total,
        "counts_by_status": counts_by_status,
        "counts_by_priority": counts_by_priority,
        "unassigned_tickets": unassigned,
        "blocked_tickets": blocked,
        "completion_percentage": round(completion_percentage, 2),
        "average_story_points": round(average_story_points, 2),
    }


def _is_blocked(ticket: Ticket, tickets_by_key: Dict[str, Ticket]) -> bool:
    """Rule-based check for whether a ticket is blocked."""
    if ticket.status == "DONE":
        return False
    has_blockers_text = bool(ticket.blockers and str(ticket.blockers).strip())
    if has_blockers_text:
        return True
    for dep_key in ticket.dependencies or []:
        dep_ticket = tickets_by_key.get(dep_key)
        # Treat missing dependency as not done, so it blocks
        if dep_ticket is None or dep_ticket.status != "DONE":
            return True
    return False


def _find_root_blockers(
    start_key: str,
    tickets_by_key: Dict[str, Ticket],
    blocked_keys: Set[str],
) -> List[Tuple[str, int]]:
    """Traverse dependencies upwards to find root blockers; handles cycles."""
    visited: Set[str] = set()
    start_ticket = tickets_by_key.get(start_key)

    def dfs(current_key: str, depth: int) -> Set[Tuple[str, int]]:
        if current_key in visited:
            # Cycle detected; treat starting ticket as its own root
            return {(start_key, depth)}
        visited.add(current_key)
        current = tickets_by_key.get(current_key)
        if current is None or current_key not in blocked_keys:
            return set()

        deps = current.dependencies or []
        blocked_deps = [d for d in deps if d in blocked_keys]
        if not deps or not blocked_deps:
            return {(current_key, depth)}

        roots: Set[Tuple[str, int]] = set()
        for dep in blocked_deps:
            roots |= dfs(dep, depth + 1)
        if not roots:
            roots.add((current_key, depth))
        return roots

    return list(dfs(start_key, 0))


@router.get("/{project_id}/blocker_analysis")
def blocker_analysis(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> dict:
    """Analyze blocked tickets and group them under root blockers."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    _assert_member(project, user)

    tickets = project.tickets or []
    if not tickets:
        return {"project_id": project.id, "project_code": project.code, "chains": []}

    tickets_by_key: Dict[str, Ticket] = {t.key: t for t in tickets}

    blocked_keys: Set[str] = {
        t.key for t in tickets if _is_blocked(t, tickets_by_key)
    }
    root_keys: Set[str] = set()
    for key in blocked_keys:
        ticket = tickets_by_key.get(key)
        if not ticket:
            continue
        deps = ticket.dependencies or []
        blocked_deps = [d for d in deps if d in blocked_keys]
        if not deps or not blocked_deps:
            root_keys.add(key)

    chains: Dict[str, dict] = {}
    for blocked_key in blocked_keys:
        roots = _find_root_blockers(blocked_key, tickets_by_key, blocked_keys)
        if not roots:
            continue
        for root_key, distance_up in roots:
            root_ticket = tickets_by_key.get(root_key)
            if not root_ticket:
                continue
            chain = chains.setdefault(
                root_key,
                {
                    "root_ticket_key": root_ticket.key,
                    "root_title": root_ticket.title,
                    "root_status": root_ticket.status,
                    "root_blockers_text": root_ticket.blockers or "",
                    "affected_tickets": [],
                },
            )
            chain["affected_tickets"].append(
                {
                    "ticket_key": blocked_key,
                    "title": tickets_by_key[blocked_key].title if blocked_key in tickets_by_key else "",
                    "status": tickets_by_key[blocked_key].status if blocked_key in tickets_by_key else "",
                    "distance": distance_up,
                }
            )

    # Sort affected tickets by distance then key for stability
    chains_list = []
    for chain in chains.values():
        chain["affected_tickets"] = sorted(chain["affected_tickets"], key=lambda x: (x["distance"], x["ticket_key"]))
        chains_list.append(chain)

    return {
        "project_id": project.id,
        "project_code": project.code,
        "chains": chains_list,
    }


# ---------------- Focus advisor ----------------


def _focus_score(ticket: Ticket, tickets_by_key: Dict[str, Ticket]) -> tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    if ticket.status == "IN_PROGRESS":
        score += 5
        reasons.append("IN_PROGRESS")
    elif ticket.status == "TODO":
        score += 3
        reasons.append("TODO")
    elif ticket.status == "DONE":
        score -= 10
        reasons.append("DONE")

    priority = getattr(ticket, "priority", "MEDIUM")
    if priority == "CRITICAL":
        score += 5
        reasons.append("CRITICAL priority")
    elif priority == "HIGH":
        score += 3
        reasons.append("HIGH priority")
    elif priority == "MEDIUM":
        score += 1
        reasons.append("MEDIUM priority")

    has_blockers = bool(ticket.blockers and str(ticket.blockers).strip())
    dep_blockers = False
    for dep in ticket.dependencies or []:
        dep_ticket = tickets_by_key.get(dep)
        if dep_ticket is None or dep_ticket.status != "DONE":
            dep_blockers = True
            break
    if has_blockers or dep_blockers:
        score += 2
        reasons.append("needs unblocking")

    return score, reasons


def _focus_summary(project: Project, top: List[dict]) -> str:
    if not settings.gemini_api_key or not top:
        return "Focus on the listed tickets first."
    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
        tickets_text = json.dumps(
            [
                {
                    "ticket_key": t.get("ticket_key"),
                    "title": t.get("title"),
                    "status": t.get("status"),
                    "priority": t.get("priority"),
                    "focus_reason": t.get("focus_reason"),
                }
                for t in top
            ],
            ensure_ascii=False,
        )
        prompt = f"""
You are a project advisor. Based on the project and the top focus tickets, say what to work on next and why.
Plain text only. No markdown, no bullets, no bold. Short paragraphs, separated by blank lines.

Project: {project.title}
Description: {project.description}

Top tickets:
{tickets_text}
"""
        response = model.generate_content(prompt)
        return response.text or "Focus on the listed tickets first."
    except Exception:
        return "Focus on the listed tickets first."


@router.get("/{project_id}/focus_ai")
def focus_ai(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> dict:
    """Return top priority tickets to work on next."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    _assert_member(project, user)
    tickets = project.tickets or []
    tickets_by_key = {t.key: t for t in tickets}

    scored: List[dict] = []
    for t in tickets:
        score, reasons = _focus_score(t, tickets_by_key)
        scored.append(
            {
                "ticket_key": t.key,
                "title": t.title,
                "status": t.status,
                "priority": getattr(t, "priority", "MEDIUM"),
                "focus_score": score,
                "focus_reason": ", ".join(reasons) or "General importance",
            }
        )
    scored.sort(key=lambda x: x["focus_score"], reverse=True)
    top = scored[:5]
    summary = _focus_summary(project, top)

    return {
        "project_id": project_id,
        "top_tickets": top,
        "summary": summary,
    }


@router.get("/{project_id}/impact_ai")
def impact_ai(project_id: str, doc_id: str = Query(...), x_user: str | None = Header(default=None, alias="X-User")) -> dict:
    """Run (or reuse) LangGraph impact analysis with change detection."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    _assert_member(project, user)
    if doc_id.lower() != "all" and not any(d.id == doc_id for d in project.documents):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found for this project")

    plan_text = project.model_dump().get("requirements_plan", "") or ""
    if doc_id.lower() == "all":
        doc_text = _load_all_docs_text(project_id)
    else:
        doc_text = _load_doc_text(project_id, doc_id)
    if plan_text:
        doc_text = plan_text + "\n\n" + doc_text
    tickets = project.tickets or []
    state_string = _state_string(project, doc_text, tickets)
    state_hash = _state_hash(state_string)
    cache_file = _cache_path(project_id, doc_id)
    cached = None
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cached = None

    if cached and cached.get("state_hash") == state_hash and cached.get("state_string") == state_string:
        return {
            "project_id": project_id,
            "doc_id": doc_id,
            "from_cache": True,
            "has_changes": False,
            "change_summary": cached.get("change_summary", "No changes detected since last analysis."),
            "impacted_tickets": cached.get("impacted_tickets", []),
            "report": cached.get("report", ""),
        }

    # Changes detected or first run -> run pipeline
    initial_state: ImpactState = {
        "project_id": project_id,
        "doc_id": doc_id,
    }
    try:
        final_state = impact_graph.invoke(initial_state)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Impact analysis failed: {exc}") from exc

    impacted = final_state.get("scored_tickets", [])
    report = final_state.get("report", "")

    # Build change summary
    if cached and cached.get("doc_text"):
        old_doc = cached.get("doc_text", "")
        old_desc = cached.get("project_description", "")
        change_summary = _change_summary(old_doc, doc_text, old_desc, project.description)
    else:
        change_summary = "Initial analysis. No previous version to compare."

    cache_payload = {
        "state_hash": state_hash,
        "state_string": state_string,
        "project_id": project_id,
        "doc_id": doc_id,
        "doc_text": doc_text,
        "project_title": project.title,
        "project_description": project.description,
        "impacted_tickets": impacted,
        "report": report,
        "change_summary": change_summary,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2))

    return {
        "project_id": project_id,
        "doc_id": doc_id,
        "from_cache": False,
        "has_changes": True,
        "change_summary": change_summary,
        "impacted_tickets": impacted,
        "report": report,
    }


@router.post("/{project_id}/impact_chat")
def impact_chat(
    project_id: str,
    payload: dict,
    doc_id: str = Query(...),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict:
    """Simple chat endpoint that answers questions about impact, citing RAG hits."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    _assert_member(project, user)
    if doc_id.lower() != "all" and not any(d.id == doc_id for d in project.documents):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found for this project")

    messages = payload.get("messages", []) if payload else []
    user_message = messages[-1]["content"] if messages else ""

    # Try to load cached impact results for richer context
    cached = None
    cache_file = _cache_path(project_id, doc_id)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cached = None

    _configure_llm()
    model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
    plan_text = project.model_dump().get("requirements_plan", "") or ""
    if doc_id.lower() == "all":
        doc_text = _load_all_docs_text(project_id)
    else:
        doc_text = _load_doc_text(project_id, doc_id)
    if plan_text:
        doc_text = plan_text + "\n\n" + doc_text
    doc_text = doc_text[:3000]
    project_context = f"Project: {project.title}\nDescription: {project.description}"
    cached_impacts = cached.get("impacted_tickets", []) if cached else []
    cached_change = cached.get("change_summary", "") if cached else ""
    citations_text = json.dumps(cached_impacts, ensure_ascii=False)
    history_text = "\n\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
    prompt = f"""
You are an impact-only assistant. Answer using project context, the current document, and cached impact results.
If the question is outside requirement change impact, respond: "This chat only covers requirement changes. Please use the main chat for other questions." Reply in plain text (no markdown or bullets). Keep answers concise and cite ticket keys only when you mention tickets.

Project context:
{project_context}

Document (truncated):
{doc_text}

Cached impacted tickets (may be empty):
{json.dumps(cached_impacts, ensure_ascii=False) if cached_impacts else "[]"}

Cached change summary:
{cached_change}

Ticket evidence (JSON):
{citations_text}

Conversation so far:
{history_text}

Answer the last user question succinctly and cite ticket keys where relevant.
"""
    response = model.generate_content(prompt)
    answer = response.text or ""
    return {
        "reply": answer,
        "citations": [],
    }


@router.post("/{project_id}/general_chat")
def general_chat(
    project_id: str,
    payload: dict,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict:
    """General project chat using RAG over tickets and project context."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    _assert_member(project, user)

    messages = payload.get("messages", []) if payload else []
    user_message = messages[-1]["content"] if messages else ""

    index = ProjectIndex(project_id)
    try:
        hits = index.search_tickets(user_message or "", k=6)
    except Exception:
        hits = []

    comments = load_ticket_comments(project_id).get("comments", []) if callable(load_ticket_comments) else []
    comment_text = "\n".join([f"{c.get('ticket_key')}: {c.get('text')}" for c in comments])[:2000]

    citations = []
    for hit in hits:
        meta = hit.get("metadata", {})
        citations.append(
            {
                "ticket_key": meta.get("ticket_key"),
                "title": meta.get("title"),
                "status": meta.get("status"),
                "epic_key": meta.get("epic_key"),
                "score": hit.get("score"),
                "excerpt": (hit.get("content") or "")[:800],
            }
        )

    _configure_llm()
    model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
    citations_text = json.dumps(citations, ensure_ascii=False)
    history_text = "\n\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
    project_context = f"Project: {project.title}\nDescription: {project.description}\nMembers: {', '.join(project.member_usernames)}"
    prompt = f"""
You are a concise project assistant. Use the project context and ticket evidence to answer.
Reply in plain text (no markdown or bullets). Keep responses brief and conversational. Cite ticket keys only when listing or referencing tickets.
Include blockers or dependencies if relevant. If the question is unrelated to this project, say so.

Project context:
{project_context}

Ticket evidence (JSON):
{citations_text}

Recent comments (if any):
{comment_text or "None"}

Conversation so far:
{history_text}

Answer the last user question succinctly.
"""
    response = model.generate_content(prompt)
    answer = response.text or ""
    return {
        "reply": answer,
        "citations": [],
    }


@router.post("/{project_id}/general_chat")
def general_chat(
    project_id: str,
    payload: dict,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict:
    """General project chat using RAG over tickets and project context."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    _assert_member(project, user)

    messages = payload.get("messages", []) if payload else []
    user_message = messages[-1]["content"] if messages else ""

    index = ProjectIndex(project_id)
    try:
        hits = index.search_tickets(user_message or "", k=6)
    except Exception:
        hits = []

    comments = load_ticket_comments(project_id).get("comments", []) if callable(load_ticket_comments) else []
    comment_text = "\n".join([f"{c.get('ticket_key')}: {c.get('text')}" for c in comments])[:2000]

    citations = []
    for hit in hits:
        meta = hit.get("metadata", {})
        citations.append(
            {
                "ticket_key": meta.get("ticket_key"),
                "title": meta.get("title"),
                "status": meta.get("status"),
                "epic_key": meta.get("epic_key"),
                "score": hit.get("score"),
                "excerpt": (hit.get("content") or "")[:800],
            }
        )

    _configure_llm()
    model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
    citations_text = json.dumps(citations, ensure_ascii=False)
    history_text = "\n\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
    project_context = f"Project: {project.title}\nDescription: {project.description}\nMembers: {', '.join(project.member_usernames)}"
    prompt = f"""
You are a concise project assistant. Use the project context and ticket evidence to answer.
Reply in plain text (no markdown or bullets). Keep responses brief and conversational. Cite ticket keys only when listing or referencing tickets.
Include blockers or dependencies if relevant. If the question is unrelated to this project, say so.

Project context:
{project_context}

Ticket evidence (JSON):
{citations_text}

Recent comments (if any):
{comment_text or "None"}

Conversation so far:
{history_text}

Answer the last user question succinctly.
"""
    response = model.generate_content(prompt)
    answer = response.text or ""
    return {
        "reply": answer,
        "citations": citations if citations and ("ticket" in (user_message or "").lower() or "list" in (user_message or "").lower()) else [],
    }


def _configure_llm() -> None:
    if not settings.gemini_api_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="GEMINI_API_KEY not configured.")
    genai.configure(api_key=settings.gemini_api_key)
