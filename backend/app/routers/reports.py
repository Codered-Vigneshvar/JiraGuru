"""Project health, impact, and advisory endpoints."""

from __future__ import annotations

import hashlib
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import google.generativeai as genai
from fastapi import APIRouter, Header, HTTPException, Query, status

from ..models import Project, Ticket
from ..storage import (
    ensure_project_doc_dir,
    get_project_by_id,
    load_users,
    load_ticket_comments,
    save_project,
    generate_new_ticket_id,
    generate_new_ticket_key,
    load_change_log,
    append_change_entry,
    load_impacted_cache,
    save_impacted_cache,
    load_plan_versions,
    save_project_plan,
)
from ..ai.impact_graph import impact_graph, ImpactState
from ..ai.story_gen import log_ai_event
from ..ai.project_index import ProjectIndex
from ..ai.impacted_tickets import generate_impacted_tickets
from ..ai.spec_conflict import check_project_alignment
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
    deps_blocked: List[str] = []
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
        # dependencies not done
        for dep in t.dependencies or []:
            dep_ticket = next((tt for tt in tickets if tt.key == dep), None)
            if dep_ticket and dep_ticket.status != "DONE":
                deps_blocked.append(t.key)
                break

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
        "dependency_blocked_tickets": deps_blocked,
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
            root_deps = []
        for dep in root_ticket.dependencies or []:
            dep_ticket = tickets_by_key.get(dep)
            if dep_ticket and dep_ticket.status == "DONE":
                continue
            root_deps.append(
                {
                    "ticket_key": dep,
                    "title": dep_ticket.title if dep_ticket else "",
                    "status": dep_ticket.status if dep_ticket else "",
                    }
                )
            unfinished_dependencies = [
                dep for dep in (tickets_by_key.get(root_key).dependencies or [])
                if dep in tickets_by_key and tickets_by_key[dep].status != "DONE"
            ]
            chain = chains.setdefault(
                root_key,
                {
                    "root_ticket_key": root_ticket.key,
                    "root_title": root_ticket.title,
                    "root_status": root_ticket.status,
                    "root_blockers_text": root_ticket.blockers or "",
                    "root_unfinished_dependencies": unfinished_dependencies,
                    "root_dependencies": root_deps,
                    "affected_tickets": [],
                },
            )
            ticket = tickets_by_key.get(blocked_key)
            deps_info = []
            if ticket:
                for dep in ticket.dependencies or []:
                    dep_ticket = tickets_by_key.get(dep)
                    if dep_ticket and dep_ticket.status == "DONE":
                        continue
                    deps_info.append(
                        {
                            "ticket_key": dep,
                            "title": dep_ticket.title if dep_ticket else "",
                            "status": dep_ticket.status if dep_ticket else "",
                        }
                    )
            chain["affected_tickets"].append(
                {
                    "ticket_key": blocked_key,
                    "title": tickets_by_key[blocked_key].title if blocked_key in tickets_by_key else "",
                    "status": tickets_by_key[blocked_key].status if blocked_key in tickets_by_key else "",
                    "distance": distance_up,
                    "dependencies": deps_info,
                }
            )

    # Sort affected tickets by distance then key for stability
    chains_list = []
    for chain in chains.values():
        chain["affected_tickets"] = sorted(chain["affected_tickets"], key=lambda x: (x["distance"], x["ticket_key"]))
        chains_list.append(chain)

    dependency_view: List[dict] = []
    for t in tickets:
        if not t.dependencies:
            continue
        deps_info = []
        for dep in t.dependencies:
            dep_ticket = tickets_by_key.get(dep)
            if dep_ticket and dep_ticket.status == "DONE":
                continue
            deps_info.append(
                {
                    "ticket_key": dep,
                    "title": dep_ticket.title if dep_ticket else "",
                    "status": dep_ticket.status if dep_ticket else "",
                }
            )
        if not deps_info:
            continue
        dependency_view.append(
            {
                "ticket_key": t.key,
                "title": t.title,
                "status": t.status,
                "dependencies": deps_info,
            }
        )

    return {
        "project_id": project.id,
        "project_code": project.code,
        "chains": chains_list,
        "dependency_view": dependency_view,
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

    # Fast-path: content unchanged -> no new impact calculation; reuse prior impact/report
    if cached:
        doc_unchanged = (doc_text.strip() == (cached.get("doc_text") or "").strip())
        desc_unchanged = (project.description.strip() == (cached.get("project_description") or "").strip())
        title_unchanged = (project.title.strip() == (cached.get("project_title") or "").strip())
        plan_unchanged = (plan_text.strip() == (cached.get("plan_text") or "").strip())
        if doc_unchanged and desc_unchanged and title_unchanged and plan_unchanged:
            return {
                "project_id": project_id,
                "doc_id": doc_id,
                "from_cache": True,
                "has_changes": False,
                "change_summary": cached.get(
                    "change_summary", "No changes detected since last analysis."
                ),
                "impacted_tickets": cached.get("impacted_tickets", []),
                "report": cached.get("report", ""),
            }

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

    # Prepare delta for changed content to focus impact
    old_doc = cached.get("doc_text", "") if cached else ""
    delta_text = ""
    if old_doc and doc_text:
        import difflib

        added_lines = [
            line[2:]
            for line in difflib.ndiff(old_doc.splitlines(), doc_text.splitlines())
            if line.startswith("+ ")
        ]
        delta_text = "\n".join(added_lines).strip()
        # Keep delta focused to avoid matching every ticket
        if len(delta_text) > 2000:
            delta_text = delta_text[:2000]

    # Changes detected or first run -> run pipeline
    initial_state: ImpactState = {
        "project_id": project_id,
        "doc_id": doc_id,
        "doc_text": doc_text,
        "old_doc_text": old_doc,
        "delta_text": delta_text,
    }
    try:
        final_state = impact_graph.invoke(initial_state)
    except HTTPException:
        raise
    except Exception as exc:
        log_ai_event(f"[Impact Error] {exc}")
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
        "plan_text": plan_text,
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

    # Try to load cached impact results for richer context, but only if state matches current
    cache_file = _cache_path(project_id, doc_id)
    cached = None
    if cache_file.exists():
        try:
            cached_loaded = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cached_loaded = None
        if cached_loaded and cached_loaded.get("project_id") == project_id:
            state_string = _state_string(project, doc_text, project.tickets or [])
            current_hash = _state_hash(state_string)
            if cached_loaded.get("state_hash") == current_hash:
                cached = cached_loaded

    project_context = f"Project: {project.title}\nDescription: {project.description}"
    cached_impacts = cached.get("impacted_tickets", []) if cached else []
    current_keys = {t.key for t in (project.tickets or [])}
    cached_impacts = [c for c in cached_impacts if c.get("ticket_key") in current_keys]
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
        hits_raw = index.search_tickets(user_message or "", k=6)
    except Exception:
        hits_raw = []

    # Filter out hits that are not part of the current project tickets
    current_keys = {t.key for t in (project.tickets or [])}
    hits = [h for h in hits_raw if h.get("metadata", {}).get("ticket_key") in current_keys]

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
                "assignee_username": meta.get("assignee_username"),
                "priority": meta.get("priority"),
                "story_points": meta.get("story_points"),
                "dependencies": meta.get("dependencies"),
                "score": hit.get("score"),
                "excerpt": (hit.get("content") or "")[:800],
            }
        )

    _configure_llm()
    model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
    citations_text = json.dumps(citations, ensure_ascii=False)
    history_text = "\n\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
    project_context = f"Project: {project.title}\nDescription: {project.description}\nMembers: {', '.join(project.member_usernames)}"
    tickets_snapshot = "\n".join(
        [
            f"{t.key} | {t.title} | assignee={t.assignee_username or 'Unassigned'} | status={t.status} | priority={getattr(t, 'priority', 'MEDIUM')} | story_points={getattr(t, 'story_points', '')}"
            for t in (project.tickets or [])
        ]
    )[:4000]
    prompt = f"""
You are a concise project assistant. Use the project context and ticket evidence to answer.
Reply in plain text (no markdown or bullets). Keep responses brief and conversational. Cite ticket keys only when listing or referencing tickets.
Include blockers or dependencies if relevant. If the question is unrelated to this project, say so.

Project context:
{project_context}

Ticket snapshot (all current tickets):
{tickets_snapshot or "No tickets yet."}

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


@router.post("/{project_id}/apply_alignment")
def apply_alignment(
    project_id: str,
    payload: dict,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict:
    """Apply a suggested alignment fix to a ticket description or requirements plan."""
    user = _require_user(x_user)
    project_dict = get_project_by_id(project_id)
    project = Project.model_validate(project_dict)
    _assert_member(project, user)

    target = (payload or {}).get("target") or ""
    target_type = (payload or {}).get("target_type") or ""
    section_label = (payload or {}).get("section_label") or ""
    content = (payload or {}).get("content") or ""
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No content provided.")

    updated = False
    if "requirements/plan" in target or "requirements_plan" in target or "plan" == target or target_type == "spec":
        save_project_plan(project_id, content)
        updated = True
    elif isinstance(target, str) and "-" in target:
        # treat as ticket key
        proj = get_project_by_id(project_id)
        tickets = proj.get("tickets", [])
        for idx, t in enumerate(tickets):
            if t.get("key") == target:
                if "acceptance" in section_label.lower():
                    lines = [line.strip() for line in content.splitlines() if line.strip()]
                    t["acceptance_criteria"] = lines
                else:
                    t["description"] = content
                t["updated_at"] = datetime.utcnow().isoformat()
                tickets[idx] = t
                proj["tickets"] = tickets
                save_project(proj)
                updated = True
                break

    if not updated:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Target not supported for auto-apply.")

    return {"project_id": project_id, "target": target, "updated": True}


@router.post("/{project_id}/impacted_tickets")
def impacted_tickets(
    project_id: str,
    payload: dict,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict:
    """Generate impacted + new tickets using Gemini + RAG over the project tickets."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    _assert_member(project, user)

    old_spec = (payload or {}).get("oldSpec", "")
    new_spec = (payload or {}).get("newSpec", "")
    log_entries = load_change_log(project_id).get("entries", [])
    # Prefer a real change (requirements_plan/description) where old != new
    meaningful_change = next(
        (e for e in reversed(log_entries) if e.get("type") in {"requirements_plan", "description"} and (e.get("old") or "") != (e.get("new") or "")),
        None,
    )
    latest_entry = log_entries[-1] if log_entries else None
    chosen_entry = meaningful_change or latest_entry

    # Auto-fill from plan versions first (use latest as new, previous as old)
    plan_versions = load_plan_versions(project_id).get("versions", [])
    latest_plan = plan_versions[-1] if plan_versions else None
    # Find the most recent different plan content for baseline
    prev_plan = None
    if plan_versions:
        latest_content = plan_versions[-1].get("content", "")
        for entry in reversed(plan_versions[:-1]):
            if entry.get("content", "") != latest_content:
                prev_plan = entry
                break
    if not new_spec:
        new_spec = (latest_plan or {}).get("content") or project.model_dump().get("requirements_plan", "") or ""
    if not old_spec:
        old_spec = (prev_plan or {}).get("content") or ""

    # If still missing, backfill from change history (description/plan)
    if not new_spec and chosen_entry:
        new_spec = chosen_entry.get("new") or chosen_entry.get("old") or ""
    if not old_spec and chosen_entry:
        old_spec = chosen_entry.get("old") or chosen_entry.get("new") or ""

    # Fallback to current project description + plan
    plan_text = project.model_dump().get("requirements_plan", "") or ""
    if not new_spec:
        new_spec = f"{project.description}\n\n{plan_text}".strip()
    if not old_spec:
        old_spec = f"{project.description}\n\n{plan_text}".strip()

    # If specs still identical but we have a prior plan version or cache, try to pick a different baseline
    if old_spec.strip() == new_spec.strip():
        if prev_plan and (prev_plan.get("content") or "").strip() != new_spec.strip():
            old_spec = prev_plan.get("content", "")
        else:
            cache_entries = load_impacted_cache(project_id).get("entries", [])
            last_cache = cache_entries[-1] if cache_entries else None
            if last_cache and (last_cache.get("new") or "").strip() != new_spec.strip():
                old_spec = last_cache.get("new", "")

    # No change -> return last cached LLM result without a new call (unless newer change exists)
    cache_entries = load_impacted_cache(project_id).get("entries", [])
    last_cache = cache_entries[-1] if cache_entries else None
    latest_change_ts = None
    if latest_plan and latest_plan.get("saved_at"):
        latest_change_ts = latest_plan["saved_at"]
    elif project.model_dump().get("requirements_plan_updated_at"):
        latest_change_ts = project.model_dump().get("requirements_plan_updated_at")
    elif chosen_entry and chosen_entry.get("saved_at"):
        latest_change_ts = chosen_entry["saved_at"]
    cache_ts = last_cache.get("saved_at") if last_cache else None
    # If no previous plan version, fall back to last cached "new" as historical baseline
    if not old_spec and last_cache:
        old_spec = last_cache.get("new") or old_spec
    # If still identical and we have a previous plan with different content, use it to force a diff
    if old_spec.strip() == new_spec.strip() and prev_plan and prev_plan.get("content", "").strip() != new_spec.strip():
        old_spec = prev_plan.get("content", "")
    # If an identical old/new pair was previously analyzed, reuse that response with a note
    pair_cache = next(
        (e for e in reversed(cache_entries) if e.get("old") == old_spec and e.get("new") == new_spec and e.get("result")),
        None,
    )
    if pair_cache:
        cached_result = pair_cache.get("result", {})
        note = f"No new changes; reusing previous impact analysis from {pair_cache.get('saved_at','recent run')}."
        return {
            "updatedTickets": cached_result.get("updatedTickets", []),
            "newTickets": cached_result.get("newTickets", []),
            "changeSummary": f"{note}\n{cached_result.get('changeSummary', pair_cache.get('change_summary',''))}",
            "evidence": cached_result.get("evidence", pair_cache.get("evidence", [])),
        }
    if old_spec.strip() == new_spec.strip() and last_cache:
        # Only reuse cache if it is at least as recent as the latest change
        if not latest_change_ts or (cache_ts and cache_ts >= latest_change_ts):
            return last_cache.get("result", {
                "updatedTickets": [],
                "newTickets": [],
                "changeSummary": last_cache.get("change_summary", "No changes detected."),
                "evidence": last_cache.get("evidence", []),
            })

    # Log what specs are being sent to the LLM to help debug identical/diff issues
    def _fingerprint(txt: str) -> str:
        import hashlib
        return hashlib.sha256((txt or "").encode("utf-8", errors="ignore")).hexdigest()[:10]

    log_ai_event(
        f"[ImpactedTickets Specs] project={project_id} | old_len={len(old_spec)} new_len={len(new_spec)} "
        f"old_hash={_fingerprint(old_spec)} new_hash={_fingerprint(new_spec)} "
        f"latest_change_ts={latest_change_ts} cache_ts={cache_ts}"
    )

    result = generate_impacted_tickets(project_id, old_spec, new_spec)

    def _ensure_keys(items: List[Dict], project_code: str) -> List[Dict]:
        patched: List[Dict] = []
        for idx, itm in enumerate(items or [], start=1):
            type_val = str(itm.get("type", "")).upper()
            key_val = itm.get("key") or itm.get("id")
            if not key_val:
                if type_val == "EPIC":
                    key_val = f"{project_code}-EP-NEW-{idx:03d}"
                else:
                    key_val = f"{project_code}-NEW-{idx:03d}"
            itm["key"] = key_val
            patched.append(itm)
        return patched

    result["updatedTickets"] = _ensure_keys(result.get("updatedTickets", []), project.code)
    result["newTickets"] = _ensure_keys(result.get("newTickets", []), project.code)
    append_change_entry(project_id, "manual_spec", old_spec, new_spec, user.get("username"))
    # If specs differ, ensure the summary reflects that (even when model claims identical)
    if old_spec.strip() != new_spec.strip():
        delta = _brief_diff(old_spec, new_spec, limit=20)
        if (
            not result.get("changeSummary")
            or "no change" in str(result.get("changeSummary", "")).lower()
            or "identical" in str(result.get("changeSummary", "")).lower()
        ):
            result["changeSummary"] = delta or "Differences detected between specs."
    # Cache this result keyed by the specs
    cache_entries.append(
        {
            "old": old_spec,
            "new": new_spec,
            "change_summary": result.get("changeSummary", ""),
            "evidence": result.get("usedTickets", []),
            "saved_at": latest_change_ts or datetime.utcnow().isoformat(),
            "result": {
                "updatedTickets": result.get("updatedTickets", []),
                "newTickets": result.get("newTickets", []),
                "changeSummary": result.get("changeSummary", ""),
                "evidence": result.get("usedTickets", []),
            },
        }
    )
    save_impacted_cache(project_id, cache_entries[-50:])
    return {
        "updatedTickets": result.get("updatedTickets", []),
        "newTickets": result.get("newTickets", []),
        "changeSummary": result.get("changeSummary", ""),
        "evidence": result.get("usedTickets", []),
    }


@router.post("/{project_id}/impacted_tickets/apply")
def apply_impacted_tickets(
    project_id: str,
    payload: dict,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict:
    """Persist selected impacted ticket changes and new tickets."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    _assert_member(project, user)
    project_dict = project.model_dump()

    updated_items = (payload or {}).get("updatedTickets") or []
    new_items = (payload or {}).get("newTickets") or []

    updated_results = []
    for item in updated_items:
        identifier = item.get("id")
        if not identifier:
            continue
        found = _find_ticket_index(project_dict, identifier)
        if not found:
            continue
        idx, ticket = found
        ticket_data = dict(ticket)
        ticket_data["title"] = item.get("title") or ticket_data.get("title")
        ticket_data["description"] = item.get("description") or ticket_data.get("description")
        new_ac = _normalize_ac(item.get("acceptanceCriteria") or item.get("acceptance_criteria"))
        ticket_data["acceptance_criteria"] = new_ac or ticket_data.get("acceptance_criteria") or []
        ticket_data["updated_at"] = datetime.utcnow().isoformat()
        project_dict["tickets"][idx] = ticket_data
        updated_results.append(ticket_data)

    new_results = []
    # First pass: create epics (including any new epics) to make keys available for stories
    latest_epic_key: str | None = None
    epic_keys_created: set[str] = set()
    for item in new_items:
        type_val_raw = str(item.get("type", "")).strip().upper()
        type_val = type_val_raw or "STORY"
        if type_val != "EPIC":
            continue
        title_val = (item.get("title") or "").strip()
        desc_val = (item.get("description") or "").strip()
        if not title_val and not desc_val:
            continue
        proposed_key = _coerce_ticket_key(project_dict, (item.get("id") or item.get("key")))
        ticket_id = generate_new_ticket_id(project_dict)
        ticket_key = proposed_key or generate_new_ticket_key(project_dict)
        epic_ticket = {
            "id": ticket_id,
            "key": ticket_key,
            "project_id": project_id,
            "type": "EPIC",
            "title": title_val or "New Epic from Impact",
            "description": desc_val or "",
            "status": "TODO",
            "assignee_username": None,
            "epic_key": None,
            "dependencies": [],
            "blockers": None,
            "linked_document_ids": [],
            "acceptance_criteria": _normalize_ac(item.get("acceptanceCriteria") or item.get("acceptance_criteria")),
            "priority": "MEDIUM",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        project_dict.setdefault("tickets", []).append(epic_ticket)
        new_results.append(epic_ticket)
        latest_epic_key = ticket_key
        epic_keys_created.add(ticket_key)

    # Second pass: create stories (and, if still needed, an auto-epic)
    auto_epic: dict | None = None
    for item in new_items:
        title_val = (item.get("title") or "").strip()
        desc_val = (item.get("description") or "").strip()
        if not title_val and not desc_val:
            # Skip empty new tickets; avoids creating unnecessary tickets
            continue
        type_val_raw = str(item.get("type", "")).strip().upper()
        type_val = type_val_raw or "STORY"
        if type_val not in {"EPIC", "STORY"}:
            continue
        if type_val == "EPIC":
            # Already processed in first pass
            continue

        # STORY handling
        epic_from_payload = (item.get("epic") or item.get("epic_key") or "").strip() or None
        # Accept epic if it exists in project or was just created
        known_epics = {t.get("key") for t in project_dict.get("tickets", []) if t.get("type") == "EPIC"}
        epic_key_for_story = epic_from_payload if epic_from_payload in known_epics else None
        epic_key_for_story = epic_from_payload or latest_epic_key
        if epic_key_for_story is None:
            # No epic provided anywhere -> create one auto (as a last resort)
            ticket_id_epic = generate_new_ticket_id(project_dict)
            ticket_key_epic = generate_new_ticket_key(project_dict)
            auto_epic = {
                "id": ticket_id_epic,
                "key": ticket_key_epic,
                "project_id": project_id,
                "type": "EPIC",
                "title": "Impact: New Epic",
                "description": (
                    "Auto-created epic grouping new impacted stories that did not map to any existing epic. "
                    "Update this epic title/description to reflect the new scope."
                ),
                "status": "TODO",
                "assignee_username": None,
                "epic_key": None,
                "dependencies": [],
                "blockers": None,
                "linked_document_ids": [],
                "acceptance_criteria": ["All key details are present and accurate.", "Output is clear, concise, and testable."],
                "priority": "MEDIUM",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            project_dict.setdefault("tickets", []).append(auto_epic)
            new_results.append(auto_epic)
            latest_epic_key = ticket_key_epic
            epic_key_for_story = ticket_key_epic

        ticket_id = generate_new_ticket_id(project_dict)
        proposed_story_key = _coerce_ticket_key(project_dict, (item.get("id") or item.get("key")))
        ticket_key = proposed_story_key or generate_new_ticket_key(project_dict)
        story_ticket = {
            "id": ticket_id,
            "key": ticket_key,
            "project_id": project_id,
            "type": "STORY",
            "title": title_val or "Story",
            "description": desc_val or "",
            "status": "TODO",
            "assignee_username": None,
            "epic_key": epic_key_for_story,
            "dependencies": [],
            "blockers": None,
            "linked_document_ids": [],
            "acceptance_criteria": _normalize_ac(item.get("acceptanceCriteria") or item.get("acceptance_criteria")),
            "priority": "MEDIUM",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        project_dict.setdefault("tickets", []).append(story_ticket)
        new_results.append(story_ticket)

    save_project(project_dict)
    try:
        index = ProjectIndex(project_id)
        for t in updated_results:
            if t.get("key"):
                index.upsert_ticket(t["key"])
        for t in new_results:
            if t.get("key"):
                index.upsert_ticket(t["key"])
    except Exception:
        pass

    return {
        "updated": updated_results,
        "created": new_results,
        "tickets": project_dict.get("tickets", []),
    }


@router.get("/{project_id}/impact_changes")
def list_impact_changes(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> dict:
    """Return saved spec/plan/description change history for impact analysis."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    _assert_member(project, user)
    entries = load_change_log(project_id).get("entries", [])
    return {"project_id": project_id, "entries": entries[-50:]}


@router.post("/{project_id}/check-alignment")
def check_alignment(
    project_id: str,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict:
    """Project-wide alignment/conflict check."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    _assert_member(project, user)
    result = check_project_alignment(project_id)
    return result


def _configure_llm() -> None:
    if not settings.gemini_api_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="GEMINI_API_KEY not configured.")
    genai.configure(api_key=settings.gemini_api_key)


# ---------------- Impacted tickets helpers ----------------
def _normalize_ac(value: str | list | None) -> list[str]:
    """Normalize acceptance criteria into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return []


def _find_ticket_index(project_dict: dict, identifier: str) -> tuple[int, dict] | None:
    """Locate a ticket by key or id."""
    tickets = project_dict.get("tickets", [])
    for idx, ticket in enumerate(tickets):
        if str(ticket.get("key")) == str(identifier) or str(ticket.get("id")) == str(identifier):
            return idx, ticket
    return None


def _coerce_ticket_key(project_dict: dict, proposed: str | None) -> str | None:
    """Ensure a ticket key matches the project code prefix (e.g., EDP-001)."""
    if not proposed:
        return None
    code = project_dict.get("code", "")
    if code and proposed.startswith(f"{code}-"):
        return proposed
    return None


def _brief_diff(old: str, new: str, limit: int = 50) -> str:
    """Return a short line-level diff summary for messaging."""
    import difflib

    old_lines = (old or "").splitlines()
    new_lines = (new or "").splitlines()
    diff = difflib.ndiff(old_lines, new_lines)
    changes: list[str] = []
    for line in diff:
        if line.startswith("+ "):
            changes.append(f"ADDED: {line[2:]}")
        elif line.startswith("- "):
            changes.append(f"REMOVED: {line[2:]}")
        if len(changes) >= limit:
            break
    return "\n".join(changes)
