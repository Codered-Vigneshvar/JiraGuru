"""Project health report endpoints."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from fastapi import APIRouter, Header, HTTPException, status

from ..models import Project, Ticket
from ..storage import get_project_by_id, load_users

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
