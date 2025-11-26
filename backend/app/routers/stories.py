"""Kanban board endpoints for project tickets."""

from __future__ import annotations

from typing import Dict, List, Literal

from fastapi import APIRouter, Header, HTTPException, status

from ..models import Project, Ticket
from ..storage import get_project_by_id, load_users

router = APIRouter(prefix="/api/projects", tags=["board"])


# Basic auth helpers (aligned with other routers)
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


def _group_tickets_by_status(tickets: List[Ticket]) -> Dict[Literal["TODO", "IN_PROGRESS", "DONE"], List[Ticket]]:
    groups: Dict[Literal["TODO", "IN_PROGRESS", "DONE"], List[Ticket]] = {
        "TODO": [],
        "IN_PROGRESS": [],
        "DONE": [],
    }
    for ticket in tickets:
        status_value = ticket.status or "TODO"
        if status_value not in groups:
            status_value = "TODO"
        groups[status_value].append(ticket)
    return groups


@router.get("/{project_id}/board")
def get_board(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> dict:
    """Return tickets grouped by status for a project (Kanban board)."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    _assert_member(project, user)

    tickets: List[Ticket] = list(project.tickets or [])
    grouped = _group_tickets_by_status(tickets)

    return {
        "project_id": project.id,
        "project_code": project.code,
        "counts": {
            "total": len(tickets),
            "TODO": len(grouped["TODO"]),
            "IN_PROGRESS": len(grouped["IN_PROGRESS"]),
            "DONE": len(grouped["DONE"]),
        },
        "columns": {
            "TODO": grouped["TODO"],
            "IN_PROGRESS": grouped["IN_PROGRESS"],
            "DONE": grouped["DONE"],
        },
    }
