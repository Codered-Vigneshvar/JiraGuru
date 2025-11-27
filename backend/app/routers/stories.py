"""Kanban board endpoints for project tickets."""

from __future__ import annotations

from typing import Dict, List, Literal

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse

from ..ai.story_gen import generate_story_payload
from ..models import Project, Ticket
from ..storage import get_project_by_id, load_users, save_project

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


@router.post("/{project_id}/generate_stories")
def generate_stories(
    project_id: str,
    regen: bool = Query(default=False, description="Regenerate stories if they already exist."),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict:
    """Generate epics and stories for a project with optional regeneration."""
    user = _require_user(x_user)
    try:
        project = Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    _assert_member(project, user)
    project_dict = project.model_dump()

    if not regen and ((project_dict.get("epics") or project_dict.get("stories"))):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Stories already exist for this project. Use regen=true to regenerate."},
        )

    if regen:
        project_dict["epics"] = []
        project_dict["stories"] = []

    try:
        payload, error = generate_story_payload(project, plan=project_dict.get("requirements_plan"))
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": exc.detail if isinstance(exc.detail, str) else "Gemini generation failed."},
        )
    if error or not payload:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"success": False, "error": error or "Invalid AI response"},
        )

    project_dict["epics"] = payload.get("epics", [])
    project_dict["stories"] = payload.get("stories", [])
    save_project(project_dict)

    return {
        "success": True,
        "message": "Stories generated successfully.",
        "epics": project_dict["epics"],
        "stories": project_dict["stories"],
    }
