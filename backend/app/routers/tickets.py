"""Ticket operations: generation, CRUD, comments, attachments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from fastapi.responses import FileResponse
from typing import List, Optional, Tuple

from fastapi import APIRouter, File, Header, HTTPException, UploadFile, status

from ..ai.story_gen import generate_epics_and_stories_for_project
from ..ai.plan_gen import generate_plan_for_project
from ..models import DocumentMeta, Project, Ticket, TicketComment, TicketUpdate, RequirementPlan
from ..storage import (
    ensure_project_doc_dir,
    generate_new_ticket_id,
    generate_new_ticket_key,
    get_project_by_id,
    load_projects,
    PROJECTS_LOCK,
    load_ticket_comments,
    load_users,
    save_project,
    save_projects,
    save_ticket_comments,
    save_project_plan,
)
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(prefix="/api/projects/{project_id}", tags=["tickets"])


def _get_user(username: str) -> Optional[dict]:
    for user in load_users():
        if user.get("username") == username:
            return user
    return None


def _require_user(x_user: str | None) -> dict:
    if not x_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-User header.")
    user = _get_user(x_user)
    if not user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unknown user.")
    return user


def _get_project_model(project_id: str) -> Project:
    try:
        return Project.model_validate(get_project_by_id(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")


def _assert_member(project: Project, user: dict) -> None:
    if user.get("is_owner"):
        return
    if user["username"] not in {project.owner_username, *project.member_usernames}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed.")


def _assert_member_dict(project: dict, user: dict) -> None:
    """Membership check for raw project dicts."""
    if user.get("is_owner"):
        return
    members = set(project.get("member_usernames", []) or [])
    owner = project.get("owner_username")
    if user["username"] not in {owner, *members}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed.")


def _find_ticket(project: Project, ticket_key: str) -> Tuple[int, Ticket]:
    for idx, ticket in enumerate(project.tickets):
        if ticket.key == ticket_key:
            return idx, ticket
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found.")


def _ensure_acceptance_criteria(ticket_data: dict) -> dict:
    """Ensure acceptance_criteria is populated with a default if missing/empty."""
    if not ticket_data.get("acceptance_criteria"):
        ticket_data["acceptance_criteria"] = [
            "All key details are present and accurate.",
            "Output is clear, concise, and testable.",
        ]
    return ticket_data


def _persist_project_dict(project_id: str, project_dict: dict) -> None:
    """Persist a project dict by replacing it in the projects list."""
    with PROJECTS_LOCK:
        projects = load_projects()
        for idx, proj in enumerate(projects):
            if str(proj.get("id")) == str(project_id):
                project_dict["id"] = str(project_id)
                projects[idx] = project_dict
                save_projects(projects)
                return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")


@router.post("/tickets/generate", response_model=List[Ticket])
def generate_tickets(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> List[Ticket]:
    """Generate epics and stories for a project using Gemini."""
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    project_dict = project.model_dump()
    plan_text = project_dict.get("requirements_plan")
    new_tickets = generate_epics_and_stories_for_project(project, plan=plan_text)

    # Merge with existing tickets
    existing_tickets = project_dict.get("tickets", [])
    existing_keys = {t.get("key") for t in existing_tickets}
    for t in new_tickets:
        if t.key in existing_keys:
            continue
        data = _ensure_acceptance_criteria(t.model_dump())
        existing_tickets.append(data)
    project_dict["tickets"] = existing_tickets
    _persist_project_dict(project_id, project_dict)
    return [Ticket.model_validate(_ensure_acceptance_criteria(t)) for t in existing_tickets]


@router.get("/tickets", response_model=List[Ticket])
def list_tickets(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> List[Ticket]:
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    return [Ticket.model_validate(_ensure_acceptance_criteria(t.model_dump())) for t in project.tickets]


@router.get("/tickets/{ticket_key}", response_model=Ticket)
def get_ticket(project_id: str, ticket_key: str, x_user: str | None = Header(default=None, alias="X-User")) -> Ticket:
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    _, ticket = _find_ticket(project, ticket_key)
    return Ticket.model_validate(_ensure_acceptance_criteria(ticket.model_dump()))


@router.patch("/tickets/{ticket_key}", response_model=Ticket)
def update_ticket(
    project_id: str,
    ticket_key: str,
    payload: TicketUpdate,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> Ticket:
    user = _require_user(x_user)
    projects = load_projects()
    proj_idx = next((i for i, p in enumerate(projects) if str(p.get("id")) == str(project_id)), None)
    if proj_idx is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    proj_dict = projects[proj_idx]
    _assert_member_dict(proj_dict, user)

    tickets = proj_dict.get("tickets", [])
    ticket_idx = next((i for i, t in enumerate(tickets) if t.get("key") == ticket_key), None)
    if ticket_idx is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found.")

    ticket_data = dict(tickets[ticket_idx])
    updates = payload.model_dump(exclude_none=True)
    allowed_fields = {
        "title",
        "description",
        "status",
        "assignee_username",
        "epic_key",
        "dependencies",
        "blockers",
        "linked_document_ids",
        "acceptance_criteria",
        "story_points",
        "priority",
    }

    if payload.priority and payload.priority not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid priority. Use one of: LOW, MEDIUM, HIGH, CRITICAL.",
        )

    for field, value in updates.items():
        if field not in allowed_fields:
            continue
        if field == "dependencies" and value is not None:
            ticket_data[field] = list(value)
        elif field == "acceptance_criteria" and value is not None:
            ticket_data[field] = [item for item in value if item]
        elif field == "story_points":
            ticket_data[field] = value
        else:
            ticket_data[field] = value

    ticket_data["updated_at"] = datetime.utcnow().isoformat()

    proj_dict["tickets"][ticket_idx] = ticket_data
    _persist_project_dict(project_id, proj_dict)
    return Ticket.model_validate(_ensure_acceptance_criteria(ticket_data))


@router.delete("/tickets/{ticket_key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ticket(project_id: str, ticket_key: str, x_user: str | None = Header(default=None, alias="X-User")) -> None:
    """Delete a single ticket."""
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    proj_dict = project.model_dump()
    filtered = [t for t in proj_dict.get("tickets", []) if t.get("key") != ticket_key]
    if len(filtered) == len(proj_dict.get("tickets", [])):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found.")
    proj_dict["tickets"] = filtered
    _persist_project_dict(project_id, proj_dict)


@router.delete("/tickets", status_code=status.HTTP_204_NO_CONTENT)
def clear_tickets(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> None:
    """Delete all tickets for a project."""
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    proj_dict = project.model_dump()
    proj_dict["tickets"] = []
    _persist_project_dict(project_id, proj_dict)


@router.get("/requirements/plan", response_model=RequirementPlan | dict)
def get_requirements_plan(project_id: str, x_user: str | None = Header(default=None, alias="X-User")):
    """Fetch saved requirement plan for a project."""
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    project_dict = project.model_dump()
    if "requirements_plan" not in project_dict:
        return {"project_id": project_id, "content": "", "updated_at": None}
    return RequirementPlan(
        project_id=project_id,
        content=project_dict.get("requirements_plan", ""),
        updated_at=datetime.fromisoformat(project_dict.get("requirements_plan_updated_at")),
    )


@router.post("/requirements/plan", response_model=RequirementPlan)
def save_requirements_plan(
    project_id: str,
    payload: dict,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> RequirementPlan:
    """Save an edited requirement plan."""
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    content = payload.get("content", "")
    updated_project = save_project_plan(project_id, content)
    return RequirementPlan(
        project_id=project_id,
        content=updated_project.get("requirements_plan", ""),
        updated_at=datetime.fromisoformat(updated_project.get("requirements_plan_updated_at")),
    )


@router.post("/requirements/plan/generate", response_model=RequirementPlan)
def generate_requirements_plan(
    project_id: str,
    payload: dict | None = None,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> RequirementPlan:
    """Generate a detailed requirement plan using Gemini."""
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    requirements = (payload or {}).get("requirements") if payload else None
    plan_text = generate_plan_for_project(project, requirements=requirements)
    updated_project = save_project_plan(project_id, plan_text)
    return RequirementPlan(
        project_id=project_id,
        content=plan_text,
        updated_at=datetime.fromisoformat(updated_project.get("requirements_plan_updated_at")),
    )





# Fallback stub generation (offline) ----------------------------------------


@router.post("/tickets/{ticket_key}/comments", response_model=TicketComment, status_code=status.HTTP_201_CREATED)
def add_comment(
    project_id: str,
    ticket_key: str,
    payload: dict,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> TicketComment:
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    _find_ticket(project, ticket_key)

    comments_data = load_ticket_comments(project_id)
    comments = comments_data.get("comments", [])
    existing_ids = [int(c["id"].split("_")[1]) for c in comments if c.get("id", "").startswith("c_")]
    new_id = f"c_{max(existing_ids or [0]) + 1}"
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Comment text required.")
    comment = TicketComment(
        id=new_id,
        ticket_key=ticket_key,
        author_username=user["username"],
        text=text,
    )
    comments.append(comment.model_dump())
    comments_data["comments"] = comments
    save_ticket_comments(project_id, comments_data)
    return comment


@router.get("/tickets/{ticket_key}/comments", response_model=List[TicketComment])
def list_comments(project_id: str, ticket_key: str, x_user: str | None = Header(default=None, alias="X-User")) -> List[TicketComment]:
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    _find_ticket(project, ticket_key)
    comments_data = load_ticket_comments(project_id)
    comments = comments_data.get("comments", [])
    return [TicketComment.model_validate(c) for c in comments if c.get("ticket_key") == ticket_key]


@router.post("/tickets/{ticket_key}/documents", response_model=Ticket)
async def upload_ticket_document(
    project_id: str,
    ticket_key: str,
    file: UploadFile = File(...),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> Ticket:
    user = _require_user(x_user)
    project = _get_project_model(project_id)
    _assert_member(project, user)
    idx, ticket = _find_ticket(project, ticket_key)

    project_dict = project.model_dump()
    documents = project_dict.get("documents", [])
    doc_ids = {d.get("id") for d in documents}

    # Create new document meta and store file
    next_doc_num = max([int(d["id"].split("_")[1]) for d in documents if d.get("id", "").startswith("doc_")] or [0]) + 1
    doc_id = f"doc_{next_doc_num}"
    suffix = Path(file.filename or "").suffix
    generated_name = f"{doc_id}{suffix}"

    doc_dir = ensure_project_doc_dir(project_id)
    file_path = doc_dir / generated_name
    content = await file.read()
    file_path.write_bytes(content)

    meta = DocumentMeta(
        id=doc_id,
        filename=generated_name,
        original_name=file.filename or generated_name,
        content_type=file.content_type or "application/octet-stream",
    )
    documents.append(meta.model_dump())
    project_dict["documents"] = documents

    ticket_data = ticket.model_dump()
    if doc_id not in ticket_data.get("linked_document_ids", []):
        ticket_data.setdefault("linked_document_ids", []).append(doc_id)
    ticket_data["updated_at"] = datetime.utcnow().isoformat()
    project_dict["tickets"][idx] = ticket_data

    _persist_project_dict(project_id, project_dict)
    return Ticket.model_validate(ticket_data)
