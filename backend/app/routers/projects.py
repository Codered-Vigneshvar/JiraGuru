"""Project management endpoints for the MVP."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)

from ..models import DocumentMeta, Project, ProjectCreate, ProjectUpdate
from ..storage import (
    ensure_project_doc_dir,
    delete_project,
    load_projects,
    load_users,
    next_document_id,
    next_project_id,
    save_projects,
    load_ticket_comments,
    save_ticket_comments,
)
from ..ai.project_index import ProjectIndex

router = APIRouter(prefix="/api/projects", tags=["projects"])


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


def _require_owner_user(x_user: str | None) -> dict:
    user = _require_user(x_user)
    if not user.get("is_owner"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Owner required.")
    return user


def _find_project(project_id: str) -> tuple[int, dict]:
    projects = load_projects()
    for idx, project in enumerate(projects):
        if project.get("id") == project_id:
            return idx, project
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")


def _to_project_models(projects: List[dict]) -> List[Project]:
    return [Project.model_validate(p) for p in projects]


@router.post("", response_model=Project, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate, x_user: str | None = Header(default=None, alias="X-User")) -> Project:
    """Create a new project for the requesting user."""
    caller = _require_user(x_user)
    projects = load_projects()
    unknown_members = [u for u in payload.member_usernames if not _get_user(u)]
    if unknown_members:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown members: {', '.join(unknown_members)}")

    project_id = next_project_id(projects)
    members = list({*payload.member_usernames, caller["username"]})
    new_project = Project(
        id=project_id,
        code=payload.code,
        title=payload.title,
        description=payload.description,
        owner_username=caller["username"],
        member_usernames=members,
        documents=[],
        tickets=[],
    )
    projects.append(new_project.model_dump())
    save_projects(projects)
    ensure_project_doc_dir(project_id)
    return new_project


@router.get("", response_model=List[Project])
def list_projects(x_user: str | None = Header(default=None, alias="X-User")) -> List[Project]:
    """List projects relevant to the requesting user."""
    user = _require_user(x_user)
    projects = load_projects()
    if user.get("is_owner"):
        return _to_project_models(projects)
    filtered = [
        p
        for p in projects
        if user["username"] == p.get("owner_username") or user["username"] in p.get("member_usernames", [])
    ]
    return _to_project_models(filtered)


@router.get("/{project_id}", response_model=Project)
def get_project(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> Project:
    """Return a single project."""
    user = _require_user(x_user)
    _, project = _find_project(project_id)
    if not user.get("is_owner") and user["username"] not in {project.get("owner_username"), *project.get("member_usernames", [])}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed.")
    return Project.model_validate(project)


@router.patch("/{project_id}", response_model=Project)
def update_project(
    project_id: str,
    payload: ProjectUpdate,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> Project:
    """Update basic project details."""
    user = _require_user(x_user)
    idx, project = _find_project(project_id)
    if not (user.get("is_owner") or user["username"] == project.get("owner_username")):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed.")

    if payload.title is not None:
        project["title"] = payload.title
    if payload.description is not None:
        project["description"] = payload.description
    if payload.member_usernames is not None:
        unknown_members = [u for u in payload.member_usernames if not _get_user(u)]
        if unknown_members:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown members: {', '.join(unknown_members)}"
            )
        project["member_usernames"] = payload.member_usernames

    projects = load_projects()
    projects[idx] = project
    save_projects(projects)
    return Project.model_validate(project)


@router.post("/{project_id}/documents", response_model=Project)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> Project:
    """Attach a document to a project."""
    user = _require_user(x_user)
    idx, project = _find_project(project_id)
    if not (user.get("is_owner") or user["username"] in {project.get("owner_username"), *project.get("member_usernames", [])}):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed.")

    documents = project.get("documents", [])
    doc_id = next_document_id(documents)
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
    project["documents"] = documents

    projects = load_projects()
    projects[idx] = project
    save_projects(projects)
    try:
        ProjectIndex(project_id).upsert_doc(doc_id)
    except Exception:
        # Swallow indexing errors to avoid breaking upload; can be logged later.
        pass
    return Project.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project_endpoint(project_id: str, x_user: str | None = Header(default=None, alias="X-User")) -> None:
    """Delete a project and its data (owner only)."""
    _require_owner_user(x_user)
    # clear comments file if exists
    try:
        comments = load_ticket_comments(project_id)
        comments["comments"] = []
        save_ticket_comments(project_id, comments)
    except FileNotFoundError:
        pass
    delete_project(project_id)
