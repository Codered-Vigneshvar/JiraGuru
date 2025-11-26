"""Sprint management endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, status

from ..config import settings
from ..models import Sprint, SprintCreate
from ..storage import ensure_dirs, load_json, save_json
from .projects import assert_project_exists

router = APIRouter(prefix="/projects/{project_id}/sprints", tags=["sprints"])

SPRINTS_FILE = Path(settings.data_dir) / "sprints.json"


def _load_sprints() -> List[Sprint]:
    data = load_json(SPRINTS_FILE, default=[])
    return [Sprint.model_validate(item) for item in data]


def _save_sprints(sprints: List[Sprint]) -> None:
    save_json(SPRINTS_FILE, [s.model_dump() for s in sprints])


@router.post("", response_model=Sprint, status_code=status.HTTP_201_CREATED)
def create_sprint(project_id: int, payload: SprintCreate) -> Sprint:
    """Create a sprint under a project."""
    ensure_dirs(settings.data_dir)
    assert_project_exists(project_id)
    sprints = _load_sprints()
    next_id = max((s.id for s in sprints), default=0) + 1
    sprint = Sprint(id=next_id, project_id=project_id, **payload.model_dump())
    sprints.append(sprint)
    _save_sprints(sprints)
    ensure_dirs(Path(settings.data_dir) / f"project_{project_id}" / f"sprint_{sprint.id}")
    return sprint


@router.get("", response_model=List[Sprint])
def list_sprints(project_id: int) -> List[Sprint]:
    """List sprints belonging to a project."""
    assert_project_exists(project_id)
    return [s for s in _load_sprints() if s.project_id == project_id]


def assert_sprint_exists(project_id: int, sprint_id: int) -> Sprint:
    """Retrieve a sprint or raise 404."""
    assert_project_exists(project_id)
    for sprint in _load_sprints():
        if sprint.id == sprint_id and sprint.project_id == project_id:
            return sprint
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sprint not found")
