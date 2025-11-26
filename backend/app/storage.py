"""Simple file-based storage helpers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List

from .config import settings


def ensure_dirs(path: Path | str) -> Path:
    """Ensure a directory exists and return it."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json(path: Path | str, default: Any) -> Any:
    """Load JSON from a path, returning default if file is missing or invalid."""
    path_obj = Path(path)
    if not path_obj.exists():
        return default
    try:
        with path_obj.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        return default


def save_json(path: Path | str, data: Any) -> None:
    """Persist data as JSON, creating parent directories as needed."""
    path_obj = Path(path)
    ensure_dirs(path_obj.parent)
    fd, tmp_path_str = tempfile.mkstemp(dir=path_obj.parent, prefix=path_obj.name, suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, path_obj)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# User storage helpers -----------------------------------------------------
USERS_FILE = settings.data_dir / "users.json"
PROJECTS_LOCK = Lock()


def ensure_default_user() -> None:
    """Ensure a default owner user exists."""
    ensure_dirs(settings.data_dir)
    payload = load_json(USERS_FILE, default={"users": []})
    users: List[Dict[str, Any]] = payload.get("users", [])
    if any(u.get("username") == "Vignesh" for u in users):
        save_json(USERS_FILE, {"users": users})
        return
    default_user = {
        "id": "u_1",
        "username": "Vignesh",
        "password": "1234",
        "is_owner": True,
    }
    users.insert(0, default_user)
    save_json(USERS_FILE, {"users": users})


def load_users() -> List[Dict[str, Any]]:
    """Load users collection."""
    payload = load_json(USERS_FILE, default={"users": []})
    return payload.get("users", [])


def save_users(users: List[Dict[str, Any]]) -> None:
    """Persist users collection."""
    save_json(USERS_FILE, {"users": users})


def next_user_id(users: List[Dict[str, Any]]) -> str:
    """Generate next user id."""
    existing = [int(u["id"].split("_")[1]) for u in users if u.get("id", "").startswith("u_")]
    next_num = max(existing or [0]) + 1
    return f"u_{next_num}"


# Project storage helpers --------------------------------------------------
PROJECTS_FILE = settings.data_dir / "projects.json"
PROJECTS_DIR = settings.data_dir / "projects"


def ensure_projects_file() -> None:
    """Ensure projects file exists."""
    ensure_dirs(settings.data_dir)
    payload = load_json(PROJECTS_FILE, default={"projects": []})
    save_json(PROJECTS_FILE, {"projects": payload.get("projects", [])})


def load_projects() -> List[Dict[str, Any]]:
    """Load projects collection."""
    payload = load_json(PROJECTS_FILE, default={"projects": []})
    return payload.get("projects", [])


def save_projects(projects: List[Dict[str, Any]]) -> None:
    """Persist projects collection."""
    save_json(PROJECTS_FILE, {"projects": projects})


def next_project_id(projects: List[Dict[str, Any]]) -> str:
    """Generate next project id."""
    existing = [int(p["id"].split("_")[1]) for p in projects if p.get("id", "").startswith("p_")]
    next_num = max(existing or [0]) + 1
    return f"p_{next_num}"


def next_document_id(documents: List[Dict[str, Any]]) -> str:
    """Generate next document id."""
    existing = [int(d["id"].split("_")[1]) for d in documents if d.get("id", "").startswith("doc_")]
    next_num = max(existing or [0]) + 1
    return f"doc_{next_num}"


def ensure_project_doc_dir(project_id: str) -> Path:
    """Ensure the documents directory for a project exists."""
    return ensure_dirs(PROJECTS_DIR / project_id / "documents")


# Project access helpers ----------------------------------------------------
def get_project_by_id(project_id: str) -> Dict[str, Any]:
    """Return a project dict or raise FileNotFoundError."""
    for project in load_projects():
        if project.get("id") == project_id:
            return project
    raise FileNotFoundError("Project not found.")


def save_project(updated_project: Dict[str, Any]) -> None:
    """Persist a single project back into projects.json."""
    target_id = str(updated_project.get("id") or "")
    if not target_id:
        raise FileNotFoundError("Project not found.")
    projects = load_projects()
    for idx, project in enumerate(projects):
        if str(project.get("id")) == target_id:
            projects[idx] = updated_project
            save_projects(projects)
            return
    raise FileNotFoundError("Project not found.")


def save_project_plan(project_id: str, content: str) -> Dict[str, Any]:
    """Save a project's requirement plan content."""
    project = get_project_by_id(project_id)
    project["requirements_plan"] = content
    project["requirements_plan_updated_at"] = datetime.utcnow().isoformat()
    save_project(project)
    return project




def delete_project(project_id: str) -> None:
    """Delete a project and its associated directory."""
    projects = load_projects()
    filtered = [p for p in projects if p.get("id") != project_id]
    if len(filtered) == len(projects):
        raise FileNotFoundError("Project not found.")
    save_projects(filtered)
    # Remove project directory if it exists
    proj_dir = PROJECTS_DIR / project_id
    if proj_dir.exists():
        shutil.rmtree(proj_dir, ignore_errors=True)


# Ticket helpers ------------------------------------------------------------
def generate_new_ticket_id(project: Dict[str, Any]) -> str:
    """Generate the next ticket id within a project (t_1, t_2...)."""
    existing = [int(t.get("id", "t_0").split("_")[1]) for t in project.get("tickets", []) if t.get("id", "").startswith("t_")]
    next_num = max(existing or [0]) + 1
    return f"t_{next_num}"


def generate_new_ticket_key(project: Dict[str, Any]) -> str:
    """Generate the next human-friendly ticket key using project code."""
    code = project.get("code", "PRJ")
    existing_keys = [t.get("key", "") for t in project.get("tickets", [])]
    existing_nums = []
    for key in existing_keys:
        if key.startswith(f"{code}-"):
            suffix = key.split("-")[-1]
            try:
                existing_nums.append(int(suffix.replace("EP", "")))
            except ValueError:
                continue
    next_num = max(existing_nums or [0]) + 1
    return f"{code}-{next_num:03d}"


def generate_new_epic_key(project: Dict[str, Any], idx: int) -> str:
    """Generate a key for an epic with EP suffix."""
    code = project.get("code", "PRJ")
    return f"{code}-EP{idx:02d}"


# Comment storage -----------------------------------------------------------
def get_ticket_comments_path(project_id: str) -> Path:
    """Return path to comments file, ensuring it exists."""
    base = ensure_dirs(PROJECTS_DIR / project_id)
    comments_path = base / "ticket_comments.json"
    if not comments_path.exists():
        save_json(comments_path, {"comments": []})
    return comments_path


def load_ticket_comments(project_id: str) -> Dict[str, Any]:
    """Load comments for a project."""
    path = get_ticket_comments_path(project_id)
    return load_json(path, default={"comments": []})


def save_ticket_comments(project_id: str, data: Dict[str, Any]) -> None:
    """Persist comments for a project."""
    path = get_ticket_comments_path(project_id)
    save_json(path, data)
