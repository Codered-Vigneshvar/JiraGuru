"""Authentication and user management endpoints (prototype, no real auth)."""

from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter, Header, HTTPException, status

from ..models import User, UserCreate, UserLogin, UserPublic
from ..storage import load_users, save_users, next_user_id

router = APIRouter(prefix="/api", tags=["auth"])


def _find_user(username: str) -> Optional[dict]:
    """Return user dict if found."""
    for user in load_users():
        if user.get("username") == username:
            return user
    return None


def _require_owner(user_header: str | None) -> dict:
    """Ensure the caller is the owner user."""
    if not user_header:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Owner required.")
    user = _find_user(user_header)
    if not user or not user.get("is_owner"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Owner required.")
    return user


@router.post("/login")
def login(payload: UserLogin):
    """Very simple credential check against users.json."""
    user = _find_user(payload.username)
    if not user or user.get("password") != payload.password:
        return {"success": False, "error": "Invalid credentials"}
    public_user = UserPublic.model_validate(user)
    return {"success": True, "user": public_user.model_dump()}


@router.get("/users", response_model=List[UserPublic])
def list_users(x_user: str | None = Header(default=None, alias="X-User")) -> List[UserPublic]:
    """List all users (owner only)."""
    _require_owner(x_user)
    return [UserPublic.model_validate(u) for u in load_users()]


@router.post("/users", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, x_user: str | None = Header(default=None, alias="X-User")) -> UserPublic:
    """Create a new user (owner only)."""
    _require_owner(x_user)
    users = load_users()
    if any(u.get("username") == payload.username for u in users):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists.")
    new_user = User(
        id=next_user_id(users),
        username=payload.username,
        password=payload.password,
        is_owner=False,
    )
    users.append(new_user.model_dump())
    save_users(users)
    return UserPublic.model_validate(new_user.model_dump())


@router.delete("/users/{username}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(username: str, x_user: str | None = Header(default=None, alias="X-User")) -> None:
    """Delete a non-owner user (owner only)."""
    _require_owner(x_user)
    users = load_users()
    user = next((u for u in users if u.get("username") == username), None)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    if user.get("is_owner"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete owner user.")
    users = [u for u in users if u.get("username") != username]
    save_users(users)

    # Also remove the user from project memberships if present
    from ..storage import load_projects, save_projects

    projects = load_projects()
    changed = False
    for proj in projects:
        members = proj.get("member_usernames", [])
        if username in members:
            proj["member_usernames"] = [m for m in members if m != username]
            changed = True
    if changed:
        save_projects(projects)
