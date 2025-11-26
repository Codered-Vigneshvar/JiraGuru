"""Placeholder story endpoints for future LangGraph/Gemini integration."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/stories", tags=["stories"])


@router.get("", response_model=dict)
def list_stories_placeholder() -> dict:
    """Temporary stub until AI-powered story generation is added."""
    return {"message": "Story generation will be added with LangGraph + Gemini soon."}
