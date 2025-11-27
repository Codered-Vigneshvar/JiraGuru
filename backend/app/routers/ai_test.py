"""Simple Gemini connectivity test endpoint."""

from __future__ import annotations

import logging

import google.generativeai as genai
from fastapi import APIRouter, HTTPException

from ..ai.story_gen import _configure_gemini, log_ai_event
from ..config import settings

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/test")
def ai_test() -> dict:
    """Call Gemini with a minimal prompt to verify connectivity."""
    prompt = "Return JSON: { 'status': 'ok' }"
    try:
        _configure_gemini()
    except HTTPException as exc:
        log_ai_event(f"[AI Test Error] {exc.detail}", level=logging.ERROR)
        raise
    try:
        log_ai_event(f"[AI Test] Request | {prompt}")
        model = genai.GenerativeModel(settings.gemini_model or "models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        raw = response.text or ""
        log_ai_event(f"[AI Test] Raw response | {raw}")
        return {"success": True, "raw_response": raw}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log_ai_event(f"[AI Test Error] {exc}", level=logging.ERROR)
        raise HTTPException(status_code=502, detail="Gemini test failed.") from exc
