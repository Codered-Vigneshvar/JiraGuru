"""Unit tests for impacted tickets prompt formatting."""

from app.ai.impacted_tickets import _ticket_block, _build_prompt
from app.models import Project


def test_ticket_block_includes_all_human_fields_no_embeddings():
    tickets = [
        {
            "id": "PRJ-1",
            "title": "Build upload page",
            "description": "Implement resume upload UI",
            "status": "TODO",
            "epic": "PRJ-EP1",
            "acceptanceCriteria": ["User can upload PDF", "Show success state"],
        },
        {
            "id": "PRJ-2",
            "title": "Process PDF",
            "description": "Extract text and structure",
            "status": "IN_PROGRESS",
            "epic": "PRJ-EP1",
            "acceptanceCriteria": "Text extracted",
        },
    ]
    block = _ticket_block(tickets)
    assert "PRJ-1" in block and "PRJ-2" in block
    assert "embedding" not in block.lower()
    assert "Acceptance Criteria" in block


def test_prompt_uses_all_tickets_block_and_no_embeddings():
    project = Project(
        id="1",
        code="PRJ",
        title="Test",
        description="Desc",
        owner_username="u",
        member_usernames=[],
        documents=[],
    )
    tickets_block = _ticket_block(
        [
            {"id": "PRJ-1", "title": "One", "description": "D1", "status": "TODO", "epic": "EP", "acceptanceCriteria": ""},
            {"id": "PRJ-2", "title": "Two", "description": "D2", "status": "TODO", "epic": "EP", "acceptanceCriteria": ""},
        ]
    )
    prompt = _build_prompt(project, "old", "new", tickets_block, "summary", "delta")
    assert "PRJ-1" in prompt and "PRJ-2" in prompt
    assert "embedding" not in prompt.lower()
    assert "TICKETS:" in prompt
