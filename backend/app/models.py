"""Pydantic models for the JiraGuru backend MVP."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# User models ----------------------------------------------------------------
class User(BaseModel):
    id: str
    username: str
    password: str
    is_owner: bool = False


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class UserLogin(BaseModel):
    username: str
    password: str


class UserPublic(BaseModel):
    id: str
    username: str
    is_owner: bool = False


# Project models --------------------------------------------------------------
class DocumentMeta(BaseModel):
    id: str
    filename: str
    original_name: str
    content_type: str


class Project(BaseModel):
    id: str
    code: str
    title: str
    description: str
    owner_username: str
    member_usernames: List[str] = Field(default_factory=list)
    documents: List[DocumentMeta] = Field(default_factory=list)
    epics: List["Epic"] = Field(default_factory=list)
    stories: List["Story"] = Field(default_factory=list)
    tickets: List["Ticket"] = Field(default_factory=list)
    requirements_plan: Optional[str] = None
    requirements_plan_updated_at: Optional[str] = None


class ProjectCreate(BaseModel):
    code: str
    title: str
    description: str
    member_usernames: List[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    member_usernames: Optional[List[str]] = None


# Epic/Story models --------------------------------------------------------
class Epic(BaseModel):
    id: str
    title: str
    description: str


class Story(BaseModel):
    id: str
    epic_id: str
    title: str
    description: str
    acceptance_criteria: List[str] = Field(default_factory=list)


# Ticket models --------------------------------------------------------------
class Ticket(BaseModel):
    id: str
    key: str
    project_id: str
    type: Literal["EPIC", "STORY"]
    title: str
    description: str
    status: Literal["TODO", "IN_PROGRESS", "DONE"] = "TODO"
    assignee_username: Optional[str] = None
    epic_key: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    blockers: Optional[str] = None
    linked_document_ids: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    story_points: Optional[int] = None
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TicketUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[Literal["TODO", "IN_PROGRESS", "DONE"]] = None
    assignee_username: Optional[str] = None
    epic_key: Optional[str] = None
    dependencies: Optional[List[str]] = None
    blockers: Optional[str] = None
    linked_document_ids: Optional[List[str]] = None
    acceptance_criteria: Optional[List[str]] = None
    story_points: Optional[int] = None
    priority: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = None


class TicketComment(BaseModel):
    id: str
    ticket_key: str
    author_username: str
    text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RequirementPlan(BaseModel):
    project_id: str
    content: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Forward references
Project.model_rebuild()
