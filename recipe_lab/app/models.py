from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User recipe R&D query")
    session_id: str = Field(default="default", description="Client session id")


class ChatResponse(BaseModel):
    answer: str
    contexts: list[str]
    model: str
