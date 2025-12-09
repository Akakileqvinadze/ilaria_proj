from pydantic import BaseModel, Field
from typing import List, Dict


class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_history: List[Message] = Field(default=[])


class Source(BaseModel):
    id: int
    content: str
    section: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    metadata: Dict = {}


class HealthResponse(BaseModel):
    status: str
    service: str
    model: str
    vector_store_docs: int
