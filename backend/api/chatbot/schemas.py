"""Chatbog schemas and customized types usedby chatbot services"""

from datetime import datetime

from pydantic import BaseModel


class ChatRecord(BaseModel):
    id: int
    role_type: str
    content: str
    created: datetime


class ChatCompletionRequest(BaseModel):
    question: str
