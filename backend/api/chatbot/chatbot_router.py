from fastapi import APIRouter, Depends

from .services import (
    gen_ai_completion,
    get_chat_history,
    gen_knowledgebase,
)
from .schemas import ChatCompletionRequest
from api.dependencies.db import DBSessionDep
from api.dependencies.auth import CurrentUserDep, validate_auth

chatbot_router = APIRouter()


@chatbot_router.post("/completion")
async def ai_completion(
    input: ChatCompletionRequest, db: DBSessionDep, user: CurrentUserDep
):
    """Generate AI completion for user question. combine info from chat history and knowledge base"""
    completion = await gen_ai_completion(db, user.id, input.question)
    return {"completion": completion}


@chatbot_router.get("/chat-history", dependencies=[Depends(validate_auth)])
async def chat_history(db: DBSessionDep, user: CurrentUserDep):
    """Load chat history belong to current user"""
    chats = await get_chat_history(db, user.id)
    return {"chat_history": chats}


@chatbot_router.post(
    "/gen-knowledgebase", dependencies=[Depends(validate_auth)]
)
async def generate_knowledgebase():
    """Generate RAG knowledge base from input webs and docs, and save into vector database"""
    await gen_knowledgebase()
