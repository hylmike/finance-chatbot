from fastapi import APIRouter, Depends

from .services import (
    gen_ai_completion,
    get_chat_history,
    gen_knowledgebase,
)
from .schemas import ChatCompletionRequest
from api.dependencies.db import DBSessionDep
from api.dependencies.auth import CurrentUserDep, valid_is_authenticated

chatbot_router = APIRouter()


@chatbot_router.post("/completion")
async def ai_completion(
    input: ChatCompletionRequest, db: DBSessionDep, user: CurrentUserDep
):
    """Generate AI completion for user question. combine info from chat history and knowledge base"""
    completion = await gen_ai_completion(db, user.id, input.question)
    return {"completion": completion}


@chatbot_router.get(
    "/chat-history", dependencies=[Depends(valid_is_authenticated)]
)
async def chat_history(db: DBSessionDep, user: CurrentUserDep):
    """Load chat history belong to current user"""
    chats = await get_chat_history(db, user.id)
    return {"chat_history": chats}


@chatbot_router.post(
    "/gen-knowledgebase", dependencies=[Depends(valid_is_authenticated)]
)
async def generate_knowledgebase(db: DBSessionDep):
    """Generate RAG knowledge base from input webs and docs, and save into vector database"""
    result = await gen_knowledgebase(db)
    return result
