import os

from chromadb import HttpClient
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore

from api.chatbot.doc_loaders import CSVLoader, PDFLoader, PPTLoader
from api.database.db import db_sync_engine
from api.utils.logger import logger
from .agents import (
    get_chroma_collection,
    build_rag_graph,
)
from api.database.db import AsyncSession
from .schemas import ChatRecord
from .models import (
    Chat as ChatModel,
    IngestedFile as IngestedFileModel,
    RoleType,
)
from api.utils.hash_file import get_file_hash

RETRIEVE_CHATS_NUM = 50
csv_file = {"file_url": "./data/tax_data.csv", "table_name": "tax"}
pdf_files = ["./data/i1040gi.pdf", "./data/usc26@118-78.pdf"]
ppt_file = "./data/MIC_3e_Ch11.pptx"


def load_csv_file(file: str):
    csv_loader = CSVLoader(db_engine=db_sync_engine)
    csv_loader.load(file_url=file["file_url"], table_name=file["table_name"])


async def load_pdf_files(file_urls: list[str], db: AsyncSession):
    collection = get_chroma_collection()
    pdf_loader = PDFLoader(collection)
    for file_url in file_urls:
        pdf_file_name = file_url.split("/")[-1]
        pdf_file_hash = get_file_hash(file_url)
        result = await IngestedFileModel.find_by_file_hash(
            db=db, file_hash=pdf_file_hash
        )
        if not result:
            pdf_loader.load(file_url)
            await IngestedFileModel.create(
                db=db, file_name=pdf_file_name, file_hash=pdf_file_hash
            )
        else:
            logger.info(f"Already ingested {file_url}, skip it")


def load_ppt_file(file_url: str, vs_client: HttpClient) -> MultiVectorRetriever:
    store = InMemoryStore()
    ppt_loader = PPTLoader(store=store, vs_client=vs_client)

    ppt_loader.load(file_url)


async def gen_knowledgebase(db: AsyncSession):
    """Ingest all raw data files, indexing and save them into DB or vector DB"""
    try:
        csv_file_name = csv_file["file_url"].split("/")[-1]
        csv_file_hash = get_file_hash(csv_file["file_url"])
        result = await IngestedFileModel.find_by_file_hash(
            db=db, file_hash=csv_file_hash
        )
        if not result:
            load_csv_file(csv_file)
            await IngestedFileModel.create(
                db=db, file_name=csv_file_name, file_hash=csv_file_hash
            )
        else:
            logger.info(f"Already ingested {csv_file_name}, skip it")

        await load_pdf_files(file_urls=pdf_files, db=db)
        chroma_host = os.getenv("CHROMA_HOST", "chromadb")
        chroma_port = os.getenv("CHROMA_PORT", "8200")
        chroma_client = HttpClient(host=chroma_host, port=int(chroma_port))

        ppt_file_name = ppt_file.split("/")[-1]
        ppt_file_hash = get_file_hash(ppt_file)
        result = await IngestedFileModel.find_by_file_hash(
            db=db, file_hash=ppt_file_hash
        )
        if not result:
            load_ppt_file(file_url=ppt_file, vs_client=chroma_client)
            await IngestedFileModel.create(
                db=db, file_name=ppt_file_name, file_hash=ppt_file_hash
            )
        else:
            logger.info(f"Already ingested {ppt_file_name}, skip it")
    except Exception as e:
        logger.error(f"Something wrong when ingesting files: {str(e)}")
        return {"status": "Failed", "error": str(e)}

    return {"status": "Success", "error": None}


async def gen_ai_completion(
    db: AsyncSession, user_id: int, question: str
) -> str:
    """Use RAG with agents to generate AI completion for given question"""
    graph = build_rag_graph()

    await ChatModel.create(
        db=db, user_id=user_id, role_type=RoleType.HUMAN, content=question
    )
    response = graph.invoke({"question": question, "chat_history": []})
    completion = response["inter_steps"][-1].log
    await ChatModel.create(
        db=db, user_id=user_id, role_type=RoleType.AI, content=completion
    )

    return completion


async def get_chat_history(db: AsyncSession, user_id: int) -> list[ChatRecord]:
    """Load chat history for given user"""
    chat_history = await ChatModel.find_by_userid(db, user_id)
    return chat_history
