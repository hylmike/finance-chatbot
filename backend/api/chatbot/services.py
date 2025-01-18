import os

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb import HttpClient, EmbeddingFunction
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

LAST_CHATS_NUM = 50
csv_file = {"file_url": "./data/tax_data.csv", "table_name": "tax"}
pdf_files = ["./data/i1040gi.pdf", "./data/usc26@118-78.pdf"]
ppt_file = "./data/MIC_3e_Ch11.pptx"


def load_csv_file(file: str):
    csv_loader = CSVLoader(db_engine=db_sync_engine)
    csv_loader.load(file_url=file["file_url"], table_name=file["table_name"])


def load_pdf_files(file_urls: list[str]):
    collection = get_chroma_collection()
    pdf_loader = PDFLoader(collection)
    for file in file_urls:
        pdf_loader.load(file)


def load_ppt_file(file_url: str, vs_client: HttpClient) -> MultiVectorRetriever:
    store = InMemoryStore()
    ppt_loader = PPTLoader(store=store, vs_client=vs_client)

    ppt_loader.load(file_url)


def gen_knowledgebase():
    try:
        load_csv_file(csv_file)

        chroma_host = os.getenv("CHROMA_HOST", "chromadb")
        chroma_port = os.getenv("CHROMA_PORT", "8200")
        chroma_client = HttpClient(host=chroma_host, port=int(chroma_port))
        load_pdf_files(file_urls=pdf_files)

        load_ppt_file(file_url=ppt_file, vs_client=chroma_client)
    except Exception as e:
        logger.error(f"Something wrong when ingesting files: {str(e)}")
        return {"status": "Failed", "error": str(e)}

    return {"status": "Success", "error": None}


def gen_ai_completion(db: AsyncSession, user_id: int, question: str) -> str:
    graph = build_rag_graph()

    response = graph.invoke({"question": question, "chat_history": []})
    logger.info(f"Final response:\n{response}")

    return response


async def get_chat_history(db: AsyncSession, user_id: int) -> list[ChatRecord]:
    return []
