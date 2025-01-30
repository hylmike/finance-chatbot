"""Loader to load pdf contents and save into vector database"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders.parsers import LLMImageBlobParser
# from langchain_openai import ChatOpenAI

from api.utils.id_generator import gen_document_id
from api.utils.logger import logger
from api.chatbot.agents import get_vector_store


class PDFLoader:
    """PDF file loader, load, chunking, indexing and save it into vector store"""

    def __init__(self, collection_name: str):
        self.vector_store = get_vector_store(collection_name)

    async def load(self, file_url: str) -> bool:
        try:
            loader = PyMuPDFLoader(
                file_path=file_url,
                mode="page",
                # images_inner_format="markdown-img",
                # images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o-mini")),
                extract_tables="markdown",
            )
            raw_docs = loader.load()
            logger.info(
                f"Loaded {len(raw_docs)} pages from PDF document {file_url}"
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=50, separators=["\n", "."]
            )
            docs = text_splitter.split_documents(raw_docs)

            await self.vector_store.aadd_documents(
                documents=docs,
                ids=[gen_document_id() for _ in range(len(docs))],
            )
            logger.info(f"Successfully load PDF file {file_url} into vector DB")
            return True
        except Exception as e:
            logger.exception(
                f"Failed to load PDF file {file_url} into vector DB: {e}"
            )
            return False

    async def load_files(self, files: list[str]):
        for file_url in files:
            await self.load(file_url)
