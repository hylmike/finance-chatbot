"""Loader to load pdf contents and save into vector database"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Collection
from api.utils.id_generator import gen_document_id
from api.utils.logger import logger


class PDFLoader:
    def __init__(self, vs_collection: Collection):
        self.collection = vs_collection

    async def load(self, file_url: str):
        try:
            loader = PyMuPDFLoader(file_url)
            raw_docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=50, separators=["\n", ".", " "]
            )
            docs = text_splitter.split_documents(raw_docs)

            self.collection.add(
                documents=[doc.page_content for doc in docs],
                ids=[gen_document_id() for _ in range(len(docs))],
            )
            logger.info(f"Successfully load PDF file {file_url} into vector DB")
        except Exception as e:
            logger.exception(
                f"Failed to load PDF file {file_url} into vector DB: {e}"
            )

    async def load_files(self, files: list[str]):
        for file_url in files:
            self.load(file_url)
