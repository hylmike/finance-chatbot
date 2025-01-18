from pptx import Presentation
from pptx.shapes.base import BaseShape
from pptx.enum.shapes import MSO_SHAPE_TYPE
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from chromadb import HttpClient
from langchain_openai import OpenAIEmbeddings

from api.utils.logger import logger
from api.utils.id_generator import gen_document_id
from .image_services import gen_image_summaries
from api.chatbot.agents import TEXT_COLLECTION_NAME, SUMMARY_COLLECTION_NAME

IMAGE_URL_PREFIX = "./data/ppt_images"


class PPTLoader:
    def __init__(self, store: InMemoryStore, vs_client: HttpClient):
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
        self.text_vector_store = Chroma(
            client=vs_client,
            collection_name=TEXT_COLLECTION_NAME,
            embedding_function=embedding_function,
        )
        self.summary_vector_store = Chroma(
            client=vs_client,
            collection_name=SUMMARY_COLLECTION_NAME,
            embedding_function=embedding_function,
        )
        self.ppt_url = None

    def extract_shape_content(
        self,
        shape: BaseShape,
        slide_index: int,
        slide_texts: list[str],
        extracted_images: list[str],
    ):
        if hasattr(shape, "text"):
            slide_texts.append(shape.text)
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for s in shape.shapes:
                self.extract_shape_content(
                    shape=s,
                    slide_index=slide_index,
                    lide_texts=slide_texts,
                    extracted_images=extracted_images,
                )
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            for picture in shape:
                image = picture.image
                # ---get image "file" contents---
                image_bytes = image.blob
                # ---assign a name for the file, e.g. 'image.jpg'---
                ppt_file_name = self.ppt_url.split("/")[-1].lower()
                image_index = len(extracted_images) + 1
                image_filename = f"{IMAGE_URL_PREFIX}/{ppt_file_name}_slide{slide_index}_image{image_index}.{image.ext}"
                with open(image_filename, "wb") as f:
                    f.write(image_bytes)
                extracted_images.append(image_filename)
                logger.info(
                    f"Extracted image {image_filename} from ppt file {self.ppt_url}"
                )

    def extract_contents(self, file_url: str) -> tuple[list[str], list[str]]:
        self.ppt_url = file_url
        ppt = Presentation(file_url)
        extracted_texts = []
        extracted_images = []

        try:
            for index, slide in enumerate(ppt.slides):
                slide_texts = []
                for shape in slide.shapes:
                    self.extract_shape_content(
                        shape=shape,
                        slide_index=index,
                        slide_texts=slide_texts,
                        extracted_images=extracted_images,
                    )
                # Combine texts in one slide into one text, as normally texts in one slide are revelent
                # later use this as chunk put into vector database
                extracted_texts.append(" ".join(slide_texts))
        except Exception as e:
            logger.exception(
                f"Failed to extract content from ppt file {file_url}: {e}"
            )

        return extracted_texts, extracted_images

    async def load(self, file_url: str) -> MultiVectorRetriever:
        extracted_texts, extracted_images = self.extract_contents(file_url)
        # Embedding all extracted texts, one record per slide
        documents = [
            Document(page_content=text, metadata={"source": "ppt"})
            for text in extracted_texts
        ]
        ids = [gen_document_id() for _ in range(len(documents))]
        self.text_vector_store.add_documents(documents=documents, ids=ids)

        # Create multi vector retriever to save image summary embeddings and raw image files
        image_base64_list, image_summaries = await gen_image_summaries(
            extracted_images
        )
        id_key = "image_id"
        multi_retriever = MultiVectorRetriever(
            vectorstore=self.summary_vector_store,
            docstore=self.store,
            id_key=id_key,
        )
        image_ids = [gen_document_id() for _ in range(len(extracted_images))]
        summary_docs = [
            Document(page_content=summary, metadata={id_key: image_ids[index]})
            for index, summary in enumerate(image_summaries)
        ]
        multi_retriever.vectorstore.add_documents(summary_docs)
        multi_retriever.docstore.mset(list(zip(image_ids, image_base64_list)))

        return multi_retriever
