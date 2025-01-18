import base64

from langchain_openai import ChatOpenAI
from api.chatbot.models import RoleType
from api.utils.logger import logger


def encode_image(image_path: str) -> str:
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def image_summarize(image_base64: str) -> str:
    """Get image summary from LLM"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01)

    summary_prompt = """You are an assistant tasked with summarizing image for retrieval. \
The summary will be embedded and used to retrieve the raw image. Give a concise summary of \
the image that is optimized for retriveal."""
    message = [
        (RoleType.SYSTEM, summary_prompt),
        (
            RoleType.HUMAN,
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ),
    ]

    response = await llm.ainvoke(message)

    return response.content


async def gen_image_summaries(
    image_urls: list[str],
) -> tuple[list[str], list[str]]:
    """Generate image summary and base64 encode string for given image urls"""
    image_base64_list = []
    image_summaries = []

    for image_url in image_urls:
        image_base64 = encode_image(image_url)
        try:
            summary = await image_summaries(image_base64)
        except Exception as e:
            logger.exception(f"Failed to summarize image {image_url}: {e}")
        logger.info(f"Successfully generated summary for image {image_url}")
        image_base64_list.append(image_base64)
        image_summaries.append(summary)

    return image_base64_list, image_summaries
