import base64
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from api.chatbot.models import RoleType
from api.utils.logger import logger


def encode_image(image_path: str) -> str:
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def image_summarize(image: dict[str, str]) -> str:
    """Get image summary from LLM"""
    image_url = image["image_url"]
    image_title = image["related_info"]
    image_base64 = encode_image(image_url)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01)

    summary_prompt = """You are an assistant tasked with summarizing image for retrieval. \
The summary will be embedded and used to retrieve the raw image. Give a concise summary of \
the given image and image title that is optimized for retriveal."""

    try:
        response = await llm.ainvoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": summary_prompt + f"\nImage title: {image_title}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ]
                )
            ]
        )
        summary = response.content
        logger.info(f"Successfully generated summary for image {image_url}")
    except Exception as e:
        logger.exception(f"Failed to summarize image {image_url}: {e}")
        return None, image_base64

    return summary, image_base64


# images: {"image_url": str, "related_info": str}[]
async def gen_image_summaries(
    images: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    """Generate image summary and base64 encode string for given image urls"""
    image_base64_list = []
    image_summaries = []
    image_summary_file_name = "./data/ppt_images/ppt_image_summaries.txt"
    
    try:
        summary_results = await asyncio.gather(*map(image_summarize, images))
        for summary, image_base64 in summary_results:
            if summary:
                image_base64_list.append(image_base64)
                image_summaries.append(summary)
                with open(image_summary_file_name, "a") as f:
                    f.write(summary)
                    f.write("\n-----------\n")
    except FileNotFoundError as e:
        logger.exception(f"Summary file {image_summary_file_name} not found: {e}")

    return image_base64_list, image_summaries
