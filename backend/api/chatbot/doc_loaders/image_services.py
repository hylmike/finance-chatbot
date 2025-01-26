import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from api.chatbot.models import RoleType
from api.utils.logger import logger


def encode_image(image_path: str) -> str:
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(image_base64: str, image_title: str) -> str:
    """Get image summary from LLM"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01)

    summary_prompt = """You are an assistant tasked with summarizing image for retrieval. \
The summary will be embedded and used to retrieve the raw image. Give a concise summary of \
the given image and image title that is optimized for retriveal."""

    response = llm.invoke(
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

    return response.content


# images: {"image_url": str, "related_info": str}[]
def gen_image_summaries(
    images: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    """Generate image summary and base64 encode string for given image urls"""
    image_base64_list = []
    image_summaries = []

    for image in images:
        image_url = image["image_url"]
        image_title = image["related_info"]
        image_base64 = encode_image(image_url)
        try:
            summary = image_summarize(image_base64, image_title)
            logger.info(f"Successfully generated summary for image {image_url}")
            image_base64_list.append(image_base64)
            image_summaries.append(summary)
            image_summary_file_name = (
                "./data/ppt_images/ppt_image_summaries.txt"
            )
            with open(image_summary_file_name, "a") as f:
                f.write(summary)
                f.write("\n-----------\n")
        except Exception as e:
            logger.exception(f"Failed to summarize image {image_url}: {e}")

    return image_base64_list, image_summaries
