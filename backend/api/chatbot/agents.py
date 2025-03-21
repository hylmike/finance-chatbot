from typing import TypedDict, Annotated
import operator
import os

from langchain_core.agents import AgentAction
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from chromadb import HttpClient, EmbeddingFunction
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langgraph.graph import StateGraph, END

from api.database.db import db_sync_engine
from .models import RoleType
from api.utils.logger import logger


TEXT_COLLECTION_NAME = "demo_text_collection"
SUMMARY_COLLECTION_NAME = "demo_summary_collection"
MAX_RETRIEVAL_RESULTS = 10


class AgentState(TypedDict):
    """Agent state used for chatbot graph, will maintain by agents"""

    question: str
    chat_history: list[str]
    inter_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


class QueryOutput(TypedDict):
    """Generated SQL query"""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01)


def get_chroma_client():
    """Utility function to get chroma DB client"""
    chroma_host = os.getenv("CHROMA_HOST", "chromadb")
    chroma_port = os.getenv("CHROMA_PORT", "8200")
    client = HttpClient(host=chroma_host, port=int(chroma_port))
    return client


def get_vector_store(collection_name: str) -> Chroma:
    vs_client = get_chroma_client()
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-large", dimensions=256
    )
    vector_store = Chroma(
        client=vs_client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    return vector_store


def get_multi_vector_retriever(
    vs_client: HttpClient, embedding_function: EmbeddingFunction
) -> MultiVectorRetriever:
    """Utility function to get multi_vector_retriever for images"""
    vector_store = Chroma(
        client=vs_client,
        collection_name=SUMMARY_COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    store = InMemoryStore()
    id_key = "image_id"

    return MultiVectorRetriever(
        vectorstore=vector_store,
        docstore=store,
        id_key=id_key,
    )


def write_query(question: str, db: SQLDatabase):
    """Generate SQL query to fetch information"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01)
    query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            # as we share DB with app for demo purpose, so here specify table name,
            # normally should use separate DB from app, then just use all tables - db.get_table_info()
            "table_info": db.get_table_info(["tax"]),
            "input": question,
        }
    )
    structure_llm = llm.with_structured_output(QueryOutput)
    response = structure_llm.invoke(prompt)

    return response["query"]


@tool("query_tax_data")
def query_tax_data(query: str):
    """Query question related data from database and return SQL query and result"""
    db = SQLDatabase(engine=db_sync_engine)

    # generate DB query based in question
    db_query = write_query(question=query, db=db)
    try:
        result = db.run(db_query)
    except Exception as e:
        logger.error(f"Failed to run {db_query} on DB: {str(e)}")
        return ""

    return f"SQL Query: {db_query}\nSQL Result: {str(result)}\n"


def query_translation(question: str) -> list[str]:
    """Use LLM to improve query content and get multiple alternative queries"""
    prompt_template = """You are AI assistant. You task is to generate five different \
versions of given question to retrieve relevant documents from a vector database. By \
generating multiple perspectives on the user question, your goal is to help the user \
overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines.
Original Question: {question}"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    try:
        queries = chain.invoke({"question": question})
    except Exception as e:
        logger.error(f"Failed to get alternative queries from LLM: {e}")
        return [question]

    return queries


def multi_queries_retriever(queries: list[str]) -> list[str]:
    vector_store = get_vector_store(collection_name=TEXT_COLLECTION_NAME)
    docs = []
    for query in queries:
        results = vector_store.similarity_search_with_score(query=query, k=3)
        for res, score in results:
            docs.append((score, res.page_content))
    docs.sort(key=lambda x: -x[0])
    final_results = set()
    for score, content in docs:
        if content not in final_results:
            final_results.add(content)
            if len(final_results) >= MAX_RETRIEVAL_RESULTS:
                break

    return list(final_results)


@tool("search_tax_code")
def search_tax_code(query: str):
    """Search question related information from given vector database and return it"""

    enhanced_queries = query_translation(query)
    retrieved_context = ""
    try:
        retrieved_results = multi_queries_retriever(enhanced_queries)
    except Exception as e:
        logger.error(
            f"Failed to retrieve results from vector database for query {enhanced_queries}: {str(e)}"
        )
        return retrieved_context

    for result in retrieved_results:
        retrieved_context += f"{result}\n"

    return retrieved_context


@tool("search_tax_data_from_images")
def search_tax_data_from_images(query: str):
    """Retrieve question related summaries from vector database, then feed summaries with related
    images as context to LLM to extract question related data from images"""
    client = get_chroma_client()
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-large", dimensions=256
    )
    multi_retriever = get_multi_vector_retriever(
        vs_client=client, embedding_function=embedding_function
    )
    images = multi_retriever.invoke(query)

    system_prompt = """You are a advisor tasked with providing information related to query. \
You will be given one or several images usually of charts or graphs. Extract query related \
information from image(s), if you can't find relevant information, just return ''."""

    human_messages = []
    for image in images:
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }
        human_messages.append(image_message)
    human_messages.append({"type": "text", "text": f"query: {query}"})

    messages = [
        (RoleType.SYSTEM, system_prompt),
        (RoleType.HUMAN, human_messages),
    ]
    try:
        response = llm.invoke(messages)
        logger.info(
            f"Successfully extract information from images {response.content}"
        )
    except Exception as e:
        logger.exception(f"Failed to retrieve data from image with LLM: {e}")

    return response.content


@tool("final_answer")
def final_answer(question: str, context: str):
    """Return a nature language response to the user, 
    based on original user question and aggregated context \
    from all tools outputs, use LLM to get final answer for question"""

    prompt_template = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. Keep the answer brief and concise.
Question: {question} 
Context: {context} 
Answer:
"""
    try:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = (
            {
                "question": RunnablePassthrough(),
                "context": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke({"question": question, "context": context})
    except Exception as e:
        logger.exception(f"Failed to get final answer with LLM: {e}")

    return answer


def create_scratchpad(inter_steps: list[AgentAction]):
    """Create scrptchpad based on inter_steps from AgentState"""
    analysis_steps = []
    for action in inter_steps:
        # If this is a tool execution
        if action.log != "TBD":
            analysis_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n-----\n".join(analysis_steps)


def router(state: AgentState):
    """return the tool name to use, if bad format got to final answer"""
    if isinstance(state["inter_steps"], list):
        return state["inter_steps"][-1].tool
    else:
        print("Router invalid format")
        return "final_answer"


def run_tool(state: AgentState):
    """Run tool node based on last state value"""
    tool_str_to_function = {
        "query_tax_data": query_tax_data,
        "search_tax_code": search_tax_code,
        "search_tax_data_from_images": search_tax_data_from_images,
        "final_answer": final_answer,
    }

    tool_name = state["inter_steps"][-1].tool
    tool_args = state["inter_steps"][-1].tool_input

    response = tool_str_to_function[tool_name].invoke(input=tool_args)
    action_output = AgentAction(
        tool=tool_name, tool_input=tool_args, log=str(response)
    )

    return {"inter_steps": [action_output]}


def build_rag_graph():
    """Build adapative RAG graph with all agent tools"""
    tools = [
        query_tax_data,
        search_tax_code,
        search_tax_data_from_images,
        final_answer,
    ]

    system_message = """You are the central processor, the great AI decision maker.
Given the user's query you must decide what to do with it based on the list of tools provided to you.

Do not use same tool (in the scratchpad) more than 2 times, also if you see that a tool has been used \
with a particular query, do NOT use that same tool with the same query again. 

You should aim to collect information from all tools if needed before providing \
the accurate answer to the user. Once you have collected enough information to \
answer the user's question (stored in the scratchpad), or you have used all tools \
searching all available data but still can not find relevent infomation, then use \
the final_answer tool to generate final answer.

Keep in mind, it is not reliable to use past information answer question related \
to future, for those situations it is better answer I don't know.
"""

    processor_prompt = ChatPromptTemplate.from_messages(
        [
            (RoleType.SYSTEM, system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            (RoleType.HUMAN, "{question}"),
            (RoleType.AI, "scratchpad: {scratchpad}"),
        ]
    )

    processor_chain = (
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "scratchpad": lambda x: create_scratchpad(
                inter_steps=x["inter_steps"]
            ),
        }
        | processor_prompt
        | llm.bind_tools(tools, tool_choice="any")
    )

    def run_processor(state: AgentState):
        """Run the process node, initialize state settings"""
        logger.info("run processor")
        logger.info(f"inter_steps: {state['inter_steps']}")

        response = processor_chain.invoke(state)
        tool_name = response.tool_calls[0]["name"]
        tool_args = response.tool_calls[0]["args"]
        action_output = AgentAction(
            tool=tool_name, tool_input=tool_args, log="TBD"
        )

        logger.info(f"Next tool: {tool_name}, tool input: {tool_args}")
        return {"inter_steps": [action_output]}

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("processor", run_processor)
    graph_builder.add_node("query_tax_data", run_tool)
    graph_builder.add_node("search_tax_code", run_tool)
    graph_builder.add_node("search_tax_data_from_images", run_tool)
    graph_builder.add_node("final_answer", run_tool)

    graph_builder.set_entry_point("processor")

    graph_builder.add_conditional_edges(source="processor", path=router)

    for tool_object in tools:
        if tool_object.name != "final_answer":
            graph_builder.add_edge(tool_object.name, "processor")

    graph_builder.add_edge("final_answer", END)

    graph = graph_builder.compile()

    return graph
