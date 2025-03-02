# Finance Chatbot Guide

**Table of Contents**

- [Finance Chatbot Guide](#finance-chatbot-guide)
  - [Overview and major functions](#overview-and-major-functions)
    - [Backend functions](#backend-functions)
      - [Adaptive RAG solution graph](#adaptive-rag-solution-graph)
    - [Frontend functions](#frontend-functions)
  - [Backend setup](#backend-setup)
  - [Frontend setup](#frontend-setup)
  - [Testing](#testing)
  - [Used dataset](#used-dataset)
  - [Other Reference](#other-reference)

## Overview and major functions
The chatbot is powered by generative AI and RAG (Retrieved Augment Generation).
- Chatbot can based on user question, search around and retrieve accurate information from different data sources (SQL database, vector DB, LLM etc) which are set up early with provided dataset, then use those contexts with LLM to formulate a final answer to user.
- LLM I use OpenAI `GPT-4o-mini` which is high efficiency and fast multi-modal large language model, data embedding also use same. 
- Backend web framework I use `FastAPI` which is a very popular and high performance Python web framework built on AsyncIO and OpenAPI.
- Frontend built a simple UI with React, to let user ingest all the raw data (indexing and put them in SQL DB or vector DB) and interact with chatbot.
- Generative AI framework use Langchain, which is very popular and powerful framework which can efficiently develop generative AI features.
- RAG part we also use LangGragh which is very powerful tool to support complicated agentic work flow and adaptive RAG.
- Use Postgres as SQL DB and Chroma as vector data in app to save different kinds of data, I use docker build environment with these tools for app.
- App includes simple user management and authentication,
- Simple frontend UI to let user ingest all the raw data (indexing and put them in SQL DB or vector DB) and interact with chatbot

### Backend functions
Backend all APIs are built with Python and `FastAPI`, I also heavily use `Langchain` and `LangGraph` tools in chatbot related modules. I include docker-compose file to easily build and launch the backend locally with docker.

Following are major functions in backend:
- Simple auth based on OAuth2.0 and JWT
- Simple user management, including add, get users etc. Chat management do need user information, this is the reason for including user management.
- Different types of document ingestion, including
  - PDF file loader based on PyMuPdf, load PDF file, chunk, indexing and save into vector database (ChromaDB)
  - CSV file loader based on Pandas, load CSV file and import it into SQL database (Postgres)
  - PPT file loader based on Python-PPTX, load PowerPoint file, extract all text and images, 
    - for text chunking based on slide, indexing and save into vector database (chromaDB), 
    - for images, we use multi-vector-retriever technology, for each image first get summary of image, and then indexing summary and save into vector database (Chroma, here use separate colllection), at same time encode image and save into object store and link with summary embedding. With this, later we can use summary match question to retrieve raw image, and feed them to LLM to get context, then we can search information from all images in PPT. 
- [Adaptive RAG graph](#adaptive-rag-solution-graph) with 4 agent / tool nodes, this is core part of this chatbot solution. It can dynamically route query to different agents to collect enough context from all data source (DB table, vector collections, images and LLM), and then get best answer with these context and LLM.
- For RAG retrieval, for input question I use query translation technique (get 3 relevant queries and retrieve top 5 of all similar contents) to improve answer accuracy
- Use same answer if same question was asked recently, for better user experience. Currently use exact match, can use max edit distance or regex search to get better question search performance
- Most services, especially database operation and web communication parts, use fully async way for better performance.
- Use file hash to record all ingested documents in DB, this can avoid duplicated work in file ingestion.
- Use docker to set up and management backend services, including API service, easy for testing and deployment.
- use small dimensions (256) instead of high default value (3072) for text indexing to improve embedding efficiency, also keep similar MTEB score (62 vs 64.6)
- Add image title as additional info feed to LLM to improve image summary in PPT file, this eventually can improve query accuracy
- Optimize file loaders code to improve performance, especially for images LLM summary part, now add concurrent processing and saved 80% time compare to before

#### Adaptive RAG solution graph
![Solution Graph](solution_graph.png)

### Frontend functions
Simple UI build based on React and Material UI, to let user ingest all the raw data (indexing and put them in SQL DB or vector DB) and interact with chatbot.

Following are major functions in Frontend:
- Auth support, all users need login to use app. Also button user can log out
- User can click `GENERATE KNOWLEDGE BASE` button to ingest all the documents given in app requirements. Ingestion may take few minutes, once done will show alert
- UI for user to chat with chatbot, also show all chat history for same user

## Backend setup
As backend use docker to set up, need docker installed before running it. 

1. After you clone repository to local. Go to backend folder to set up .env file with contents in env.example file. You need openai api key for this app.

2. To start backend, just run following command in app root folder (/finance-chatbot):
```
npm run dev:backend
```
3. Then it will load all services in docker and start backend api service. Once started, you use following url in browser to view all API
```
http://localhost:3100/docs
```

## Frontend setup
After you clone repository to local. 

1. Go to frontend folder to set up .env file with contents in env.example file. Just one line to set up backend api url config.

2. To start frontend, just run following command in app root folder (/finance-chatbot):
```
npm run dev:frontend
```

3. Once it is done, you can use following url in browser to view all API
```
http://localhost:4000
```
As we already auto create an admin user in backend for testing, so you can use following username and password to login
- username: admin
- password: 54321

4. You need first setup knowledge base to finish ingestion of all data files. For this just click `GENERATE KNOWLEDGE BASE` button in top left corner. Normally it will take few minutes to finish everything, then you can chat with chatbot for any question.

## Testing
Please refer to another [test question and answer](test%20questions%20and%20answers.txt) file in this repo for test samples and example answers during test, you can use them to test chatbot. Notice wording of the answers may be slightly different each time due to non-deterministic nature of LLM, but should cover the answer for question.

## Used dataset
- Tabular dataset: Financial information (CSV format)
- Unstructured financial reports: (PDF format, some are very big 7000+ pages)
- Financial presentations: (PPT format, with text, images and table inside)
All dataset files are also included in this repo (/backend/data), for easily testing and verification.

## Other Reference
[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/html/2401.18059v1)
Will introduce RAPTOR for complex documents indexing and retrieval in next version



