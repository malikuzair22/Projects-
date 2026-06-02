from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")


loader = PyPDFLoader(r"C:\Users\Hp\Desktop\langchain_models\Rag_based_project\multimodel_rag\GPT-4(Technical Report).pdf")

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap= 100)

text_chunks = text_splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(text_chunks, embeddings)

retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {'k': 5, 'lambda_mult': 0.5}
)

@tool
def search_knowledge_base(query: str)->str:
    """Search stored knowledge base for questions about
    AI, GPT-4, machine learning, and technical research."""
    docs = retriever.invoke(query)
    return "\n\n" .join(doc.page_content for doc in docs)


@tool
def get_latest_news(query: str)->str:
    """Get latest real-time news articles about 
    current events, recent developments, or 
    anything happening in the world right now."""

    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize=5"

    response = requests.get(url)
    data = response.json()
    result = []
    for article in data.get("articles", []):
        title = article["title"]
        description = article.get("description", "No description")
        url = article.get("url", "")
        result.append(f"Title: {title}\nDescription: {description}\nURL: {url}\n")
    return "\n\n".join(result)


agent = create_react_agent(
    model = model,
    tools = [search_knowledge_base, get_latest_news],
    prompt="You are an intelligent assistant that has access to "
           "a knowledge base about AI research AND live news data. "
           "Use search_knowledge_base for technical AI questions. "
           "Use get_latest_news for current events and recent news. "
           "Combine both when needed."
)

result = agent.invoke({"messages": [("human", "What is GPT-4 and what are the latest AI news in 2025?")]})
print(result["messages"][-1].content)
