from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

load_dotenv()

web_search = TavilySearch()

model = ChatGroq(model="llama-3.3-70b-versatile")


loader = PyPDFLoader(r"C:\Users\Hp\Desktop\langchain_models\21.Contextual_compression\GPT-4(Technical Report).pdf")

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap= 200
)

text_chunks = text_splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(text_chunks, embeddings)
print("✅ Vector store created successfully")

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs= {'k':5, 'lambda_mult': 0.5}
)

class CorrectiveRagState(TypedDict):
    question : str
    documents : list
    generation : str
    is_relevant : str

def retrieve_node(state: CorrectiveRagState) -> CorrectiveRagState:
    documents = retriever.invoke(state['question'])
    return {'documents': documents}

def decide_relevance_node(state: CorrectiveRagState) -> CorrectiveRagState:
    question = state['question']
    docs = state['documents']

    doc_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""Are these retrieved documents relevant to the question?
    Reply with yes or no.
    question: {question}
    Documents: {doc_text}"""

    result = model.invoke(prompt)
    return {"is_relevant": result.content.strip().lower()}

def web_search_node(state: CorrectiveRagState) -> CorrectiveRagState:
    search_results = web_search.invoke(state['question'])

    print(f"Tavily result type: {type(search_results)}")
    print(f"Tavily result: {search_results}")

    docs = [Document(page_content= r['content']) for r in search_results["results"]]
    return {"documents": docs}
 
def generate_answer_node(state: CorrectiveRagState) -> CorrectiveRagState:  
    question = state['question']
    docs = state['documents']

    doc_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""Generate a concise answer based ONLY on the document. 
    If the document doesn't contain the answer, say "I don't know".
    Question : {question}
    documents: {doc_text}"""
    result = model.invoke(prompt)
    return {"generation": result.content.strip()}

def route_relevance(state: CorrectiveRagState) -> str:
    if state["is_relevant"] == "yes":
        return "generate_answer_node"
    else:
        return "web_search_node"


builder = StateGraph(CorrectiveRagState)

builder.add_node("retrieve_node", retrieve_node)
builder.add_node("decide_relevance_node",decide_relevance_node)
builder.add_node("web_search_node",web_search_node)
builder.add_node("generate_answer_node",generate_answer_node)

builder.set_entry_point("retrieve_node")

# retrieve → grade
builder.add_edge("retrieve_node", "decide_relevance_node")

# grade → conditional (yes=generate, no=web search)
builder.add_conditional_edges("decide_relevance_node", route_relevance)

# web search → generate
builder.add_edge("web_search_node", "generate_answer_node")

# generate → END
builder.add_edge("generate_answer_node", END)

graph = builder.compile()


result = graph.invoke({
    "question": "How does GPT-4 perform on MMLU?",
    "documents": [],
    "generation": "",
    "is_relevant": ""
})
print(result["generation"])



