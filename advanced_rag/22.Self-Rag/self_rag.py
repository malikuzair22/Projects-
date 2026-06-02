
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END


load_dotenv()


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


class selfrag(TypedDict):
    question: str
    documents: list
    generation: str
    should_retrieve: str
    is_relevant: str
    is_grounded: str



def retrieve_node(state: selfrag) -> selfrag:
    docs = retriever.invoke(state['question'])
    return {'documents': docs}

def decide_retrieval_node(state: selfrag) -> selfrag:
    prompt = f"""Should I search a document to answer this question?
    Reply with only 'Yes' or 'No'-
    Question: {state['question']}"""
    result = model.invoke(prompt)
    return {"should_retrieve": result.content.strip().lower()}

def grade_relevance_node(state: selfrag)-> selfrag:
    question = state['question']
    docs = state['documents']

    doc_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""Are these retrieved documents relevant to the question?
    Reply with yes or no.
    question: {question}
    Documents: {doc_text}"""

    result = model.invoke(prompt)
    return {"is_relevant": result.content.strip().lower()}


def generate_node(state: selfrag)-> selfrag:
    question = state['question']
    docs = state['documents']
    if docs:
        doc_text = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"""Answer the question using the context below.
        If not in context say 'I don't know'.
        Context: {doc_text}
        Question: {question}"""
    else:
        # No retrieval needed — answer from memory
        prompt = f"""Answer this question directly:
        Question: {question}"""
    result = model.invoke(prompt)
    return {'generation': result.content}

def grade_grounding_node(state: selfrag) -> selfrag:
    generation = state['generation']
    docs = state['documents']
    doc_text = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Is this Answer supported by the context
    Reply with only 'yes' or 'no'
    context: {doc_text}
    Answer: {generation}"""
    result = model.invoke(prompt)
    return {"is_grounded": result.content.strip().lower()}

def route_retrieval(state: selfrag)-> selfrag:
    if state["should_retrieve"]=="yes":
        return "retrieve_node"
    else:
        return "generate_node"

def route_relevance(state: selfrag)-> selfrag:
    if state["is_relevant"] == "yes":
        return "generate_node"
    else:
        return "retrieve_node"
    
def route_grounding(state: selfrag)-> selfrag:
    if state ["is_grounded"] == "yes":
        return END
    else:
        return "generate_node"



builder = StateGraph(selfrag)
builder.add_node("retrieve_node", retrieve_node)
builder.add_node("decide_retrieval_node", decide_retrieval_node)
builder.add_node("grade_relevance_node",grade_relevance_node)
builder.add_node("generate_node", generate_node)
builder.add_node("grade_grounding_node", grade_grounding_node)


builder.set_entry_point("decide_retrieval_node")

builder.add_conditional_edges("decide_retrieval_node", route_retrieval)
builder.add_conditional_edges("grade_relevance_node",route_relevance)
builder.add_conditional_edges("grade_grounding_node", route_grounding)

builder.add_edge("retrieve_node", "grade_relevance_node")
builder.add_edge("generate_node", "grade_grounding_node")

graph = builder.compile()

result = graph.invoke({
    "question": "What is 2+2?",
    "documents": [],
    "generation": "",
    "should_retrieve": "",
    "is_relevant": "",
    "is_grounded": ""
})

print(result["generation"])