from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,  RunnableLambda
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

# ── 1. MODEL ──────────────────────────────────────────────────────────────────
model = ChatGroq(model="llama-3.3-70b-versatile")

# ── 2. LOAD PDF ───────────────────────────────────────────────────────────────
loader = PyPDFLoader(r"C:\Users\Hp\Desktop\langchain_models\19.Hyde_Rag\GPT-4(Technical Report).pdf")
documents = loader.load()
print(f"✅ Total pages loaded: {len(documents)}")

# ── 3. FILTER OUT BOILERPLATE PARAGRAPHS ─────────────────────────────────────
BOILERPLATE = "Technology continues to evolve rapidly across industries."

filtered_documents = [
    doc for doc in documents
    if BOILERPLATE not in doc.page_content[:200]
]
print(f"✅ Pages after filtering boilerplate: {len(filtered_documents)}")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents([Document(page_content="init")],embeddings)

docsstore = InMemoryStore()

# ── 4. CHUNK THE DOCUMENT ─────────────────────────────────────────────────────
Child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


Parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000
)

retriever = ParentDocumentRetriever(
    vectorstore= vector_store,
    docstore= docsstore,
    child_splitter= Child_splitter,
    parent_splitter= Parent_splitter
)

retriever.add_documents(filtered_documents)

hyde_prompt = PromptTemplate(
    template= """Generate a short 2-3 Sentences hypothetical answer
    for this question. Be consice and stay on topic.
    Question: {question}
    Hypothetical Answer:""",
    input_variables= ["question"]
)


# ── 8. PROMPT ─────────────────────────────────────────────────────────────────
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided context.
If the answer is not in context, say: "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)



# ── 9. HELPER FUNCTIONS ───────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def hyde_retrieve(question: str):
    fake_answer = (hyde_prompt | model | StrOutputParser()).invoke(
        {"question": question}
    )
    print(f"\n🔮 Fake Answer:\n{fake_answer}\n")
    docs = retriever.invoke(fake_answer)  # ✅ uses Parent Document Retriever!
    return docs


def extract_question(input):
    if isinstance(input, list):
        return input[0]   # if it arrives as a list, take the first element
    return str(input)     # otherwise just ensure it's a string

# ── 10. RAG CHAIN ─────────────────────────────────────────────────────────────
rag_chain = (
    RunnableParallel({
        "context": RunnableLambda(hyde_retrieve) | RunnableLambda(format_docs),
        "question": RunnableLambda(extract_question)   
    })
    | prompt
    | model
    | StrOutputParser()
)

# ── 11. DEBUG: CHECK WHAT RETRIEVER FETCHES ───────────────────────────────────
question = "How does GPT-4 perform on the MMLU benchmark?"

print(f"\n🔍 Retrieving chunks for: '{question}'")
test_docs = retriever.invoke(question)
for i, doc in enumerate(test_docs):
    print(f"\n── Chunk {i+1} ──")
    print(doc.page_content[:300])

# ── 12. RUN THE CHAIN ─────────────────────────────────────────────────────────
print("\n⏳ Running RAG chain...\n")
try:
    result = rag_chain.invoke(question)
    print("── Answer ──────────────────────────────────────")
    print(result)
    print("────────────────────────────────────────────────\n")
except Exception as e:
    print(f"\n❌ Something went wrong: {e}\n")