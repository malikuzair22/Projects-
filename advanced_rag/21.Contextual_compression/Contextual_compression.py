from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,  RunnableLambda
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from dotenv import load_dotenv

load_dotenv()

# ── 1. MODEL ──────────────────────────────────────────────────────────────────
model = ChatGroq(model="llama-3.3-70b-versatile")

# ── 2. LOAD PDF ───────────────────────────────────────────────────────────────
loader = PyPDFLoader(r"C:\Users\Hp\Desktop\langchain_models\21.Contextual_compression\GPT-4(Technical Report).pdf")
documents = loader.load()
print(f"✅ Total pages loaded: {len(documents)}")

# ── 3. FILTER OUT BOILERPLATE PARAGRAPHS ─────────────────────────────────────
# FIX 2: Your PDF has repetitive filler text that pollutes retrieval results.
# We remove any page whose content starts with the known boilerplate phrase.
BOILERPLATE = "Technology continues to evolve rapidly across industries."

filtered_documents = [
    doc for doc in documents
    if BOILERPLATE not in doc.page_content[:200]
]
print(f"✅ Pages after filtering boilerplate: {len(filtered_documents)}")

# ── 4. CHUNK THE DOCUMENT ─────────────────────────────────────────────────────
# Smaller chunks = more focused = better retrieval accuracy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
text_chunks = text_splitter.split_documents(filtered_documents)
print(f"✅ Total chunks created: {len(text_chunks)}")

# ── 5. EMBEDDINGS ─────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ── 6. VECTOR STORE ───────────────────────────────────────────────────────────
# FAISS handles embedding internally — do NOT call embed_documents() manually
vector_store = FAISS.from_documents(text_chunks, embeddings)
print("✅ Vector store created successfully")

# ── 7. RETRIEVER ──────────────────────────────────────────────────────────────
retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 10, 'lambda_mult': 0.5}
)

compressor = LLMChainExtractor.from_llm(llm=model)

compression_retriever = ContextualCompressionRetriever(
    base_compressor= compressor,
    base_retriever= retriever
)

# ── 8. PROMPT ─────────────────────────────────────────────────────────────────
prompt = PromptTemplate(
    template="""
You are a helpful assistant analyzing AI research.

Answer ONLY from the provided context.
Note: Some context may be in table format with numbers.
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

def extract_question(input):
    if isinstance(input, list):
        return input[0]   # if it arrives as a list, take the first element
    return str(input)     # otherwise just ensure it's a string

# ── 10. RAG CHAIN ─────────────────────────────────────────────────────────────
rag_chain = (
    RunnableParallel({
        "context": compression_retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(extract_question)   
    })
    | prompt
    | model
    | StrOutputParser()
)

# ── 11. DEBUG: CHECK WHAT RETRIEVER FETCHES ───────────────────────────────────
# Keep this during development so you can verify chunk quality anytime
question = "How does GPT-4 perform on the MMLU benchmark?"

print(f"\n🔍 Retrieving chunks for: '{question}'")
test_docs = compression_retriever.invoke(question)
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