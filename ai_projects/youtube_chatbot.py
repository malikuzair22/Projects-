from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# 1. GET TRANSCRIPT
# -----------------------------
video_id = "Gfr50f6ZBvo"

try:
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    text = " ".join(chunk.text for chunk in transcript)

except TranscriptsDisabled:
    raise Exception("No captions available for this video")


# -----------------------------
# 2. SPLIT TEXT
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.create_documents([text])


# -----------------------------
# 3. EMBEDDINGS + VECTOR DB
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(documents, embedding)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# -----------------------------
# 4. LLM
# -----------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")


# -----------------------------
# 5. PROMPT
# -----------------------------
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


# -----------------------------
# 6. FORMAT FUNCTION
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------
# 7. RAG CHAIN
# -----------------------------
rag_chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)


# -----------------------------
# 8. ASK QUESTION
# -----------------------------
question = "Is the topic of aliens discussed in this video?"

result = rag_chain.invoke(question)

print(result)