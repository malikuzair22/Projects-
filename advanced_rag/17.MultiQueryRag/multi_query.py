from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


model = ChatGroq(model="llama-3.3-70b-versatile")

loader = PyPDFLoader(r"C:\Users\Hp\Desktop\langchain_models\17.MultiQueryRag\GPT-4(Technical Report).pdf")

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap= 100
)

chunk_text = text_splitter.split_documents(document)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunk_text, embedding)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever= vector_store.as_retriever(),
    llm = model
)

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_question(input):
    if isinstance(input, list):
        return input[0]   # if it arrives as a list, take the first element
    return str(input)  

rag_chain = (
    RunnableParallel({
      "context" :   multi_query_retriever | RunnableLambda(format_docs),
      "question": RunnableLambda(extract_question)
    }) 
    | prompt
    | model
    | StrOutputParser()
)

question = "How does GPT-4 perform on the MMLU benchmark?"

try:
    result = rag_chain.invoke(question)
    print("\n── Answer ──────────────────────────")
    print(result)
    print("────────────────────────────────────")
except Exception as e:
    print(f"❌ Error: {e}")



