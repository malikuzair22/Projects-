import fitz
import base64
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

vision_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

model = ChatGroq(model="llama-3.3-70b-versatile")

all_chunks = []

docs = fitz.open(r"C:\Users\Hp\Desktop\langchain_models\Rag_based_project\multimodel_rag\GPT-4(Technical Report).pdf")

#print(docs[0].get_text())
#print("Total Pages:", docs.page_count)



for page_range in range(docs.page_count):
    page = docs[page_range]
    text = page.get_text()
    if text.strip():
        all_chunks.append(text.strip())
    images = page.get_images(full=True)
    for img in images:
        xref = img[0]
        baseimg = docs.extract_image(xref)
        image_bytes = baseimg["image"]
        ext = baseimg["ext"]
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            model_response = vision_model.invoke([{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe everything you see in this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{image_base64}"}
                    }
                ]
            }])
            all_chunks.append(model_response.content)
        except Exception as e:
            print(f"❌ Error processing image: {e}")
    


for page_range in range(docs.page_count):
    page = docs[page_range]
    tables = page.find_tables()
    for table in tables:
        data = table.extract()
        table_text = "\n".join(" | ".join(str(cell) if cell else "" for cell in row)
        for row in data
        )
        all_chunks.append(table_text)

#print(f"✅ Total chunks collected: {len(all_chunks)}")


documents = [Document(page_content= chunks) for chunks in all_chunks]
#print(f"✅ Total documents: {len(documents)}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

text_chunks = text_splitter.split_documents(documents)
#print(f"✅ Total chunks created: {len(text_chunks)}")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(text_chunks, embeddings)
print("✅ Vector store created successfully")


retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'lambda_mult': 0.5}
)

prompt = PromptTemplate(
    template=""" You are a multi model rag assistant.

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


chain  = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x[0] if isinstance(x, list) else str(x))
    })
    | prompt
    | model
    | StrOutputParser()
)

question = "How does GPT-4 perform on the MMLU benchmark?"

try:
    result = chain.invoke(question)
    print("\n── Answer ──────────────────────────")
    print(result)
    print("────────────────────────────────────")
except Exception as e:
    print(f"❌ Error: {e}")