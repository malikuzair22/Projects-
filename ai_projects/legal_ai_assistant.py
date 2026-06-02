from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import json

from dotenv import load_dotenv
load_dotenv()

# ── 1. MODEL ──────────────────────────────────────────────────────────────────
model = ChatGroq(model="llama-3.3-70b-versatile")


# ── 2. LOAD PDF ───────────────────────────────────────────────────────────────
loader = PyPDFLoader(r"C:\Users\Hp\Desktop\langchain_models\Rag_based_project\TechCorp_ServiceAgreement_Contract.pdf")

documents = loader.load()

print(f"✅ Total pages loaded: {len(documents)}")

contract_text = "\n\n".join([doc.page_content for doc in documents])

prompt_template = PromptTemplate(
    template="""
You are a legal analyst.
Read this contract and return ONLY a valid JSON object with these fields:
- parties (list of names)
- payment_terms (string)
- termination_clause (string)
- risks (list of strings)

Return ONLY raw JSON. No explanation. No markdown. No backticks.

Contract:
{contract_text}
""",
    input_variables=["contract_text"]
)

response = model.invoke(prompt_template)
print("✅ Raw model response:")


try:
    parsed = json.loads(response.content)
    print("\n✅ Parsed JSON output:")
    for key, value in parsed.items():
        print(f"\n  [{key}]")
        if isinstance(value, list):
            for item in value:
                print(f"    • {item}")
        else:
            print(f"    {value}")
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing failed: {e}")
    print("Raw model output was:")
    print(response.content)







