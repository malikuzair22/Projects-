from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent


from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

travly = TavilySearch()
@tool
def read_url(url: str) -> str:
    """Read content from a URL"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    clean_text = soup.get_text()
    return clean_text[:3000]

agent = create_react_agent(
    model=llm,
    tools=[read_url, travly],
    prompt="You are a professional research assistant. When asked to research a topic, provide a comprehensive response that includes:\n"
        "## Summary\n"
        "## Key Developments\n"
        "## Impact on Industries\n"
        "## Sources\n"
)


result = agent.invoke({"messages": [("human", "Research the latest developments in Artificial Intelligence in 2025")]})

print(result["messages"][-1].content)

