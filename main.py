from dotenv import load_dotenv

load_dotenv() 
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches the query on the web and returns the results.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query)


llm = ChatOllama(
    model="llama3.2:3b",
)
tools = [TavilySearch()]
agent = create_agent(llm, tools)

def main():
    print("Hello from langchain-course!")
    theSearch = input("What do you want to search for? ")
    result = agent.invoke({"messages": HumanMessage(content=theSearch)})
    print(result)


if __name__ == "__main__":
    main()
