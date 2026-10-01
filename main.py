import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = "Elon Musk"

    summary_template = """
        given the information {information} about a person, I want you to tell:
        1. A short summary of the person in 2-3 sentences.
        2. two interesting facts about the person.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # Ollama cloud API (free tier), authenticated with OLLAMA_API_KEY
    ollama_llm = ChatOllama(
        model="gpt-oss:120b",
        base_url="https://ollama.com",
        client_kwargs={"headers": {"Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"}},
        temperature=0,
    )
    chain = summary_prompt_template | ollama_llm

    try:
        ollama_response = chain.invoke({"information": information})
    except Exception as e:
        print("\n Ollama request failed")
        print(f"  Type    : {type(e).__name__}")
        print(f"  Status  : {getattr(e, 'status_code', 'n/a')}")
        print(f"  Message : {getattr(e, 'error', None) or str(e)}")
        print("  Hint    : check OLLAMA_API_KEY in .env and that the model name exists on ollama.com")
        return

    print("Ollama Response:")
    print(ollama_response.content)


if __name__ == "__main__":
    main()
