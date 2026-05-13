from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY


def get_llm():

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm