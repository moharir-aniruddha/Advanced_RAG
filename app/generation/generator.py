from app.retrieval.retriever import Retriever

class ResponseGenerator:

    def __init__(self, llm):

        self.llm = llm
        self.retriever = Retriever()

    def generate_response(self, query):

        docs = self.retriever.retrieve(query)

        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        prompt = f"""
    ```
    
    You are a helpful AI assistant.
    
    Answer ONLY from the provided context.
    Say "I do not know the answer" if relevant answers are absent in the given context.
    Context:
    {context}
    
    Question:
    {query}
    """


        response = self.llm.invoke(prompt)

        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in docs
        ]))

        return {
            "answer": response,
            "sources": sources
        }

