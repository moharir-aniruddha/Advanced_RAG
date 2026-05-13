from app.retrieval.hybrid_retriever import HybridRetriever


class ResponseGenerator:

    def __init__(self, llm):

        self.llm = llm

        self.retriever = HybridRetriever()

    def generate_response(self, query):

        docs = self.retriever.retrieve(query)

        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in context,
say:
"I could not find relevant information."

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
            "answer": response.content,
            "sources": sources
        }