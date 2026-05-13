from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranking.reranker import Reranker


class ResponseGenerator:

    def __init__(self, llm):

        self.llm = llm

        self.retriever = HybridRetriever()

        self.reranker = Reranker()

    def generate_response(self, query):

        retrieved_docs = self.retriever.retrieve(
            query
        )

        reranked_docs = self.reranker.rerank(
            query,
            retrieved_docs
        )

        context = "\n\n".join([
            doc.page_content
            for doc in reranked_docs
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
            for doc in reranked_docs
        ]))

        return {
            "answer": response.content,
            "sources": sources
        }