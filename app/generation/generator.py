from app.retrieval.hybrid_retriever import HybridRetriever

from app.reranking.reranker import Reranker

from app.query_transform.query_rewriter import QueryRewriter


class ResponseGenerator:

    def __init__(self, llm):

        self.llm = llm

        self.retriever = HybridRetriever()

        self.reranker = Reranker()

        self.query_rewriter = QueryRewriter(
            llm
        )

    def generate_response(self, query):

        rewritten_query = self.query_rewriter.rewrite(
            query
        )

        print("\nRewritten Query:\n")
        print(rewritten_query)

        retrieved_docs = self.retriever.retrieve(
            rewritten_query
        )

        reranked_docs = self.reranker.rerank(
            rewritten_query,
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