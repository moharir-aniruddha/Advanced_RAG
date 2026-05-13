from app.retrieval.hybrid_retriever import HybridRetriever

from app.reranking.reranker import Reranker

from app.query_transform.query_rewriter import QueryRewriter

from app.memory.conversation_memory import (
    ConversationMemory
)


class ResponseGenerator:

    def __init__(self, llm):

        self.llm = llm

        self.retriever = HybridRetriever()

        self.reranker = Reranker()

        self.query_rewriter = QueryRewriter(
            llm
        )

        self.memory = ConversationMemory()

    def generate_response(self, query):

        chat_history = self.memory.get_history()

        rewritten_query = self.query_rewriter.rewrite(
            query,
            chat_history
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

Use the provided context to answer the question.

If the answer is not available in context,
say:
"I could not find relevant information."

Conversation History:
{chat_history}

Context:
{context}

Question:
{query}
"""

        print("\nStreaming Response:\n")

        response = self.llm.stream(prompt)

        final_answer = ""

        for chunk in response:

            content = chunk.content

            print(content, end="", flush=True)

            final_answer += content

        print()

        self.memory.add_message(
            "User",
            query
        )

        self.memory.add_message(
            "Assistant",
            final_answer
        )

        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in reranked_docs
        ]))

        return {
            "answer": final_answer,
            "sources": sources
        }