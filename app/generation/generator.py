import time

from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

from app.reranking.reranker import (
    Reranker
)

from app.query_transform.query_rewriter import (
    QueryRewriter
)

from app.memory.conversation_memory import (
    ConversationMemory
)

from app.observability.logger import (
    logger
)

from app.observability.evaluator import (
    RetrievalEvaluator
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

        start_time = time.time()

        logger.info(f"User Query: {query}")

        chat_history = self.memory.get_history()

        rewritten_query = self.query_rewriter.rewrite(
            query,
            chat_history
        )

        logger.info(
            f"Rewritten Query: "
            f"{rewritten_query}"
        )

        print("\nRewritten Query:\n")
        print(rewritten_query)

        retrieved_docs = self.retriever.retrieve(
            rewritten_query
        )

        evaluation = (
            RetrievalEvaluator.evaluate_retrieval(
                query,
                rewritten_query,
                retrieved_docs
            )
        )

        logger.info(
            f"Retrieval Evaluation: "
            f"{evaluation}"
        )

        reranked_docs = self.reranker.rerank(
            rewritten_query,
            retrieved_docs
        )

        context = "\n\n".join([
            doc.page_content
            for doc in reranked_docs
        ])

        logger.info(
            f"Final Context Length: "
            f"{len(context)}"
        )

        prompt = f"""
You are a helpful AI assistant.

Use the provided context to answer
the question.

If the answer is not available,
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

        end_time = time.time()

        latency = round(
            end_time - start_time,
            2
        )

        logger.info(
            f"Response Latency: "
            f"{latency} seconds"
        )

        logger.info(
            f"Final Answer: "
            f"{final_answer}"
        )

        self.memory.add_message(
            "User",
            query
        )

        self.memory.add_message(
            "Assistant",
            final_answer
        )

        sources = list(set([
            doc.metadata.get(
                "source",
                "Unknown"
            )
            for doc in reranked_docs
        ]))

        logger.info(
            f"Sources: {sources}"
        )

        return {
            "answer": final_answer,
            "sources": sources
        }