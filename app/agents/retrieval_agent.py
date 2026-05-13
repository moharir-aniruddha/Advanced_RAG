from app.config import (
    MAX_RETRIEVAL_RETRIES,
    MIN_RELEVANCE_SCORE
)

from app.observability.logger import (
    logger
)


class RetrievalAgent:

    def __init__(
        self,
        retriever,
        reranker,
        query_rewriter
    ):

        self.retriever = retriever

        self.reranker = reranker

        self.query_rewriter = query_rewriter

    def run_agentic_retrieval(
        self,
        query,
        chat_history
    ):

        current_query = query

        for attempt in range(
            MAX_RETRIEVAL_RETRIES
        ):

            logger.info(
                f"Retrieval Attempt: "
                f"{attempt + 1}"
            )

            rewritten_query = (
                self.query_rewriter.rewrite(
                    current_query,
                    chat_history
                )
            )

            logger.info(
                f"Agent Query: "
                f"{rewritten_query}"
            )

            retrieved_docs = (
                self.retriever.retrieve(
                    rewritten_query
                )
            )

            reranked_docs = (
                self.reranker.rerank(
                    rewritten_query,
                    retrieved_docs
                )
            )

            if not reranked_docs:

                logger.info(
                    "No reranked docs found."
                )

                continue

            top_doc = reranked_docs[0]

            top_score = (
                self.reranker.model.predict([
                    (
                        rewritten_query,
                        top_doc.page_content
                    )
                ])[0]
            )

            logger.info(
                f"Top Relevance Score: "
                f"{top_score}"
            )

            if top_score >= MIN_RELEVANCE_SCORE:

                logger.info(
                    "Retrieval successful."
                )

                return (
                    rewritten_query,
                    reranked_docs
                )

            logger.info(
                "Weak retrieval detected. "
                "Retrying..."
            )

            current_query = (
                f"{query} detailed explanation"
            )

        logger.info(
            "Max retries reached."
        )

        return (
            rewritten_query,
            reranked_docs
        )