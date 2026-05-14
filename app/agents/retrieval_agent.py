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
        rewritten_queries = []
        reranked_docs = []

        for attempt in range(MAX_RETRIEVAL_RETRIES):
            logger.info(f"Retrieval Attempt: {attempt + 1}")

            rewritten_queries = self.query_rewriter.rewrite(
                current_query,
                chat_history
            )

            logger.info(f"Agent Sub-Queries: {rewritten_queries}")

            all_retrieved_docs = []
            for sub_q in rewritten_queries:
                docs = self.retriever.retrieve(sub_q)
                all_retrieved_docs.extend(docs)

            unique_docs = []
            seen_content = set()
            for doc in all_retrieved_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)

            if not unique_docs:
                logger.info("No documents found in database.")
                # If the database is physically empty, stop retrying immediately
                return (", ".join(rewritten_queries), [])

            reranked_docs = self.reranker.rerank(
                query,
                unique_docs
            )

            if not reranked_docs:
                continue

            top_doc = reranked_docs[0]
            top_score = self.reranker.model.predict([
                (query, top_doc.page_content)
            ])[0]

            logger.info(f"Top Relevance Score: {top_score}")

            if top_score >= MIN_RELEVANCE_SCORE:
                logger.info("Retrieval successful.")
                return (
                    ", ".join(rewritten_queries),
                    reranked_docs
                )

            logger.info("Weak retrieval detected. Retrying...")
            current_query = f"{query} detailed explanation"

        # --- GUARDRAIL FIX ---
        # If we exit the loop without hitting MIN_RELEVANCE_SCORE,
        # return an empty list to signal the generator to block the LLM.
        logger.info("Max retries reached without relevant results. Returning empty list.")
        return (", ".join(rewritten_queries) if rewritten_queries else query, [])