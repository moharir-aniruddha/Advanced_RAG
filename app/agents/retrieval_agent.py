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
        rewritten_queries = []  # Store the latest list of queries
        reranked_docs = []

        for attempt in range(MAX_RETRIEVAL_RETRIES):
            logger.info(f"Retrieval Attempt: {attempt + 1}")

            # 1. Get a list of sub-queries (e.g., ["AI definition", "LLM definition"])
            rewritten_queries = self.query_rewriter.rewrite(
                current_query,
                chat_history
            )

            logger.info(f"Agent Sub-Queries: {rewritten_queries}")

            all_retrieved_docs = []

            # 2. Retrieve documents for EACH sub-query to ensure source diversity
            for sub_q in rewritten_queries:
                docs = self.retriever.retrieve(sub_q)
                all_retrieved_docs.extend(docs)

            # 3. Deduplicate documents based on content
            unique_docs = []
            seen_content = set()
            for doc in all_retrieved_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)

            if not unique_docs:
                logger.info("No documents found in initial retrieval.")
                current_query = f"{query} detailed explanation"
                continue

            # 4. Rerank the combined set against the ORIGINAL query
            # This ensures the final ranking respects the user's main intent.
            reranked_docs = self.reranker.rerank(
                query,
                unique_docs
            )

            if not reranked_docs:
                logger.info("No reranked docs found.")
                continue

            # 5. Evaluate the quality of the top result
            top_doc = reranked_docs[0]
            top_score = self.reranker.model.predict([
                (query, top_doc.page_content)
            ])[0]

            logger.info(f"Top Relevance Score: {top_score}")

            if top_score >= MIN_RELEVANCE_SCORE:
                logger.info("Retrieval successful.")
                return (
                    ", ".join(rewritten_queries),  # Return joined string for logging
                    reranked_docs
                )

            logger.info("Weak retrieval detected. Retrying with expanded query...")
            current_query = f"{query} detailed explanation"

        logger.info("Max retries reached.")
        return (
            ", ".join(rewritten_queries) if rewritten_queries else query,
            reranked_docs
        )