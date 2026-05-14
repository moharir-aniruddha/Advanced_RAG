from app.retrieval.vector_retriever import VectorRetriever
from langchain_community.retrievers import BM25Retriever
from app.vectordb.vector_store import VectorStoreManager
from app.observability.logger import logger


class HybridRetriever:
    def __init__(self):
        # 1. Initialize the defensive VectorRetriever
        self.vector_retriever = VectorRetriever()

        # 2. Try to load documents for BM25
        docs = VectorStoreManager.load_documents()

        if docs:
            # BM25 requires at least one document to initialize
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = 5
            logger.info("HybridRetriever: BM25 initialized with existing documents.")
        else:
            self.bm25_retriever = None
            logger.info("HybridRetriever: No documents found for BM25. Setting to None.")

    def retrieve(self, query):
        """Combines results from Vector and BM25 search."""
        # Get vector results (will be [] if DB is empty)
        vector_results = self.vector_retriever.retrieve(query)

        # Get BM25 results if available
        bm25_results = []
        if self.bm25_retriever:
            try:
                bm25_results = self.bm25_retriever.invoke(query)
            except Exception as e:
                logger.error(f"HybridRetriever: BM25 search failed: {e}")

        # Combine and deduplicate
        all_results = vector_results + bm25_results

        # Simple deduplication based on page content
        unique_results = []
        seen_content = set()
        for doc in all_results:
            if doc.page_content not in seen_content:
                unique_results.append(doc)
                seen_content.add(doc.page_content)

        return unique_results