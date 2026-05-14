from app.vectordb.vector_store import VectorStoreManager
from app.config import VECTOR_TOP_K
from app.observability.logger import logger

class VectorRetriever:

    def __init__(self):
        # This now returns None instead of crashing if the disk is empty
        # because of the defensive update we made to VectorStoreManager
        self.vectorstore = VectorStoreManager.load_vector_store()

    def retrieve(self, query):
        """Safely searches the vector store or returns an empty list if no data exists."""
        if self.vectorstore is None:
            logger.info("VectorRetriever: No vector store found. Returning empty results.")
            return []

        try:
            return self.vectorstore.similarity_search(
                query,
                k=VECTOR_TOP_K
            )
        except Exception as e:
            logger.error(f"VectorRetriever Error during search: {e}")
            return []