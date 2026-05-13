from app.vectordb.vector_store import VectorStoreManager
from app.config import TOP_K_RESULTS

class Retriever:


    def __init__(self):

        self.vectorstore = VectorStoreManager.load_vector_store()

    def retrieve(self, query):

        results = self.vectorstore.similarity_search(
            query,
            k=TOP_K_RESULTS
        )

        return results

