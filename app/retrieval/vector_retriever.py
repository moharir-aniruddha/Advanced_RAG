from app.vectordb.vector_store import VectorStoreManager
from app.config import VECTOR_TOP_K


class VectorRetriever:

    def __init__(self):

        self.vectorstore = VectorStoreManager.load_vector_store()

    def retrieve(self, query):

        return self.vectorstore.similarity_search(
            query,
            k=VECTOR_TOP_K
        )