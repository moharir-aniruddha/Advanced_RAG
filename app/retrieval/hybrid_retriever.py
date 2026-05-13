from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25RetrieverManager

from app.vectordb.vector_store import VectorStoreManager


class HybridRetriever:

    def __init__(self):

        self.vector_retriever = VectorRetriever()

        documents = VectorStoreManager.load_documents()

        self.bm25_retriever = BM25RetrieverManager.create(
            documents
        )

    def retrieve(self, query):

        vector_results = self.vector_retriever.retrieve(query)

        bm25_results = self.bm25_retriever.invoke(query)

        combined_results = vector_results + bm25_results

        unique_results = []

        seen_content = set()

        for doc in combined_results:

            if doc.page_content not in seen_content:

                unique_results.append(doc)

                seen_content.add(doc.page_content)

        return unique_results