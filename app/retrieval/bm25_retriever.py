from langchain_community.retrievers import BM25Retriever

from app.config import BM25_TOP_K


class BM25RetrieverManager:

    @staticmethod
    def create(documents):

        retriever = BM25Retriever.from_documents(
            documents
        )

        retriever.k = BM25_TOP_K

        return retriever