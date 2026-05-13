import pickle

from langchain_community.vectorstores import FAISS

from app.embeddings.embedding_model import get_embedding_model
from app.config import VECTOR_DB_PATH


DOCUMENT_STORE_PATH = "vectorstore/documents.pkl"


class VectorStoreManager:

    @staticmethod
    def create_vector_store(documents):

        embedding_model = get_embedding_model()

        vectorstore = FAISS.from_documents(
            documents,
            embedding_model
        )

        vectorstore.save_local(VECTOR_DB_PATH)

        with open(DOCUMENT_STORE_PATH, "wb") as f:
            pickle.dump(documents, f)

        return vectorstore

    @staticmethod
    def load_vector_store():

        embedding_model = get_embedding_model()

        return FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    @staticmethod
    def load_documents():

        with open(DOCUMENT_STORE_PATH, "rb") as f:
            return pickle.load(f)