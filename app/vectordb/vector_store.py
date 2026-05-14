import pickle
import os
import shutil
from langchain_community.vectorstores import FAISS
from app.embeddings.embedding_model import get_embedding_model
from app.config import VECTOR_DB_PATH

DOCUMENT_STORE_PATH = "vectorstore/documents.pkl"


class VectorStoreManager:
    @staticmethod
    def create_vector_store(new_chunks):
        embedding_model = get_embedding_model()

        # 1. Update/Merge the FAISS Index
        if os.path.exists(VECTOR_DB_PATH) and os.path.isdir(VECTOR_DB_PATH):
            print(f"Updating existing vector store at {VECTOR_DB_PATH}")
            vectorstore = FAISS.load_local(
                VECTOR_DB_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(new_chunks)
        else:
            print("Creating fresh vector store...")
            vectorstore = FAISS.from_documents(new_chunks, embedding_model)

        vectorstore.save_local(VECTOR_DB_PATH)

        # 2. Update/Merge the Documents List (for BM25)
        all_docs = []
        if os.path.exists(DOCUMENT_STORE_PATH):
            with open(DOCUMENT_STORE_PATH, "rb") as f:
                all_docs = pickle.load(f)

        all_docs.extend(new_chunks)

        with open(DOCUMENT_STORE_PATH, "wb") as f:
            pickle.dump(all_docs, f)

        return vectorstore

    @staticmethod
    def clear_database():
        """Helper to delete old data before a new test run."""
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
            os.makedirs("vectorstore")
            print("Database cleared successfully.")

    @staticmethod
    def load_vector_store():
        embedding_model = get_embedding_model()
        return FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

    @staticmethod
    def load_documents():
        if not os.path.exists(DOCUMENT_STORE_PATH): return []
        with open(DOCUMENT_STORE_PATH, "rb") as f: return pickle.load(f)