from langchain_community.vectorstores import FAISS
from app.embeddings.embedding_model import get_embedding_model
from app.config import VECTOR_DB_PATH

class VectorStoreManager:

    @staticmethod
    def create_vector_store(documents):

        embedding_model = get_embedding_model()

        vectorstore = FAISS.from_documents(
            documents,
            embedding_model
        )

        vectorstore.save_local(VECTOR_DB_PATH)

        return vectorstore

    @staticmethod
    def load_vector_store():

        embedding_model = get_embedding_model()

        return FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

