from sentence_transformers import CrossEncoder

from app.config import (
    RERANKER_MODEL,
    FINAL_TOP_K
)


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            RERANKER_MODEL
        )

    def rerank(self, query, documents):

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        scored_docs = list(
            zip(documents, scores)
        )

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        reranked_docs = [
            doc
            for doc, score in scored_docs[:FINAL_TOP_K]
        ]

        for idx, (doc, score) in enumerate(scored_docs[:FINAL_TOP_K]):
            print(f"\nRANK {idx + 1}")
            print(f"SCORE: {score}")
            print(doc.page_content[:300])

        return reranked_docs