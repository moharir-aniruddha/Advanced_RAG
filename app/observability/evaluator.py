class RetrievalEvaluator:

    @staticmethod
    def evaluate_retrieval(
        query,
        rewritten_query,
        retrieved_docs
    ):

        evaluation = {
            "query": query,
            "rewritten_query": rewritten_query,
            "retrieved_chunks": len(retrieved_docs),
            "sources": list(set([
                doc.metadata.get(
                    "source",
                    "Unknown"
                )
                for doc in retrieved_docs
            ]))
        }

        return evaluation