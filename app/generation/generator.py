import time
from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranking.reranker import Reranker
from app.query_transform.query_rewriter import QueryRewriter
from app.memory.conversation_memory import ConversationMemory
from app.agents.retrieval_agent import RetrievalAgent
from app.observability.logger import logger
from app.observability.evaluator import RetrievalEvaluator


class ResponseGenerator:

    def __init__(self, llm, memory):
        self.llm = llm
        # FIX: Ensure we use the shared memory object, NOT a new one.
        self.memory = memory

        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.query_rewriter = QueryRewriter(llm)

        self.retrieval_agent = RetrievalAgent(
            self.retriever,
            self.reranker,
            self.query_rewriter
        )

    def generate_response(self, query, session_id="default"):
        start_time = time.time()
        logger.info(f"User Query: {query}")

        chat_history = self.memory.get_history(session_id)

        (rewritten_query, reranked_docs) = self.retrieval_agent.run_agentic_retrieval(
            query,
            chat_history
        )

        # --- THE GUARDRAIL GUARD ---
        if not reranked_docs:
            logger.warning("No relevant info found. Guardrail triggered.")

            final_answer = "I'm sorry, but my knowledge base is currently empty or contains no relevant information to answer that question. Please ingest some relevant URLs to get started!"

            # Save the rejection to memory so the bot knows it couldn't answer
            self.memory.add_message(session_id, "User", query)
            self.memory.add_message(session_id, "Assistant", final_answer)

            return {
                "answer": final_answer,
                "sources": []
            }
        # ---------------------------

        # Proceed normally if documents are found
        context_parts = []
        for doc in reranked_docs:
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"SOURCE: {source}\nCONTENT: {doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
        You are a professional Corporate AI Assistant. Use ONLY the provided context.
        If the answer is not present, say "I don't know." Do not use your own knowledge.

        Conversation History:
        {chat_history}

        Context:
        {context}

        User Question:
        {query}
        """

        # Non-streaming invocation as requested
        response = self.llm.invoke(prompt)
        final_answer = response.content

        # Update memory
        self.memory.add_message(session_id, "User", query)
        self.memory.add_message(session_id, "Assistant", final_answer)

        sources = list(set([doc.metadata.get("source", "Unknown") for doc in reranked_docs]))

        return {
            "answer": final_answer,
            "sources": sources
        }