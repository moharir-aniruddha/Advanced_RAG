import time

from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

from app.reranking.reranker import (
    Reranker
)

from app.query_transform.query_rewriter import (
    QueryRewriter
)

from app.memory.conversation_memory import (
    ConversationMemory
)

from app.agents.retrieval_agent import (
    RetrievalAgent
)

from app.observability.logger import (
    logger
)

from app.observability.evaluator import (
    RetrievalEvaluator
)


class ResponseGenerator:

    def __init__(self, llm, memory):

        self.llm = llm

        self.memory = memory

        self.retriever = HybridRetriever()

        self.reranker = Reranker()

        self.query_rewriter = QueryRewriter(
            llm
        )

        self.memory = ConversationMemory()

        self.retrieval_agent = RetrievalAgent(
            self.retriever,
            self.reranker,
            self.query_rewriter
        )

    def generate_response(self, query, session_id="default"):

        start_time = time.time()

        logger.info(f"User Query: {query}")

        chat_history = self.memory.get_history(session_id)

        (
            rewritten_query,
            reranked_docs
        ) = (
            self.retrieval_agent
            .run_agentic_retrieval(
                query,
                chat_history
            )
        )

        logger.info(
            f"Final Agent Query: "
            f"{rewritten_query}"
        )

        evaluation = (
            RetrievalEvaluator
            .evaluate_retrieval(
                query,
                rewritten_query,
                reranked_docs
            )
        )

        logger.info(
            f"Retrieval Evaluation: "
            f"{evaluation}"
        )
        unique_sources = list(set([
            doc.metadata.get("source", "Unknown source")
            for doc in reranked_docs
        ]))

        context_parts = []
        for doc in reranked_docs:
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"SOURCE: {source}\nCONTENT: {doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        logger.info(
            f"Final Context Length: "
            f"{len(context)}"
        )

        # 4. Professional Synthesis Prompt
        prompt = f"""
        You are a professional Corporate AI Assistant. Your goal is to provide a clean, 
        well-structured, and authoritative response based on the provided context.

        CRITICAL INSTRUCTIONS:
        1. **Response Style**: Write in a clear, professional prose or use bullet points where appropriate. 
        2. **No Inline Citations**: DO NOT include URLs, bracketed sources, or "(Source: ...)" tags 
           within the body of your response. The main text should be completely clean.
        3. **Synthesis**: Compare and combine information from different provided sources naturally.
        
        4. If the answer is not present in the context just say "I don't know the answer.". Strictly Do not stretch it much

        Conversation History:
        {chat_history}

        Context from Multiple Sources:
        {context}

        User Question:
        {query}
        """

        print("\nStreaming Response:\n")

        response = self.llm.stream(prompt)

        final_answer = ""

        for chunk in response:

            content = chunk.content

            print(content, end="", flush=True)

            final_answer += content

        print()

        end_time = time.time()

        latency = round(
            end_time - start_time,
            2
        )

        logger.info(
            f"Response Latency: "
            f"{latency} seconds"
        )

        logger.info(
            f"Final Answer: "
            f"{final_answer}"
        )

        try:
            self.memory.add_message(session_id, "User", query)
            self.memory.add_message(session_id, "Assistant", final_answer)
        except Exception as e:
            print(f"\nMemory Error: {e}")

        sources = list(set([
            doc.metadata.get(
                "source",
                "Unknown"
            )
            for doc in reranked_docs
        ]))

        logger.info(
            f"Sources: {sources}"
        )

        return {
            "answer": final_answer,
            "sources": sources
        }