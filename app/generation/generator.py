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

    def __init__(self, llm):

        self.llm = llm

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

    def generate_response(self, query):

        start_time = time.time()

        logger.info(f"User Query: {query}")

        chat_history = self.memory.get_history()

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

        prompt = f"""
        You are a sophisticated AI assistant capable of cross-document synthesis.
        Use the provided context to answer the question. 

        Guidelines:
        1. If the information comes from different sources, compare and contrast them in your answer.
        2. Always mention which source (URL) the information comes from.
        3. If the sources contradict each other, highlight that contradiction.
        4. Use ONLY the provided context. If the answer isn't there, say you can't find it.
        
        CRITICAL INSTRUCTION: 
Your context contains information from MULTIPLE domains ({list(set(unique_sources))}). 
If the user asks a broad or comparative question, you MUST look for relevant 
details in every source provided. Do not rely on just one URL if others 
contain overlapping information.

        if answer is not found in the context just say "I don't know" 

        Conversation History:
        {chat_history}

        Context:
        {context}

        Question:
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

        self.memory.add_message(
            "User",
            query
        )

        self.memory.add_message(
            "Assistant",
            final_answer
        )

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