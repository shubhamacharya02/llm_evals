from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq

from src.settings import settings
from src.prompts import RAG_PROMPT_TEMPLATE
from src.document_process import retrieve_relevant_chunks

_LLM_CLIENT = None


def get_llm():
    """
    Initializes and returns the configured Chat LLM instance (cached singleton).
    Defaults to Groq with LLaMA 3.3 70B, with fallback to OpenAI if configured.
    """
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT

    if settings.GROQ_API_KEY:
        _LLM_CLIENT = ChatGroq(
            model=settings.MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        return _LLM_CLIENT

    if settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        _LLM_CLIENT = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        return _LLM_CLIENT

    # Fallback to local Groq client without explicit key if GROQ_API_KEY is in env
    _LLM_CLIENT = ChatGroq(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS
    )
    return _LLM_CLIENT


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved chunks into a clean, numbered context block.
    """
    if not retrieved_chunks:
        return "No relevant policy documents found in the database."

    formatted_sections = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        content = chunk.get("content", "").strip()
        metadata = chunk.get("metadata", {})
        page = metadata.get("page")
        page_info = f" (Page {page + 1})" if page is not None else ""
        formatted_sections.append(f"--- [Excerpt {idx}{page_info}] ---\n{content}")

    return "\n\n".join(formatted_sections)


def generate_answer(
    query: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    End-to-end RAG generation pipeline:
    1. Retrieves top-K chunks from vector database based on settings.TOP_K (if not provided).
    2. Formats retrieved chunks into grounded context.
    3. Invokes LLM with strict enterprise HR policy prompt.
    4. Returns answer, sources, and metadata.
    """
    if retrieved_chunks is None:
        retrieved_chunks = retrieve_relevant_chunks(query=query)

    context_str = format_context(retrieved_chunks)
    llm = get_llm()
    chain = RAG_PROMPT_TEMPLATE | llm

    response = chain.invoke({
        "context": context_str,
        "question": query
    })
    answer_text = response.content if hasattr(response, "content") else str(response)

    # Extract unique page sources
    sources = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        page = meta.get("page")
        if page is not None:
            source_label = f"Page {page + 1}"
            if source_label not in sources:
                sources.append(source_label)

    return {
        "query": query,
        "answer": answer_text,
        "model": settings.MODEL_NAME,
        "retrieved_chunks_count": len(retrieved_chunks),
        "sources": sources,
        "retrieved_chunks": retrieved_chunks
    }
