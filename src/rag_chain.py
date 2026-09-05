"""
Conversational RAG chain.

This module connects retrieval, conversation history, the prompt template,
and the OpenAI chat model:

    User Question
        + Conversation History
            → Retriever → Relevant Chunks
            → RAG Prompt → OpenAI → Answer + Sources

Memory (conversation history) helps with follow-up questions.
Retrieval (document chunks) provides factual information from your files.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.schemas import RAGResponse, SourceCitation
from src.vectorstore import format_docs, get_retriever, load_vectorstore

load_dotenv()

DEFAULT_PROMPT_PATH = Path("prompts/rag_prompt.txt")
DEFAULT_CHAT_MODEL = "gpt-4o-mini"


def load_rag_prompt(prompt_path: str | Path = DEFAULT_PROMPT_PATH) -> PromptTemplate:
    """
    Load the RAG prompt text from disk and turn it into a PromptTemplate.

    Keeping the prompt in a .txt file makes it easy for beginners to edit
    without changing Python code.

    Args:
        prompt_path (str | Path): Path to the prompt file.

    Returns:
        PromptTemplate: Prompt with context, chat_history, and question variables.

    Raises:
        FileNotFoundError: If the prompt file is missing.

    Example:
        prompt = load_rag_prompt()
        print(prompt.input_variables)
    """
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"RAG prompt file not found: {path}")

    template_text = path.read_text(encoding="utf-8")
    return PromptTemplate(
        template=template_text,
        input_variables=["context", "chat_history", "question"],
    )


def get_llm() -> ChatOpenAI:
    """
    Create the OpenAI chat model using values from .env.

    Returns:
        ChatOpenAI: Configured OpenAI chat client.

    Raises:
        ValueError: If OPENAI_API_KEY is missing.
        RuntimeError: If the LLM client cannot be created.

    Example:
        llm = get_llm()
        print(llm.model_name)
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your .env file before chatting."
        )

    model_name = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    try:
        # temperature=0 keeps answers more focused for educational demos.
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            temperature=0,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not create the OpenAI chat client. "
            "Check OPENAI_API_KEY and OPENAI_CHAT_MODEL in your .env file."
        ) from exc


def get_llm_display_name() -> str:
    """
    Return a short LLM label for the Streamlit status panel.

    Returns:
        str: Display string such as 'OpenAI / gpt-4o-mini'.

    Example:
        print(get_llm_display_name())
    """
    model_name = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    return f"OpenAI / {model_name}"


def format_chat_history(messages: list[dict]) -> str:
    """
    Convert Streamlit-style chat messages into plain text for the prompt.

    Conversation history is separate from retrieved document context:
    - history = what the user and assistant already said
    - context = facts retrieved from the knowledge base

    Args:
        messages (list[dict]): Chat messages with 'role' and 'content' keys.

    Returns:
        str: Readable conversation history, or a placeholder if empty.

    Example:
        text = format_chat_history(
            [{"role": "user", "content": "What is the refund period?"}]
        )
    """
    if not messages:
        return "No previous conversation."

    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")

    return "\n".join(lines) if lines else "No previous conversation."


def extract_sources(docs: list[Document]) -> list[SourceCitation]:
    """
    Build unique source citations from retrieved document metadata.

    Args:
        docs (list[Document]): Retrieved chunks from the retriever.

    Returns:
        list[SourceCitation]: Deduplicated citations for the UI.

    Example:
        sources = extract_sources(retrieved_docs)
        for source in sources:
            print(source.filename, source.page)
    """
    citations: list[SourceCitation] = []
    seen: set[tuple[str, int | None]] = set()

    for doc in docs:
        filename = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (str(filename), page)
        if key in seen:
            continue
        seen.add(key)

        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."

        citations.append(
            SourceCitation(
                filename=str(filename),
                page=page,
                preview=preview,
            )
        )

    return citations


def ask_question(
    question: str,
    chat_history: list[dict] | None = None,
    k: int | None = None,
) -> RAGResponse:
    """
    Run one conversational RAG turn and return an answer with sources.

    Steps:
        1. Load the FAISS knowledge base
        2. Retrieve the top-k relevant chunks
        3. Combine conversation history + retrieved context in the prompt
        4. Call the OpenAI chat model
        5. Return the answer and source citations

    Args:
        question (str): The user's current question.
        chat_history (list[dict] | None): Previous chat messages for memory.
        k (int | None): Optional override for how many chunks to retrieve.

    Returns:
        RAGResponse: Answer text, source citations, and chunk count.

    Raises:
        FileNotFoundError: If the knowledge base has not been built yet.
        ValueError: If the API key is missing or the question is empty.
        RuntimeError: If retrieval or generation fails.

    Example:
        response = ask_question(
            "What about international customers?",
            chat_history=[
                {"role": "user", "content": "What is the refund policy?"},
                {"role": "assistant", "content": "The refund period is 30 days."},
            ],
        )
        print(response.answer)
    """
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        raise ValueError("Please enter a question.")

    history = chat_history or []

    # Load knowledge base and retrieve relevant chunks for this question.
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, k=k)

    try:
        retrieved_docs = retriever.invoke(cleaned_question)
    except Exception as exc:
        raise RuntimeError(
            "Retrieval failed. Try rebuilding the knowledge base."
        ) from exc

    context = format_docs(retrieved_docs)
    history_text = format_chat_history(history)
    prompt = load_rag_prompt()
    llm = get_llm()

    # Fill the prompt with:
    # - retrieved document context (facts)
    # - conversation history (memory for follow-ups)
    # - the current user question
    final_prompt = prompt.format(
        context=context,
        chat_history=history_text,
        question=cleaned_question,
    )

    try:
        result = llm.invoke(final_prompt)
        answer = result.content if hasattr(result, "content") else str(result)
    except Exception as exc:
        message = str(exc).lower()
        if "api key" in message or "authentication" in message or "unauthorized" in message:
            raise ValueError(
                "OpenAI authentication failed. Check OPENAI_API_KEY in your .env file."
            ) from exc
        if "rate limit" in message:
            raise RuntimeError(
                "OpenAI rate limit reached. Please wait a moment and try again."
            ) from exc
        raise RuntimeError(
            "The LLM failed to generate an answer. "
            "Check your OpenAI API key, model name, and internet connection."
        ) from exc

    return RAGResponse(
        answer=str(answer).strip(),
        sources=extract_sources(retrieved_docs),
        num_chunks=len(retrieved_docs),
    )
