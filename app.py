"""
Local RAG Knowledge Assistant — Streamlit UI

Run with:
    streamlit run app.py

This app shows the full educational RAG pipeline:

    Documents → Load → Split → Embed → FAISS → Retriever → Prompt → OpenAI → Answer + Sources
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.embeddings import get_embedding_model_name
from src.loaders import SUPPORTED_EXTENSIONS, count_source_files
from src.rag_chain import ask_question, get_llm_display_name
from src.vectorstore import (
    build_vectorstore,
    clear_vectorstore,
    get_chunk_count,
    rebuild_vectorstore,
    vectorstore_exists,
)

load_dotenv()

DOCUMENTS_DIR = Path("data/documents")
VECTORSTORE_DIR = Path("data/vectorstore")


def ensure_folders() -> None:
    """
    Create the documents and vectorstore folders if they do not exist yet.

    Returns:
        None

    Example:
        ensure_folders()
    """
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(uploaded_files) -> list[str]:
    """
    Save uploaded PDF/DOCX files into the local documents folder.

    Args:
        uploaded_files: Streamlit UploadedFile objects from the file uploader.

    Returns:
        list[str]: Names of files that were saved successfully.

    Raises:
        ValueError: If an unsupported file type is uploaded.

    Example:
        saved = save_uploaded_files(uploaded_files)
        print(saved)
    """
    saved_names: list[str] = []

    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {uploaded.name}. "
                "Only PDF and DOCX files are supported."
            )

        destination = DOCUMENTS_DIR / uploaded.name
        destination.write_bytes(uploaded.getbuffer())
        saved_names.append(uploaded.name)

    return saved_names


def init_session_state() -> None:
    """
    Initialize Streamlit session values used by the chat UI.

    Returns:
        None

    Example:
        init_session_state()
    """
    if "messages" not in st.session_state:
        # Chat history lives only in this Streamlit session (not a database).
        st.session_state.messages = []
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = get_chunk_count()


def render_sidebar() -> None:
    """
    Render the knowledge-base controls and status panel.

    Returns:
        None
    """
    with st.sidebar:
        st.header("Knowledge Base")
        st.caption("Upload documents, then build a local FAISS index.")

        uploaded_files = st.file_uploader(
            "Upload PDF / DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Files are stored under data/documents/",
        )

        if uploaded_files:
            if st.button("Save Uploaded Files", use_container_width=True):
                try:
                    saved = save_uploaded_files(uploaded_files)
                    st.success(f"Saved {len(saved)} file(s).")
                except ValueError as exc:
                    st.error(str(exc))

        col1, col2 = st.columns(2)
        with col1:
            build_clicked = st.button(
                "Build Knowledge Base",
                use_container_width=True,
                type="primary",
            )
        with col2:
            rebuild_clicked = st.button(
                "Rebuild Knowledge Base",
                use_container_width=True,
            )

        if build_clicked:
            with st.spinner("Building knowledge base with OpenAI embeddings..."):
                try:
                    _, num_docs, num_chunks = build_vectorstore()
                    st.session_state.chunk_count = num_chunks
                    st.success(
                        f"Knowledge base ready — {num_docs} document(s), {num_chunks} chunk(s)."
                    )
                except ValueError as exc:
                    st.error(str(exc))
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unexpected error while building the knowledge base: {exc}")

        if rebuild_clicked:
            with st.spinner("Rebuilding knowledge base from scratch..."):
                try:
                    _, num_docs, num_chunks = rebuild_vectorstore()
                    st.session_state.chunk_count = num_chunks
                    st.success(
                        f"Knowledge base rebuilt — {num_docs} document(s), {num_chunks} chunk(s)."
                    )
                except ValueError as exc:
                    st.error(str(exc))
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unexpected error while rebuilding: {exc}")

        if st.button("Clear Vector Store", use_container_width=True):
            clear_vectorstore()
            st.session_state.chunk_count = 0
            st.info("Vector store cleared. Build it again before chatting.")

        st.divider()
        st.subheader("Status")
        doc_count = count_source_files(str(DOCUMENTS_DIR))
        chunk_count = st.session_state.chunk_count
        store_status = "Ready" if vectorstore_exists() else "Not built"

        st.markdown(
            f"""
**Documents:** {doc_count}  
**Chunks:** {chunk_count}  
**Vector Store:** FAISS ({store_status})  
**Embedding Model:**  
`{get_embedding_model_name()}`  
**LLM:**  
`{get_llm_display_name()}`
"""
        )

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_sources(sources, num_chunks: int) -> None:
    """
    Display source citations under an assistant answer.

    Args:
        sources: List of SourceCitation objects.
        num_chunks (int): Number of chunks retrieved for this answer.

    Returns:
        None
    """
    st.caption(f"Retrieved {num_chunks} chunk(s)")
    if not sources:
        st.markdown("_No sources were returned for this answer._")
        return

    st.markdown("**Sources**")
    for source in sources:
        if source.page is not None:
            st.markdown(f"- 📄 `{source.filename}` — Page {source.page}")
        else:
            st.markdown(f"- 📄 `{source.filename}`")


def main() -> None:
    """
    Launch the Streamlit Local RAG Knowledge Assistant.

    Returns:
        None

    Example:
        # From the project root:
        # streamlit run app.py
        main()
    """
    st.set_page_config(
        page_title="Local RAG Knowledge Assistant",
        page_icon="📚",
        layout="centered",
    )

    ensure_folders()
    init_session_state()

    st.title("📚 Local RAG Knowledge Assistant")
    st.write(
        "A beginner-friendly RAG app using LangChain, "
        "OpenAI embeddings, FAISS, and OpenAI chat."
    )

    render_sidebar()

    if not vectorstore_exists():
        st.info(
            "No knowledge base yet. Upload PDF/DOCX files in the sidebar "
            "(or use the sample documents), then click **Build Knowledge Base**."
        )

    # Show the conversation so far.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources") is not None:
                render_sources(message["sources"], message.get("num_chunks", 0))

    # Chat input for the next user question.
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Add the user message to session history first.
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer..."):
                try:
                    # Pass prior turns only (exclude the just-added user message)
                    # so memory stays separate from the current question.
                    prior_history = st.session_state.messages[:-1]
                    response = ask_question(question, chat_history=prior_history)

                    st.markdown(response.answer)
                    render_sources(response.sources, response.num_chunks)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources,
                            "num_chunks": response.num_chunks,
                        }
                    )
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.session_state.messages.pop()
                except ConnectionError as exc:
                    st.error(str(exc))
                    st.session_state.messages.pop()
                except ValueError as exc:
                    st.error(str(exc))
                    st.session_state.messages.pop()
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.session_state.messages.pop()
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")
                    st.session_state.messages.pop()


if __name__ == "__main__":
    main()
