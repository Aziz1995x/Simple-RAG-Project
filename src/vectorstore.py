"""
FAISS vector store helpers.

This module handles the knowledge-base lifecycle:

    Documents → Split → Embed → Create FAISS index → Save locally

It can also load an existing index from disk and rebuild it from scratch.
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.embeddings import get_embedding_model
from src.loaders import load_documents, split_documents

load_dotenv()

DEFAULT_VECTORSTORE_DIR = Path("data/vectorstore")
DEFAULT_DOCUMENTS_DIR = Path("data/documents")
INDEX_NAME = "index"


def vectorstore_exists(persist_directory: str | Path = DEFAULT_VECTORSTORE_DIR) -> bool:
    """
    Check whether a saved FAISS index already exists on disk.

    Args:
        persist_directory (str | Path): Folder where the FAISS index is stored.

    Returns:
        bool: True if an index appears to be present.

    Example:
        if vectorstore_exists():
            print("Knowledge base is ready")
    """
    folder = Path(persist_directory)
    # FAISS.save_local writes files such as index.faiss and index.pkl.
    return (folder / f"{INDEX_NAME}.faiss").exists() and (
        folder / f"{INDEX_NAME}.pkl"
    ).exists()


def build_vectorstore(
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    persist_directory: str | Path = DEFAULT_VECTORSTORE_DIR,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[FAISS, int, int]:
    """
    Build a FAISS knowledge base from documents and save it locally.

    Args:
        documents_dir (str | Path): Folder containing PDF/DOCX files.
        persist_directory (str | Path): Folder where FAISS files will be saved.
        chunk_size (int | None): Optional override for chunk size.
        chunk_overlap (int | None): Optional override for chunk overlap.

    Returns:
        tuple[FAISS, int, int]: The vector store, number of source files used,
        and number of chunks created.

    Raises:
        ValueError: If documents are missing or empty.
        RuntimeError: If embedding or FAISS creation fails.

    Example:
        vectorstore, docs, chunks = build_vectorstore()
        print(docs, chunks)
    """
    selected_chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
    selected_chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "150"))

    # 1) Load raw documents from disk.
    documents = load_documents(str(documents_dir))
    num_files = len({doc.metadata.get("filename") for doc in documents})

    # 2) Split into chunks before embedding.
    chunks = split_documents(
        documents,
        chunk_size=selected_chunk_size,
        chunk_overlap=selected_chunk_overlap,
    )

    try:
        # 3) Create embeddings and store them in FAISS.
        embeddings = get_embedding_model()
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        raise RuntimeError(
            "Failed to create embeddings or build the FAISS index. "
            "Check your OPENAI_API_KEY and internet connection, then try again."
        ) from exc

    # 4) Persist the index so we can reload it later without rebuilding.
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path), index_name=INDEX_NAME)

    return vectorstore, num_files, len(chunks)


def load_vectorstore(
    persist_directory: str | Path = DEFAULT_VECTORSTORE_DIR,
) -> FAISS:
    """
    Load an existing FAISS knowledge base from disk.

    Args:
        persist_directory (str | Path): Folder containing the saved FAISS index.

    Returns:
        FAISS: The loaded vector store.

    Raises:
        FileNotFoundError: If no saved index exists.
        RuntimeError: If the index cannot be loaded.

    Example:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    """
    if not vectorstore_exists(persist_directory):
        raise FileNotFoundError(
            "No knowledge base found. Please build the knowledge base first."
        )

    try:
        embeddings = get_embedding_model()
        # allow_dangerous_deserialization is required by LangChain when loading
        # the local FAISS pickle metadata. This is safe here because we only load
        # an index created by this same application on the local machine.
        vectorstore = FAISS.load_local(
            str(persist_directory),
            embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the FAISS knowledge base. "
            "Try rebuilding the knowledge base."
        ) from exc

    return vectorstore


def rebuild_vectorstore(
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    persist_directory: str | Path = DEFAULT_VECTORSTORE_DIR,
) -> tuple[FAISS, int, int]:
    """
    Delete the existing FAISS index and rebuild it from current documents.

    Args:
        documents_dir (str | Path): Folder containing PDF/DOCX files.
        persist_directory (str | Path): Folder where FAISS files are stored.

    Returns:
        tuple[FAISS, int, int]: The new vector store, file count, and chunk count.

    Example:
        vectorstore, docs, chunks = rebuild_vectorstore()
    """
    clear_vectorstore(persist_directory)
    return build_vectorstore(documents_dir, persist_directory)


def clear_vectorstore(persist_directory: str | Path = DEFAULT_VECTORSTORE_DIR) -> None:
    """
    Remove the saved FAISS index files from disk.

    Args:
        persist_directory (str | Path): Folder containing the FAISS index.

    Example:
        clear_vectorstore()
    """
    folder = Path(persist_directory)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def get_retriever(
    vectorstore: FAISS,
    k: int | None = None,
) -> VectorStoreRetriever:
    """
    Convert a FAISS vector store into a LangChain retriever.

    The retriever finds the top-k chunks most similar to a user question:

        User Question → Retriever → Top K Relevant Chunks

    Args:
        vectorstore (FAISS): Loaded or newly built FAISS store.
        k (int | None): Number of chunks to retrieve. Defaults to TOP_K from .env.

    Returns:
        VectorStoreRetriever: Retriever ready for similarity search.

    Example:
        retriever = get_retriever(vectorstore, k=4)
        docs = retriever.invoke("What is the refund period?")
    """
    top_k = k or int(os.getenv("TOP_K", "4"))

    # Convert the FAISS vector store into a retriever for similarity search.
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def get_chunk_count(persist_directory: str | Path = DEFAULT_VECTORSTORE_DIR) -> int:
    """
    Estimate how many chunks are stored in the current FAISS index.

    Args:
        persist_directory (str | Path): Folder containing the FAISS index.

    Returns:
        int: Number of vectors/chunks in the index, or 0 if unavailable.

    Example:
        print(get_chunk_count())
    """
    if not vectorstore_exists(persist_directory):
        return 0

    try:
        vectorstore = load_vectorstore(persist_directory)
        # FAISS exposes the number of stored vectors through the index object.
        return int(vectorstore.index.ntotal)
    except Exception:
        return 0


def format_docs(docs: list[Document]) -> str:
    """
    Join retrieved document chunks into one context string for the prompt.

    Args:
        docs (list[Document]): Retrieved LangChain documents.

    Returns:
        str: Combined context text with simple source labels.

    Example:
        context = format_docs(retrieved_docs)
    """
    if not docs:
        return "No relevant context was retrieved."

    parts = []
    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{filename}"
        if page is not None:
            location = f"{filename} (page {page})"
        parts.append(f"[Source {i}: {location}]\n{doc.page_content}")

    return "\n\n".join(parts)
