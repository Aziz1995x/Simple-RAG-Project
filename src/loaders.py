"""
Document loading helpers for PDF and DOCX files.

This module turns raw files into LangChain Document objects that keep useful
metadata such as the source filename and page number (when available).
"""

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_documents(folder_path: str) -> list[Document]:
    """
    Load all supported PDF and DOCX documents from a folder.

    Args:
        folder_path (str): Path to the folder that contains uploaded documents.

    Returns:
        list[Document]: A list of LangChain Document objects with metadata.

    Raises:
        FileNotFoundError: If the folder does not exist.
        ValueError: If the folder has no supported documents or files are empty.

    Example:
        documents = load_documents("data/documents")
        print(len(documents))
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found: {folder_path}")

    files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise ValueError(
            "No supported documents found. Please upload PDF or DOCX files."
        )

    documents: list[Document] = []
    for file_path in files:
        loaded = _load_single_document(file_path)
        documents.extend(loaded)

    # Remove empty pages / sections that would create useless embeddings.
    documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]

    if not documents:
        raise ValueError(
            "Documents were found, but they appear to be empty or unreadable."
        )

    return documents


def _load_single_document(file_path: Path) -> list[Document]:
    """
    Load one PDF or DOCX file and normalize its metadata.

    Args:
        file_path (Path): Path to a single document file.

    Returns:
        list[Document]: One or more Document objects from the file.

    Raises:
        ValueError: If the file type is unsupported or the file cannot be read.

    Example:
        docs = _load_single_document(Path("data/documents/refund_policy.pdf"))
    """
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".pdf":
            # PyPDFLoader creates one Document per page and includes page numbers.
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        elif suffix == ".docx":
            # DOCX loaders usually do not provide page numbers.
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
        else:
            raise ValueError(
                f"Unsupported file type: {file_path.name}. "
                "Only PDF and DOCX files are supported."
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not read '{file_path.name}'. "
            "The file may be invalid or corrupted."
        ) from exc

    # Keep metadata simple and beginner-friendly for source citations later.
    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["filename"] = file_path.name
        # PDF loaders usually set "page" (0-based). Convert to 1-based for display.
        if "page" in doc.metadata and doc.metadata["page"] is not None:
            try:
                doc.metadata["page"] = int(doc.metadata["page"]) + 1
            except (TypeError, ValueError):
                doc.metadata["page"] = None
        else:
            doc.metadata["page"] = None

    return docs


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split documents into smaller overlapping chunks for embedding.

    chunk_size controls how many characters go into each chunk.
    chunk_overlap keeps some shared text between neighboring chunks so meaning
    is less likely to be cut off at chunk boundaries.

    Args:
        documents (list[Document]): Raw documents loaded from disk.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Overlapping characters between consecutive chunks.

    Returns:
        list[Document]: Chunked documents ready for embedding.

    Example:
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=150)
        print(len(chunks))
    """
    if not documents:
        raise ValueError("No documents to split.")

    # RecursiveCharacterTextSplitter tries to split on paragraphs, then sentences,
    # then words — which usually produces cleaner chunks than a fixed character cut.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # Split large documents into smaller chunks before creating embeddings.
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    return chunks


def count_source_files(folder_path: str) -> int:
    """
    Count supported PDF/DOCX files currently stored in the documents folder.

    Args:
        folder_path (str): Path to the document folder.

    Returns:
        int: Number of supported files found.

    Example:
        print(count_source_files("data/documents"))
    """
    folder = Path(folder_path)
    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
