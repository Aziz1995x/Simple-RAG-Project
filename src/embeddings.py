"""
OpenAI embedding model setup.

Text chunks are converted into numeric vectors so we can search by meaning
instead of exact keywords:

    Text Chunk → Embedding Model → Vector

OpenAI embeddings keep the project beginner-friendly: one API key, no local
model downloads, and no PyTorch / transformers install.
"""

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

# Small, affordable OpenAI embedding model — good default for learning RAG.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding_model(model_name: str | None = None) -> OpenAIEmbeddings:
    """
    Create an OpenAI embedding client using the API key from .env.

    Args:
        model_name (str | None): Optional OpenAI embedding model name. If omitted,
            the value from EMBEDDING_MODEL in .env is used.

    Returns:
        OpenAIEmbeddings: Ready-to-use embedding model.

    Raises:
        ValueError: If OPENAI_API_KEY is missing.
        RuntimeError: If the embedding client cannot be created.

    Example:
        embeddings = get_embedding_model()
        vector = embeddings.embed_query("What is the refund policy?")
        print(len(vector))
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your .env file before "
            "building the knowledge base."
        )

    selected_model = model_name or os.getenv(
        "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )

    try:
        embeddings = OpenAIEmbeddings(
            model=selected_model,
            api_key=SecretStr(api_key),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create the OpenAI embedding client. "
            "Check OPENAI_API_KEY and EMBEDDING_MODEL in your .env file."
        ) from exc

    return embeddings


def get_embedding_model_name() -> str:
    """
    Return the configured embedding model name for UI display.

    Returns:
        str: OpenAI embedding model name currently configured.

    Example:
        print(get_embedding_model_name())
    """
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
