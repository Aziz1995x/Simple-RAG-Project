"""
Simple data shapes used by the Local RAG application.

Keeping these in one small file makes the RAG response structure easy to follow
without adding complex validation frameworks.
"""

from dataclasses import dataclass, field


@dataclass
class SourceCitation:
    """
    A single source citation extracted from a retrieved document chunk.

    Attributes:
        filename (str): Name of the source file (for example, refund_policy.pdf).
        page (int | None): Page number when available (PDF). None for DOCX.
        preview (str): Short text preview of the retrieved chunk.
    """

    filename: str
    page: int | None = None
    preview: str = ""


@dataclass
class RAGResponse:
    """
    Final answer returned by the conversational RAG pipeline.

    Attributes:
        answer (str): The LLM-generated answer.
        sources (list[SourceCitation]): Unique source citations used for the answer.
        num_chunks (int): How many document chunks were retrieved for this question.

    Example:
        response = RAGResponse(
            answer="The refund period is 30 days.",
            sources=[SourceCitation(filename="refund_policy.pdf", page=1)],
            num_chunks=4,
        )
        print(response.answer)
    """

    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    num_chunks: int = 0
