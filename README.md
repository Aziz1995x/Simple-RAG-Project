# Local RAG Knowledge Assistant

A beginner-friendly **RAG (Retrieval-Augmented Generation)** application built with LangChain, FAISS, OpenAI embeddings, OpenAI chat, and Streamlit.

This project is designed for learning — not production. The goal is to help you understand the full RAG pipeline by reading simple, readable Python code.

---

## Project Overview

**RAG** means the LLM answers questions using information retrieved from your documents, instead of relying only on what it memorized during training.

High-level flow:

```text
Documents
    → Loaders
    → Chunking
    → Embeddings
    → FAISS
    → Retriever
    → RAG Prompt
    → OpenAI Chat
    → Answer + Sources
```

You can upload PDF/DOCX files, build a local FAISS knowledge base, ask questions in a chat UI, see source citations, and use conversational memory for follow-up questions.

---

## Why RAG?

An LLM alone is often not enough when you want answers about:

- private company documents
- custom policies
- notes that were never in the model’s training data

Without RAG, the model may guess or say it does not know.

With RAG:

1. Your documents are split into chunks
2. Chunks are turned into vectors (embeddings)
3. Similar chunks are retrieved for each question
4. Those chunks are added to the prompt
5. The LLM answers using that retrieved context

---

## Architecture

```text
                DOCUMENT INGESTION
                       │
                       ▼
                Load PDF / DOCX
                       │
                       ▼
                  Split Text
                       │
                       ▼
                Create Embeddings (OpenAI)
                       │
                       ▼
                  FAISS Store
                       │
                       ▼
                   USER QUERY
                       │
                       ▼
                    Retriever
                       │
                       ▼
               Relevant Chunks
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
 Conversation History      Document Context
   (memory)                 (facts)
           │                       │
           └───────────┬───────────┘
                       ▼
                   RAG Prompt
                       │
                       ▼
                   OpenAI Chat
                       │
                       ▼
                Answer + Sources
```

---

## Project Structure

```text
.
├── app.py                 # Streamlit UI entry point
├── requirements.txt       # Python dependencies (CPU-friendly)
├── .env.example           # Example configuration
├── README.md              # This file
├── prompts/
│   └── rag_prompt.txt     # Editable RAG prompt template
├── data/
│   ├── documents/         # PDF / DOCX files (samples included)
│   └── vectorstore/       # Saved FAISS index
├── scripts/
│   └── create_sample_documents.py
└── src/
    ├── loaders.py         # Load + split documents
    ├── embeddings.py      # OpenAI embeddings (simple API, no PyTorch)
    ├── vectorstore.py     # Build / load / rebuild FAISS
    ├── rag_chain.py       # Retrieve + memory + LLM answer
    └── schemas.py         # Simple response data shapes
```

Every file has one clear job so beginners can follow the pipeline step by step.

---

## Prerequisites

- **Python 3.10+** (3.11/3.12 recommended; 3.13 may work depending on package wheels)
- An **OpenAI API key**
- A normal laptop is enough — **GPU is NOT required**

Notes:

- OpenAI is used for **both** embeddings and chat answers
- FAISS runs locally on CPU to store and search document vectors
- This project does **not** use Ollama, PyTorch, or Hugging Face Transformers

---

## OpenAI Setup

1. Create an API key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Put it in your `.env` file:

```text
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

Defaults are beginner-friendly and low-cost. You can change the chat or embedding model names in `.env` if you want.

---

## Installation

From the project root:

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the example environment file and add your OpenAI API key:

```bash
copy .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```text
OPENAI_API_KEY=sk-...
```

---

## Running the Application

```bash
streamlit run app.py
```

Then:

1. Confirm sample documents exist in `data/documents/` (or upload your own PDF/DOCX)
2. Click **Build Knowledge Base**
3. Ask questions in the chat panel

---

## Configuration

Values are read from `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | _(required)_ | API key for embeddings and chat |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `TOP_K` | `4` | How many chunks to retrieve |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between neighboring chunks |

### What do `chunk_size` and `chunk_overlap` mean?

- **`chunk_size`**: how large each text piece is before embedding. Larger chunks keep more context; smaller chunks can retrieve more precisely.
- **`chunk_overlap`**: shared text between consecutive chunks so meaning is less likely to be cut awkwardly at boundaries.

---

## How RAG Works in This Project

### 1) Ingestion (Build Knowledge Base)

```text
PDF / DOCX
  → Document Loader
  → Raw Documents
  → RecursiveCharacterTextSplitter
  → Chunks
  → OpenAI Embeddings (text-embedding-3-small)
  → FAISS index saved under data/vectorstore/
```

### 2) Query (Ask a Question)

```text
User Question
  → Retriever (top-k similar chunks)
  → Relevant Context
  + Conversation History
  → RAG Prompt
  → OpenAI Chat
  → Answer + Sources
```

---

## Memory

This app is a **conversational RAG** system.

- **Retrieval** provides factual document context
- **Memory** provides conversation context for follow-ups

Example:

```text
User: What is the company's refund policy?
AI: The refund period is 30 days...

User: What about international customers?
AI: For international customers, the same 30-day period applies...
```

The second question uses conversation history to understand that "international customers" still refers to the refund policy.

Chat history is stored only in the Streamlit session. It is cleared with **Clear Chat** and is not saved to a database.

---

## Sources / Citations

Answers include source citations from retrieved chunk metadata:

```text
Sources
- refund_policy.pdf — Page 2
- remote_work_guidelines.docx
```

- PDF chunks usually include page numbers
- DOCX files often do not have page metadata, so only the filename is shown

---

## Example Questions

Try these after building the knowledge base:

```text
What is the refund period?
Can employees work remotely?
How many days of leave are available?
What happens if I cancel after 30 days?
What about international customers?
```

Ask the last question **after** asking about the refund policy to see conversational memory in action.

Other useful questions:

```text
What benefits does the company provide?
What are the core collaboration hours?
Do international customers get refunds in their original currency?
```

---

## Sample Documents

The repository includes:

- `data/documents/company_handbook.pdf`
- `data/documents/refund_policy.pdf`
- `data/documents/remote_work_guidelines.docx`

To regenerate them:

```bash
pip install fpdf2 python-docx
python scripts/create_sample_documents.py
```

---

## Common Errors

| Message | What to do |
|---|---|
| `OPENAI_API_KEY is missing...` | Add your key to `.env` |
| OpenAI authentication failed | Check that the API key is valid |
| OpenAI rate limit reached | Wait a moment and retry |
| No documents uploaded | Add PDF/DOCX files under `data/documents/` |
| No knowledge base found | Click **Build Knowledge Base** |
| Unsupported file type | Upload only `.pdf` or `.docx` |
| Embedding / FAISS build failed | Check `OPENAI_API_KEY` and rebuild the knowledge base |

---

## Teaching Map

This project is intentionally arranged so you can teach:

```text
Document Loaders
 → Document Splitting
 → Embeddings
 → Vector Stores (FAISS)
 → Retrievers
 → Prompt Templates
 → LLM (OpenAI)
 → Conversational Memory
 → Source / Citation Handling
 → Streamlit UI
```

Students should be able to open `src/` and trace:

```text
User Question → Retriever → Relevant Documents → Prompt → OpenAI → Answer → Sources
```

---

## Future Improvements

Not included in Version 1 (on purpose):

- better retrieval / hybrid search / BM25
- reranking
- query rewriting
- RAG evaluation
- LangGraph
- agents and tool calling
- cloud vector databases
- production deployment

Those are great follow-up playlist topics after learners understand this baseline.

---
