# xTal.ai — Crystallography AI Assistant

A local AI-powered Q&A assistant for crystallography researchers and graduate students.
Uses semantic search over a curated knowledge base to answer questions with full source citations.

## Features

- Exhaustive semantic search over all Q&A pairs (FAISS + sentence-transformers)
- Automatic sub-topic grouping with section headers
- Full source citations (e.g. "Q47 from Q&A_2026.docx")
- ChatGPT-style conversation history with auto-generated titles
- Tolerant to typos, abbreviations, and vague phrasing
- Runs fully offline — no external API calls

## Tech Stack

- Python 3.11+
- Streamlit (UI)
- FAISS (vector similarity search)
- Sentence Transformers — `all-MiniLM-L6-v2`
- Ollama — `llama3.2:latest` (local LLM)
- docx2txt (knowledge base parsing)
- rapidfuzz + NLTK (query normalization)

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK data (first run only)
```python
import nltk
nltk.download('stopwords')
```

### 3. Pull the LLM model
```bash
ollama pull llama3.2:latest
```

### 4. Run the app
```bash
streamlit run app.py
```

## Knowledge Base

Place `.docx` Q&A files in `data/docx/`. Two formats are supported:
- `Q. <question>\nA. <answer>` (original format)
- `Question N . <question>\nAnswer . <answer>` (2026 format)

## Project Structure

```
app.py                  # Streamlit UI + conversation management
src/
  qa_parser.py          # Parses .docx files into Q&A pairs with source metadata
  embeddings.py         # Builds/caches sentence embeddings (SHA256 hash invalidation)
  vector_store.py       # FAISS index (cosine similarity)
  rag_pipeline.py       # Retrieval + topic grouping + LLM synthesis
  topic_grouper.py      # Keyword-overlap clustering into sub-topics
  query_processor.py    # Query normalization, spell correction, abbreviation expansion
data/docx/              # Knowledge base .docx files
assets/                 # UI assets (logo, background image)
database/               # Embedding cache (auto-generated)
```

## Notes

- Embedding cache is stored at `database/qa_embeddings.pkl` and auto-invalidates when source files change.
- Ollama must be running locally with `llama3.2:latest` pulled.
- The app runs fully offline — no data leaves the machine.
