"""
Integration + unit tests for the RAG pipeline.
- Out-of-scope path never calls LLM
- Citation present in answer for all sources
- Threshold filtering correctness
- source_file present in all returned sources
"""
import numpy as np
import pytest
import unittest.mock as mock

from src.rag_pipeline import retrieve_and_respond, MID_CONFIDENCE, LOW_CONFIDENCE, OUT_OF_SCOPE_MSG


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_qa_pairs(n=10):
    return [
        {
            "question":    f"Question about topic {i}",
            "answer":      f"Answer for topic {i} with details.",
            "keywords":    [f"topic{i}", "crystal", "diffraction"],
            "source_file": "Q&A.docx" if i % 2 == 0 else "Q&A_2026.docx",
            "pair_index":  i + 1,
        }
        for i in range(n)
    ]


def _make_mock_index(scores_to_return, n_pairs):
    """Return a mock FAISS index whose search() returns given scores."""
    index = mock.MagicMock()
    indices = np.arange(n_pairs, dtype=np.int64)
    scores_arr = np.array(scores_to_return, dtype=np.float32)
    index.search.return_value = (scores_arr.reshape(1, -1), indices.reshape(1, -1))
    return index


# ── Out-of-scope: LLM never called ───────────────────────────────────────────

def test_out_of_scope_never_calls_llm():
    """When all scores < LOW_CONFIDENCE, LLM must not be called."""
    qa_pairs = _make_qa_pairs(5)
    scores = [0.10, 0.12, 0.08, 0.15, 0.11]
    index = _make_mock_index(scores, 5)

    with mock.patch("ollama.chat") as mock_llm:
        result = retrieve_and_respond("unrelated query", index, qa_pairs)

    mock_llm.assert_not_called()
    assert result["answer"] == OUT_OF_SCOPE_MSG
    assert result["sources"] == []
    assert result["clarification_needed"] is True


def test_out_of_scope_with_alternatives_never_calls_llm():
    """Scores in LOW..MID range → alternatives populated, LLM still not called."""
    qa_pairs = _make_qa_pairs(5)
    scores = [0.38, 0.36, 0.37, 0.39, 0.10]
    index = _make_mock_index(scores, 5)

    with mock.patch("ollama.chat") as mock_llm:
        result = retrieve_and_respond("vague query", index, qa_pairs)

    mock_llm.assert_not_called()
    assert result["answer"] == OUT_OF_SCOPE_MSG
    assert len(result["alternatives"]) > 0
    assert result["sources"] == []


# ── Threshold filtering ───────────────────────────────────────────────────────

def test_sources_only_above_mid_confidence():
    qa_pairs = _make_qa_pairs(5)
    scores = [0.80, 0.60, 0.44, 0.36, 0.10]
    index = _make_mock_index(scores, 5)

    with mock.patch("src.rag_pipeline._build_answer", return_value="### General\n\nSynthesized.\n\n*Source: Q1 from Q&A.docx (80%)*"):
        result = retrieve_and_respond("test query", index, qa_pairs)

    for src in result["sources"]:
        assert src["confidence"] >= MID_CONFIDENCE, f"Source below threshold: {src['confidence']}"


def test_alternatives_only_in_low_mid_band():
    qa_pairs = _make_qa_pairs(5)
    scores = [0.80, 0.60, 0.44, 0.36, 0.10]
    index = _make_mock_index(scores, 5)

    with mock.patch("src.rag_pipeline._build_answer", return_value="### General\n\nSynthesized.\n\n*Source: Q1 from Q&A.docx (80%)*"):
        result = retrieve_and_respond("test query", index, qa_pairs)

    for alt in result["alternatives"]:
        assert LOW_CONFIDENCE <= alt["confidence"] < MID_CONFIDENCE


# ── Citations ─────────────────────────────────────────────────────────────────

def test_all_sources_have_citation_field():
    qa_pairs = _make_qa_pairs(5)
    scores = [0.80, 0.70, 0.60, 0.50, 0.10]
    index = _make_mock_index(scores, 5)

    with mock.patch("src.rag_pipeline._build_answer", return_value="### General\n\nSynthesized.\n\n*Source: Q1 from Q&A.docx (80%)*"):
        result = retrieve_and_respond("test query", index, qa_pairs)

    for src in result["sources"]:
        assert "citation" in src
        assert src["citation"].startswith("Q")
        assert "from" in src["citation"]


def test_source_file_present_in_all_sources():
    """Regression: source_file must be populated on every returned source."""
    qa_pairs = _make_qa_pairs(5)
    scores = [0.80, 0.70, 0.60, 0.50, 0.10]
    index = _make_mock_index(scores, 5)

    with mock.patch("src.rag_pipeline._build_answer", return_value="### General\n\nSynthesized.\n\n*Source: Q1 from Q&A.docx (80%)*"):
        result = retrieve_and_respond("test query", index, qa_pairs)

    for src in result["sources"]:
        assert src.get("source_file"), f"Missing source_file on: {src}"
        assert src.get("pair_index", 0) >= 1


# ── Integration: real files ───────────────────────────────────────────────────

def test_integration_real_files():
    """End-to-end smoke test with real .docx files."""
    from src.qa_parser import load_all_qa
    from src.embeddings import build_and_cache_embeddings
    from src.vector_store import create_vector_store
    from src.query_processor import build_word_vocab

    qa_pairs = load_all_qa("data/docx")
    assert len(qa_pairs) > 0

    build_word_vocab(qa_pairs)
    embeddings = build_and_cache_embeddings(qa_pairs)
    index = create_vector_store(embeddings)

    # All pairs must have source metadata
    for p in qa_pairs:
        assert p.get("source_file"), "Missing source_file"
        assert p.get("pair_index", 0) >= 1, "Invalid pair_index"

    # A known topic should return sources with citations
    result = retrieve_and_respond("omega phi rotation", index, qa_pairs)
    if result["sources"]:
        for src in result["sources"]:
            assert "citation" in src
            assert src["source_file"] in {"Q&A.docx", "Q&A_2026.docx"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
