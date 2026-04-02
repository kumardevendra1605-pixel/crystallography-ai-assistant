"""
Tests for src/qa_parser.py
- source_file and pair_index present on all pairs
- pair_index is 1-based and unique within each file
- deduplication preserves first occurrence metadata
- graceful handling of missing/empty files
"""
import os
import pytest
from src.qa_parser import parse_qa_file, load_all_qa


# ── parse_qa_file ─────────────────────────────────────────────────────────────

def test_parse_real_files():
    """Both docx files parse into non-empty lists with required fields."""
    data_dir = "data/docx"
    for fname in os.listdir(data_dir):
        if not fname.endswith(".docx"):
            continue
        pairs = parse_qa_file(os.path.join(data_dir, fname))
        assert len(pairs) > 0, f"{fname} returned no pairs"
        for p in pairs:
            assert p["source_file"] == fname, "source_file must be basename"
            assert isinstance(p["pair_index"], int) and p["pair_index"] >= 1
            assert p["question"].strip()
            assert p["answer"].strip()


def test_pair_index_unique_per_file():
    """pair_index values are unique within each file."""
    data_dir = "data/docx"
    for fname in os.listdir(data_dir):
        if not fname.endswith(".docx"):
            continue
        pairs = parse_qa_file(os.path.join(data_dir, fname))
        indices = [p["pair_index"] for p in pairs]
        assert len(indices) == len(set(indices)), f"Duplicate pair_index in {fname}"


def test_pair_index_starts_at_one():
    data_dir = "data/docx"
    for fname in os.listdir(data_dir):
        if not fname.endswith(".docx"):
            continue
        pairs = parse_qa_file(os.path.join(data_dir, fname))
        if pairs:
            assert pairs[0]["pair_index"] == 1


def test_nonexistent_file_returns_empty():
    result = parse_qa_file("data/docx/nonexistent_file.docx")
    assert result == []


# ── load_all_qa ───────────────────────────────────────────────────────────────

def test_load_all_qa_has_source_metadata():
    pairs = load_all_qa("data/docx")
    assert len(pairs) > 0
    for p in pairs:
        assert "source_file" in p and p["source_file"]
        assert "pair_index" in p and p["pair_index"] >= 1


def test_load_all_qa_no_duplicates():
    pairs = load_all_qa("data/docx")
    questions = [p["question"].lower().strip() for p in pairs]
    assert len(questions) == len(set(questions)), "Duplicate questions found after dedup"


def test_deduplication_preserves_first_occurrence(tmp_path):
    """When same question appears in two files, first file's metadata is kept."""
    # Create two minimal docx-like text files (we'll test via parse directly)
    # Instead, test the dedup logic with synthetic data
    from src.qa_parser import load_all_qa
    import unittest.mock as mock

    fake_pairs_a = [{"question": "What is twinning?", "answer": "A1",
                     "keywords": [], "source_file": "Q&A.docx", "pair_index": 1}]
    fake_pairs_b = [{"question": "What is twinning?", "answer": "A2",
                     "keywords": [], "source_file": "Q&A_2026.docx", "pair_index": 5}]

    with mock.patch("src.qa_parser.parse_qa_file", side_effect=[fake_pairs_a, fake_pairs_b]):
        with mock.patch("os.listdir", return_value=["Q&A.docx", "Q&A_2026.docx"]):
            result = load_all_qa("data/docx")

    assert len(result) == 1
    assert result[0]["source_file"] == "Q&A.docx"
    assert result[0]["pair_index"] == 1


def test_load_all_qa_missing_dir_returns_empty():
    result = load_all_qa("data/nonexistent_dir")
    assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
