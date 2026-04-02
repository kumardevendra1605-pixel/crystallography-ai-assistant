"""
Tests for src/topic_grouper.py
- Partition invariant: union of groups == input sources
- ≤ 2 sources → single "General" group
- Labels are non-empty strings
- Groups ordered by descending max confidence
"""
import pytest
from src.topic_grouper import group_by_subtopic


def _make_source(question, keywords, confidence, source_file="Q&A.docx", pair_index=1):
    return {
        "question":    question,
        "answer":      "Some answer.",
        "keywords":    keywords,
        "confidence":  confidence,
        "source_file": source_file,
        "pair_index":  pair_index,
        "citation":    f"Q{pair_index} from {source_file}",
    }


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_sources_returns_empty():
    result = group_by_subtopic([], "anything")
    assert result == {}


def test_single_source_returns_general():
    src = _make_source("What is twinning?", ["twinning", "crystal"], 0.9)
    result = group_by_subtopic([src], "twinning")
    assert list(result.keys()) == ["General"]
    assert result["General"] == [src]


def test_two_sources_returns_general():
    s1 = _make_source("Q1", ["twinning"], 0.9, pair_index=1)
    s2 = _make_source("Q2", ["diffraction"], 0.8, pair_index=2)
    result = group_by_subtopic([s1, s2], "twinning")
    assert list(result.keys()) == ["General"]
    assert set(id(s) for s in result["General"]) == {id(s1), id(s2)}


# ── Partition invariant ───────────────────────────────────────────────────────

def test_partition_invariant_all_sources_present():
    sources = [
        _make_source("Q1", ["twinning", "crystal", "detection"], 0.9, pair_index=1),
        _make_source("Q2", ["twinning", "crystal", "refinement"], 0.85, pair_index=2),
        _make_source("Q3", ["phase", "problem", "diffraction"], 0.7, pair_index=3),
        _make_source("Q4", ["phase", "problem", "solution"], 0.65, pair_index=4),
        _make_source("Q5", ["absorption", "correction", "crystal"], 0.6, pair_index=5),
    ]
    result = group_by_subtopic(sources, "crystallography")

    # All sources appear exactly once across all groups
    all_in_groups = [s for group in result.values() for s in group]
    assert len(all_in_groups) == len(sources)
    assert set(id(s) for s in all_in_groups) == set(id(s) for s in sources)


def test_no_source_duplicated_across_groups():
    sources = [
        _make_source("Q1", ["twinning", "crystal", "detection"], 0.9, pair_index=1),
        _make_source("Q2", ["phase", "problem", "diffraction"], 0.8, pair_index=2),
        _make_source("Q3", ["absorption", "correction", "sample"], 0.7, pair_index=3),
        _make_source("Q4", ["twinning", "refinement", "crystal"], 0.65, pair_index=4),
    ]
    result = group_by_subtopic(sources, "test")
    all_ids = [id(s) for group in result.values() for s in group]
    assert len(all_ids) == len(set(all_ids)), "Source appears in multiple groups"


# ── Labels ────────────────────────────────────────────────────────────────────

def test_all_labels_non_empty():
    sources = [
        _make_source("Q1", ["twinning", "crystal", "detection"], 0.9, pair_index=1),
        _make_source("Q2", ["phase", "problem", "diffraction"], 0.8, pair_index=2),
        _make_source("Q3", ["absorption", "correction", "sample"], 0.7, pair_index=3),
    ]
    result = group_by_subtopic(sources, "test")
    for label in result.keys():
        assert isinstance(label, str) and label.strip(), f"Empty label found: {label!r}"


def test_labels_are_unique():
    sources = [
        _make_source(f"Q{i}", ["twinning", "crystal", f"kw{i}"], 0.9 - i * 0.05, pair_index=i)
        for i in range(1, 6)
    ]
    result = group_by_subtopic(sources, "test")
    labels = list(result.keys())
    assert len(labels) == len(set(labels)), "Duplicate group labels"


# ── Ordering ──────────────────────────────────────────────────────────────────

def test_groups_ordered_by_descending_max_confidence():
    sources = [
        _make_source("Q1", ["twinning", "crystal", "detection"], 0.5, pair_index=1),
        _make_source("Q2", ["twinning", "crystal", "refinement"], 0.55, pair_index=2),
        _make_source("Q3", ["phase", "problem", "diffraction"], 0.9, pair_index=3),
        _make_source("Q4", ["phase", "problem", "solution"], 0.85, pair_index=4),
    ]
    result = group_by_subtopic(sources, "test")
    max_confs = [max(s["confidence"] for s in grp) for grp in result.values()]
    assert max_confs == sorted(max_confs, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
