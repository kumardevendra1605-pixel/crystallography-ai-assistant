import os
import re
import logging
import docx2txt

logger = logging.getLogger(__name__)


def parse_qa_file(filepath):
    """Read a .docx file and pull out all Q&A pairs.

    Handles two formats we've seen in the wild:
    - The original style: Q. ... A. ...
    - The 2026 workshop style: Question 1 . ... Answer . ...

    Each pair gets tagged with the filename and its position in that file.
    """
    filename = os.path.basename(filepath)
    try:
        raw_text = docx2txt.process(filepath)
    except Exception as e:
        logger.warning(f"Couldn't open {filename}: {e}")
        return []

    # Try the original format first, fall back to the newer one
    pairs = _parse_original_format(raw_text, filename) or _parse_workshop_format(raw_text, filename)

    if not pairs:
        logger.warning(f"No Q&A pairs found in {filename}, skipping it.")

    return pairs


def _parse_original_format(text, filename):
    """Handles the Q. / A. style used in Q&A.docx."""
    blocks = re.split(r'\n\s*Q\.\s*', text)
    pairs = []
    idx = 1

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        parts = re.split(r'\n\s*A\d*\.\s*', block, maxsplit=1)
        if len(parts) != 2:
            continue

        q, a = parts[0].strip(), parts[1].strip()
        if q and a:
            pairs.append(_build_pair(q, a, filename, idx))
            idx += 1

    return pairs


def _parse_workshop_format(text, filename):
    """Handles the 'Question N . / Answer .' style from Q&A_2026.docx."""
    blocks = re.split(r'\n\s*Question\s+\d+\s*\.\s*', text)
    pairs = []
    idx = 1

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        parts = re.split(r'\n\s*Answer\s*\.\s*', block, maxsplit=1)
        if len(parts) != 2:
            continue

        q, a = parts[0].strip(), parts[1].strip()
        if q and a:
            pairs.append(_build_pair(q, a, filename, idx))
            idx += 1

    return pairs


def _build_pair(question, answer, source_file, pair_index):
    return {
        "question": question,
        "answer": answer,
        "keywords": _extract_keywords(question + " " + answer),
        "source_file": source_file,
        "pair_index": pair_index,
    }


def load_all_qa(data_dir="data/docx"):
    """Load every .docx file in the given folder and merge all pairs.

    Deduplicates by question text — if the same question appears in both
    files, we keep the first one we saw.
    """
    all_pairs = []

    try:
        files = sorted(os.listdir(data_dir))
    except FileNotFoundError:
        logger.warning(f"Data folder not found: {data_dir}")
        return []

    for fname in files:
        if not fname.endswith(".docx"):
            continue
        pairs = parse_qa_file(os.path.join(data_dir, fname))
        all_pairs.extend(pairs)

    # Deduplicate — keep first occurrence
    seen = set()
    unique = []
    for p in all_pairs:
        key = p["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def _extract_keywords(text):
    """Pull out meaningful words from a piece of text.

    Skips short words and common filler words that don't help with matching.
    """
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    boring_words = {
        "what", "when", "where", "which", "that", "this", "with",
        "from", "have", "does", "will", "your", "about", "there",
        "their", "they", "been", "were", "would", "could", "should",
        "also", "some", "more", "than", "then", "just", "like",
        "very", "only", "such", "each", "both", "into", "over",
    }
    # dict.fromkeys preserves order while deduplicating
    return list(dict.fromkeys(w for w in words if w not in boring_words))
