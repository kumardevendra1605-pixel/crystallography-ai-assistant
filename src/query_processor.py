import re
import string
from rapidfuzz import process as fuzz_process, fuzz

try:
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Fallback if NLTK data isn't downloaded yet
    _STOPWORDS = {"a", "an", "the", "is", "it", "in", "on", "at", "to",
                  "for", "of", "and", "or", "but", "not", "with", "this"}

# Crystallography shorthand that people commonly type
ABBREVIATIONS = {
    "xrd":   "x-ray diffraction",
    "cryo":  "cryogenic",
    "res":   "resolution",
    "diff":  "diffraction",
    "snr":   "signal to noise ratio",
    "i/sig": "intensity signal ratio",
    "asu":   "asymmetric unit",
    "rmsd":  "root mean square deviation",
    "mtz":   "reflection data file",
    "cu":    "copper",
    "mo":    "molybdenum",
}

# Built from the question vocabulary at startup
_word_vocab = []


def build_word_vocab(qa_pairs):
    """Collect all unique words from the question set.

    Used later for spell correction — we correct against words that actually
    appear in our dataset rather than a generic dictionary.
    """
    global _word_vocab
    words = set()
    for p in qa_pairs:
        for w in re.findall(r'\b[a-zA-Z]{3,}\b', p["question"].lower()):
            words.add(w)
    _word_vocab = sorted(words)
    return _word_vocab


def normalize_query(query):
    """Clean up a raw query string.

    Lowercases, expands known abbreviations, strips punctuation (but keeps
    hyphens since they matter in crystallography terms like 'X-ray').
    """
    q = query.lower().strip()

    for abbr, expansion in ABBREVIATIONS.items():
        q = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, q)

    # Remove punctuation except hyphens
    translator = str.maketrans({c: ' ' for c in string.punctuation if c != '-'})
    q = q.translate(translator)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def _fix_word(word, threshold=82):
    """Try to correct a single misspelled word against our vocabulary.

    Only corrects if the match is confident enough — we don't want to
    silently mangle words that are just unusual.
    """
    if not _word_vocab or len(word) < 4:
        return word
    match = fuzz_process.extractOne(word, _word_vocab, scorer=fuzz.ratio, score_cutoff=threshold)
    return match[0] if match else word


def spell_correct(query):
    """Run word-level spell correction on a normalized query.

    Returns (corrected_text, was_anything_changed).
    """
    words = query.split()
    fixed = [_fix_word(w) for w in words]
    changed = fixed != words
    return " ".join(fixed), changed


def process_query(query, vocabulary=None):
    """Full query processing pipeline.

    Takes a raw user query and returns a dict with all the variants we
    might want to use — the original, normalized, spell-corrected, and
    the 'effective' version we'll actually embed.
    """
    normalized = normalize_query(query)
    corrected, was_corrected = spell_correct(normalized)

    return {
        "original":  query,
        "normalized": normalized,
        "corrected":  corrected if was_corrected else None,
        "effective":  corrected,  # always use the corrected version for embedding
    }
