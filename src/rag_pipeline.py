import os
import re
import numpy as np
from collections import OrderedDict

from src.query_processor import process_query, build_word_vocab
from src.embeddings import embed_query
from src.vector_store import search
from src.topic_grouper import group_by_subtopic

# Pick up model config from environment, with sensible defaults
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")

# Similarity thresholds
MID_CONFIDENCE = 0.45   # above this → include in the answer
LOW_CONFIDENCE = 0.35   # above this → show as a related question
                        # below LOW_CONFIDENCE → ignore entirely

OUT_OF_SCOPE_MSG = "This specific topic isn't covered in the current Q&A dataset."

# Keep a reference to the last qa_pairs we saw so we only rebuild the
# word vocabulary when the dataset actually changes
_last_qa_pairs = []


def retrieve_and_respond(user_query, index, qa_pairs, top_k=None):
    """Main entry point — takes a user question and returns a full response dict.

    The pipeline is:
    1. Normalize and spell-correct the query
    2. Embed it
    3. Search the entire dataset (no early stopping)
    4. Filter results by confidence threshold
    5. Group related results by sub-topic
    6. Send everything to the LLM in one call
    7. Return the answer with citations and metadata
    """
    global _last_qa_pairs
    if _last_qa_pairs is not qa_pairs:
        build_word_vocab(qa_pairs)
        _last_qa_pairs = qa_pairs

    # Step 1 — clean up the query
    processed = process_query(user_query)
    query_to_embed = processed["effective"]

    # Step 2 — embed
    query_embedding = embed_query(query_to_embed)

    # Step 3 — search everything
    n = len(qa_pairs)
    k = min(top_k or n, n)
    scores, indices = search(query_embedding, index, top_k=k)

    # Step 4 — split results into sources (used in answer) and alternatives (shown as suggestions)
    sources = []
    alternatives = []

    for score, idx in zip(scores, indices):
        score, idx = float(score), int(idx)
        if idx < 0 or idx >= n:
            continue

        pair = qa_pairs[idx]
        citation = f"Q{pair.get('pair_index', '?')} from {pair.get('source_file', 'unknown')}"

        if score >= MID_CONFIDENCE:
            sources.append({
                "question":    pair["question"],
                "answer":      pair["answer"],
                "keywords":    pair.get("keywords", []),
                "confidence":  round(score, 3),
                "source_file": pair.get("source_file", "unknown"),
                "pair_index":  pair.get("pair_index", 0),
                "citation":    citation,
            })
        elif score >= LOW_CONFIDENCE:
            alternatives.append({
                "question":   pair["question"],
                "confidence": round(score, 3),
                "citation":   citation,
            })

    sources.sort(key=lambda x: x["confidence"], reverse=True)

    # Nothing relevant found — return the out-of-scope message without touching the LLM
    if not sources:
        return {
            "answer":               OUT_OF_SCOPE_MSG,
            "confidence":           0.0,
            "matched_question":     None,
            "sources":              [],
            "alternatives":         alternatives,
            "topic_groups":         OrderedDict(),
            "did_you_mean":         alternatives[0]["question"] if alternatives else None,
            "clarification_needed": True,
            "query_info":           _query_info(processed),
        }

    best = sources[0]

    # Step 5 — group by sub-topic
    topic_groups = group_by_subtopic(sources, user_query)

    # Step 6 — one LLM call for everything
    answer = _build_answer(user_query, topic_groups)

    return {
        "answer":               answer,
        "confidence":           best["confidence"],
        "matched_question":     best["question"],
        "sources":              sources,
        "alternatives":         alternatives,
        "topic_groups":         topic_groups,
        "did_you_mean":         None if best["confidence"] >= 0.75 else best["question"],
        "clarification_needed": False,
        "query_info":           _query_info(processed),
    }


def _build_answer(user_query, topic_groups):
    """Send all topic groups to the LLM in a single call and get back a structured answer.

    We pass the full grouped context and ask the model to write one section per
    topic. This is faster than one call per group and gives the model enough
    context to avoid repeating itself across sections.

    Falls back to a plain text dump if Ollama isn't available.
    """
    if not topic_groups:
        return OUT_OF_SCOPE_MSG

    # Build the context block
    group_blocks = []
    for label, group_sources in topic_groups.items():
        entries = []
        for src in group_sources:
            entries.append(
                f"  [{src['citation']} · {src['confidence']:.0%}]\n"
                f"  Q: {src['question']}\n"
                f"  A: {src['answer']}"
            )
        group_blocks.append(f"=== {label} ===\n" + "\n\n".join(entries))

    context = "\n\n".join(group_blocks)

    # Pre-build citation footers so we can append them after the LLM responds
    citation_footers = {}
    for label, group_sources in topic_groups.items():
        citation_footers[label] = "\n".join(
            f"*Source: {src['citation']} ({src['confidence']:.0%})*"
            for src in group_sources
        )

    try:
        import ollama

        section_list = "\n".join(f"- {label}" for label in topic_groups)

        prompt = f"""You are an expert crystallographer writing for researchers and graduate students.

USER QUESTION: {user_query}

The knowledge base has {len(topic_groups)} relevant sub-topic(s):
{section_list}

KNOWLEDGE BASE (grouped by sub-topic):
{context}

INSTRUCTIONS:
- Write one ### section per sub-topic, in the same order as listed above.
- Each section header must be exactly: ### <sub-topic name>
- Only use information from the entries under that sub-topic — don't mix across sections.
- Use only what's in the knowledge base — no outside knowledge, no guessing.
- Keep all the technical details, caveats, and "it depends" nuances from the sources.
- Write in clear, expert prose. Bullet points are fine where they help.
- Be thorough but don't pad it out.

ANSWER:"""

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response["message"]["content"].strip()

        # Append citation lines after each section
        for label, footer in citation_footers.items():
            answer = re.sub(
                rf'(### {re.escape(label)}.*?)(\n### |\Z)',
                lambda m: m.group(1) + "\n\n" + footer + "\n\n---\n\n" + m.group(2).lstrip("\n"),
                answer,
                flags=re.DOTALL,
            )

        return answer.strip()

    except Exception:
        # Ollama not available — just format the raw Q&A nicely
        sections = []
        for label, group_sources in topic_groups.items():
            parts = [f"**{src['question']}**\n\n{src['answer']}" for src in group_sources]
            footer = citation_footers[label]
            sections.append(f"### {label}\n\n" + "\n\n".join(parts) + "\n\n" + footer)
        return "\n\n---\n\n".join(sections)


def _query_info(processed):
    return {
        "original":   processed["original"],
        "normalized": processed["normalized"],
        "corrected":  processed.get("corrected"),
    }
