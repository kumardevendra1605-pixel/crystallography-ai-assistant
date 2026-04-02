# Requirements: Crystallography Q&A Chatbot

## Introduction

This document defines the functional and non-functional requirements for the crystallography Q&A chatbot feature. The system extends the existing Python/Streamlit application to provide exhaustive semantic retrieval, sub-topic grouping, expert-level synthesis, and full source citation from two curated `.docx` knowledge bases.

---

## Requirements

### 1. Knowledge Base Parsing with Source Tracking

**User Story**: As a researcher, I want the chatbot to know which file and which question number each answer came from, so I can verify the source of any information.

#### Acceptance Criteria

- 1.1 `parse_qa_file(filepath)` MUST attach `source_file` (basename of the file) and `pair_index` (1-based integer, unique within that file) to every returned Q&A pair dict.
- 1.2 `load_all_qa("data/docx")` MUST parse both `Q&A.docx` and `Q&A_2026.docx` and return a combined list where every entry has `source_file` and `pair_index` populated.
- 1.3 Deduplication MUST preserve the `source_file` and `pair_index` of the first occurrence when duplicate questions are found across files.
- 1.4 If a `.docx` file contains no valid Q./A. delimited pairs, the system MUST log a warning and continue loading the other file without crashing.

---

### 2. Exhaustive Semantic Retrieval

**User Story**: As a researcher asking about a broad topic like "twinning" or "phase problem", I want the chatbot to find every relevant Q&A pair in the dataset, not just the top few.

#### Acceptance Criteria

- 2.1 The FAISS search call in `retrieve_and_respond` MUST pass `top_k = len(qa_pairs)` — the full dataset size — on every query.
- 2.2 The system MUST search against both question text and answer text: embeddings MUST be built from `question + " " + answer[:300]` rather than question text alone.
- 2.3 All pairs with cosine similarity score `>= 0.45` (MID_CONFIDENCE) MUST be included in `result["sources"]`.
- 2.4 Pairs with score `0.35 <= score < 0.45` MUST be included in `result["alternatives"]` and MUST NOT appear in `result["sources"]`.
- 2.5 Pairs with score `< 0.35` MUST be discarded entirely and MUST NOT appear in any result field.

---

### 3. Sub-Topic Grouping

**User Story**: As a researcher, I want related answers grouped under clear headings so I can navigate a multi-faceted topic without reading a wall of text.

#### Acceptance Criteria

- 3.1 When `result["sources"]` contains more than 2 entries, the system MUST group them into sub-topic clusters using keyword overlap.
- 3.2 Each group MUST have a non-empty human-readable label (e.g. "Phase Problem — Fundamentals", "Data Collection").
- 3.3 The union of all groups MUST equal `result["sources"]` — no source may be lost or duplicated across groups (partition invariant).
- 3.4 When `result["sources"]` contains 1 or 2 entries, the system MAY use a single group labelled "General".
- 3.5 Groups MUST be ordered by descending maximum confidence score.

---

### 4. Expert-Level Synthesis Without Topic Blending

**User Story**: As a researcher, I want a coherent expert answer, not a raw dump of Q&A pairs, and I never want answers from different topics mixed into the same paragraph.

#### Acceptance Criteria

- 4.1 The LLM (Ollama llama3.2:latest) MUST be called once per topic group, receiving only the sources from that group as context.
- 4.2 The synthesis prompt MUST instruct the LLM to use ONLY the provided knowledge entries and not add outside knowledge.
- 4.3 The final `result["answer"]` MUST contain one `###`-prefixed section per topic group.
- 4.4 Sources from different topic groups MUST NOT be passed together in a single LLM call.
- 4.5 If Ollama is unavailable, the system MUST fall back to a formatted raw Q&A display (existing fallback behavior) without crashing.

---

### 5. Source Citations

**User Story**: As a researcher, I want to see exactly which question and file each answer block came from, formatted as "Q47 from Q&A_2026.docx".

#### Acceptance Criteria

- 5.1 Every entry in `result["sources"]` MUST include a `citation` string formatted as `"Q{pair_index} from {source_file}"`.
- 5.2 Every source that contributes to `result["answer"]` MUST have its citation string present in the answer text.
- 5.3 The Streamlit UI MUST display the citation alongside each answer block, not only inside the expander.
- 5.4 The confidence score MUST be shown next to each citation (e.g. `"Q47 from Q&A_2026.docx (87%)"`).

---

### 6. Out-of-Scope Handling

**User Story**: As a researcher, I want the chatbot to tell me honestly when a topic isn't covered, rather than making something up.

#### Acceptance Criteria

- 6.1 When all cosine scores are `< 0.35`, the system MUST return the exact message: `"This specific topic isn't covered in the current Q&A dataset."` without calling the LLM.
- 6.2 When scores exist in the `0.35–0.45` range but none meet the `0.45` threshold, the system MUST return the out-of-scope message AND populate `result["alternatives"]` with those weak matches.
- 6.3 The LLM MUST NOT be called when `result["sources"]` is empty.
- 6.4 The system MUST NOT add any information not present in the knowledge base to any response.

---

### 7. Embedding Cache Validity

**User Story**: As a developer, I want the embedding cache to automatically rebuild when the knowledge base content changes, so stale embeddings never cause wrong results.

#### Acceptance Criteria

- 7.1 The cache MUST be invalidated and rebuilt when the content hash of all question+answer texts changes, not only when the pair count changes.
- 7.2 If the cache file is corrupt or unreadable, the system MUST silently delete it and rebuild from scratch.
- 7.3 The cache MUST be stored at `database/qa_embeddings.pkl` and MUST include both the content hash and the embeddings array.

---

### 8. Streamlit UI Rendering

**User Story**: As a researcher using the chat interface, I want a clean, navigable response with section headers, citations, and expandable details.

#### Acceptance Criteria

- 8.1 The UI MUST render one section per topic group with a `###` header visible in the chat.
- 8.2 Each section MUST show inline citation lines below the synthesized paragraph.
- 8.3 The sources expander MUST show the full question, answer, confidence, and citation for each contributing source.
- 8.4 The alternatives expander MUST show related questions with their confidence scores.
- 8.5 A spell-correction hint MUST be shown when the query was auto-corrected.
- 8.6 The startup spinner MUST display the total number of Q&A pairs loaded from both files.
