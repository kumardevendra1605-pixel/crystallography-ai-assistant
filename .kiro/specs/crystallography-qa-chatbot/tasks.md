# Tasks: Crystallography Q&A Chatbot

## Task List

- [x] 1. Extend qa_parser.py with source tracking metadata
  - [x] 1.1 Add `source_file` (basename) and `pair_index` (1-based, per-file) fields to every dict returned by `parse_qa_file`
  - [x] 1.2 Update `load_all_qa` to pass `filepath` into `parse_qa_file` so metadata is attached before merging
  - [x] 1.3 Ensure deduplication preserves the first occurrence's `source_file` and `pair_index`
  - [x] 1.4 Add graceful error handling: log warning and skip file if no valid Q./A. pairs found

- [x] 2. Extend embeddings.py to embed question + answer text
  - [x] 2.1 Change `build_and_cache_embeddings` to embed `question + " " + answer[:300]` per pair instead of question only
  - [x] 2.2 Replace count-based cache invalidation with a content hash (SHA256 of all question+answer texts)
  - [x] 2.3 Add corrupt-cache recovery: catch pickle errors, delete cache file, rebuild from scratch

- [x] 3. Create src/topic_grouper.py
  - [x] 3.1 Implement `group_by_subtopic(sources, query) -> OrderedDict[str, list[dict]]`
  - [x] 3.2 Use keyword overlap (shared crystallography terms) as primary clustering signal
  - [x] 3.3 Return single "General" group when sources list has ≤ 2 entries
  - [x] 3.4 Derive human-readable group labels from dominant keywords in each cluster
  - [x] 3.5 Order groups by descending maximum confidence score

- [x] 4. Extend rag_pipeline.py with grouping, per-group synthesis, and citations
  - [x] 4.1 Add `citation` field (`"Q{pair_index} from {source_file}"`) to every SourceEntry before returning
  - [x] 4.2 Call `group_by_subtopic` after threshold filtering and add `topic_groups` to the response dict
  - [x] 4.3 Replace single `_build_answer` call with per-group synthesis loop (one LLM call per group)
  - [x] 4.4 Update synthesis prompt to include citation header per source block and instruct no cross-group blending
  - [x] 4.5 Assemble final `answer` by joining per-group sections with `\n\n---\n\n` separator
  - [x] 4.6 Update out-of-scope path to return exact message: `"This specific topic isn't covered in the current Q&A dataset."`

- [x] 5. Update app.py UI rendering
  - [x] 5.1 Update `render_result` to iterate `result["topic_groups"]` and render one section per group with `###` header
  - [x] 5.2 Render inline citation lines (`Q{n} from {file} ({pct}%)`) below each group's synthesized paragraph
  - [x] 5.3 Update sources expander to show `citation` field alongside question, answer, and confidence
  - [x] 5.4 Update startup spinner caption to show total pair count from both files

- [x] 6. Write tests
  - [x] 6.1 Add `test_qa_parser.py`: assert `source_file` and `pair_index` present on all pairs; test deduplication
  - [x] 6.2 Add `test_topic_grouper.py`: assert partition invariant; test ≤2 sources edge case; test label non-empty
  - [x] 6.3 Extend `test_pipeline.py`: mock FAISS; assert out-of-scope path never calls LLM; assert citation in answer
