# Flashcard Vectorizer

An open-source, local-first study tool. Import Anki decks, vectorize the cards, and study/review them through semantic search and a knowledge graph.

## Project Objectives
- Take an individual's flashcards and vectorize them so that semantic querying can be used for studying
- Provide a knowledge graph where an individual can observe the connections between the concepts they have studied
- Support optional spaced-repetition review (FSRS) and query-driven study sessions
- Stay lightweight enough to run on modest hardware (no GPU required), since users' hardware is unknown

## Specifications
- Very fast query runtime (hard ceiling 5 seconds; target < 1 second at 100k cards on a 4-core CPU)
- Responsive UI for the knowledge graph feature
- Scale target: a library of up to ~100,000 cards total, accumulated across many decks over time (individual decks are expected to be much smaller, typically a few thousand cards)
- Import: embedding a few thousand cards should feel quick; a full 100k library embedded in < ~30 minutes on CPU. Imports are background, resumable, and incremental
- Memory: < ~800 MB at 100k cards
- Hardware floor: works on a 4-core laptop with 8 GB RAM and no GPU
- Local-first and private by default; any API-based option is clearly opt-in

## Architecture

### Form factor
Local-first web app. Python (FastAPI) backend, React/TypeScript frontend. The frontend is prebuilt and bundled so end users do not need Node. Distribution target: `pip`/`pipx` install, with Docker as a secondary option.

### Storage
- Single SQLite database per user. Cards from all decks live together with a `deck` column, so search, study sessions, and the graph work across decks or scoped to one or more decks
- Vectors stored as float16 blobs and loaded into memory as a NumPy matrix
- Each vector records the embedding model name/version that produced it; changing models triggers re-embedding
- Content hash per card so re-importing an updated deck only embeds new or changed cards

### Embeddings
- Local model, default small (e.g. `bge-small-en-v1.5` or `all-MiniLM-L6-v2`, 384 dims), run via ONNX Runtime with int8 quantization for fast CPU inference and no heavy PyTorch dependency
- Embed question+answer combined, and also each side separately, so queries can match either
- Larger models selectable in config (requires re-embedding)
- Cards must be cleaned before embedding: strip HTML, handle cloze deletions (`{{c1::...}}`), handle multiple Anki note types

### Search
- Dense vector search: brute-force cosine similarity over the in-memory matrix (~20-60 ms at 100k). Hidden behind a small interface so an ANN index (hnswlib / sqlite-vec) can be swapped in if libraries grow far beyond target
- Keyword search: SQLite FTS5 (BM25)
- Hybrid: merge both result lists with reciprocal rank fusion. Helps with exact terms (names, formulas, acronyms) where embeddings are weak
- Optional deck filters on every query
- Keyword search is usable immediately after import while vectors are still being computed

### Knowledge graph
- Nodes are cards; edges are k-nearest-neighbor links above a similarity threshold (k about 5-10), precomputed at import time and stored, updated incrementally as decks are added
- Community detection (Leiden/Louvain) provides clusters and cluster colors
- Hierarchical view for responsiveness at scale: show clusters as super-nodes, expand into cards on click; deck-level coloring/filtering available
- Neighborhood mode: show the 1-2 hop neighborhood of a selected card or query result
- Layout precomputed and cached (server-side or in a web worker), not simulated live over the full graph
- WebGL rendering (Sigma.js + graphology)
- Phase 2 (optional): LLM-extracted explicit concepts/relations for labeled edges

### Study features
- **Semantic study sessions:** build a study session from a query result or cluster ("quiz me on everything related to X"), optionally scoped to decks
- **FSRS (togglable):** uses `py-fsrs`. When off, plain review with no scheduling. On Anki import, replay existing review history to seed each card's FSRS memory state
- **Weak-spot detection:** color graph nodes/clusters by retention or lapse rate to show weak regions of the knowledge graph (depends on FSRS data and the graph)

### Answer checking (tiered, LLM optional)
The grader only *suggests* a grade; the user confirms or overrides it, because FSRS depends on honest ratings. Grading has its own, more relaxed time budget than search.
1. Embedding similarity between the user's answer and the reference (free; settles clear matches/misses)
2. Small NLI/cross-encoder model (~100M params, ONNX) for the middle band
3. Optional LLM judge, configured as a plain endpoint (Ollama, llama.cpp, or an API key). A 3-4B model is recommended over 1B, which is unreliable on partial credit. Only borderline answers reach this tier

Everything works with tier 3 disabled; tiers 1-2 are the default.

## Out of scope for now
- Duplicate / near-duplicate card detection
- Gap suggestions (finding sparse areas of the graph)

## Import
- Anki `.apkg` (zip containing SQLite) is the primary source; decks are added incrementally to the existing library
- Handle HTML, cloze deletions, media references, multiple note types
- Import review history for FSRS seeding
- Other formats (CSV/TSV, Quizlet export) are later additions

## Milestones
1. **Ingest and search:** `.apkg` import, cleaning, embedding job (background/resumable), hybrid search, minimal UI. Measure query latency here against the targets
2. **Study sessions:** query-driven sessions and the FSRS toggle with history seeding
3. **Knowledge graph:** kNN edges, clustering, hierarchical/neighborhood views in Sigma.js
4. **Weak-spot detection:** retention/lapse coloring on the graph
5. **Answer checking:** tiered grader, with the optional LLM judge

## Open-source considerations
- Permissive license (MIT or Apache-2.0); verify model and dependency licenses (MiniLM/bge are permissive; `py-fsrs` is MIT)
- Small config file for model choices, LLM endpoint, and the FSRS toggle
- Synthetic dataset generator (up to 100k cards across many decks) plus a benchmark script, so performance claims are verifiable and regressions are caught
- Explicitly test against the hardware floor (4 cores, 8 GB RAM, no GPU)
