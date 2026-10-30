# Flashcard Vectorizer

A local-first study tool: import Anki decks, vectorize the cards, and search them semantically.
See [project.md](project.md) for the full plan. **Milestone 1 (ingest and search) is implemented**;
study sessions, the knowledge graph, weak-spot detection and answer checking come later.

## Install and run

```bash
pip install -e .            # Python 3.11+; pipx works too
flashcard-vectorizer serve  # http://127.0.0.1:8765
```

Open the page, click **Import .apkg**, and start searching straight away: keyword search works as
soon as the import finishes, and semantic search turns on as the background embedding job catches
up (progress is shown at the top; the job survives restarts and resumes where it stopped).

The first embedding run downloads the model (~35 MB, int8 ONNX, from Hugging Face). That is the
only network access the app makes. Everything else stays on your machine.

Command line equivalents:

```bash
flashcard-vectorizer import deck.apkg [more.apkg ...]   # import, then embed
flashcard-vectorizer search "why does osmosis happen"   # --mode hybrid|dense|keyword
flashcard-vectorizer status
```

Data lives in `~/.flashcard_vectorizer` (`$FV_HOME`, or `--data-dir`): one SQLite file,
`library.db`, plus the downloaded `models/`.

## How it works

| Piece | Implementation |
|---|---|
| Import | `.apkg` in every Anki format (legacy schema 11, current zstd schema 18). Card templates are rendered, so *Basic*, *reversed*, *cloze* (one card per cloze number, nested clozes too) and custom note types all become question/answer pairs. HTML, sounds, images and MathJax delimiters are cleaned out of the text; media filenames are recorded. |
| Incremental | Cards are identified by deck + note GUID + card number and hashed on their cleaned text. Re-importing an updated deck adds new cards, updates edited ones, and only re-embeds what changed. Nothing is deleted. |
| Embeddings | `bge-small-en-v1.5` (int8, ONNX Runtime, no PyTorch). Each card gets three vectors: question+answer, question, answer, so a query can match either side. Stored as float16 together with the model id; changing `model` re-embeds. |
| Search | Dense: exact brute-force cosine over an in-memory matrix, behind a small `VectorIndex` interface so an ANN index can replace it. Keyword: SQLite FTS5 (BM25). Hybrid: reciprocal rank fusion of both. Every mode takes an optional deck filter. |
| Jobs | One background worker runs imports then embedding in committed batches of 256 cards, so it is resumable and never blocks queries. |

## Configuration

Optional `config.toml` in the data directory; all keys are optional:

```toml
model = "bge-small-en-v1.5"   # or "all-MiniLM-L6-v2", "bge-base-en-v1.5" (768-d, slower)
max_seq_len = 256             # tokens per text
batch_size = 64
embed_threads = 0             # 0 = available CPUs minus one
index_dtype = "float32"       # "float16" halves the in-memory matrix, slightly slower
host = "127.0.0.1"
port = 8765
```

## Tests and benchmarks

```bash
pip install -e ".[dev]"
pytest                                   # offline: uses a hash embedder, no model download
python benchmarks/bench.py --cards 100000 --cores 4   # needs: pip install psutil
```

`benchmarks/bench.py` builds a synthetic library (`flashcard_vectorizer.synthetic`), measures
import speed, index load time, memory and query latency, measures real embedding throughput on a
sample, and prints pass/fail against the targets in `project.md`. `--cores 4` pins the process to
four CPUs to approximate the hardware floor. See the script's docstring for what is real and what
is stood in (the 100k run uses random vectors; query embedding and embedding throughput use the
real model).

`tests/make_fixtures.py` regenerates the real Anki exports in `tests/fixtures/` (needs
`pip install anki`); the tests themselves do not need it.

## Known limitations (milestone 1)

- Media files (images, audio) are not extracted from the package yet, only referenced. The UI shows text.
- Review history in the package is not read yet (milestone 2, FSRS seeding).
- Only `.apkg`; CSV/TSV and Quizlet imports are later additions.
- The UI is a single static page. The React/TypeScript frontend arrives with the graph view (milestone 3).
