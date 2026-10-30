# Flashcard Vectorizer

A local-first study tool: import Anki decks, vectorize the cards, and search them semantically.
See [project.md](project.md) for the full plan. **Milestones 1 (ingest and search) and 2 (study
sessions with optional FSRS) are implemented**; the knowledge graph, weak-spot detection and answer
checking come later.

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

## Studying

Type a query and press **Quiz me** to turn the matching cards into a study session ("quiz me on
everything related to osmosis"), optionally scoped to the decks you tick. With an empty query it
quizzes the selected decks (or the whole library). Show the answer with space, grade with 1-4.
Sessions are saved, so you can close the tab and resume them from the **Unfinished sessions** list.

* **Spaced repetition (FSRS) on:** grades are Again / Hard / Good / Easy, every answer updates the
  card's FSRS memory state ([py-fsrs](https://github.com/open-spaced-repetition/py-fsrs)), and
  you can restrict a session to *due & new* cards. Sessions without a query put the most overdue
  cards first. The status line shows how many cards are due.
* **FSRS off:** a plain quiz (Missed / Got it), nothing is scheduled. Answers are still logged, and
  switching FSRS back on folds them into the schedule.
* **Again** puts the card back at the end of the session until you get it.
* **Anki history:** the review history in an `.apkg` is imported and replayed through FSRS to seed
  each card's memory state, so a deck you already studied in Anki is scheduled sensibly from the
  first session. Re-importing never duplicates reviews; re-importing a deck you imported before
  history support existed backfills it. Only real answers count (learn / review / relearn); cram
  sessions in filtered decks and manual reschedules are ignored.

## How it works

| Piece | Implementation |
|---|---|
| Import | `.apkg` in every Anki format (legacy schema 11, current zstd schema 18). Card templates are rendered, so *Basic*, *reversed*, *cloze* (one card per cloze number, nested clozes too) and custom note types all become question/answer pairs. HTML, sounds, images and MathJax delimiters are cleaned out of the text; media filenames are recorded. |
| Incremental | Cards are identified by deck + note GUID + card number and hashed on their cleaned text. Re-importing an updated deck adds new cards, updates edited ones, and only re-embeds what changed. Nothing is deleted. |
| Embeddings | `bge-small-en-v1.5` (int8, ONNX Runtime, no PyTorch). Each card gets three vectors: question+answer, question, answer, so a query can match either side. Stored as float16 together with the model id; changing `model` re-embeds. |
| Search | Dense: exact brute-force cosine over an in-memory matrix, behind a small `VectorIndex` interface so an ANN index can replace it. Keyword: SQLite FTS5 (BM25). Hybrid: reciprocal rank fusion of both. Every mode takes an optional deck filter. |
| Reviews | Every answer (from Anki's revlog or from this app) is a row in `reviews`. A card's FSRS state is always rebuilt by replaying its reviews in time order, so Anki history, in-app answers and later re-imports combine consistently. Fuzzing is off so replays are deterministic. |
| Sessions | A session is a saved queue of cards (from hybrid search, or from decks), resumable after a restart. |
| Jobs | One background worker runs imports then embedding in committed batches of 256 cards, so it is resumable and never blocks queries. |

## Configuration

Optional `config.toml` in the data directory; all keys are optional:

```toml
model = "bge-small-en-v1.5"   # or "all-MiniLM-L6-v2", "bge-base-en-v1.5" (768-d, slower)
max_seq_len = 256             # tokens per text
batch_size = 64
embed_threads = 0             # 0 = available CPUs minus one
index_dtype = "float32"       # "float16" halves the in-memory matrix, slightly slower
fsrs = true                   # default for the in-app FSRS toggle (once you use the toggle, it wins)
desired_retention = 0.9       # FSRS target recall probability, 0.7-0.99
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

## Known limitations

- Media files (images, audio) are not extracted from the package yet, only referenced. The UI shows text.
- FSRS state is computed with py-fsrs's default parameters; it is not optimised per user from the history.
- Semantic search always returns the nearest cards, even for a query unrelated to anything in the
  library, so a "quiz me on X" session for an unknown topic is filled with the closest (weak) matches.
  Use keyword mode if you want an empty result instead.
- Sessions built from a graph cluster arrive with the knowledge graph (milestone 3); sessions are
  built from queries and decks for now.
- Only `.apkg`; CSV/TSV and Quizlet imports are later additions.
- The UI is a single static page. The React/TypeScript frontend arrives with the graph view (milestone 3).
