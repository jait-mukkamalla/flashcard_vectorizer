# Flashcard Vectorizer

A local-first study tool: import Anki decks, vectorize the cards, search them semantically, and
explore how your cards connect in a knowledge graph.
See [project.md](project.md) for the full plan. **Milestones 1 (ingest and search), 2 (study
sessions with optional FSRS) and 3 (knowledge graph) are implemented**; weak-spot detection and
answer checking come later.

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
flashcard-vectorizer import deck.apkg [more.apkg ...]   # import, embed, then update the graph
flashcard-vectorizer search "why does osmosis happen"   # --mode hybrid|dense|keyword
flashcard-vectorizer graph [--rebuild]                  # build or update the knowledge graph
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

## Knowledge graph

Open **Knowledge graph →** in the header (or `/graph/`). Every card is a node; each card is linked to
its nearest neighbours, and related cards are grouped into clusters.

* **Overview:** each cluster is a bubble, sized by its cards, labelled with its most characteristic
  words, and joined to the clusters it is most similar to. Bubbles are coloured by cluster, or by deck
  (their dominant deck).
* **Open a cluster:** click a bubble and it opens into its cards and the links among them (zoom in to
  read them). Several clusters can be open at once; **Collapse** or **Collapse all** closes them.
* **Neighbourhood:** click a card to read it and see which cards it is linked to, then
  **Show its neighbourhood** for its 1 or 2 hop surroundings. Or type a topic in the search box to see
  the cards around the best matches, which shows where that topic sits in your library.
* **Decks:** the deck chips filter the overview, the open clusters and neighbourhoods to the decks you pick.
* **Study from the graph:** **Quiz this cluster** and **Quiz this neighbourhood** start a study session
  (the same sessions as before, with FSRS on or off) from those cards. It is also available as
  `POST /api/sessions` with `cluster` or `card_ids`.

The graph is built by the background job after embedding finishes, and kept up to date as decks are
added: only new or changed cards are searched for neighbours, and existing cards just learn about any
new ones that are closer. Clusters and layout are then rebuilt. Progress shows in the status line, and an
interrupted build resumes where it stopped.

## How it works

| Piece | Implementation |
|---|---|
| Import | `.apkg` in every Anki format (legacy schema 11, current zstd schema 18). Card templates are rendered, so *Basic*, *reversed*, *cloze* (one card per cloze number, nested clozes too) and custom note types all become question/answer pairs. HTML, sounds, images and MathJax delimiters are cleaned out of the text; media filenames are recorded. |
| Incremental | Cards are identified by deck + note GUID + card number and hashed on their cleaned text. Re-importing an updated deck adds new cards, updates edited ones, and only re-embeds what changed. Nothing is deleted. |
| Embeddings | `bge-small-en-v1.5` (int8, ONNX Runtime, no PyTorch). Each card gets three vectors: question+answer, question, answer, so a query can match either side. Stored as float16 together with the model id; changing `model` re-embeds. |
| Search | Dense: exact brute-force cosine over an in-memory matrix, behind a small `VectorIndex` interface so an ANN index can replace it. Keyword: SQLite FTS5 (BM25). Hybrid: reciprocal rank fusion of both. Every mode takes an optional deck filter. |
| Reviews | Every answer (from Anki's revlog or from this app) is a row in `reviews`. A card's FSRS state is always rebuilt by replaying its reviews in time order, so Anki history, in-app answers and later re-imports combine consistently. Fuzzing is off so replays are deterministic. |
| Sessions | A session is a saved queue of cards (from hybrid search, or from decks), resumable after a restart. |
| Graph edges | Each card's `graph_k` (8) nearest neighbours by cosine similarity of its combined vector, found exactly by brute force over the in-memory index and stored in SQLite. An edge is a neighbour pair at or above the similarity threshold (default depends on the model, 0.6 for bge-small), undirected. Adding cards updates only what changed (see below). |
| Clusters | [Louvain](https://en.wikipedia.org/wiki/Louvain_method) community detection (written in NumPy: the fast libraries are GPL), then each community is split into its connected parts. Clusters under 3 cards are pooled into one "Loose cards" cluster. Each cluster is named after the deck (at any level of its `A::B` path, the most specific one wins) or, failing that, the tag that at least 60% of its cards share, e.g. "Chemistry". Clusters that share a subject get keywords added ("History: roman"); with no dominant deck or tag the name is the cluster's most characteristic words (TF-IDF style). Nothing is LLM-generated. |
| Layout | Computed on the server and stored, never simulated in the browser: a force layout places cards inside each cluster, and another places the clusters (with overlaps removed). Neighbourhoods are small enough to be laid out per request. |
| Rendering | React + TypeScript + [Sigma.js](https://www.sigmajs.org/) (WebGL) + graphology. The frontend in `frontend/` is prebuilt into the package, so users do not need Node. |
| Jobs | One background worker runs imports, then embedding in committed batches of 256 cards, then the graph update, so it is resumable and never blocks queries. |

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
graph_k = 8                   # neighbours per card in the knowledge graph (1-50)
graph_threshold = 0.6         # minimum cosine similarity for an edge; omit for the model's default
graph_resolution = 1.0        # Louvain resolution: higher = more, smaller clusters
graph_threads = 1             # BLAS threads while building the graph (see below)
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
import speed, index load time, memory and query latency, builds the knowledge graph (and times adding
one more deck and reading the graph back), measures real embedding throughput on a sample, and prints
pass/fail against the targets in `project.md`. `--cores 4` pins the process to four CPUs to
approximate the hardware floor; `--index-dtype float16` measures the lower-memory setting. See the
script's docstring for what is real and what is stood in (the 100k run uses stand-in vectors with
topic structure; query embedding and embedding throughput use the real model).

`tests/make_fixtures.py` regenerates the real Anki exports in `tests/fixtures/` (needs
`pip install anki`); the tests themselves do not need it.

### Frontend

The knowledge graph UI in `frontend/` is React + TypeScript + Sigma.js. The built bundle is committed
under `src/flashcard_vectorizer/static/graph/`, so you only need Node to change the UI:

```bash
cd frontend
npm install
npm run dev        # hot-reloading UI on :5173, proxied to a running `flashcard-vectorizer serve`
npm test           # unit tests for how graphs are built from the API data
npm run build      # type-check, then rebuild the bundle into the Python package
```

## Known limitations

- Media files (images, audio) are not extracted from the package yet, only referenced. The UI shows text.
- FSRS state is computed with py-fsrs's default parameters; it is not optimised per user from the history.
- Semantic search always returns the nearest cards, even for a query unrelated to anything in the
  library, so a "quiz me on X" session for an unknown topic is filled with the closest (weak) matches.
  Use keyword mode if you want an empty result instead.
- A session draws from at most the top ~200 search hits and holds at most 100 cards, so very broad
  queries are truncated to the best matches.
- Grading is self-reported: you choose Again/Hard/Good/Easy. Suggested grades from answer checking
  arrive in milestone 5.
- Anki history is replayed assuming four answer buttons everywhere. Collections last scheduled with
  Anki's long-removed v1 scheduler (three buttons while learning) may be seeded slightly off.
- FSRS fuzzing is disabled so replays are deterministic; due dates are therefore not spread out
  the way Anki spreads them.
- Re-importing never deletes: cards or reviews removed from a deck in Anki stay in the library.
- Memory is at the budget at 100k cards. In the benchmark the app sits at about 750 MB (466 MB of it
  the vector matrix), and the knowledge graph build pushes the process to about 804 MB, and adding a
  3,000-card deck to about 834 MB, so those two steps slightly exceed the 800 MB target (the
  benchmark reports this as a FAIL). It is a short-lived peak in the background job. The
  `index_dtype = "float16"` option halves the vector matrix (by arithmetic, not measured here) and is
  the way to get headroom.
- Importing a large review history takes time: replaying reviews runs at about 5,000 per second
  (1M reviews is roughly 3 minutes), inside the background import.
- Benchmark figures at 100k cards use stand-in vectors (query embedding and embedding throughput are
  measured with the real model), and were taken on one machine, not on dedicated 4-core / 8 GB
  hardware. Pinning to 4 logical CPUs there means two physical cores. Timings vary a lot between runs
  on that machine (the graph build took 150-300 s, and embedding throughput 35-96 cards/s depending on
  what else was running).
- Graph build at 100k cards (4 CPUs, one BLAS thread): about 2.5-5 minutes, nearly all of it the exact
  neighbour search; clustering and layout take 10-15 s. Adding a 3,000-card deck takes about 35 s.
  Reading the graph back (overview, opening a cluster, a neighbourhood) takes under 100 ms.
- BLAS threads: `graph_threads` defaults to 1 because multi-threaded BLAS was unreliable when measured
  (up to 5-10x slower than one thread when CPUs are shared or limited, only ~1.5x faster otherwise).
  Raise it if you know your machine does better.
- Only `.apkg`; CSV/TSV and Quizlet imports are later additions.
- The search and study page is still one static HTML page; only the knowledge graph is the React app.
  The two are separate pages that link to each other.

Knowledge graph:

- Clusters are communities of the neighbour graph, so they follow how densely cards link, not a
  fixed topic list. On a 96-card demo they came out at the level of subjects (chemistry, physics,
  ...) with the finer sub-topics merged; raise `graph_resolution` for smaller clusters. Louvain has a
  resolution limit, and cluster ids and colours are not stable across rebuilds.
- The similarity threshold is a weak filter for short Q&A cards: on the demo, cards in different
  topics were nearly as similar (median 0.63) as cards in the same topic (0.66) for bge-small. What
  separates clusters is link density, so tune `graph_threshold` by looking at the graph if you need to.
  Cards with no neighbour above the threshold, and clusters under 3 cards, land in the "Loose cards"
  bubble.
- Cluster names come from your deck and tag names, so they are only as good as those: a deck called
  "Default" says nothing, and a cluster that mixes decks is named by keywords ("acid, solution, water")
  instead. An LLM-written subject name is a later option.
- Opening a cluster hides its links to other clusters (they would cut across its cards); very large
  clusters show only their first 20,000 cards. Cluster-to-cluster links are drawn for the strongest 4,000
  pairs.
- Neighbours are exact, but a card's neighbour list is only recomputed when it (or a card it pointed to)
  changes. After a bulk re-import that edits many cards, the next build takes longer.
- Changing the embedding model or `graph_k` rebuilds every neighbour list from scratch.
- Weak-spot colouring (retention and lapses on the graph) is milestone 4.
