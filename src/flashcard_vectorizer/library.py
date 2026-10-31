"""The library: database + embedder + in-memory index + background jobs, behind one object."""
from __future__ import annotations

import queue
import sqlite3
import threading
import time
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np

from .cleaning import card_text
from .config import Config
from .db import connect, init_db
from .embedder import HASH_MODEL, MODELS, ModelSpec, load_embedder
from .graph import GraphService
from .importer import ImportResult, import_apkg
from .search import (
    CANDIDATES,
    KIND_ANSWER,
    KIND_COMBINED,
    KIND_QUESTION,
    BruteForceIndex,
    keyword_search,
    load_index,
    rrf,
)
from .study import StudyService

EMBED_CHUNK = 256  # cards per committed batch; also the granularity of resuming


def resolve_spec(cfg: Config) -> ModelSpec:
    if cfg.model == HASH_MODEL.name:
        return HASH_MODEL
    if cfg.model not in MODELS:
        raise ValueError(f"unknown model {cfg.model!r}; choose one of {sorted(MODELS) + [HASH_MODEL.name]}")
    return MODELS[cfg.model]


class Library:
    def __init__(self, cfg: Config, embedder=None, log=print):
        self.cfg = cfg
        self.log = log
        self.spec = resolve_spec(cfg)
        init_db(cfg.db_path)
        self._local = threading.local()
        self._embedder = embedder
        self._embedder_lock = threading.Lock()
        self._index: BruteForceIndex | None = None
        self.index_lock = threading.RLock()  # serialises "commit vectors + update index" vs loading
        self.jobs = JobManager(self)
        self.study = StudyService(self)
        self.graph = GraphService(self)

    # -- plumbing --------------------------------------------------------------------------
    def conn(self) -> sqlite3.Connection:
        """Per-thread connection."""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = connect(self.cfg.db_path)
        return c

    @property
    def embedder(self):
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = load_embedder(self.cfg, self.log)
            return self._embedder

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def index(self) -> BruteForceIndex:
        with self.index_lock:
            if self._index is None:
                t = time.perf_counter()
                self._index = load_index(self.conn(), self.model_id, self.spec.dim, self.cfg.index_dtype)
                self.log(f"index loaded: {len(self._index)} vectors in {time.perf_counter() - t:.2f}s")
            return self._index

    def close(self) -> None:
        self.jobs.stop()

    # -- stats -----------------------------------------------------------------------------
    def stats(self) -> dict:
        c = self.conn()
        cards = c.execute("SELECT count(*) FROM cards").fetchone()[0]
        embedded = c.execute(
            "SELECT count(*) FROM cards WHERE embedded_hash = content_hash AND embedded_model = ?",
            (self.model_id,),
        ).fetchone()[0]
        fsrs = self.study.fsrs_enabled()
        due = new = None  # only meaningful with FSRS on
        if fsrs:
            due = c.execute("SELECT count(*) FROM fsrs_state WHERE due <= ?", (int(time.time() * 1000),)).fetchone()[0]
            new = cards - c.execute("SELECT count(*) FROM fsrs_state").fetchone()[0]
        return {
            "fsrs": fsrs,
            "due": due,  # reviewed cards whose next review is due
            "new": new,  # never-reviewed cards
            "cards": cards,
            "decks": c.execute("SELECT count(*) FROM decks").fetchone()[0],
            "embedded": embedded,
            "pending": cards - embedded,
            "model": self.model_id,
            "index_rows": len(self._index) if self._index is not None else 0,
            "graph": self.graph.status(),
            "job": self.jobs.state(),
        }

    def decks(self) -> list[dict]:
        rows = self.conn().execute(
            "SELECT d.id, d.name, count(c.id) AS cards, "
            "coalesce(sum(c.embedded_hash = c.content_hash AND c.embedded_model = ?), 0) AS embedded "
            "FROM decks d LEFT JOIN cards c ON c.deck_id = d.id GROUP BY d.id ORDER BY d.name",
            (self.model_id,),
        )
        return [dict(r) for r in rows]

    # -- embedding -------------------------------------------------------------------------
    def pending_ids(self) -> list[int]:
        rows = self.conn().execute(
            "SELECT id FROM cards WHERE embedded_hash IS NOT content_hash OR embedded_model IS NOT ? ORDER BY id",
            (self.model_id,),
        )
        return [r[0] for r in rows]

    def embed_pending(self, should_stop=lambda: False, progress=None) -> int:
        """Embed every card whose vectors are missing or stale. Safe to interrupt and re-run."""
        ids = self.pending_ids()
        if not ids:
            return 0
        embedder, index, conn = self.embedder, self.index(), self.conn()
        done = 0
        for start in range(0, len(ids), EMBED_CHUNK):
            if should_stop():
                break
            chunk = ids[start : start + EMBED_CHUNK]
            marks = ",".join("?" * len(chunk))
            cards = conn.execute(
                f"SELECT id, deck_id, question, answer FROM cards WHERE id IN ({marks})", chunk
            ).fetchall()
            plan: list[tuple[int, int, str]] = []  # (card id, kind, text)
            for c in cards:
                q, a, both = card_text(c["question"], c["answer"])
                plan.append((c["id"], KIND_COMBINED, both))
                plan.append((c["id"], KIND_QUESTION, q))
                if a:
                    plan.append((c["id"], KIND_ANSWER, a))
            unique = list(dict.fromkeys(t for _, _, t in plan))
            vecs = embedder.encode(unique, batch_size=self.cfg.batch_size)
            by_text = dict(zip(unique, vecs))
            mat = np.stack([by_text[t] for _, _, t in plan])
            blobs = mat.astype(np.float16)
            deck_of = {c["id"]: c["deck_id"] for c in cards}
            with self.index_lock:
                conn.execute(f"DELETE FROM vectors WHERE card_id IN ({marks})", chunk)
                conn.executemany(
                    "INSERT INTO vectors(card_id, kind, model, vec) VALUES (?,?,?,?)",
                    [(cid, kind, self.model_id, blobs[i].tobytes()) for i, (cid, kind, _) in enumerate(plan)],
                )
                conn.execute(
                    f"UPDATE cards SET embedded_hash = content_hash, embedded_model = ? WHERE id IN ({marks})",
                    [self.model_id, *chunk],
                )
                conn.commit()
                index.remove(chunk)
                index.add(
                    [cid for cid, _, _ in plan],
                    [deck_of[cid] for cid, _, _ in plan],
                    [k for _, k, _ in plan],
                    mat.astype(index.dtype),
                )
            done += len(chunk)
            if progress:
                progress(done, len(ids))
        return done

    # -- search ----------------------------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid", decks: Sequence[int] | None = None, limit: int = 20) -> dict:
        if mode not in ("hybrid", "dense", "keyword"):
            raise ValueError(f"unknown search mode {mode!r}")
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        dense: list[tuple[int, float]] = []
        kw: list[tuple[int, float]] = []
        dense_ready = False
        if mode in ("hybrid", "dense") and query.strip():
            index = self.index()
            dense_ready = len(index) > 0
            if dense_ready:
                t = time.perf_counter()
                qvec = self.embedder.encode([query], is_query=True)[0]
                timings["embed_ms"] = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                dense = index.search(qvec, CANDIDATES, decks)
                timings["dense_ms"] = (time.perf_counter() - t) * 1000
        if mode in ("hybrid", "keyword"):
            t = time.perf_counter()
            kw = keyword_search(self.conn(), query, CANDIDATES, decks)
            timings["keyword_ms"] = (time.perf_counter() - t) * 1000

        if mode == "dense":
            order = dense
        elif mode == "keyword":
            order = kw
        else:
            order = rrf([[i for i, _ in dense], [i for i, _ in kw]])
        order = order[:limit]
        dense_score = dict(dense)
        kw_rank = {i: r + 1 for r, (i, _) in enumerate(kw)}
        dense_rank = {i: r + 1 for r, (i, _) in enumerate(dense)}

        t = time.perf_counter()
        results = self._hydrate([i for i, _ in order])
        for r, (i, score) in zip(results, order):
            r["score"] = score
            r["dense_score"] = dense_score.get(i)
            r["dense_rank"] = dense_rank.get(i)
            r["keyword_rank"] = kw_rank.get(i)
        timings["hydrate_ms"] = (time.perf_counter() - t) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return {
            "query": query,
            "mode": mode,
            "dense_ready": dense_ready,
            "results": results,
            "timings": {k: round(v, 2) for k, v in timings.items()},
        }

    def _hydrate(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn().execute(
            "SELECT c.id, d.name AS deck, c.question, c.answer, c.tags, c.notetype "
            f"FROM cards c JOIN decks d ON d.id = c.deck_id WHERE c.id IN ({marks})",
            ids,
        )
        by_id = {r["id"]: dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]


class JobManager:
    """Single background worker: imports and embedding run one at a time, never blocking requests."""

    def __init__(self, lib: Library):
        self.lib = lib
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state: dict = {"phase": "idle", "import": None, "embed": None, "graph": None, "error": None}
        self._embed_queued = False
        self._graph_queued = False
        self._thread: threading.Thread | None = None

    def state(self) -> dict:
        with self._lock:
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self._state.items()}

    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="fv-jobs", daemon=True)
            self._thread.start()

    def submit_import(self, path: str | Path, name: str | None = None, delete_after: bool = False) -> None:
        self._ensure_thread()
        self._q.put(("import", Path(path), name or Path(path).name, delete_after))
        self._set(phase="queued" if self._state["phase"] == "idle" else self._state["phase"])

    def submit_embed(self) -> None:
        with self._lock:
            if self._embed_queued:
                return
            self._embed_queued = True
        self._ensure_thread()
        self._q.put(("embed",))

    def submit_graph(self) -> None:
        with self._lock:
            if self._graph_queued:
                return
            self._graph_queued = True
        self._ensure_thread()
        self._q.put(("graph",))

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the queue drains (used by the CLI and tests)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                idle = self._q.empty() and self._state["phase"] in ("idle", "error")
            if idle:
                return True
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._q.put(("quit",))
            self._thread.join(timeout=30)

    def _run(self) -> None:
        while not self._stop.is_set():
            task = self._q.get()
            if task[0] == "quit":
                return
            try:
                if task[0] == "import":
                    self._do_import(*task[1:])
                elif task[0] == "embed":
                    with self._lock:
                        self._embed_queued = False
                    self._do_embed()
                elif task[0] == "graph":
                    with self._lock:
                        self._graph_queued = False
                    self._do_graph()
            except Exception as e:  # report to the UI instead of dying silently
                self.lib.log(traceback.format_exc())
                self._set(phase="error", error=f"{type(e).__name__}: {e}")
            else:
                with self._lock:
                    if self._q.empty() and self._state["phase"] != "error":
                        self._state["phase"] = "idle"

    def _do_import(self, path: Path, name: str, delete_after: bool) -> None:
        self._set(phase="importing", error=None, **{"import": {"file": name, "cards": 0}})
        try:
            result: ImportResult = import_apkg(
                self.lib.conn(),
                path,
                lambda n: self._set(**{"import": {"file": name, "cards": n}}),
                self.lib.cfg.desired_retention,
            )
        finally:
            if delete_after:
                path.unlink(missing_ok=True)
        self._set(
            **{
                "import": {
                    "file": name,
                    "cards": result.total,
                    "added": result.added,
                    "updated": result.updated,
                    "unchanged": result.unchanged + result.refreshed,
                    "reviews": result.reviews,
                    "decks": sorted(result.decks),
                    "done": True,
                }
            }
        )
        self.submit_embed()

    def _do_embed(self) -> None:
        self._set(phase="embedding", error=None, embed={"done": 0, "total": len(self.lib.pending_ids())})
        self.lib.embed_pending(
            should_stop=self._stop.is_set,
            progress=lambda d, t: self._set(embed={"done": d, "total": t}),
        )
        self._set(embed={"done": 0, "total": 0})
        if not self._stop.is_set():
            self.submit_graph()

    def _do_graph(self) -> None:
        self._set(phase="graphing", error=None, graph={"stage": "neighbours", "done": 0, "total": 0})
        try:
            self.lib.graph.build(
                should_stop=self._stop.is_set,
                progress=lambda stage, d, t: self._set(graph={"stage": stage, "done": d, "total": t}),
            )
        finally:
            self._set(graph=None)
