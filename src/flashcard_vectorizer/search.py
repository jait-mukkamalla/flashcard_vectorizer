"""Dense (vector), keyword (FTS5/BM25) and hybrid (reciprocal rank fusion) search."""
from __future__ import annotations

import re
import sqlite3
import threading
from typing import Iterable, Protocol, Sequence

import numpy as np

KIND_COMBINED, KIND_QUESTION, KIND_ANSWER = 0, 1, 2

RRF_K = 60
CANDIDATES = 100  # per retriever, before fusion


class VectorIndex(Protocol):
    """Nearest-neighbour index over card vectors.

    Brute force is enough for ~100k cards; an ANN index (hnswlib, sqlite-vec) can implement
    the same interface if libraries grow far beyond that.
    """

    def __len__(self) -> int: ...
    def add(self, card_ids: Sequence[int], deck_ids: Sequence[int], kinds: Sequence[int], vecs: np.ndarray) -> None: ...
    def remove(self, card_ids: Iterable[int]) -> None: ...
    def search(self, qvec: np.ndarray, k: int, deck_ids: Sequence[int] | None = None) -> list[tuple[int, float]]: ...


class BruteForceIndex:
    """Exact cosine search: one matrix product over all rows (vectors are unit length).

    Each card owns up to three rows (combined / question / answer); a card scores as its best row.
    """

    _CHUNK = 32768

    def __init__(self, dim: int, dtype: str = "float32", capacity: int = 1024):
        self.dim = dim
        self.dtype = np.dtype(dtype)
        self._n = 0
        capacity = max(capacity, 16)
        self._mat = np.empty((capacity, dim), dtype=self.dtype)
        self._cards = np.empty(capacity, dtype=np.int64)
        self._decks = np.empty(capacity, dtype=np.int64)
        self._kinds = np.empty(capacity, dtype=np.int8)
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return self._n

    @property
    def nbytes(self) -> int:
        return self._mat.nbytes + self._cards.nbytes + self._decks.nbytes + self._kinds.nbytes

    def _reserve(self, extra: int) -> None:
        need = self._n + extra
        cap = len(self._cards)
        if need <= cap:
            return
        while cap < need:
            cap *= 2
        mat = np.empty((cap, self.dim), dtype=self.dtype)
        mat[: self._n] = self._mat[: self._n]
        self._mat = mat
        for name in ("_cards", "_decks", "_kinds"):
            old = getattr(self, name)
            arr = np.empty(cap, dtype=old.dtype)
            arr[: self._n] = old[: self._n]
            setattr(self, name, arr)

    def add(self, card_ids, deck_ids, kinds, vecs) -> None:
        n = len(card_ids)
        if n == 0:
            return
        with self._lock:
            self._reserve(n)
            s = slice(self._n, self._n + n)
            self._mat[s] = vecs
            self._cards[s] = card_ids
            self._decks[s] = deck_ids
            self._kinds[s] = kinds
            self._n += n

    def remove(self, card_ids) -> None:
        ids = np.fromiter(card_ids, dtype=np.int64)
        if len(ids) == 0:
            return
        with self._lock:
            keep = ~np.isin(self._cards[: self._n], ids)
            kept = int(keep.sum())
            self._mat[:kept] = self._mat[: self._n][keep]
            self._cards[:kept] = self._cards[: self._n][keep]
            self._decks[:kept] = self._decks[: self._n][keep]
            self._kinds[:kept] = self._kinds[: self._n][keep]
            self._n = kept

    # -- row access for the knowledge graph (brute-force specific: an ANN index would build its own kNN) --
    def kind_rows(self, kind: int) -> np.ndarray:
        """Row numbers of every vector of one kind (e.g. KIND_COMBINED), in storage order."""
        return np.flatnonzero(self._kinds[: self._n] == kind)

    def card_of_rows(self, rows: np.ndarray) -> np.ndarray:
        return self._cards[rows]

    def rows_f32(self, rows: np.ndarray) -> np.ndarray:
        """Copy of the given rows as float32, ready for a matrix product."""
        return self._mat[rows].astype(np.float32, copy=False)

    def _scores(self, q: np.ndarray) -> np.ndarray:
        n = self._n
        if self.dtype == np.float32:
            return self._mat[:n] @ q.astype(np.float32)
        out = np.empty(n, dtype=np.float32)  # float16 has no fast BLAS path: widen chunk by chunk
        q32 = q.astype(np.float32)
        for s in range(0, n, self._CHUNK):
            out[s : s + self._CHUNK] = self._mat[s : min(s + self._CHUNK, n)].astype(np.float32) @ q32
        return out

    def search(self, qvec, k, deck_ids=None):
        with self._lock:
            n = self._n
            if n == 0 or k <= 0:
                return []
            scores = self._scores(qvec)
            if deck_ids is not None:
                scores[~np.isin(self._decks[:n], np.asarray(list(deck_ids), dtype=np.int64))] = -np.inf
            cards = self._cards[:n]
            rows = min(n, k * 4)
            while True:
                top = np.argpartition(scores, n - rows)[n - rows :] if rows < n else np.arange(n)
                top = top[np.argsort(-scores[top])]
                best: dict[int, float] = {}
                for r in top:
                    s = float(scores[r])
                    if s == -np.inf:
                        break
                    best.setdefault(int(cards[r]), s)  # rows are sorted: first hit is the best
                if len(best) >= k or rows >= n:
                    return list(best.items())[:k]
                rows = min(n, rows * 4)


def load_index(conn: sqlite3.Connection, model_id: str, dim: int, dtype: str = "float32") -> BruteForceIndex:
    """Build the in-memory index from stored vectors that are current for ``model_id``."""
    where = "v.model = ? AND c.embedded_hash = c.content_hash AND c.embedded_model = v.model"
    total = conn.execute(
        f"SELECT count(*) FROM vectors v JOIN cards c ON c.id = v.card_id WHERE {where}", (model_id,)
    ).fetchone()[0]
    index = BruteForceIndex(dim, dtype, capacity=total)  # exact size: no power-of-two slack
    cur = conn.execute(
        f"SELECT v.card_id, c.deck_id, v.kind, v.vec FROM vectors v JOIN cards c ON c.id = v.card_id WHERE {where}",
        (model_id,),
    )
    while rows := cur.fetchmany(20000):
        vecs = np.frombuffer(b"".join(r[3] for r in rows), dtype=np.float16).reshape(len(rows), dim)
        index.add([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], vecs.astype(index.dtype))
    return index


# --------------------------------------------------------------------------- keyword

_STOP = frozenset(
    "a an and are as at be by for from how in is it of on or that the this to was what when where which who why with".split()
)


def fts_match_expr(text: str) -> str | None:
    """Turn free text into a safe FTS5 OR-query of quoted terms (stop words dropped)."""
    terms = re.findall(r"\w+", text.lower())
    kept = [t for t in terms if t not in _STOP] or terms
    kept = list(dict.fromkeys(kept))[:32]
    return " OR ".join(f'"{t}"' for t in kept) if kept else None


def keyword_search(
    conn: sqlite3.Connection, text: str, k: int = CANDIDATES, deck_ids: Sequence[int] | None = None
) -> list[tuple[int, float]]:
    expr = fts_match_expr(text)
    if expr is None:
        return []
    sql = "SELECT f.rowid, bm25(cards_fts, 2.0, 1.0, 0.5) AS s FROM cards_fts f"
    args: list = [expr]
    if deck_ids is not None:
        marks = ",".join("?" * len(deck_ids))
        sql += f" JOIN cards c ON c.id = f.rowid WHERE cards_fts MATCH ? AND c.deck_id IN ({marks})"
        args += list(deck_ids)
    else:
        sql += " WHERE cards_fts MATCH ?"
    sql += " ORDER BY s LIMIT ?"
    args.append(k)
    return [(r[0], -r[1]) for r in conn.execute(sql, args)]  # higher = better


# --------------------------------------------------------------------------- fusion

def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion of several best-first lists of ids."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            fused[id_] = fused.get(id_, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])
