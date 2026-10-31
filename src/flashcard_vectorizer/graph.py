"""The knowledge graph: nodes are cards, edges are nearest-neighbour links, clusters are communities.

Building happens in two stages, both run by the background job after embedding:

1. **Neighbours** (incremental). Every card gets its ``graph_k`` nearest neighbours by cosine similarity
   of its combined question+answer vector, found by exact brute force over the in-memory index.
   Only new or changed cards are searched; cards already in the graph just learn about new
   neighbours, so adding a deck costs ``new x library`` instead of ``library x library``. Work is
   committed block by block, so an interrupted build resumes where it stopped.
2. **Clusters and layout.** Edges are the neighbour pairs at or above the similarity threshold,
   undirected. Louvain finds communities, tiny ones are pooled into one "loose cards" cluster, and a
   force layout positions the cards inside each cluster and the clusters relative to each other.
   Positions, labels and cluster links are stored, so the browser only draws.

The threshold is applied when clusters are built, not when neighbours are stored, so changing it
(or the resolution) re-clusters without recomputing neighbours.
"""
from __future__ import annotations

import math
import re
import time
from array import array
from collections import Counter
from typing import Callable, Iterator, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from .layout import force_layout, separate_circles
from .louvain import louvain
from .search import KIND_COMBINED, _STOP

# Tall blocks matter: BLAS is slow on products with few rows (measured: 13 vs 300 GFLOPS at 256 vs 1024 rows).
BLOCK = 512  # cards whose neighbours are searched together; also the unit of resuming
CHUNK = 2048  # candidate rows per matrix product (keeps the score matrix and its sort under ~12 MB)
MERGE_CHUNK = 4096  # existing cards compared with a block of new ones at a time
FETCH = 50_000  # rows pulled from SQLite at a time (each becomes a Python tuple, ~150 bytes)
MIN_CLUSTER = 3  # communities smaller than this are pooled into the "loose" cluster
CARD_EPOCHS = 120
CLUSTER_EPOCHS = 250
LABEL_WORDS = 3
LOOSE_LABEL = "Loose cards"
SUBJECT_SHARE = 0.6  # a deck or tag names a cluster if at least this share of its cards belong to it
SUBJECT_WORDS = 2  # keywords added to a subject name when several clusters share it
# Deck/tag names that say nothing about the subject (Anki's defaults, bookkeeping tags).
_GENERIC = frozenset(
    "default all anki deck decks my cards card flashcards unsorted imported new misc other leech marked "
    "suspended edited".split()
)

_LABEL_STOP = _STOP | frozenset(
    """about after all also any been before being both can could did does done each else even every few
    give given had has have her here him his its just like list many may might more most much must name
    not now one only other our out over per same she should some such than that their them then there
    these they those through too two under use used uses using very was were will would you your
    called define defined definition describe difference different example examples explain following
    meaning means mean type types term terms known main two three""".split()
)
_WORD = re.compile(r"[a-z][a-z']{2,}")

Progress = Callable[[str, int, int], None]


def _chunks(seq: Sequence, n: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def topk_rows(index, rows: np.ndarray, qpos: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` most similar rows (excluding the query itself) for each query position.

    ``rows`` maps a position to a row of ``index``; ``qpos`` are the positions to look up. Returns
    (similarities, positions), best first; missing slots are -inf / -1.
    """
    n, m = len(rows), len(qpos)
    queries = index.rows_f32(rows[qpos])
    best_s = np.full((m, k), -np.inf, dtype=np.float32)
    best_i = np.full((m, k), -1, dtype=np.int64)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        sc = queries @ index.rows_f32(rows[s:e]).T
        own = np.flatnonzero((qpos >= s) & (qpos < e))
        sc[own, qpos[own] - s] = -np.inf
        width = e - s
        kk = min(k, width)
        part = np.argpartition(sc, width - kk, axis=1)[:, width - kk :]
        cs = np.concatenate([best_s, np.take_along_axis(sc, part, 1)], axis=1)
        ci = np.concatenate([best_i, part + s], axis=1)
        keep = np.argpartition(cs, cs.shape[1] - k, axis=1)[:, -k:]
        best_s, best_i = np.take_along_axis(cs, keep, 1), np.take_along_axis(ci, keep, 1)
    order = np.argsort(-best_s, axis=1)
    best_s, best_i = np.take_along_axis(best_s, order, 1), np.take_along_axis(best_i, order, 1)
    best_i[~np.isfinite(best_s)] = -1
    return best_s, best_i


class GraphService:
    def __init__(self, lib):
        self.lib = lib

    # -- parameters and bookkeeping --------------------------------------------------------
    @property
    def k(self) -> int:
        return self.lib.cfg.graph_k

    @property
    def threshold(self) -> float:
        t = self.lib.cfg.graph_threshold
        return self.lib.spec.graph_threshold if t is None else t

    def _sig(self) -> str:
        return f"{self.lib.model_id}|k={self.k}|thr={self.threshold}|res={self.lib.cfg.graph_resolution}"

    def _setting(self, key: str) -> str | None:
        row = self.lib.conn().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _put_setting(self, key: str, value: str) -> None:
        self.lib.conn().execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def status(self) -> dict:
        # The UI polls this every second or so, so it reads the few hundred cluster rows, not every card.
        clusters, nodes = self.lib.conn().execute("SELECT count(*), coalesce(sum(size), 0) FROM graph_clusters").fetchone()
        return {"ready": clusters > 0, "nodes": nodes, "clusters": clusters, "k": self.k, "threshold": self.threshold}

    def needs_build(self) -> bool:
        """True if embedded cards are missing from the graph, changed, or the settings changed."""
        c = self.lib.conn()
        embedded = c.execute(
            "SELECT count(*) FROM cards WHERE embedded_hash = content_hash AND embedded_model = ?",
            (self.lib.model_id,),
        ).fetchone()[0]
        if embedded == 0:
            return False
        current = c.execute(
            "SELECT count(*) FROM cards c JOIN graph_nodes g ON g.card_id = c.id "
            "WHERE g.hash = c.embedded_hash AND c.embedded_hash = c.content_hash AND c.embedded_model = ? "
            "AND g.cluster_id IS NOT NULL",
            (self.lib.model_id,),
        ).fetchone()[0]
        return current != embedded or self._setting("graph_sig") != self._sig()

    def reset(self) -> None:
        """Forget the whole graph; the next build starts from scratch."""
        c = self.lib.conn()
        with c:
            c.execute("DELETE FROM knn")
            c.execute("DELETE FROM graph_nodes")
            c.execute("DELETE FROM graph_clusters")
            c.execute("DELETE FROM settings WHERE key IN ('graph_sig', 'graph_knn_sig')")

    # -- building --------------------------------------------------------------------------
    def build(self, should_stop: Callable[[], bool] = lambda: False, progress: Progress | None = None) -> dict:
        """Bring the graph up to date with the embedded cards. Safe to interrupt and re-run."""
        t0 = time.perf_counter()
        progress = progress or (lambda *_: None)
        with threadpool_limits(limits=self.lib.cfg.graph_threads, user_api="blas"):
            changed, finished = self._update_neighbours(should_stop, progress)
        if not finished:
            return {"complete": False}
        timings = {"neighbours_s": round(time.perf_counter() - t0, 2)}
        if changed or self._setting("graph_sig") != self._sig():
            t = time.perf_counter()
            self._cluster(progress)
            timings["clusters_s"] = round(time.perf_counter() - t, 2)
            with self.lib.conn():
                self._put_setting("graph_sig", self._sig())
        return {"complete": True, **self.status(), **timings}

    def _update_neighbours(self, should_stop, progress: Progress) -> tuple[bool, bool]:
        """Returns (anything changed, ran to the end)."""
        lib, c, k = self.lib, self.lib.conn(), self.k
        index = lib.index()
        knn_sig = f"{lib.model_id}|{k}"
        if self._setting("graph_knn_sig") != knn_sig:  # different model or k: every stored list is wrong
            with c:
                c.execute("DELETE FROM knn")
                c.execute("DELETE FROM graph_nodes")
                c.execute("DELETE FROM graph_clusters")
                self._put_setting("graph_knn_sig", knn_sig)

        # Which cards are in the graph is decided in SQL, so no per-card Python objects pile up (at 100k
        # cards a dict of hashes alone is tens of MB, and this process is near its memory budget).
        current = np.fromiter(
            (r[0] for r in c.execute(
                "SELECT id FROM cards WHERE embedded_hash = content_hash AND embedded_model = ?", (lib.model_id,)
            )),
            dtype=np.int64,
        )
        rows = index.kind_rows(KIND_COMBINED)
        ids = index.card_of_rows(rows)
        usable = np.isin(ids, current)
        rows, ids = rows[usable], ids[usable]
        n = len(ids)
        del current, usable

        changed = self._invalidate()
        done = np.isin(
            ids,
            np.fromiter(
                (r[0] for r in c.execute(
                    "SELECT g.card_id FROM graph_nodes g JOIN cards c ON c.id = g.card_id "
                    "WHERE g.hash = c.embedded_hash AND c.embedded_hash = c.content_hash AND c.embedded_model = ?",
                    (lib.model_id,),
                )),
                dtype=np.int64,
            ),
        )
        todo = np.flatnonzero(~done)
        changed = changed or len(todo) > 0
        if changed:  # until clustering finishes again, the stored clusters no longer match the neighbours
            with c:
                c.execute("DELETE FROM settings WHERE key = 'graph_sig'")
        if len(todo) == 0:
            return changed, True

        order = np.argsort(ids)
        sorted_ids = ids[order]

        def pos_of(card_ids: np.ndarray) -> np.ndarray:
            p = np.minimum(np.searchsorted(sorted_ids, card_ids), n - 1)
            return np.where(sorted_ids[p] == card_ids, order[p], -1)

        clean = np.flatnonzero(done)
        nbr = np.full((n, k), -1, dtype=np.int32)
        sim = np.full((n, k), -np.inf, dtype=np.float32)
        if len(clean):
            self._load_lists(pos_of, nbr, sim)

        for s in range(0, len(todo), BLOCK):
            if should_stop():
                return changed, False
            blk = todo[s : s + BLOCK]
            blk_sim, blk_nbr = topk_rows(index, rows, blk, k)
            touched = self._merge_into(index, rows, clean, blk, nbr, sim, k) if len(clean) else np.zeros(0, np.int64)
            with c:
                self._write_lists(ids, blk, blk_nbr, blk_sim)
                if len(touched):
                    self._write_lists(ids, touched, nbr[touched], sim[touched])
                c.executemany(
                    "INSERT INTO graph_nodes(card_id, hash) SELECT id, embedded_hash FROM cards WHERE id = ? "
                    "ON CONFLICT(card_id) DO UPDATE SET hash = excluded.hash",
                    [(i,) for i in ids[blk].tolist()],
                )
            progress("neighbours", s + len(blk), len(todo))
        return changed, True

    def _invalidate(self) -> bool:
        """Drop graph data of cards that vanished or changed, and mark cards that pointed at them for a redo."""
        c = self.lib.conn()
        gone = [
            r[0]
            for r in c.execute(
                "SELECT g.card_id FROM graph_nodes g LEFT JOIN cards c ON c.id = g.card_id "
                "WHERE c.id IS NULL OR NOT (c.embedded_hash = c.content_hash AND c.embedded_model = ?) "
                "OR (g.hash IS NOT NULL AND g.hash IS NOT c.embedded_hash)",
                (self.lib.model_id,),
            )
        ]
        for batch in _chunks(gone, 400):
            marks = ",".join("?" * len(batch))
            with c:
                # cards whose neighbour list contained a dropped card need a fresh search
                c.execute(
                    f"UPDATE graph_nodes SET hash = NULL WHERE card_id IN "
                    f"(SELECT src FROM knn WHERE dst IN ({marks}))",
                    batch,
                )
                c.execute(f"DELETE FROM knn WHERE src IN ({marks}) OR dst IN ({marks})", [*batch, *batch])
                c.execute(f"DELETE FROM graph_nodes WHERE card_id IN ({marks})", batch)
        return bool(gone)

    def _load_lists(self, pos_of, nbr: np.ndarray, sim: np.ndarray) -> None:
        """Fill ``nbr``/``sim`` (positions, similarities) from the stored neighbour lists, a chunk at a time."""
        n, k = nbr.shape
        filled = np.zeros(n, dtype=np.int32)  # neighbours already placed in each card's list
        # The table is read in primary-key order, so each card's rows are contiguous (a run may straddle chunks).
        cur = self.lib.conn().execute("SELECT src, dst, sim FROM knn ORDER BY src")
        while chunk := cur.fetchmany(FETCH):
            arr = np.array(chunk, dtype=np.float64)
            src, dst = pos_of(arr[:, 0].astype(np.int64)), pos_of(arr[:, 1].astype(np.int64))
            ok = (src >= 0) & (dst >= 0)
            src, dst, s = src[ok], dst[ok], arr[ok, 2].astype(np.float32)
            if not len(src):
                continue
            at = np.arange(len(src))
            run_start = np.maximum.accumulate(np.where(np.r_[True, src[1:] != src[:-1]], at, 0))
            slot = filled[src] + (at - run_start)
            fit = slot < k
            nbr[src[fit], slot[fit]] = dst[fit]
            sim[src[fit], slot[fit]] = s[fit]
            filled += np.bincount(src, minlength=n).astype(np.int32)

    def _merge_into(self, index, rows, clean, blk, nbr, sim, k) -> np.ndarray:
        """Offer the cards in ``blk`` as neighbours to every clean card; returns the positions that changed."""
        n = len(rows)
        queries = index.rows_f32(rows[blk])
        col = np.full(n, -1, dtype=np.int64)
        col[blk] = np.arange(len(blk))
        touched = []
        for chunk in _chunks(clean, MERGE_CHUNK):
            sc = index.rows_f32(rows[chunk]) @ queries.T
            cur_n, cur_s = nbr[chunk], sim[chunk]
            seen = np.where(cur_n >= 0, col[np.maximum(cur_n, 0)], -1)  # candidates already in a list
            r, j = np.nonzero(seen >= 0)
            sc[r, seen[r, j]] = -np.inf
            upd = np.flatnonzero(sc.max(axis=1) > cur_s.min(axis=1))
            if len(upd) == 0:
                continue
            cs = np.concatenate([cur_s[upd], sc[upd]], axis=1)
            ci = np.concatenate([cur_n[upd], np.broadcast_to(blk, (len(upd), len(blk)))], axis=1)
            keep = np.argpartition(cs, cs.shape[1] - k, axis=1)[:, -k:]
            new_s, new_i = np.take_along_axis(cs, keep, 1), np.take_along_axis(ci, keep, 1)
            new_i[~np.isfinite(new_s)] = -1
            nbr[chunk[upd]], sim[chunk[upd]] = new_i, new_s
            touched.append(chunk[upd])
        return np.concatenate(touched) if touched else np.zeros(0, dtype=np.int64)

    def _write_lists(self, ids: np.ndarray, positions: np.ndarray, nbrs: np.ndarray, sims: np.ndarray) -> None:
        c = self.lib.conn()
        owners = ids[positions]
        for batch in _chunks(owners.tolist(), 400):
            c.execute(f"DELETE FROM knn WHERE src IN ({','.join('?' * len(batch))})", batch)
        valid = nbrs >= 0
        src = np.repeat(owners, nbrs.shape[1]).reshape(nbrs.shape)[valid]
        c.executemany(
            "INSERT INTO knn(src, dst, sim) VALUES (?, ?, ?)",
            zip(src.tolist(), ids[nbrs[valid]].tolist(), sims[valid].astype(float).tolist()),
        )

    # -- clusters and layout ---------------------------------------------------------------
    def _cluster(self, progress: Progress) -> None:
        c = self.lib.conn()
        ids = np.array(
            [r[0] for r in c.execute("SELECT card_id FROM graph_nodes WHERE hash IS NOT NULL ORDER BY card_id")],
            dtype=np.int64,
        )
        n = len(ids)
        if n == 0:
            with c:
                c.execute("DELETE FROM graph_clusters")
            return
        progress("clustering", 0, 1)
        ea, eb, ew = self._edges(ids)
        labels = louvain(n, ea, eb, ew, self.lib.cfg.graph_resolution)
        cluster, count, loose = self._order_clusters(labels)
        sizes = np.bincount(cluster, minlength=count)

        progress("layout", 0, 1)
        intra = cluster[ea] == cluster[eb]
        local = force_layout(n, ea[intra], eb[intra], ew[intra], cluster, CARD_EPOCHS)
        links = self._cluster_links(cluster, ea, eb, ew, loose)
        centres, radii = self._place_clusters(sizes, links, loose)

        del ea, eb, ew, intra, labels  # not needed again; labelling is the next memory peak
        labels_text, deck_counts = self._describe(ids, cluster, sizes, loose)
        with c:
            c.execute("DELETE FROM graph_clusters")  # cascades to decks and links
            c.executemany(
                "INSERT INTO graph_clusters(id, size, label, x, y, radius, loose) VALUES (?,?,?,?,?,?,?)",
                [
                    (i, int(sizes[i]), labels_text[i], float(centres[i, 0]), float(centres[i, 1]), float(radii[i]),
                     int(i == loose))
                    for i in range(count)
                ],
            )
            c.executemany("INSERT INTO graph_cluster_decks(cluster_id, deck_id, n) VALUES (?,?,?)", deck_counts)
            c.executemany("INSERT INTO graph_links(a, b, edges, weight) VALUES (?,?,?,?)", links)
            c.executemany(
                "UPDATE graph_nodes SET cluster_id=?, x=?, y=? WHERE card_id=?",
                zip(cluster.tolist(), local[:, 0].astype(float).tolist(), local[:, 1].astype(float).tolist(), ids.tolist()),
            )
        progress("layout", 1, 1)

    def _edges(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Undirected card-level edges (positions in ``ids``) at or above the threshold, each once."""
        n = len(ids)
        # A pair that is each other's neighbour is stored twice; keep one row per pair.
        cur = self.lib.conn().execute(
            "SELECT k.src, k.dst, k.sim FROM knn k WHERE k.sim >= ?1 AND (k.src < k.dst OR NOT EXISTS "
            "(SELECT 1 FROM knn r WHERE r.src = k.dst AND r.dst = k.src AND r.sim >= ?1))",
            (self.threshold,),
        )
        ea, eb, ew = [], [], []
        while chunk := cur.fetchmany(FETCH):
            arr = np.array(chunk, dtype=np.float64)
            src, dst = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
            a = np.minimum(np.searchsorted(ids, src), n - 1)
            b = np.minimum(np.searchsorted(ids, dst), n - 1)
            ok = (ids[a] == src) & (ids[b] == dst) & (a != b)  # both cards are in the graph
            ea.append(np.minimum(a, b)[ok].astype(np.int32))
            eb.append(np.maximum(a, b)[ok].astype(np.int32))
            ew.append(arr[ok, 2])
        if not ea:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0)
        return np.concatenate(ea), np.concatenate(eb), np.maximum(np.concatenate(ew), 1e-6)

    @staticmethod
    def _order_clusters(labels: np.ndarray) -> tuple[np.ndarray, int, int | None]:
        """Number clusters biggest first; pool the tiny ones. Returns (cluster per node, count, loose id)."""
        n = len(labels)
        sizes = np.bincount(labels)
        first = np.full(len(sizes), n, dtype=np.int64)
        np.minimum.at(first, labels, np.arange(n))
        real = np.flatnonzero(sizes >= MIN_CLUSTER)
        real = real[np.lexsort((first[real], -sizes[real]))]
        remap = np.full(len(sizes), -1, dtype=np.int64)
        remap[real] = np.arange(len(real))
        cluster = remap[labels]
        loose = None
        if (cluster < 0).any():
            loose = len(real)
            cluster[cluster < 0] = loose
        return cluster, len(real) + (loose is not None), loose

    @staticmethod
    def _cluster_links(cluster, ea, eb, ew, loose) -> list[tuple[int, int, int, float]]:
        ca, cb = cluster[ea], cluster[eb]
        cross = ca != cb
        if loose is not None:
            cross &= (ca != loose) & (cb != loose)
        lo, hi = np.minimum(ca[cross], cb[cross]), np.maximum(ca[cross], cb[cross])
        if not len(lo):
            return []
        width = int(hi.max()) + 1
        keys, inv = np.unique(lo * width + hi, return_inverse=True)
        edges = np.bincount(inv)
        weight = np.bincount(inv, weights=ew[cross])
        return [
            (int(k // width), int(k % width), int(e), float(w)) for k, e, w in zip(keys, edges, weight)
        ]

    @staticmethod
    def _place_clusters(sizes, links, loose) -> tuple[np.ndarray, np.ndarray]:
        """Cluster centres and radii in world units (a card occupies about one unit of area)."""
        count = len(sizes)
        radii = np.sqrt(sizes.astype(np.float64)) * 1.1 + 1.0
        if count == 1:
            return np.zeros((1, 2)), radii
        la = np.array([l[0] for l in links], dtype=np.int64)
        lb = np.array([l[1] for l in links], dtype=np.int64)
        lw = np.array([math.log1p(l[2]) for l in links])
        unit = force_layout(count, la, lb, lw, np.zeros(count, dtype=np.int64), CLUSTER_EPOCHS)
        scale = math.sqrt(float((radii**2).sum())) * 1.5
        return separate_circles(unit * scale, radii, pad=radii.max() * 0.05), radii

    def _describe(self, ids, cluster, sizes, loose) -> tuple[list[str], list[tuple[int, int, int]]]:
        """Label each cluster with its most characteristic words; count cards per (cluster, deck)."""
        count, n = len(sizes), len(ids)
        vocab: dict[str, int] = {}
        pair_cluster, pair_word = array("i"), array("i")  # one entry per (card, distinct word)
        deck_n: Counter[tuple[int, int]] = Counter()
        tag_n: Counter[tuple[int, str]] = Counter()
        cur = self.lib.conn().execute("SELECT id, deck_id, question, tags FROM cards")
        while chunk := cur.fetchmany(5_000):
            card_ids = np.fromiter((r[0] for r in chunk), dtype=np.int64, count=len(chunk))
            at = np.minimum(np.searchsorted(ids, card_ids), n - 1)
            member = (ids[at] == card_ids).tolist()
            owner = cluster[at].tolist()
            for (_, deck, question, tags), ok, ci in zip(chunk, member, owner):
                if not ok:
                    continue
                deck_n[(ci, deck)] += 1
                for tag in set(tags.split()):
                    tag_n[(ci, tag)] += 1
                for w in {w for w in _WORD.findall(f"{question} {tags}".lower()) if w not in _LABEL_STOP}:
                    pair_cluster.append(ci)
                    pair_word.append(vocab.setdefault(w, len(vocab)))
        labels = [f"Cluster {ci + 1}" for ci in range(count)]
        if loose is not None:
            labels[loose] = LOOSE_LABEL
        if vocab:
            v = len(vocab)
            words = list(vocab)  # insertion order is id order
            pc = np.frombuffer(pair_cluster, dtype=np.int32)
            pw = np.frombuffer(pair_word, dtype=np.int32)
            df = np.bincount(pw, minlength=v)  # cards containing each word
            # Cards in each cluster containing each word: sort the (cluster, word) keys in place and count
            # runs, which needs far less memory than np.unique at ~1M pairs.
            key = pc.astype(np.int64)
            key *= v
            key += pw
            key.sort()
            first = np.concatenate([[0], np.flatnonzero(key[1:] != key[:-1]) + 1])
            tf = np.diff(np.concatenate([first, [len(key)]]))
            key = key[first]
            kc, kw = key // v, key % v
            score = tf * np.log1p(n / df[kw])
            keep = tf >= np.where(sizes[kc] >= 6, 2, 1)
            kc, kw, score = kc[keep], kw[keep], score[keep]
            best = np.lexsort((-score, kc))  # by cluster, best score first
            first = np.searchsorted(kc[best], np.arange(count))
            last = np.searchsorted(kc[best], np.arange(count), side="right")
            for ci in range(count):
                if ci != loose and last[ci] > first[ci]:
                    picked = best[first[ci] : min(last[ci], first[ci] + LABEL_WORDS)]
                    labels[ci] = ", ".join(words[w] for w in kw[picked])
        # Prefer a subject name taken from the user's own decks and tags; keywords describe it or stand in.
        subjects = self._subjects(deck_n, tag_n, sizes, loose)
        taken = Counter(x for x in subjects if x)
        for ci, subject in enumerate(subjects):
            if subject is None:
                continue
            if taken[subject] > 1 and labels[ci] != f"Cluster {ci + 1}":  # told apart by keywords
                labels[ci] = f"{subject}: {', '.join(labels[ci].split(', ')[:SUBJECT_WORDS])}"
            else:
                labels[ci] = subject
        return labels, [(ci, deck, m) for (ci, deck), m in sorted(deck_n.items())]

    def _subjects(self, deck_n, tag_n, sizes, loose) -> list[str | None]:
        """Per cluster, the deck (any level of its path) or tag that most of its cards share, if any.

        The most specific deck level covering at least SUBJECT_SHARE of the cluster wins, so a cluster
        inside "Science::Chemistry" is "Chemistry"; tags are tried only when no deck qualifies.
        """
        names = dict(self.lib.conn().execute("SELECT id, name FROM decks"))
        by_path: Counter[tuple[int, tuple[str, ...]]] = Counter()
        for (ci, deck), m in deck_n.items():
            parts = tuple(x.strip() for x in names.get(deck, "").split("::") if x.strip())
            for depth in range(1, len(parts) + 1):
                by_path[(ci, parts[:depth])] += m
        found: list[str | None] = [None] * len(sizes)
        depth_of = [0] * len(sizes)
        for (ci, path), m in by_path.items():
            name = path[-1]
            if ci == loose or name.lower() in _GENERIC or m < SUBJECT_SHARE * sizes[ci]:
                continue
            if len(path) > depth_of[ci]:
                found[ci], depth_of[ci] = name, len(path)
        best_tag: dict[int, tuple[int, str]] = {}
        for (ci, tag), m in tag_n.items():
            name = tag.split("::")[-1].replace("_", " ").strip()
            if found[ci] is None and ci != loose and name.lower() not in _GENERIC and m >= SUBJECT_SHARE * sizes[ci]:
                if m > best_tag.get(ci, (0, ""))[0]:
                    best_tag[ci] = (m, name)
        for ci, (_, name) in best_tag.items():
            found[ci] = name
        return found

    # -- reading ---------------------------------------------------------------------------
    def overview(self, decks: Sequence[int] | None = None, max_links: int = 4000) -> dict:
        """Clusters as super-nodes plus the strongest links between them."""
        c = self.lib.conn()
        clusters = c.execute("SELECT id, size, label, x, y, radius, loose FROM graph_clusters ORDER BY id").fetchall()
        per_deck: dict[int, list[tuple[int, int]]] = {}
        for cid, deck, m in c.execute("SELECT cluster_id, deck_id, n FROM graph_cluster_decks"):
            if decks is None or deck in decks:
                per_deck.setdefault(cid, []).append((deck, m))
        out = []
        for r in clusters:
            mix = sorted(per_deck.get(r["id"], []), key=lambda d: -d[1])
            size = sum(m for _, m in mix)
            if size == 0:
                continue
            out.append({
                "id": r["id"], "label": r["label"], "size": size, "total": r["size"], "x": r["x"], "y": r["y"],
                "radius": r["radius"], "loose": bool(r["loose"]), "decks": [list(d) for d in mix[:6]],
            })
        visible = {o["id"] for o in out}
        links = [
            {"a": a, "b": b, "edges": e, "weight": w}
            for a, b, e, w in c.execute("SELECT a, b, edges, weight FROM graph_links ORDER BY weight DESC LIMIT ?", (max_links,))
            if a in visible and b in visible
        ]
        return {**self.status(), "clusters": out, "links": links, "cards": sum(o["size"] for o in out)}

    def cluster(self, cluster_id: int, decks: Sequence[int] | None = None, limit: int = 20000) -> dict:
        """The cards of one cluster, positioned in world coordinates, with the edges among them."""
        c = self.lib.conn()
        row = c.execute("SELECT * FROM graph_clusters WHERE id = ?", (cluster_id,)).fetchone()
        if row is None:
            raise LookupError("no such cluster")
        sql = (
            "SELECT g.card_id, g.x, g.y, c.deck_id, c.question FROM graph_nodes g JOIN cards c ON c.id = g.card_id "
            "WHERE g.cluster_id = ?"
        )
        args: list = [cluster_id]
        if decks:
            sql += f" AND c.deck_id IN ({','.join('?' * len(decks))})"
            args += list(decks)
        rows = c.execute(sql + " ORDER BY g.card_id LIMIT ?", [*args, limit + 1]).fetchall()
        truncated = len(rows) > limit
        rows = rows[:limit]
        nodes = [
            {"id": r[0], "x": row["x"] + r[1] * row["radius"], "y": row["y"] + r[2] * row["radius"], "deck": r[3],
             "label": r[4][:140]}
            for r in rows
        ]
        keep = {n["id"] for n in nodes}
        edges: dict[tuple[int, int], float] = {}
        for a, b, s in c.execute(
            "SELECT k.src, k.dst, k.sim FROM knn k JOIN graph_nodes a ON a.card_id = k.src "
            "JOIN graph_nodes b ON b.card_id = k.dst WHERE a.cluster_id = ? AND b.cluster_id = ? AND k.sim >= ?",
            (cluster_id, cluster_id, self.threshold),
        ):
            if a in keep and b in keep:
                edges[(min(a, b), max(a, b))] = s
        return {
            "id": cluster_id, "label": row["label"], "size": row["size"], "truncated": truncated, "nodes": nodes,
            "edges": [{"a": a, "b": b, "sim": s} for (a, b), s in edges.items()],
        }

    def cluster_card_ids(self, cluster_id: int, decks: Sequence[int] | None = None) -> list[int]:
        c = self.lib.conn()
        if c.execute("SELECT 1 FROM graph_clusters WHERE id = ?", (cluster_id,)).fetchone() is None:
            raise LookupError("no such cluster")
        sql = "SELECT g.card_id FROM graph_nodes g JOIN cards c ON c.id = g.card_id WHERE g.cluster_id = ?"
        args: list = [cluster_id]
        if decks:
            sql += f" AND c.deck_id IN ({','.join('?' * len(decks))})"
            args += list(decks)
        return [r[0] for r in c.execute(sql, args)]

    def neighbourhood(
        self, seeds: Sequence[int], hops: int = 1, limit: int = 150, decks: Sequence[int] | None = None
    ) -> dict:
        """Cards within ``hops`` edges of the seed cards (strongest links first, at most ``limit`` cards)."""
        c = self.lib.conn()
        seeds = list(dict.fromkeys(seeds))
        found = set()
        for batch in _chunks(seeds, 400):
            found |= {r[0] for r in c.execute(f"SELECT id FROM cards WHERE id IN ({','.join('?' * len(batch))})", batch)}
        seeds = [s for s in seeds if s in found][:limit]
        if not seeds:
            raise LookupError("no such card")
        hop = {s: 0 for s in seeds}
        frontier = list(seeds)
        thr = self.threshold
        for depth in range(1, hops + 1):
            if not frontier or len(hop) >= limit:
                break
            best: dict[int, float] = {}
            for batch in _chunks(frontier, 400):
                marks = ",".join("?" * len(batch))
                for src, dst, s in c.execute(
                    f"SELECT src, dst, sim FROM knn WHERE (src IN ({marks}) OR dst IN ({marks})) AND sim >= ?",
                    [*batch, *batch, thr],
                ):
                    other = dst if src in hop else src
                    if other not in hop and s > best.get(other, -1.0):
                        best[other] = s
            if decks:
                allowed = set()
                for batch in _chunks(list(best), 400):
                    marks = ",".join("?" * len(batch))
                    allowed |= {
                        r[0] for r in c.execute(
                            f"SELECT id FROM cards WHERE id IN ({marks}) AND deck_id IN ({','.join('?' * len(decks))})",
                            [*batch, *decks],
                        )
                    }
                best = {i: s for i, s in best.items() if i in allowed}
            added = sorted(best, key=lambda i: -best[i])[: limit - len(hop)]
            hop.update({i: depth for i in added})
            frontier = added

        members = sorted(hop)
        edges: dict[tuple[int, int], float] = {}
        for batch in _chunks(members, 400):
            for src, dst, s in c.execute(
                f"SELECT src, dst, sim FROM knn WHERE src IN ({','.join('?' * len(batch))}) AND sim >= ?", [*batch, thr]
            ):
                if dst in hop:
                    edges[(min(src, dst), max(src, dst))] = s

        index = {cid: i for i, cid in enumerate(members)}
        ea = np.array([index[a] for a, _ in edges], dtype=np.int64)
        eb = np.array([index[b] for _, b in edges], dtype=np.int64)
        unit = force_layout(len(members), ea, eb, np.array(list(edges.values())) - thr + 0.05,
                            np.zeros(len(members), dtype=np.int64), epochs=150)
        info: dict[int, tuple] = {}
        for batch in _chunks(members, 400):
            for r in c.execute(
                "SELECT c.id, c.deck_id, c.question, g.cluster_id FROM cards c "
                f"LEFT JOIN graph_nodes g ON g.card_id = c.id WHERE c.id IN ({','.join('?' * len(batch))})", batch
            ):
                info[r[0]] = (r[1], r[2][:140], r[3])
        radius = 6.0 * math.sqrt(len(members))
        return {
            "seeds": seeds,
            "nodes": [
                {"id": cid, "x": float(unit[i, 0]) * radius, "y": float(unit[i, 1]) * radius, "hop": hop[cid],
                 "deck": info[cid][0], "label": info[cid][1], "cluster": info[cid][2]}
                for cid, i in index.items()
            ],
            "edges": [{"a": a, "b": b, "sim": s} for (a, b), s in edges.items()],
        }
