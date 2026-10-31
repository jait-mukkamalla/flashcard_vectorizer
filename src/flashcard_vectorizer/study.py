"""Study sessions: a persisted queue of cards built from a search query and/or decks.

With FSRS on, every answer is logged and updates the card's schedule. With it off the session is a
plain quiz: answers are still logged (cheap, and they become history if FSRS is switched on later)
but nothing is scheduled.
"""
from __future__ import annotations

import json
import random
import time
from typing import Sequence

from . import scheduling

POOL = 200  # most search hits considered when building a session
MAX_SIZE = 100
SCOPES = ("all", "due")


def now_ms() -> int:
    return int(time.time() * 1000)


class StudyService:
    def __init__(self, lib):
        self.lib = lib

    def conn(self):
        return self.lib.conn()

    # -- FSRS toggle -----------------------------------------------------------------------
    def fsrs_enabled(self) -> bool:
        row = self.conn().execute("SELECT value FROM settings WHERE key='fsrs'").fetchone()
        return self.lib.cfg.fsrs if row is None else row[0] == "1"

    def set_fsrs(self, enabled: bool) -> dict:
        c = self.conn()
        c.execute(
            "INSERT INTO settings(key, value) VALUES ('fsrs', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("1" if enabled else "0",),
        )
        c.commit()
        # Reviews logged while scheduling was off were never folded into state: catch up now.
        caught_up = scheduling.refresh_stale(c, self.lib.cfg.desired_retention) if enabled else 0
        return {"fsrs": enabled, "caught_up": caught_up}

    def settings(self) -> dict:
        return {"fsrs": self.fsrs_enabled(), "desired_retention": self.lib.cfg.desired_retention}

    # -- building sessions -----------------------------------------------------------------
    def create_session(
        self,
        query: str | None = None,
        mode: str = "hybrid",
        decks: Sequence[int] | None = None,
        size: int = 20,
        scope: str = "all",
        title: str | None = None,
        cluster: int | None = None,
        card_ids: Sequence[int] | None = None,
    ) -> dict:
        """Start a session from a query ("quiz me on X"), optionally scoped to decks.

        Without a query it quizzes the given decks (or the whole library). With FSRS on, queryless
        sessions are ordered most-overdue first, then new cards; with it off they are shuffled.
        ``scope='due'`` keeps only new cards and cards that are due.

        Instead of a query, a session can be built from a knowledge-graph ``cluster`` (scoped to
        ``decks`` if given) or from an explicit list of ``card_ids`` (e.g. a neighbourhood). Those
        are ordered like a queryless session. At most one of query, cluster and card_ids may be set.
        """
        query = (query or "").strip()
        decks = list(decks) if decks else None
        card_ids = list(dict.fromkeys(card_ids)) if card_ids else None
        if sum(x is not None for x in (query or None, cluster, card_ids)) > 1:
            raise ValueError("give at most one of query, cluster and cards")
        if scope not in SCOPES:
            raise ValueError(f"unknown scope {scope!r}; choose one of {list(SCOPES)}")
        if not 1 <= size <= MAX_SIZE:
            raise ValueError(f"size must be between 1 and {MAX_SIZE}")
        fsrs = self.fsrs_enabled()
        if scope == "due" and not fsrs:
            raise ValueError("scope 'due' needs spaced repetition (FSRS) to be turned on")
        now = now_ms()

        if query:
            pool = [r["id"] for r in self.lib.search(query, mode, decks, POOL)["results"]]
            if scope == "due":
                due = self._due_among(pool, now)
                pool = [i for i in pool if i in due]
            ids = pool[:size]
        elif cluster is not None:
            ids = self._pick(self.lib.graph.cluster_card_ids(cluster, decks), size, scope == "due", fsrs, now)
            title = title or f"Cluster: {self._cluster_label(cluster)}"
        elif card_ids:
            ids = self._pick(card_ids, size, scope == "due", fsrs, now)
            title = title or "Selected cards"
        else:
            ids = self._deck_cards(decks, size, scope == "due", fsrs, now)
        if not ids:
            raise ValueError("no cards matched" + (" (none are due)" if scope == "due" else ""))

        c = self.conn()
        with c:
            cur = c.execute(
                "INSERT INTO sessions(title, query, mode, decks, scope, created) VALUES (?,?,?,?,?,?)",
                (title or query or "Deck review", query or None, mode if query else None,
                 json.dumps(decks) if decks else None, scope, now),
            )
            sid = cur.lastrowid
            c.executemany(
                "INSERT INTO session_cards(session_id, position, card_id) VALUES (?,?,?)",
                [(sid, pos, cid) for pos, cid in enumerate(ids)],
            )
        return self.get_session(sid)

    def _due_among(self, ids: list[int], now: int) -> set[int]:
        if not ids:
            return set()
        marks = ",".join("?" * len(ids))
        rows = self.conn().execute(
            f"SELECT c.id FROM cards c LEFT JOIN fsrs_state s ON s.card_id = c.id "
            f"WHERE c.id IN ({marks}) AND (s.card_id IS NULL OR s.due <= ?)",
            [*ids, now],
        )
        return {r[0] for r in rows}

    def _cluster_label(self, cluster: int) -> str:
        row = self.conn().execute("SELECT label FROM graph_clusters WHERE id=?", (cluster,)).fetchone()
        return row[0] if row else str(cluster)

    def _pick(self, candidates: list[int], size: int, due_only: bool, fsrs: bool, now: int) -> list[int]:
        """The first ``size`` of ``candidates`` in study order: overdue first with FSRS, else shuffled."""
        found: list[tuple[int, int | None]] = []  # (card id, due)
        for start in range(0, len(candidates), 500):
            batch = candidates[start : start + 500]
            sql = (
                "SELECT c.id, s.due FROM cards c LEFT JOIN fsrs_state s ON s.card_id = c.id "
                f"WHERE c.id IN ({','.join('?' * len(batch))})"
            )
            for cid, due in self.conn().execute(sql, batch):
                if not due_only or due is None or due <= now:
                    found.append((cid, due))
        if fsrs:
            found.sort(key=lambda r: (r[1] is None, r[1] or 0, r[0]))  # due cards by date, then new cards
        else:
            random.shuffle(found)
        return [cid for cid, _ in found[:size]]

    def _deck_cards(self, decks, size: int, due_only: bool, fsrs: bool, now: int) -> list[int]:
        where, args = [], []
        if decks:
            where.append(f"c.deck_id IN ({','.join('?' * len(decks))})")
            args += decks
        if due_only:
            where.append("(s.card_id IS NULL OR s.due <= ?)")
            args.append(now)
        order = "s.card_id IS NULL, s.due, c.id" if fsrs else "random()"
        sql = (
            "SELECT c.id FROM cards c LEFT JOIN fsrs_state s ON s.card_id = c.id"
            + (" WHERE " + " AND ".join(where) if where else "")
            + f" ORDER BY {order} LIMIT ?"
        )
        return [r[0] for r in self.conn().execute(sql, [*args, size])]

    # -- running a session -----------------------------------------------------------------
    def get_session(self, sid: int) -> dict:
        c = self.conn()
        s = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if s is None:
            raise LookupError("no such session")
        rows = c.execute(
            "SELECT rating, count(*) FROM session_cards WHERE session_id=? GROUP BY rating", (sid,)
        ).fetchall()
        ratings = {r[0]: r[1] for r in rows}
        remaining = ratings.pop(None, 0)
        answered = sum(ratings.values())
        passed = sum(n for rating, n in ratings.items() if rating >= 2)  # Anki's definition of "recalled"
        total = c.execute("SELECT count(DISTINCT card_id) FROM session_cards WHERE session_id=?", (sid,)).fetchone()[0]
        return {
            "id": s["id"],
            "title": s["title"],
            "query": s["query"],
            "mode": s["mode"],
            "decks": json.loads(s["decks"]) if s["decks"] else None,
            "scope": s["scope"],
            "created": s["created"],
            "cards": total,
            "answered": answered,
            "remaining": remaining,
            "ratings": {str(k): ratings.get(k, 0) for k in (1, 2, 3, 4)},
            "recall_rate": round(passed / answered, 3) if answered else None,
            "done": remaining == 0,
        }

    def list_sessions(self, limit: int = 50) -> list[dict]:
        ids = [r[0] for r in self.conn().execute("SELECT id FROM sessions ORDER BY id DESC LIMIT ?", (limit,))]
        return [self.get_session(i) for i in ids]

    def delete_session(self, sid: int) -> None:
        c = self.conn()
        with c:
            if c.execute("DELETE FROM sessions WHERE id=?", (sid,)).rowcount == 0:
                raise LookupError("no such session")

    def next_card(self, sid: int) -> dict:
        summary = self.get_session(sid)
        row = self.conn().execute(
            "SELECT c.id, d.name AS deck, c.question, c.answer, c.tags, c.notetype "
            "FROM session_cards sc JOIN cards c ON c.id = sc.card_id JOIN decks d ON d.id = c.deck_id "
            "WHERE sc.session_id=? AND sc.rating IS NULL ORDER BY sc.position LIMIT 1",
            (sid,),
        ).fetchone()
        card = None
        if row is not None:
            card = dict(row)
            card["schedule"] = self._schedule(card["id"]) if self.fsrs_enabled() else None
        return {"session": summary, "card": card}

    def _schedule(self, card_id: int) -> dict:
        row = self.conn().execute("SELECT * FROM fsrs_state WHERE card_id=?", (card_id,)).fetchone()
        return scheduling.describe(row, scheduling.make_scheduler(self.lib.cfg.desired_retention))

    def answer(self, sid: int, card_id: int, rating: int, duration_ms: int = 0) -> dict:
        """Grade the next pending occurrence of ``card_id``. 'Again' puts the card back at the end."""
        if rating not in (1, 2, 3, 4):
            raise ValueError("rating must be 1 (again), 2 (hard), 3 (good) or 4 (easy)")
        c = self.conn()
        entry = c.execute(
            "SELECT position FROM session_cards WHERE session_id=? AND card_id=? AND rating IS NULL "
            "ORDER BY position LIMIT 1",
            (sid, card_id),
        ).fetchone()
        if entry is None:
            raise LookupError("that card is not waiting to be answered in this session")
        fsrs = self.fsrs_enabled()
        duration_ms = min(max(duration_ms, 0), 600_000)
        ts = now_ms()
        with c:
            c.execute(
                "UPDATE session_cards SET rating=? WHERE session_id=? AND position=?", (rating, sid, entry[0])
            )
            if rating == 1:
                c.execute(
                    "INSERT INTO session_cards(session_id, position, card_id) "
                    "SELECT ?, max(position) + 1, ? FROM session_cards WHERE session_id=?",
                    (sid, card_id, sid),
                )
            while not c.execute(  # (card, ts) is unique; nudge in the unlikely event of a same-ms clash
                "INSERT OR IGNORE INTO reviews(card_id, ts, rating, duration_ms, source) VALUES (?,?,?,?,'app')",
                (card_id, ts, rating, duration_ms),
            ).rowcount:
                ts += 1
            if fsrs:
                scheduling.rebuild(c, [card_id], self.lib.cfg.desired_retention)
        return {"schedule": self._schedule(card_id) if fsrs else None, "session": self.get_session(sid)}
