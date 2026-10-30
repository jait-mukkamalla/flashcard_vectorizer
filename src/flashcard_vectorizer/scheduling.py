"""FSRS scheduling via py-fsrs.

A card's FSRS state is a pure function of its review history: it is always (re)built by replaying
the ``reviews`` table in time order. That is what seeds state from an Anki import, and it keeps the
state correct when reviews arrive from several places (Anki history, this app, a later re-import)
or when scheduling is switched off for a while and back on.
"""
from __future__ import annotations

import itertools
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Sequence

from fsrs import Card, Rating, Scheduler, State

BATCH = 500


def make_scheduler(retention: float = 0.9) -> Scheduler:
    # Fuzzing is off: with it, replaying the same history twice would give different due dates.
    return Scheduler(desired_retention=retention, enable_fuzzing=False)


def to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def replay(scheduler: Scheduler, reviews: Iterable[tuple[int, int, int]]) -> Card | None:
    """Fold ``(ts_ms, rating, duration_ms)`` reviews, oldest first, into a card. None if empty."""
    card: Card | None = None
    for ts, rating, duration in reviews:
        card, _ = scheduler.review_card(card or Card(), Rating(rating), to_dt(ts), duration or None)
    return card


def rebuild(conn: sqlite3.Connection, card_ids: Sequence[int], retention: float = 0.9) -> int:
    """Recompute ``fsrs_state`` for these cards from their reviews. The caller commits."""
    scheduler = make_scheduler(retention)
    ids = list(card_ids)
    for start in range(0, len(ids), BATCH):
        chunk = ids[start : start + BATCH]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT card_id, ts, rating, duration_ms FROM reviews WHERE card_id IN ({marks}) "
            "ORDER BY card_id, ts",
            chunk,
        )
        built: dict[int, Card] = {}
        for card_id, group in itertools.groupby(rows, key=lambda r: r[0]):
            card = replay(scheduler, ((r[1], r[2], r[3]) for r in group))
            if card is not None:
                built[card_id] = card
        conn.execute(
            f"DELETE FROM fsrs_state WHERE card_id IN ({marks})",
            chunk,
        )
        conn.executemany(
            "INSERT INTO fsrs_state(card_id, state, step, stability, difficulty, due, last_ts) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (cid, c.state.value, c.step, c.stability, c.difficulty, to_ms(c.due), to_ms(c.last_review))
                for cid, c in built.items()
            ],
        )
    return len(ids)


def stale_card_ids(conn: sqlite3.Connection) -> list[int]:
    """Cards with a review newer than their stored state (or reviews but no state)."""
    rows = conn.execute(
        "SELECT r.card_id FROM reviews r LEFT JOIN fsrs_state s ON s.card_id = r.card_id "
        "GROUP BY r.card_id HAVING s.card_id IS NULL OR max(r.ts) > s.last_ts"
    )
    return [r[0] for r in rows]


def refresh_stale(conn: sqlite3.Connection, retention: float = 0.9) -> int:
    ids = stale_card_ids(conn)
    rebuild(conn, ids, retention)
    conn.commit()
    return len(ids)


def card_from_row(row: sqlite3.Row) -> Card:
    return Card(
        state=State(row["state"]),
        step=row["step"],
        stability=row["stability"],
        difficulty=row["difficulty"],
        due=to_dt(row["due"]),
        last_review=to_dt(row["last_ts"]),
    )


STATE_NAMES = {State.Learning.value: "learning", State.Review.value: "review", State.Relearning.value: "relearning"}


def describe(row: sqlite3.Row | None, scheduler: Scheduler, now: datetime | None = None) -> dict:
    """Schedule summary for a card's state row (None = never reviewed)."""
    if row is None:
        return {"state": "new"}
    return {
        "state": STATE_NAMES[row["state"]],
        "due": row["due"],
        "stability": round(row["stability"], 2),
        "retrievability": round(scheduler.get_card_retrievability(card_from_row(row), now), 4),
    }
