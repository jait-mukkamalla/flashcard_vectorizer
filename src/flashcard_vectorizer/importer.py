"""Incremental import of parsed cards into the library."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import scheduling
from .apkg import ParsedCard, iter_apkg

COMMIT_EVERY = 2000


@dataclass
class ImportResult:
    added: int = 0
    updated: int = 0  # text changed, so it will be re-embedded
    refreshed: int = 0  # only tags/markup changed; vectors stay valid
    unchanged: int = 0
    reviews: int = 0  # review-history entries newly stored (already-known ones are skipped)
    decks: set[str] = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.added + self.updated + self.refreshed + self.unchanged


def content_hash(question: str, answer: str) -> str:
    """Identity of the text that gets embedded; unchanged hash => keep existing vectors."""
    return hashlib.sha1(f"{question}\x1f{answer}".encode()).hexdigest()


def _deck_id(conn: sqlite3.Connection, cache: dict[str, int], name: str) -> int:
    if name not in cache:
        conn.execute("INSERT OR IGNORE INTO decks(name) VALUES (?)", (name,))
        cache[name] = conn.execute("SELECT id FROM decks WHERE name=?", (name,)).fetchone()[0]
    return cache[name]


def import_cards(
    conn: sqlite3.Connection,
    cards: Iterable[ParsedCard],
    progress: Callable[[int], None] | None = None,
    retention: float = 0.9,
) -> ImportResult:
    """Add or update cards. Only new or text-changed cards will need embedding afterwards.

    Review history is stored for every card (even unchanged ones, so a library imported before
    history support is backfilled) and FSRS state is rebuilt for cards that gained reviews.
    """
    result = ImportResult()
    decks: dict[str, int] = {}
    pending = 0
    reviewed: set[int] = set()  # cards that gained reviews since the last commit

    def flush() -> None:
        if reviewed:
            scheduling.rebuild(conn, sorted(reviewed), retention)
            reviewed.clear()
        conn.commit()

    for c in cards:
        deck_id = _deck_id(conn, decks, c.deck)
        result.decks.add(c.deck)
        h = content_hash(c.question, c.answer)
        media = json.dumps(c.media)
        row = conn.execute(
            "SELECT id, content_hash, tags, question_html, answer_html FROM cards "
            "WHERE deck_id=? AND note_guid=? AND ord=?",
            (deck_id, c.guid, c.ord),
        ).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO cards(deck_id, note_guid, ord, notetype, question, answer, question_html, "
                "answer_html, tags, media, content_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (deck_id, c.guid, c.ord, c.notetype, c.question, c.answer, c.question_html,
                 c.answer_html, c.tags, media, h),
            )
            conn.execute(
                "INSERT INTO cards_fts(rowid, question, answer, tags) VALUES (?,?,?,?)",
                (cur.lastrowid, c.question, c.answer, c.tags),
            )
            card_id = cur.lastrowid
            result.added += 1
        elif (row["content_hash"], row["tags"], row["question_html"], row["answer_html"]) == (
            h, c.tags, c.question_html, c.answer_html
        ):
            card_id = row["id"]
            result.unchanged += 1
        else:
            conn.execute(
                "UPDATE cards SET notetype=?, question=?, answer=?, question_html=?, answer_html=?, "
                "tags=?, media=?, content_hash=? WHERE id=?",
                (c.notetype, c.question, c.answer, c.question_html, c.answer_html, c.tags, media, h, row["id"]),
            )
            conn.execute("DELETE FROM cards_fts WHERE rowid=?", (row["id"],))
            conn.execute(
                "INSERT INTO cards_fts(rowid, question, answer, tags) VALUES (?,?,?,?)",
                (row["id"], c.question, c.answer, c.tags),
            )
            card_id = row["id"]
            if row["content_hash"] != h:
                result.updated += 1
            else:
                result.refreshed += 1
        if c.reviews:
            stored = conn.executemany(
                "INSERT OR IGNORE INTO reviews(card_id, ts, rating, duration_ms, source) VALUES (?,?,?,?,'anki')",
                [(card_id, r.ts, r.rating, r.duration_ms) for r in c.reviews],
            ).rowcount
            if stored > 0:
                result.reviews += stored
                reviewed.add(card_id)
        pending += 1
        if pending >= COMMIT_EVERY:
            flush()
            pending = 0
            if progress:
                progress(result.total)
    flush()
    if progress:
        progress(result.total)
    return result


def import_apkg(
    conn: sqlite3.Connection,
    path: str | Path,
    progress: Callable[[int], None] | None = None,
    retention: float = 0.9,
) -> ImportResult:
    return import_cards(conn, iter_apkg(path), progress, retention)
