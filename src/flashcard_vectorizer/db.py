"""SQLite storage: one database per user, all decks together."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decks (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS cards (
    id             INTEGER PRIMARY KEY,
    deck_id        INTEGER NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
    note_guid      TEXT NOT NULL,
    ord            INTEGER NOT NULL,
    notetype       TEXT NOT NULL,
    question       TEXT NOT NULL,      -- cleaned plain text (what gets embedded/searched)
    answer         TEXT NOT NULL,
    question_html  TEXT NOT NULL,      -- rendered Anki HTML, for the study UI later
    answer_html    TEXT NOT NULL,
    tags           TEXT NOT NULL DEFAULT '',
    media          TEXT NOT NULL DEFAULT '[]',   -- JSON list of referenced media filenames
    content_hash   TEXT NOT NULL,
    embedded_hash  TEXT,               -- content_hash the stored vectors were computed from
    embedded_model TEXT,
    UNIQUE (deck_id, note_guid, ord)
);
CREATE INDEX IF NOT EXISTS cards_deck ON cards(deck_id);

-- kind: 0 = question+answer combined, 1 = question, 2 = answer
CREATE TABLE IF NOT EXISTS vectors (
    card_id  INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    kind     INTEGER NOT NULL,
    model    TEXT NOT NULL,
    vec      BLOB NOT NULL,            -- float16, L2-normalised
    PRIMARY KEY (card_id, kind)
) WITHOUT ROWID;

CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
    question, answer, tags,
    tokenize = 'porter unicode61 remove_diacritics 2'
);   -- rowid = cards.id
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection (one per thread). WAL lets the importer write while queries read."""
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: str | Path) -> None:
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
