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

-- Every answer ever given, from Anki's revlog or from this app. The source of truth for scheduling.
CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY,
    card_id      INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    ts           INTEGER NOT NULL,     -- ms since the epoch
    rating       INTEGER NOT NULL,     -- 1 again, 2 hard, 3 good, 4 easy
    duration_ms  INTEGER NOT NULL DEFAULT 0,
    source       TEXT NOT NULL,        -- 'anki' | 'app'
    UNIQUE (card_id, ts)
);

-- FSRS memory state: a pure function of a card's reviews (see scheduling.py). No row = new card.
CREATE TABLE IF NOT EXISTS fsrs_state (
    card_id     INTEGER PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
    state       INTEGER NOT NULL,      -- fsrs.State: 1 learning, 2 review, 3 relearning
    step        INTEGER,
    stability   REAL NOT NULL,
    difficulty  REAL NOT NULL,
    due         INTEGER NOT NULL,      -- ms since the epoch
    last_ts     INTEGER NOT NULL       -- ts of the newest review folded into this state
);
CREATE INDEX IF NOT EXISTS fsrs_due ON fsrs_state(due);

CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

-- A study session is a persisted queue, so it can be resumed after a restart.
CREATE TABLE IF NOT EXISTS sessions (
    id       INTEGER PRIMARY KEY,
    title    TEXT NOT NULL,
    query    TEXT,
    mode     TEXT,
    decks    TEXT,                     -- JSON list of deck ids the session was scoped to, or NULL
    scope    TEXT NOT NULL,            -- 'all' | 'due'
    created  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS session_cards (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    card_id     INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    rating      INTEGER,               -- NULL until answered
    PRIMARY KEY (session_id, position)
) WITHOUT ROWID;
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
