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

-- Knowledge graph (see graph.py). knn holds each card's nearest neighbours by cosine similarity of
-- its combined vector; the graph's edges are these pairs above a similarity threshold, undirected.
CREATE TABLE IF NOT EXISTS knn (
    src  INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    dst  INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    sim  REAL NOT NULL,
    PRIMARY KEY (src, dst)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS knn_dst ON knn(dst);

-- One row per card whose neighbour list is up to date. hash = the content hash it was computed for.
-- cluster_id and x, y (position inside the cluster, unit disc) are filled in by clustering.
CREATE TABLE IF NOT EXISTS graph_nodes (
    card_id     INTEGER PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
    hash        TEXT,
    cluster_id  INTEGER,
    x           REAL,
    y           REAL
);
CREATE INDEX IF NOT EXISTS graph_nodes_cluster ON graph_nodes(cluster_id);

CREATE TABLE IF NOT EXISTS graph_clusters (
    id      INTEGER PRIMARY KEY,
    size    INTEGER NOT NULL,
    label   TEXT NOT NULL,
    x       REAL NOT NULL,             -- centre and radius in world coordinates
    y       REAL NOT NULL,
    radius  REAL NOT NULL,
    loose   INTEGER NOT NULL DEFAULT 0 -- 1 = the pool of cards that belong to no real cluster
);
CREATE TABLE IF NOT EXISTS graph_cluster_decks (
    cluster_id  INTEGER NOT NULL REFERENCES graph_clusters(id) ON DELETE CASCADE,
    deck_id     INTEGER NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
    n           INTEGER NOT NULL,
    PRIMARY KEY (cluster_id, deck_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS graph_links (
    a      INTEGER NOT NULL REFERENCES graph_clusters(id) ON DELETE CASCADE,
    b      INTEGER NOT NULL REFERENCES graph_clusters(id) ON DELETE CASCADE,
    edges  INTEGER NOT NULL,           -- card-level edges between the two clusters
    weight REAL NOT NULL,              -- their summed similarity
    PRIMARY KEY (a, b)
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
