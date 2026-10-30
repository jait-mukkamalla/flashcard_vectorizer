"""Synthetic flashcard libraries for tests and benchmarks (no network, deterministic per seed).

The text is pseudo-language, not meaningful prose. It has realistic lengths and word-frequency
structure, and every deck draws on its own topic vocabulary, which is what search latency and
FTS selectivity depend on.
"""
from __future__ import annotations

import json
import random
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

from .apkg import ParsedCard

_SYLLABLES = "ba be bi bo bu ca ce ci co cu da de di do du fa fe fi fo ga ge gi go la le li lo lu ma me mi mo mu na ne ni no nu pa pe pi po ra re ri ro sa se si so ta te ti to tra tri pro pre con ver sym der".split()
_COMMON = "the of and a to in is that it for as with by on this are from be at or an which has can used when each between".split()


def _make_vocab(rng: random.Random, size: int) -> list[str]:
    words: set[str] = set()
    while len(words) < size:
        words.add("".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 4))))
    return sorted(words)


def generate_cards(n: int, decks: int = 20, seed: int = 0, vocab: list[str] | None = None) -> Iterator[ParsedCard]:
    """Yield ``n`` cards spread over ``decks`` decks.

    By default words are pronounceable gibberish, which wordpiece tokenisers split into many
    pieces. Pass ``vocab`` (real words) when token counts matter, e.g. for embedding throughput.
    """
    rng = random.Random(seed)
    vocab = sorted(vocab) if vocab else _make_vocab(rng, 8000)
    topics = [rng.sample(vocab, 400) for _ in range(decks)]
    for i in range(n):
        d = i % decks
        words = topics[d]

        def phrase(lo: int, hi: int) -> str:
            out = []
            for _ in range(rng.randint(lo, hi)):
                out.append(rng.choice(_COMMON) if rng.random() < 0.35 else (
                    rng.choice(words) if rng.random() < 0.8 else rng.choice(vocab)))
            return " ".join(out)

        q = f"What is {phrase(1, 2)} {rng.choice(['in', 'for', 'of'])} {phrase(3, 8)}?"
        a = phrase(5, 25).capitalize() + "."
        tags = " ".join(rng.sample(words[:20], rng.randint(0, 2)))
        yield ParsedCard(
            deck=f"Deck {d:03d}", guid=f"g{seed}-{i}", ord=0, notetype="Basic",
            question_html=q, answer_html=a, question=q, answer=a, tags=tags,
        )


def build_apkg(path: str | Path, cards: list[tuple[str, str, str]], guid_prefix: str = "n") -> Path:
    """Write a legacy-format ``.apkg`` of Basic cards from ``(deck, front, back)`` triples."""
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        dbfile = Path(tmp) / "collection.anki2"
        db = sqlite3.connect(dbfile)
        db.executescript(
            """
            CREATE TABLE col (id integer primary key, crt integer, mod integer, scm integer, ver integer,
                dty integer, usn integer, ls integer, conf text, models text, decks text, dconf text, tags text);
            CREATE TABLE notes (id integer primary key, guid text, mid integer, mod integer, usn integer,
                tags text, flds text, sfld text, csum integer, flags integer, data text);
            CREATE TABLE cards (id integer primary key, nid integer, did integer, ord integer, mod integer,
                usn integer, type integer, queue integer, due integer, ivl integer, factor integer,
                reps integer, lapses integer, left integer, odue integer, odid integer, flags integer, data text);
            CREATE TABLE revlog (id integer primary key, cid integer, usn integer, ease integer, ivl integer,
                lastIvl integer, factor integer, time integer, type integer);
            """
        )
        model = {
            "1": {
                "name": "Basic", "type": 0,
                "flds": [{"name": "Front", "ord": 0}, {"name": "Back", "ord": 1}],
                "tmpls": [{"name": "Card 1", "ord": 0, "qfmt": "{{Front}}",
                           "afmt": "{{FrontSide}}<hr id=answer>{{Back}}"}],
            }
        }
        deck_ids: dict[str, int] = {}
        for deck, _, _ in cards:
            deck_ids.setdefault(deck, len(deck_ids) + 10)
        decks = {str(i): {"name": n} for n, i in deck_ids.items()}
        db.execute("INSERT INTO col VALUES (1,0,0,0,11,0,0,0,'{}',?,?,'{}','{}')",
                   (json.dumps(model), json.dumps(decks)))
        for i, (deck, front, back) in enumerate(cards, start=1):
            db.execute("INSERT INTO notes VALUES (?,?,1,0,0,'',?,?,0,0,'')",
                       (i, f"{guid_prefix}{i}", f"{front}\x1f{back}", front))
            db.execute("INSERT INTO cards VALUES (?,?,?,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'')", (i, i, deck_ids[deck]))
        db.commit()
        db.close()
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(dbfile, "collection.anki2")
            zf.writestr("media", "{}")
    return path
