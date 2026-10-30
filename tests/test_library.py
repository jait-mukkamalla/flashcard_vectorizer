import dataclasses

import pytest

from flashcard_vectorizer import library as library_mod
from flashcard_vectorizer.importer import import_apkg
from flashcard_vectorizer.synthetic import build_apkg

from conftest import FIXTURES

CARDS = [
    ("Bio", "What organelle produces ATP?", "The mitochondrion produces ATP through respiration"),
    ("Bio", "What does the ribosome do?", "Ribosomes synthesise proteins from amino acids"),
    ("Chem", "What is the formula for table salt?", "NaCl, sodium chloride"),
    ("Chem", "Define an acid", "A proton donor that lowers pH"),
]


def make(lib, tmp_path, cards=CARDS, name="d.apkg"):
    return import_apkg(lib.conn(), build_apkg(tmp_path / name, cards))


def test_keyword_search_works_before_any_embedding(lib, tmp_path):
    make(lib, tmp_path)
    assert lib.stats()["pending"] == 4
    res = lib.search("ribosome", mode="keyword")
    assert [r["question"] for r in res["results"]] == ["What does the ribosome do?"]
    hybrid = lib.search("ribosome")  # no vectors yet: hybrid degrades to keyword
    assert hybrid["dense_ready"] is False and hybrid["results"]


def test_embed_then_semantic_search_and_deck_filter(lib, tmp_path):
    make(lib, tmp_path)
    assert lib.embed_pending() == 4
    s = lib.stats()
    assert (s["embedded"], s["pending"], s["index_rows"]) == (4, 0, 12)  # combined + question + answer per card

    top = lib.search("proton donor pH", mode="dense", limit=1)["results"][0]
    assert top["question"] == "Define an acid" and top["dense_score"] > 0.3

    chem = {d["name"]: d["id"] for d in lib.decks()}["Chem"]
    res = lib.search("organelle ATP", mode="hybrid", decks=[chem])["results"]
    assert res and all(r["deck"] == "Chem" for r in res)


def test_search_matches_either_side(lib, tmp_path):
    make(lib, tmp_path)
    lib.embed_pending()
    # query only overlaps with the answer text
    assert lib.search("sodium chloride", mode="dense", limit=1)["results"][0]["question"].startswith("What is the formula")


def test_reimport_is_incremental(lib, tmp_path):
    r1 = make(lib, tmp_path)
    assert (r1.added, r1.unchanged) == (4, 0)
    lib.embed_pending()

    r2 = make(lib, tmp_path, name="again.apkg")
    assert (r2.added, r2.updated, r2.unchanged) == (0, 0, 4)
    assert lib.stats()["pending"] == 0  # nothing to re-embed

    changed = [CARDS[0], CARDS[1], CARDS[2], ("Chem", "Define an acid", "Donates H+ ions")] + [("Bio", "New card", "x")]
    r3 = make(lib, tmp_path, changed, "changed.apkg")
    assert (r3.added, r3.updated, r3.unchanged) == (1, 1, 3)
    assert lib.stats()["pending"] == 2  # only the new and the edited card
    assert lib.embed_pending() == 2
    assert lib.search("H+ ions", mode="keyword")["results"][0]["answer"] == "Donates H+ ions"
    assert lib.search("proton donor", mode="keyword")["results"] == []  # old text left the FTS index


def test_embedding_is_resumable(lib, tmp_path, monkeypatch):
    monkeypatch.setattr(library_mod, "EMBED_CHUNK", 2)
    make(lib, tmp_path)
    calls = []

    def stop_after_first(done=None, total=None):
        calls.append(done)

    assert lib.embed_pending(should_stop=lambda: len(calls) >= 1, progress=stop_after_first) == 2
    assert lib.stats()["embedded"] == 2
    # a fresh library object (a "restart") picks up where it stopped, without duplicating vectors
    fresh = library_mod.Library(lib.cfg, log=lambda *_: None)
    assert fresh.embed_pending() == 2
    s = fresh.stats()
    assert (s["embedded"], s["pending"], s["index_rows"]) == (4, 0, 12)
    assert fresh.conn().execute("SELECT count(*) FROM vectors").fetchone()[0] == 12
    fresh.close()


def test_changing_model_triggers_reembedding(lib, tmp_path):
    make(lib, tmp_path)
    lib.embed_pending()
    assert lib.stats()["pending"] == 0
    lib.spec = dataclasses.replace(lib.spec, version="2")  # as if a different model were configured
    lib._index = None
    assert lib.stats()["pending"] == 4
    assert lib.search("ribosome", mode="dense")["dense_ready"] is False  # stale vectors are not used
    assert lib.embed_pending() == 4
    assert lib.conn().execute("SELECT DISTINCT model FROM vectors").fetchall()[0][0] == "hash-test@2"


def test_import_real_anki_exports(lib):
    r = import_apkg(lib.conn(), FIXTURES / "modern.apkg")
    assert r.added == 8 and r.decks == {"Biology", "Biology::Cells"}
    r = import_apkg(lib.conn(), FIXTURES / "legacy.apkg")  # same notes, other format
    assert (r.added, r.unchanged) == (0, 8)
    lib.embed_pending()
    assert lib.search("ribosome protein", mode="hybrid")["results"][0]["question"].startswith("The")


def test_background_job_runs_import_then_embed(lib, tmp_path):
    p = build_apkg(tmp_path / "bg.apkg", CARDS)
    lib.jobs.submit_import(p, delete_after=False)
    assert lib.jobs.wait(30)
    s = lib.stats()
    assert s["pending"] == 0 and s["embedded"] == 4
    assert s["job"]["import"]["added"] == 4 and s["job"]["error"] is None


def test_bad_import_reports_error_and_recovers(lib, tmp_path):
    bad = tmp_path / "bad.apkg"
    bad.write_text("not a zip")
    lib.jobs.submit_import(bad)
    assert lib.jobs.wait(30)
    assert "not a valid" in lib.stats()["job"]["error"]
    lib.jobs.submit_import(build_apkg(tmp_path / "ok.apkg", CARDS), delete_after=False)
    assert lib.jobs.wait(30)
    assert lib.stats()["job"]["error"] is None and lib.stats()["embedded"] == 4


def test_unknown_search_mode(lib):
    with pytest.raises(ValueError):
        lib.search("x", mode="nope")
