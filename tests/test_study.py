from datetime import datetime, timedelta, timezone

import pytest

from flashcard_vectorizer import scheduling
from flashcard_vectorizer.importer import import_apkg
from flashcard_vectorizer.library import Library
from flashcard_vectorizer.synthetic import build_apkg

from conftest import FIXTURES, with_revlog

DAY = 86_400_000
T0 = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

CARDS = [
    ("Bio", "What organelle produces ATP?", "The mitochondrion produces ATP through respiration"),
    ("Bio", "What does the ribosome do?", "Ribosomes synthesise proteins from amino acids"),
    ("Chem", "What is the formula for table salt?", "NaCl, sodium chloride"),
    ("Chem", "Define an acid", "A proton donor that lowers pH"),
]
# Anki revlog rows (ts, ease, time_ms, type): learn, then two reviews (one failed), then a relearn
# (revlog ids are unique across a collection, so a second card's history is shifted with `shift`)
HISTORY = [(T0, 3, 4000, 0), (T0 + 1 * DAY, 3, 3000, 1), (T0 + 5 * DAY, 1, 9000, 1), (T0 + 5 * DAY + 600_000, 3, 5000, 2)]


def shift(rows, ms=1):
    return [(ts + ms, *rest) for ts, *rest in rows]


def load(lib, tmp_path, reviews=None, name="d.apkg", cards=CARDS):
    return import_apkg(lib.conn(), build_apkg(tmp_path / name, cards, reviews=reviews))


def rows(lib, sql, *args):
    return lib.conn().execute(sql, args).fetchall()


# --------------------------------------------------------------------------- history import

def test_history_is_imported_and_seeds_fsrs_state(lib, tmp_path):
    r = load(lib, tmp_path, {0: HISTORY})
    assert r.reviews == 4
    assert [(x["rating"], x["source"]) for x in rows(lib, "SELECT * FROM reviews ORDER BY ts")] == [
        (3, "anki"), (3, "anki"), (1, "anki"), (3, "anki")]

    # state equals replaying the same history straight through py-fsrs
    expected = scheduling.replay(scheduling.make_scheduler(), [(ts, e, ms) for ts, e, ms, _ in HISTORY])
    (st,) = rows(lib, "SELECT * FROM fsrs_state")
    assert st["stability"] == pytest.approx(expected.stability)
    assert st["difficulty"] == pytest.approx(expected.difficulty)
    assert st["due"] == scheduling.to_ms(expected.due) and st["last_ts"] == HISTORY[-1][0]
    # the other three cards were never reviewed: no state, i.e. new
    assert lib.stats()["new"] == 3


def test_only_real_answers_count(lib, tmp_path):
    noise = [
        (T0, 3, 4000, 0),
        (T0 + 1000, 0, 0, 4),  # manual reschedule
        (T0 + 2000, 3, 1000, 3),  # cram in a filtered deck: does not change memory
        (T0 + 3000, 2, 1000, 5),  # reschedule
        (T0 + DAY, 4, 99_999_999, 1),  # absurd duration is clamped
    ]
    assert load(lib, tmp_path, {0: noise}).reviews == 2
    assert [x["duration_ms"] for x in rows(lib, "SELECT * FROM reviews ORDER BY ts")] == [4000, 600_000]


def test_reimport_does_not_duplicate_history_and_backfills(lib, tmp_path):
    load(lib, tmp_path)  # a library imported before it had any history
    assert rows(lib, "SELECT count(*) FROM reviews")[0][0] == 0

    r = load(lib, tmp_path, {1: HISTORY}, name="with.apkg")
    assert (r.added, r.unchanged, r.reviews) == (0, 4, 4)  # unchanged cards still get their history
    assert len(rows(lib, "SELECT * FROM fsrs_state")) == 1
    assert lib.stats()["pending"] == 4  # history does not disturb the embedding queue

    r = load(lib, tmp_path, {1: HISTORY}, name="again.apkg")
    assert r.reviews == 0 and rows(lib, "SELECT count(*) FROM reviews")[0][0] == 4


def test_new_reviews_in_a_later_export_extend_the_state(lib, tmp_path):
    load(lib, tmp_path, {0: HISTORY[:2]})
    before = rows(lib, "SELECT * FROM fsrs_state")[0]
    r = load(lib, tmp_path, {0: HISTORY}, name="later.apkg")
    assert r.reviews == 2
    after = rows(lib, "SELECT * FROM fsrs_state")[0]
    assert after["last_ts"] == HISTORY[-1][0] and after["stability"] != before["stability"]


@pytest.mark.parametrize("fixture", ["modern.apkg", "legacy.apkg"])
def test_history_from_real_anki_exports(lib, tmp_path, fixture):
    src = with_revlog(FIXTURES / fixture, tmp_path / fixture, {"powerhouse": HISTORY, "nucleus": shift(HISTORY[:2])})
    r = import_apkg(lib.conn(), src)
    assert (r.added, r.reviews) == (8, 6)
    owners = {x["question"] for x in rows(
        lib, "SELECT c.question FROM cards c JOIN fsrs_state s ON s.card_id = c.id")}
    assert owners == {"What is the powerhouse of the cell?", "nucleus"}


# --------------------------------------------------------------------------- sessions

@pytest.fixture
def filled(lib, tmp_path):
    load(lib, tmp_path, {0: HISTORY, 2: shift(HISTORY[:1])})
    lib.embed_pending()
    return lib


def test_session_from_query_is_scoped_to_decks(filled):
    s = filled.study.create_session(query="ribosome amino acids", size=3)
    assert s["cards"] == 3 and s["remaining"] == 3 and s["query"] == "ribosome amino acids"
    assert filled.study.next_card(s["id"])["card"]["question"] == "What does the ribosome do?"

    chem = {d["name"]: d["id"] for d in filled.decks()}["Chem"]
    s = filled.study.create_session(query="what is", decks=[chem])
    assert s["cards"] == 2 and s["decks"] == [chem]
    seen = set()
    while card := filled.study.next_card(s["id"])["card"]:
        seen.add(card["deck"])
        filled.study.answer(s["id"], card["id"], 3)
    assert seen == {"Chem"}


def test_again_requeues_the_card_and_every_answer_is_logged(filled):
    s = filled.study.create_session(query="acid proton donor", size=1)
    cid = filled.study.next_card(s["id"])["card"]["id"]
    out = filled.study.answer(s["id"], cid, 1)
    assert out["session"]["remaining"] == 1 and out["session"]["cards"] == 1  # same card, back at the end
    assert filled.study.next_card(s["id"])["card"]["id"] == cid
    out = filled.study.answer(s["id"], cid, 3)["session"]
    assert out["done"] and out["answered"] == 2 and out["recall_rate"] == 0.5
    logged = rows(filled, "SELECT rating FROM reviews WHERE card_id=? AND source='app' ORDER BY ts", cid)
    assert [r["rating"] for r in logged] == [1, 3]
    assert filled.study.next_card(s["id"])["card"] is None


def test_fsrs_on_schedules_answers_and_off_does_not(filled):
    study = filled.study
    assert study.fsrs_enabled()  # default
    s = study.create_session(query="table salt formula", size=1)
    card = study.next_card(s["id"])["card"]
    sched = study.answer(s["id"], card["id"], 4)["schedule"]
    assert sched["state"] in ("learning", "review")

    study.set_fsrs(False)
    s = study.create_session(query="what does the ribosome do", size=1)
    card = study.next_card(s["id"])["card"]
    assert card["schedule"] is None
    assert study.answer(s["id"], card["id"], 3)["schedule"] is None
    assert not rows(filled, "SELECT * FROM fsrs_state WHERE card_id=?", card["id"])  # no scheduling...
    assert rows(filled, "SELECT count(*) FROM reviews WHERE card_id=? AND source='app'", card["id"])[0][0] == 1  # ...but logged
    assert filled.stats()["due"] is None

    assert study.set_fsrs(True)["caught_up"] == 1  # the answer given while off is folded in on re-enable
    assert rows(filled, "SELECT * FROM fsrs_state WHERE card_id=?", card["id"])


def test_in_app_reviews_merge_with_anki_history(filled):
    study = filled.study
    cid = rows(filled, "SELECT card_id FROM reviews GROUP BY card_id HAVING count(*) = 4")[0][0]
    s = study.create_session(query="mitochondrion ATP", size=1)
    assert study.next_card(s["id"])["card"]["id"] == cid
    study.answer(s["id"], cid, 3)

    anki_only = scheduling.replay(scheduling.make_scheduler(), [(ts, e, ms) for ts, e, ms, _ in HISTORY])
    got = rows(filled, "SELECT * FROM fsrs_state WHERE card_id=?", cid)[0]
    assert got["stability"] != pytest.approx(anki_only.stability)  # the new answer was added to the old history
    assert datetime.now(timezone.utc) - scheduling.to_dt(got["last_ts"]) < timedelta(minutes=1)


def test_due_scope_keeps_new_and_due_cards_only(filled):
    study = filled.study
    s = study.create_session(query="what", size=10, scope="due")  # all four: two long overdue, two new
    assert s["cards"] == 4
    cid = study.next_card(s["id"])["card"]["id"]
    study.answer(s["id"], cid, 4)  # reviewed just now, so no longer due
    s2 = study.create_session(query="what", size=10, scope="due")
    assert s2["cards"] == 3
    seen = []
    while card := study.next_card(s2["id"])["card"]:
        seen.append(card["id"])
        study.answer(s2["id"], card["id"], 3)
    assert cid not in seen
    study.set_fsrs(False)
    with pytest.raises(ValueError, match="FSRS"):
        study.create_session(query="what", scope="due")


def test_queryless_session_orders_most_overdue_first(filled):
    study = filled.study
    s = study.create_session(size=10)
    order = []
    while card := study.next_card(s["id"])["card"]:
        order.append(card["question"])
        study.answer(s["id"], card["id"], 3)
    # reviewed cards first, whichever came due earliest (the salt card's lone learn step), then new ones
    assert order[:2] == ["What is the formula for table salt?", "What organelle produces ATP?"]
    assert set(order[2:]) == {"What does the ribosome do?", "Define an acid"}  # new cards last


def test_session_errors_and_resume(filled, cfg):
    study = filled.study
    for bad in (dict(scope="weekly"), dict(size=0), dict(query="zzzzqqqq nonexistent", mode="keyword")):
        with pytest.raises(ValueError):
            study.create_session(**bad)
    with pytest.raises(LookupError):
        study.get_session(999)
    s = study.create_session(query="what", size=2)
    with pytest.raises(LookupError):
        study.answer(s["id"], 12345, 3)
    card = study.next_card(s["id"])["card"]
    with pytest.raises(ValueError):
        study.answer(s["id"], card["id"], 7)

    study.answer(s["id"], card["id"], 3)
    restarted = Library(cfg, log=lambda *_: None)  # sessions survive a restart
    try:
        assert restarted.study.get_session(s["id"])["remaining"] == 1
        assert restarted.study.list_sessions()[0]["id"] == s["id"]
        restarted.study.delete_session(s["id"])
        with pytest.raises(LookupError):
            restarted.study.get_session(s["id"])
    finally:
        restarted.close()


def test_invalid_retention_is_rejected(tmp_path):
    from flashcard_vectorizer.config import Config

    with pytest.raises(ValueError, match="desired_retention"):
        Config(data_dir=tmp_path, desired_retention=1.5)
