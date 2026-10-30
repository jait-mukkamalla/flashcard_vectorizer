import zipfile

import pytest

from flashcard_vectorizer.apkg import ApkgError, iter_apkg

from conftest import FIXTURES


def rows(name):
    return [(c.deck, c.notetype, c.ord, c.question, c.answer, c.tags, c.media) for c in iter_apkg(FIXTURES / name)]


def test_modern_and_legacy_exports_agree():
    # The same collection exported in Anki's schema-18 (zstd) and schema-11 formats.
    assert rows("modern.apkg") == rows("legacy.apkg")


def test_fixture_contents():
    all_rows = rows("modern.apkg")
    assert len(all_rows) == 8
    cards = {(r[0], r[1], r[2], r[3]): r for r in all_rows}
    basic = cards[("Biology", "Basic", 0, "What is the powerhouse of the cell?")]
    assert basic[4:6] == ("The mitochondrion", "cells")
    # the reversed card is its own card with the sides swapped
    rev = "Basic (and reversed card)"
    assert cards[("Biology::Cells", rev, 0, "nucleus")][4] == "Organelle that stores DNA"
    assert cards[("Biology::Cells", rev, 1, "Organelle that stores DNA")][4] == "nucleus"
    # one card per cloze number
    c1 = cards[("Biology::Cells", "Cloze", 0, "The [...] synthesises protein.")]
    c2 = cards[("Biology::Cells", "Cloze", 1, "The ribosome synthesises [product].")]
    assert c1[4] == c2[4] == "The ribosome synthesises protein.\nFound in all cells"


def test_media_and_html_are_handled():
    by_q = {r[3]: r for r in rows("modern.apkg")}
    assert by_q["Define osmosis."][4] == "Diffusion of water across a semipermeable membrane.\nPassive transport."
    assert by_q["Define osmosis."][6] == ["osmosis.mp3"]
    assert by_q["What is shown?"][6] == ["diagram.png"]  # <img> tag removed from the text


def test_not_a_zip(tmp_path):
    bad = tmp_path / "bad.apkg"
    bad.write_text("nope")
    with pytest.raises(ApkgError, match="not a valid"):
        list(iter_apkg(bad))


def test_zip_without_collection(tmp_path):
    p = tmp_path / "empty.apkg"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("media", "{}")
    with pytest.raises(ApkgError, match="no Anki collection"):
        list(iter_apkg(p))
