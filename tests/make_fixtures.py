"""Regenerate the real Anki exports in tests/fixtures using the official `anki` package.

Dev-only (``pip install anki``); the committed fixtures mean the test suite does not need it.
"""
import os
import sys
import tempfile
from pathlib import Path

from anki.collection import Collection, ExportAnkiPackageOptions
from anki.decks import DeckId  # noqa: F401
from anki.import_export_pb2 import ExportLimit  # noqa: F401

OUT = Path(__file__).parent / "fixtures"


def build(col: Collection):
    basic = col.models.by_name("Basic")
    rev = col.models.by_name("Basic (and reversed card)")
    cloze = col.models.by_name("Cloze")
    deck = col.decks.id("Biology")
    sub = col.decks.id("Biology::Cells")

    def add(model, did, fields, tags=()):
        n = col.new_note(model)
        for k, v in fields.items():
            n[k] = v
        n.tags = list(tags)
        col.add_note(n, did)
        return n

    add(basic, deck, {"Front": "What is the powerhouse of the cell?", "Back": "The <b>mitochondrion</b>"}, ["cells"])
    add(basic, deck, {"Front": "Define <i>osmosis</i>.", "Back": "Diffusion of water across a semipermeable membrane.<br>Passive transport.&nbsp;[sound:osmosis.mp3]"})
    add(rev, sub, {"Front": "nucleus", "Back": "Organelle that stores DNA"})
    add(cloze, sub, {"Text": "The {{c1::ribosome}} synthesises {{c2::protein::product}}.", "Back Extra": "Found in all cells"})
    add(basic, sub, {"Front": '<img src="diagram.png"> What is shown?', "Back": "A plant cell"})
    # MathJax delimiters in the back field
    add(basic, deck, {"Front": "Photosynthesis equation", "Back": r"6CO2 + 6H2O \(\to\) C6H12O6 + 6O2"})


def main():
    with tempfile.TemporaryDirectory() as d:
        col = Collection(os.path.join(d, "c.anki2"))
        build(col)
        did = col.decks.id("Biology")
        for name, legacy in (("modern.apkg", False), ("legacy.apkg", True)):
            opts = ExportAnkiPackageOptions(with_scheduling=True, with_media=False, legacy=legacy)
            from anki.import_export_pb2 import ExportLimit
            limit = ExportLimit(deck_id=did)
            col.export_anki_package(out_path=str(OUT / name), options=opts, limit=limit)
            print("wrote", OUT / name)
        col.close()


if __name__ == "__main__":
    sys.exit(main())
