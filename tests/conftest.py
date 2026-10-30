from pathlib import Path

import pytest

from flashcard_vectorizer.config import Config
from flashcard_vectorizer.library import Library

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg(tmp_path):
    """Config using the offline hash embedder, so tests never touch the network."""
    return Config(data_dir=tmp_path / "data", model="hash-test", batch_size=16)


@pytest.fixture
def lib(cfg):
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    library = Library(cfg, log=lambda *_: None)
    yield library
    library.close()


def with_revlog(src: Path, dst: Path, by_note_text: dict) -> Path:
    """Copy a real Anki export, adding revlog rows ``(ts_ms, ease, time_ms, type)`` to the first card of
    the note whose fields contain the given text. Handles both the zstd (schema 18) and legacy layouts."""
    import sqlite3
    import tempfile
    import zipfile

    import zstandard

    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        member = next(m for m in ("collection.anki21b", "collection.anki21", "collection.anki2") if m in names)
        raw = zin.read(member)
        zstd = member.endswith("b")
        db_path = Path(tmp) / "c.sqlite"
        db_path.write_bytes(zstandard.ZstdDecompressor().decompressobj().decompress(raw) if zstd else raw)
        db = sqlite3.connect(db_path)
        try:
            for text, rows in by_note_text.items():
                (cid,) = db.execute(
                    "SELECT c.id FROM cards c JOIN notes n ON n.id = c.nid WHERE n.flds LIKE ? ORDER BY c.ord LIMIT 1",
                    (f"%{text}%",),
                ).fetchone()
                for ts, ease, ms, kind in rows:
                    db.execute("INSERT INTO revlog VALUES (?,?,0,?,0,0,2500,?,?)", (ts, cid, ease, ms, kind))
            db.commit()
        finally:
            db.close()
        data = db_path.read_bytes()
        if zstd:
            data = zstandard.ZstdCompressor().compress(data)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for n in names:
                zout.writestr(n, data if n == member else zin.read(n))
    return dst
