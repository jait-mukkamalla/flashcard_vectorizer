"""Read Anki ``.apkg`` exports (legacy and current formats) into cleaned cards.

Anki has shipped three collection layouts inside an ``.apkg`` zip:

* ``collection.anki2``   - legacy (schema 11); models/decks are JSON in the ``col`` table.
  Newer Anki writes a one-note "please update" placeholder under this name.
* ``collection.anki21``  - same schema 11, written when "support older versions" is ticked.
* ``collection.anki21b`` - zstd-compressed schema 18; notetypes, templates and decks live in
  their own tables with protobuf ``config`` blobs.
"""
from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import zstandard

from .cleaning import cloze_numbers, media_refs, render_cloze, strip_html


class ApkgError(Exception):
    pass


@dataclass
class ParsedCard:
    deck: str
    guid: str
    ord: int
    notetype: str
    question_html: str
    answer_html: str
    question: str
    answer: str
    tags: str
    media: list[str] = field(default_factory=list)


@dataclass
class _Template:
    qfmt: str
    afmt: str


@dataclass
class _Notetype:
    name: str
    is_cloze: bool
    field_names: list[str]
    templates: list[_Template]


# --------------------------------------------------------------------------- templates

_SECTION = re.compile(r"\{\{([#^])\s*([^{}]+?)\s*\}\}((?:(?!\{\{[#^]).)*?)\{\{/\s*\2\s*\}\}", re.S)
_TOKEN = re.compile(r"\{\{\s*([^{}#^/][^{}]*?)\s*\}\}")
_ANSWER_HR = re.compile(r"<hr\s+id\s*=\s*[\"']?answer[\"']?\s*/?>", re.I)


def render_template(
    tmpl: str,
    fields: dict[str, str],
    *,
    front_side: str = "",
    cloze_num: int | None = None,
    reveal: bool = False,
    tags: str = "",
) -> str:
    """Render an Anki card template (the subset needed to recover question/answer text)."""

    def value(token: str) -> str:
        *filters, name = [p.strip() for p in token.split(":")]
        if name == "FrontSide":
            return front_side
        if name == "Tags":
            return tags
        raw = fields.get(name, "")
        for f in reversed(filters):
            if f == "cloze":
                raw = render_cloze(raw, cloze_num or 1, reveal)
            elif f in ("type", "hint"):
                return ""
            elif f == "text":
                raw = strip_html(raw)
        return raw

    def section(m):
        present = bool(strip_html(value(m.group(2))))
        return m.group(3) if present == (m.group(1) == "#") else ""

    prev = None
    while prev != tmpl:  # innermost conditional sections first
        prev = tmpl
        tmpl = _SECTION.sub(section, tmpl)
    return _TOKEN.sub(lambda m: value(m.group(1)), tmpl)


def _split_answer(a_render: str, q_render: str) -> str:
    parts = _ANSWER_HR.split(a_render, maxsplit=1)
    if len(parts) == 2:
        return parts[1]
    if q_render and a_render.startswith(q_render):
        return a_render[len(q_render):]
    return a_render


# --------------------------------------------------------------------------- protobuf

def _pb_read(buf: bytes) -> Iterator[tuple[int, int | bytes]]:
    """Minimal protobuf reader: yields (field_number, value) for varint and bytes fields."""
    i, n = 0, len(buf)

    def varint():
        nonlocal i
        shift = result = 0
        while True:
            b = buf[i]
            i += 1
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7

    while i < n:
        key = varint()
        num, wire = key >> 3, key & 7
        if wire == 0:
            yield num, varint()
        elif wire == 2:
            size = varint()
            yield num, buf[i : i + size]
            i += size
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        else:
            return


# --------------------------------------------------------------------------- collections

def _unicase(a: str, b: str) -> int:
    a, b = a.casefold(), b.casefold()
    return (a > b) - (a < b)


def _legacy_meta(db: sqlite3.Connection):
    models_json, decks_json = db.execute("SELECT models, decks FROM col").fetchone()
    notetypes: dict[int, _Notetype] = {}
    for mid, m in json.loads(models_json).items():
        flds = sorted(m["flds"], key=lambda f: f["ord"])
        tmpls = sorted(m["tmpls"], key=lambda t: t["ord"])
        notetypes[int(mid)] = _Notetype(
            m["name"],
            m.get("type") == 1,
            [f["name"] for f in flds],
            [_Template(t["qfmt"], t["afmt"]) for t in tmpls],
        )
    decks = {int(did): d["name"] for did, d in json.loads(decks_json).items()}
    return notetypes, decks


def _modern_meta(db: sqlite3.Connection):
    notetypes: dict[int, _Notetype] = {}
    for mid, name, config in db.execute("SELECT id, name, config FROM notetypes").fetchall():
        kind = next((v for n, v in _pb_read(config) if n == 1), 0)
        fields = [r[0] for r in db.execute("SELECT name FROM fields WHERE ntid=? ORDER BY ord", (mid,))]
        tmpls = []
        for (cfg,) in db.execute("SELECT config FROM templates WHERE ntid=? ORDER BY ord", (mid,)):
            parts = {n: v for n, v in _pb_read(cfg)}
            tmpls.append(_Template(parts.get(1, b"").decode("utf-8"), parts.get(2, b"").decode("utf-8")))
        notetypes[mid] = _Notetype(name, kind == 1, fields, tmpls)
    decks = {did: name.replace("\x1f", "::") for did, name in db.execute("SELECT id, name FROM decks")}
    return notetypes, decks


def _open_collection(apkg: Path, tmpdir: Path) -> tuple[sqlite3.Connection, bool]:
    try:
        zf = zipfile.ZipFile(apkg)
    except zipfile.BadZipFile as e:
        raise ApkgError(f"{apkg.name} is not a valid .apkg (zip) file") from e
    with zf:
        names = set(zf.namelist())
        dest = tmpdir / "collection.sqlite"
        if "collection.anki21b" in names:
            with zf.open("collection.anki21b") as src, open(dest, "wb") as out:
                zstandard.ZstdDecompressor().copy_stream(src, out)
            modern = True
        elif "collection.anki21" in names or "collection.anki2" in names:
            member = "collection.anki21" if "collection.anki21" in names else "collection.anki2"
            with zf.open(member) as src, open(dest, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            modern = False
        else:
            raise ApkgError(f"{apkg.name} has no Anki collection inside")
    db = sqlite3.connect(dest)
    db.create_collation("unicase", _unicase)
    return db, modern


def iter_apkg(path: str | Path) -> Iterator[ParsedCard]:
    """Yield every non-empty card in the package, with question/answer cleaned to plain text."""
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="fv-apkg-") as tmp:
        db, modern = _open_collection(path, Path(tmp))
        try:
            try:
                notetypes, decks = _modern_meta(db) if modern else _legacy_meta(db)
            except (sqlite3.DatabaseError, KeyError, TypeError) as e:
                raise ApkgError(f"could not read the collection in {path.name}: {e}") from e
            if not modern:
                n = db.execute("SELECT count(*) FROM notes").fetchone()[0]
                if n == 1 and "update to the latest Anki" in db.execute("SELECT flds FROM notes").fetchone()[0]:
                    raise ApkgError(
                        "this package only contains Anki's compatibility placeholder; "
                        "re-export it from Anki (a current .apkg, or with 'support older Anki versions')"
                    )
            rows = db.execute(
                "SELECT c.ord, c.did, c.odid, n.guid, n.mid, n.flds, n.tags "
                "FROM cards c JOIN notes n ON n.id = c.nid ORDER BY c.id"
            )
            for ord_, did, odid, guid, mid, flds, tags in rows:
                card = _build_card(notetypes.get(mid), decks.get(odid or did, "Default"), guid, ord_, flds, tags)
                if card:
                    yield card
        finally:
            db.close()


def _build_card(nt: _Notetype | None, deck: str, guid: str, ord_: int, flds: str, tags: str) -> ParsedCard | None:
    if nt is None:
        return None
    values = flds.split("\x1f")
    fields = dict(zip(nt.field_names, values + [""] * (len(nt.field_names) - len(values))))
    tag_str = " ".join(tags.split())

    if nt.is_cloze:
        cloze_num = ord_ + 1
        if not nt.templates or not any(cloze_num in cloze_numbers(v) for v in fields.values()):
            return None
        tmpl = nt.templates[0]
    else:
        cloze_num = None
        if ord_ >= len(nt.templates):
            return None
        tmpl = nt.templates[ord_]

    q_html = render_template(tmpl.qfmt, fields, cloze_num=cloze_num, tags=tag_str)
    a_full = render_template(tmpl.afmt, fields, front_side=q_html, cloze_num=cloze_num, reveal=True, tags=tag_str)
    a_html = _split_answer(a_full, q_html)
    question, answer = strip_html(q_html), strip_html(a_html)
    if not question:
        return None
    return ParsedCard(
        deck=deck,
        guid=guid,
        ord=ord_,
        notetype=nt.name,
        question_html=q_html,
        answer_html=a_html,
        question=question,
        answer=answer,
        tags=tag_str,
        media=media_refs(q_html + a_html),
    )
