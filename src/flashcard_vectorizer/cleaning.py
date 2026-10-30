"""Turn Anki field HTML (including cloze deletions) into plain text for embedding."""
from __future__ import annotations

import html
import re

_STYLE_SCRIPT = re.compile(r"<(style|script)\b.*?</\1\s*>", re.I | re.S)
_BLOCK_BREAK = re.compile(r"<\s*/?\s*(br|p|div|li|ul|ol|tr|table|h[1-6]|hr)\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_SOUND = re.compile(r"\[sound:([^\]]+)\]")
_IMG_SRC = re.compile(r"""<img\b[^>]*?\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.I)
_ANKI_MATH = re.compile(r"\[/?\$\$?\]|\[/?latex\]")
_MATHJAX = re.compile(r"\\[()\[\]]")
_WS_LINE = re.compile(r"[ \t\u00a0\u200b]+")
_CLOZE_OPEN = re.compile(r"\{\{c(\d+)::", re.I)


def media_refs(raw: str) -> list[str]:
    """Filenames referenced by <img> tags and [sound:] tags, in order, de-duplicated."""
    found = [next(g for g in m.groups() if g) for m in _IMG_SRC.finditer(raw)]
    found += _SOUND.findall(raw)
    return list(dict.fromkeys(html.unescape(f) for f in found))


def strip_html(raw: str) -> str:
    text = _STYLE_SCRIPT.sub("", raw)
    text = _SOUND.sub("", text)
    text = _BLOCK_BREAK.sub("\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    text = _ANKI_MATH.sub("", text)
    text = _MATHJAX.sub("", text)
    lines = (_WS_LINE.sub(" ", ln).strip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if ln)


def _parse_cloze(text: str):
    """Yield ('text', str) and ('cloze', num, inner, hint) parts; supports nested clozes."""
    pos = 0
    while True:
        m = _CLOZE_OPEN.search(text, pos)
        if not m:
            if pos < len(text):
                yield ("text", text[pos:])
            return
        if m.start() > pos:
            yield ("text", text[pos : m.start()])
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text.startswith("{{", i):
                depth += 1
                i += 2
            elif text.startswith("}}", i):
                depth -= 1
                i += 2
            else:
                i += 1
        if depth:  # unterminated: treat the rest as plain text
            yield ("text", text[m.start() :])
            return
        body = text[m.end() : i - 2]
        inner, hint = body, ""
        # hint is the last top-level '::' segment
        d, cut = 0, -1
        for j in range(len(body) - 1):
            if body.startswith("{{", j):
                d += 1
            elif body.startswith("}}", j):
                d -= 1
            elif d == 0 and body.startswith("::", j):
                cut = j
        if cut >= 0:
            inner, hint = body[:cut], body[cut + 2 :]
        yield ("cloze", int(m.group(1)), inner, hint)
        pos = i


def cloze_numbers(raw: str) -> list[int]:
    nums: set[int] = set()

    def walk(text):
        for part in _parse_cloze(text):
            if part[0] == "cloze":
                nums.add(part[1])
                walk(part[2])

    walk(raw)
    return sorted(nums)


def render_cloze(raw: str, active: int, reveal: bool) -> str:
    """Render a cloze field for card number ``active`` (1-based).

    The active cloze becomes ``[...]`` (or ``[hint]``) on the question side and its text on
    the answer side; all other clozes show their text on both sides.
    """
    out = []
    for part in _parse_cloze(raw):
        if part[0] == "text":
            out.append(part[1])
            continue
        _, num, inner, hint = part
        inner = render_cloze(inner, active, reveal)
        if num == active and not reveal:
            out.append(f"[{strip_html(hint) or '...'}]")
        else:
            out.append(inner)
    return "".join(out)


def card_text(question: str, answer: str) -> tuple[str, str, str]:
    """(question, answer, combined) texts to embed for a cleaned card."""
    combined = f"{question}\n{answer}" if answer else question
    return question, answer, combined
