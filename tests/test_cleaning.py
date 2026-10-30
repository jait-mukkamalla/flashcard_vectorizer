from flashcard_vectorizer.cleaning import cloze_numbers, media_refs, render_cloze, strip_html


def test_strip_html_basics():
    assert strip_html("The <b>mito</b>chondrion&nbsp;is&amp;more") == "The mitochondrion is&more"
    assert strip_html("line one<br>line two<div>three</div>") == "line one\nline two\nthree"
    assert strip_html("<style>.x{color:red}</style>text<script>alert(1)</script>") == "text"
    assert strip_html("hello [sound:a.mp3] world") == "hello world"
    assert strip_html("    spaced \t out  ") == "spaced out"


def test_strip_html_math_delimiters():
    assert strip_html(r"x \(\to\) y") == r"x \to y"
    assert strip_html("[$]E=mc^2[/$]") == "E=mc^2"


def test_media_refs():
    raw = '<img src="a.png"> and <img class=x src=\'b.jpg\'> [sound:c.mp3] <img src="a.png">'
    assert media_refs(raw) == ["a.png", "b.jpg", "c.mp3"]


def test_cloze_render():
    raw = "The {{c1::ribosome}} makes {{c2::protein::product}}."
    assert render_cloze(raw, 1, reveal=False) == "The [...] makes protein."
    assert render_cloze(raw, 2, reveal=False) == "The ribosome makes [product]."
    assert render_cloze(raw, 2, reveal=True) == "The ribosome makes protein."
    assert cloze_numbers(raw) == [1, 2]


def test_cloze_nested_and_spanning_markup():
    raw = "{{c1::outer {{c2::inner}} text}} and {{c1::<b>bold</b>}}"
    assert render_cloze(raw, 1, reveal=False) == "[...] and [...]"
    assert render_cloze(raw, 2, reveal=False) == "outer [...] text and <b>bold</b>"
    assert cloze_numbers(raw) == [1, 2]


def test_cloze_unterminated_is_plain_text():
    assert render_cloze("a {{c1::oops", 1, reveal=False) == "a {{c1::oops"
