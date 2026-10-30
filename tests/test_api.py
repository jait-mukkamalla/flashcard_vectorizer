import pytest
from fastapi.testclient import TestClient

from flashcard_vectorizer.api import create_app

from conftest import FIXTURES


@pytest.fixture
def client(lib):
    with TestClient(create_app(library=lib), base_url="http://127.0.0.1:8765") as c:
        yield c


def test_refuses_foreign_host_and_cross_origin_requests(client):
    # DNS rebinding: a hostile name resolving to 127.0.0.1
    assert client.get("/api/status", headers={"host": "evil.example:8765"}).status_code == 403
    # cross-site form post from a page the user happens to have open
    r = client.post("/api/import", files={"file": ("x.apkg", b"x")}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    # the app's own page is fine
    assert client.get("/api/status", headers={"origin": "http://127.0.0.1:8765"}).status_code == 200


def test_upload_import_and_search(client, lib):
    with open(FIXTURES / "modern.apkg", "rb") as f:
        r = client.post("/api/import", files={"file": ("modern.apkg", f)})
    assert r.status_code == 202
    assert lib.jobs.wait(30)

    status = client.get("/api/status").json()
    assert status["cards"] == 8 and status["pending"] == 0 and status["job"]["phase"] == "idle"
    assert {d["name"] for d in client.get("/api/decks").json()} == {"Biology", "Biology::Cells"}

    res = client.get("/api/search", params={"q": "mitochondrion powerhouse"}).json()
    assert res["dense_ready"] and res["results"][0]["question"] == "What is the powerhouse of the cell?"
    assert res["timings"]["total_ms"] > 0

    deck = next(d["id"] for d in client.get("/api/decks").json() if d["name"] == "Biology::Cells")
    only = client.get("/api/search", params={"q": "cell", "deck": deck}).json()["results"]
    assert only and {r["deck"] for r in only} == {"Biology::Cells"}

    card = client.get(f"/api/cards/{res['results'][0]['id']}").json()
    assert "mitochondrion" in card["answer_html"]


def test_rejects_non_apkg_and_bad_params(client):
    assert client.post("/api/import", files={"file": ("notes.txt", b"x")}).status_code == 400
    assert client.get("/api/search", params={"q": "x", "mode": "bogus"}).status_code == 400
    assert client.get("/api/search", params={"q": "x", "limit": 1000}).status_code == 422
    assert client.get("/api/cards/999").status_code == 404


def test_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "Flashcard Vectorizer" in r.text
