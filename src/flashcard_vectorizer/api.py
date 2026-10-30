"""HTTP API and the bundled single-page UI."""
from __future__ import annotations

import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import Config, load_config
from .library import Library


class SessionRequest(BaseModel):
    query: str | None = None
    mode: str = "hybrid"
    decks: list[int] | None = None
    size: int = Field(20, ge=1, le=100)
    scope: str = "all"
    title: str | None = None


class AnswerRequest(BaseModel):
    card_id: int
    rating: int = Field(ge=1, le=4)
    duration_ms: int = 0


class SettingsRequest(BaseModel):
    fsrs: bool


STATIC = Path(__file__).parent / "static"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def create_app(cfg: Config | None = None, library: Library | None = None, warm: bool = True) -> FastAPI:
    lib = library or Library(cfg or load_config())

    def warm_up() -> None:
        try:
            lib.index()
            if lib.stats()["embedded"]:
                lib.embedder  # noqa: B018 - load the model now so the first query is fast
        except Exception as e:  # keyword search still works without the model
            lib.log(f"warm-up failed: {e}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm:
            threading.Thread(target=warm_up, name="fv-warmup", daemon=True).start()
            if lib.stats()["pending"]:
                lib.jobs.submit_embed()  # resume an interrupted embedding run
        yield
        lib.close()

    app = FastAPI(title="Flashcard Vectorizer", lifespan=lifespan)
    app.state.library = lib

    if lib.cfg.host in LOOPBACK:
        # The app is meant to be reachable only from this machine. Browsers will still let any
        # web page the user visits send requests to localhost, so refuse requests whose Host
        # (DNS rebinding) or Origin (cross-site form posts) is not the app itself.
        @app.middleware("http")
        async def local_only(request: Request, call_next):
            host = request.headers.get("host", "")
            if host.rsplit(":", 1)[0].strip("[]") not in LOOPBACK:
                return JSONResponse({"detail": "unexpected Host header"}, status_code=403)
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != host:
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
            return await call_next(request)

    @app.get("/api/status")
    def status():
        return lib.stats()

    @app.get("/api/decks")
    def decks():
        return lib.decks()

    @app.post("/api/import", status_code=202)
    def import_deck(file: Annotated[UploadFile, File()]):  # sync: runs in the threadpool, not the event loop
        name = file.filename or "upload.apkg"
        if not name.lower().endswith(".apkg"):
            raise HTTPException(400, "only Anki .apkg files are supported")
        tmp_dir = lib.cfg.data_dir / "uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dest = tmp_dir / f"{uuid.uuid4().hex}.apkg"
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out, 1 << 20)
        lib.jobs.submit_import(dest, name=name, delete_after=True)
        return {"queued": name}

    @app.get("/api/search")
    def search(
        q: str,
        mode: str = "hybrid",
        deck: Annotated[list[int] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        try:
            return lib.search(q, mode, deck, limit)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/cards/{card_id}")
    def card(card_id: int):
        row = lib.conn().execute(
            "SELECT c.id, d.name AS deck, c.question, c.answer, c.question_html, c.answer_html, "
            "c.tags, c.notetype, c.media FROM cards c JOIN decks d ON d.id = c.deck_id WHERE c.id = ?",
            (card_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "no such card")
        return dict(row)

    @app.get("/api/settings")
    def get_settings():
        return lib.study.settings()

    @app.put("/api/settings")
    def put_settings(body: SettingsRequest):
        lib.study.set_fsrs(body.fsrs)
        return lib.study.settings()

    # Study sessions. ValueError -> 400 (bad request), LookupError -> 404.
    def guarded(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except LookupError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/sessions", status_code=201)
    def create_session(body: SessionRequest):
        return guarded(lib.study.create_session, **body.model_dump())

    @app.get("/api/sessions")
    def list_sessions():
        return lib.study.list_sessions()

    @app.get("/api/sessions/{sid}")
    def get_session(sid: int):
        return guarded(lib.study.get_session, sid)

    @app.get("/api/sessions/{sid}/next")
    def next_card(sid: int):
        return guarded(lib.study.next_card, sid)

    @app.post("/api/sessions/{sid}/answer")
    def answer(sid: int, body: AnswerRequest):
        return guarded(lib.study.answer, sid, body.card_id, body.rating, body.duration_ms)

    @app.delete("/api/sessions/{sid}", status_code=204)
    def delete_session(sid: int):
        guarded(lib.study.delete_session, sid)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
