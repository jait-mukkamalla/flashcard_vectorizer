"""Command line entry point: ``flashcard-vectorizer serve | import | embed | search | status``."""
from __future__ import annotations

import argparse
import sys
import time

from .config import load_config
from .library import Library


def _progress(prefix: str):
    last = [0.0]

    def cb(done: int, total: int | None = None) -> None:
        now = time.monotonic()
        if now - last[0] > 0.5 or (total is not None and done >= total):
            last[0] = now
            tail = f"/{total}" if total else ""
            print(f"\r{prefix}: {done}{tail}", end="", flush=True)

    return cb


def cmd_serve(lib: Library, args) -> int:
    import uvicorn

    from .api import create_app

    cfg = lib.cfg
    host, port = args.host or cfg.host, args.port or cfg.port
    print(f"serving on http://{host}:{port}  (data in {cfg.data_dir})")
    uvicorn.run(create_app(library=lib), host=host, port=port, log_level="warning")
    return 0


def cmd_import(lib: Library, args) -> int:
    from .importer import import_apkg

    for path in args.files:
        result = import_apkg(lib.conn(), path, _progress(f"importing {path}"))
        print(
            f"\n{path}: {result.added} added, {result.updated} changed, "
            f"{result.unchanged + result.refreshed} unchanged ({len(result.decks)} decks)"
        )
    if not args.no_embed:
        return cmd_embed(lib, args)
    return 0


def cmd_embed(lib: Library, args) -> int:
    pending = len(lib.pending_ids())
    if not pending:
        print("nothing to embed")
        return 0
    t = time.perf_counter()
    lib.embed_pending(progress=_progress("embedding"))
    dt = time.perf_counter() - t
    print(f"\nembedded {pending} cards in {dt:.1f}s ({pending / dt:.0f} cards/s)")
    return 0


def cmd_search(lib: Library, args) -> int:
    res = lib.search(" ".join(args.query), args.mode, None, args.limit)
    for r in res["results"]:
        print(f"[{r['deck']}] {r['question']}\n    -> {r['answer']}")
    print(f"\n{len(res['results'])} results; timings (ms): {res['timings']}")
    return 0


def cmd_status(lib: Library, args) -> int:
    s = lib.stats()
    print(f"{s['cards']} cards, {s['decks']} decks, {s['embedded']} embedded ({s['pending']} pending), model {s['model']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flashcard-vectorizer", description=__doc__)
    p.add_argument("--data-dir", help="library location (default: $FV_HOME or ~/.flashcard_vectorizer)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("import", help="import .apkg files, then embed")
    s.add_argument("files", nargs="+")
    s.add_argument("--no-embed", action="store_true", help="import only; keyword search works immediately")
    s.set_defaults(fn=cmd_import)

    sub.add_parser("embed", help="embed cards that have no (current) vectors").set_defaults(fn=cmd_embed)

    s = sub.add_parser("search", help="search from the terminal")
    s.add_argument("query", nargs="+")
    s.add_argument("--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(fn=cmd_search)

    sub.add_parser("status", help="library summary").set_defaults(fn=cmd_status)

    args = p.parse_args(argv)
    lib = Library(load_config(args.data_dir))
    try:
        return args.fn(lib, args)
    except KeyboardInterrupt:
        print("\ninterrupted - re-run to resume")
        return 130
    finally:
        lib.close()


if __name__ == "__main__":
    sys.exit(main())
