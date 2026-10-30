"""Benchmark ingest and query performance against the project targets.

    python benchmarks/bench.py --cards 100000 --cores 4

What is measured for real, and what is not:

* Import, FTS indexing, index load, memory and query latency run on a synthetic library of
  ``--cards`` cards. For the 100k-card run the stored vectors are random unit vectors (embedding
  100k cards for real takes tens of minutes); brute-force search cost depends only on matrix
  shape, not on vector content. Query *embedding* uses the real model.
* Embedding throughput is measured on a real sample (``--embed-sample`` cards, 3 texts each)
  and extrapolated to the full library.

``--cores N`` pins the process to N CPUs to approximate the hardware floor (4 cores).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from flashcard_vectorizer.config import Config
from flashcard_vectorizer.embedder import ensure_model_files
from flashcard_vectorizer.importer import import_cards
from flashcard_vectorizer.library import Library
from flashcard_vectorizer.synthetic import generate_cards

TARGET_CARDS = 100_000
TARGETS = {"query_p95_ms": 1000, "query_ceiling_ms": 5000, "memory_mb": 800, "full_embed_min": 30}


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def real_vocab(model_root: Path) -> list[str]:
    """Whole English-ish words from the model's own tokenizer, so one word is about one token."""
    with open(model_root / "tokenizer.json", encoding="utf-8") as f:
        vocab = json.load(f)["model"]["vocab"]
    return [t for t in vocab if t.isalpha() and t.isascii() and len(t) >= 3]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", type=int, default=100_000)
    ap.add_argument("--decks", type=int, default=50)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--model", default="bge-small-en-v1.5")
    ap.add_argument("--embed-sample", type=int, default=2000)
    ap.add_argument("--cores", type=int, default=0, help="pin to this many CPUs (0 = no pinning)")
    ap.add_argument("--workdir", default=".fv_bench")
    ap.add_argument("--keep", action="store_true", help="reuse an existing workdir library if present")
    args = ap.parse_args()

    if args.cores:
        psutil.Process().cpu_affinity(list(range(args.cores)))
    work = Path(args.workdir)
    db_exists = (work / "library.db").exists()
    if work.exists() and not args.keep:
        import shutil

        for p in work.iterdir():  # start from an empty library but keep downloaded models
            if p.name != "models":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        db_exists = False
    cores = len(psutil.Process().cpu_affinity())
    cfg = Config(data_dir=work, model=args.model, embed_threads=max(1, cores - 1))
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"cores={cores} cards={args.cards} decks={args.decks} model={args.model} base_rss={rss_mb():.0f}MB\n")

    lib = Library(cfg, log=lambda *_: None)
    spec = lib.spec
    vocab = real_vocab(ensure_model_files(spec, cfg.models_dir))
    conn = lib.conn()
    report: dict[str, float] = {}

    # ---- import (parse results -> SQLite + FTS) -------------------------------------------
    if not (args.keep and db_exists):
        t = time.perf_counter()
        res = import_cards(conn, generate_cards(args.cards, args.decks, vocab=vocab))
        dt = time.perf_counter() - t
        report["import_cards_per_s"] = res.added / dt
        print(f"import:        {res.added} cards in {dt:.1f}s ({res.added / dt:.0f}/s) incl. FTS indexing")

        # ---- stand-in vectors for the whole library ---------------------------------------
        t = time.perf_counter()
        rng = np.random.default_rng(0)
        ids = [r[0] for r in conn.execute("SELECT id FROM cards ORDER BY id")]
        for s in range(0, len(ids), 5000):
            chunk = ids[s : s + 5000]
            v = rng.normal(size=(len(chunk) * 3, spec.dim)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            v = v.astype(np.float16)
            conn.executemany(
                "INSERT INTO vectors(card_id, kind, model, vec) VALUES (?,?,?,?)",
                [(cid, k, lib.model_id, v[i * 3 + k].tobytes()) for i, cid in enumerate(chunk) for k in range(3)],
            )
        conn.execute("UPDATE cards SET embedded_hash = content_hash, embedded_model = ?", (lib.model_id,))
        conn.commit()
        print(f"stand-in vectors written in {time.perf_counter() - t:.1f}s")

    # ---- embedding throughput on a real sample --------------------------------------------
    embedder = lib.embedder
    sample = list(generate_cards(args.embed_sample, 10, seed=99, vocab=vocab))
    texts = []
    for c in sample:
        texts += [f"{c.question}\n{c.answer}", c.question, c.answer]
    embedder.encode(texts[:64], batch_size=cfg.batch_size)  # warm-up
    t = time.perf_counter()
    embedder.encode(texts, batch_size=cfg.batch_size)
    dt = time.perf_counter() - t
    cps = len(sample) / dt
    full_min = TARGET_CARDS / cps / 60  # always extrapolated to the 100k target library
    report["embed_cards_per_s"] = cps
    report["full_embed_min"] = full_min
    words = statistics.mean(len(t.split()) for t in texts)
    print(f"embedding:     {len(texts)} texts of ~{words:.0f} words ({len(sample)} cards) in {dt:.1f}s = {cps:.0f} cards/s "
          f"-> {TARGET_CARDS} cards in {full_min:.1f} min")

    # ---- index load + memory ----------------------------------------------------------------
    before = rss_mb()
    t = time.perf_counter()
    index = lib.index()
    load_s = time.perf_counter() - t
    mem = rss_mb()
    report["memory_mb"] = mem
    print(f"index load:    {len(index)} vectors in {load_s:.1f}s; matrix {index.nbytes / 1e6:.0f}MB; "
          f"RSS {before:.0f} -> {mem:.0f}MB")

    # ---- query latency ----------------------------------------------------------------------
    rng = np.random.default_rng(1)
    pool = list(generate_cards(min(args.cards, 5000), args.decks, seed=0, vocab=vocab))
    queries = []
    for _ in range(args.queries):
        c = pool[int(rng.integers(len(pool)))]
        words = (c.question + " " + c.answer).replace("?", " ").replace(".", " ").split()
        queries.append(" ".join(rng.choice(words, size=min(len(words), int(rng.integers(2, 7))), replace=False)))
    decks = [r[0] for r in conn.execute("SELECT id FROM decks LIMIT 3")]

    lib.search("warm up query", "hybrid")
    print()
    for label, mode, deck_filter in (
        ("hybrid", "hybrid", None),
        ("hybrid, 3-deck filter", "hybrid", decks),
        ("dense", "dense", None),
        ("keyword", "keyword", None),
    ):
        lat = {"total_ms": [], "embed_ms": [], "dense_ms": [], "keyword_ms": []}
        for q in queries:
            tm = lib.search(q, mode, deck_filter, 20)["timings"]
            for k in lat:
                if k in tm:
                    lat[k].append(tm[k])
        t = lat["total_ms"]
        parts = " ".join(f"{k[:-3]}={statistics.median(v):.1f}" for k, v in lat.items() if k != "total_ms" and v)
        print(f"query {label:22s} p50={pct(t, .5):7.1f}ms p95={pct(t, .95):7.1f}ms max={max(t):7.1f}ms  (median parts: {parts})")
        if mode == "hybrid" and deck_filter is None:
            report["query_p95_ms"] = pct(t, .95)
            report["query_max_ms"] = max(t)

    # ---- verdict ----------------------------------------------------------------------------
    print("\ntargets (only meaningful at --cards 100000, --cores 4):")
    checks = [
        ("query p95 < 1000 ms", report["query_p95_ms"] < TARGETS["query_p95_ms"]),
        ("query max < 5000 ms", report["query_max_ms"] < TARGETS["query_ceiling_ms"]),
        (f"memory < {TARGETS['memory_mb']} MB", report["memory_mb"] < TARGETS["memory_mb"]),
        (f"full embed < {TARGETS['full_embed_min']} min", report["full_embed_min"] < TARGETS["full_embed_min"]),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    lib.close()
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
