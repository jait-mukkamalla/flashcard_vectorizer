"""Configuration: defaults, overridden by ``config.toml`` in the data directory."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


def default_data_dir() -> Path:
    env = os.environ.get("FV_HOME")
    return Path(env) if env else Path.home() / ".flashcard_vectorizer"


def available_cpus() -> int:
    """CPUs this process may actually use (honours affinity / container limits, unlike cpu_count)."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@dataclass
class Config:
    data_dir: Path = None  # type: ignore[assignment]
    # Embedding model (see embedder.MODELS). Changing it triggers re-embedding.
    model: str = "bge-small-en-v1.5"
    max_seq_len: int = 256
    batch_size: int = 64
    # ONNX threads for the background embedding job; 0 = all cores minus one.
    embed_threads: int = 0
    # In-memory dtype of the vector matrix: float32 is fastest, float16 halves RAM.
    index_dtype: str = "float32"
    # Spaced repetition. `fsrs` is the default for the in-app toggle (the toggle, once used, wins).
    fsrs: bool = True
    desired_retention: float = 0.9
    # Knowledge graph: each card links to its `graph_k` nearest neighbours whose cosine similarity is at
    # least `graph_threshold` (None = a default suited to the embedding model). `graph_resolution`
    # steers community detection: higher gives more, smaller clusters.
    graph_k: int = 8
    graph_threshold: float | None = None
    graph_resolution: float = 1.0
    # BLAS threads used while building the graph. One thread is deliberate: it reaches ~60 GFLOPS
    # reliably, while multi-threaded BLAS gains at most ~1.5x and, when cores are shared or limited,
    # can run 5-10x slower than a single thread. It also leaves the other cores to the web UI.
    graph_threads: int = 1
    host: str = "127.0.0.1"
    port: int = 8765

    def __post_init__(self):
        self.data_dir = Path(self.data_dir) if self.data_dir else default_data_dir()
        if not 0.7 <= self.desired_retention <= 0.99:
            raise ValueError(f"desired_retention must be between 0.7 and 0.99, got {self.desired_retention}")
        if not 1 <= self.graph_k <= 50:
            raise ValueError(f"graph_k must be between 1 and 50, got {self.graph_k}")
        if self.graph_threshold is not None and not -1.0 <= self.graph_threshold <= 1.0:
            raise ValueError(f"graph_threshold must be a cosine similarity in [-1, 1], got {self.graph_threshold}")
        if self.graph_threads < 1:
            raise ValueError(f"graph_threads must be at least 1, got {self.graph_threads}")
        if self.graph_resolution <= 0:
            raise ValueError(f"graph_resolution must be positive, got {self.graph_resolution}")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "library.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def resolved_embed_threads(self) -> int:
        return self.embed_threads or max(1, available_cpus() - 1)


def load_config(data_dir: str | Path | None = None) -> Config:
    base = Path(data_dir) if data_dir else default_data_dir()
    values: dict = {}
    path = base / "config.toml"
    if path.exists():
        values = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    values.pop("data_dir", None)
    cfg = Config(data_dir=base, **values)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
