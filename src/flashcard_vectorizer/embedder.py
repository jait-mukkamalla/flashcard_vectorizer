"""Text embedding on CPU via ONNX Runtime (int8-quantised models, no PyTorch)."""
from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Config


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str  # Hugging Face repo holding transformers.js-style ONNX exports
    onnx_file: str
    dim: int
    pooling: str  # "cls" | "mean"
    query_prefix: str
    # Bump when the files/preprocessing change so stored vectors get recomputed.
    version: str = "onnx-int8-1"

    @property
    def model_id(self) -> str:
        return f"{self.name}@{self.version}"


_BGE_PREFIX = "Represent this sentence for searching relevant passages: "

MODELS: dict[str, ModelSpec] = {
    s.name: s
    for s in (
        ModelSpec("bge-small-en-v1.5", "Xenova/bge-small-en-v1.5", "onnx/model_quantized.onnx", 384, "cls", _BGE_PREFIX),
        ModelSpec("all-MiniLM-L6-v2", "Xenova/all-MiniLM-L6-v2", "onnx/model_quantized.onnx", 384, "mean", ""),
        ModelSpec("bge-base-en-v1.5", "Xenova/bge-base-en-v1.5", "onnx/model_quantized.onnx", 768, "cls", _BGE_PREFIX),
    )
}

# Offline stand-in for tests and CI; not a real semantic model.
HASH_MODEL = ModelSpec("hash-test", "", "", 256, "", "", version="1")

_TOKENIZER_FILE = "tokenizer.json"


def _download(url: str, dest: Path, log: Callable[[str], None]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "flashcard-vectorizer"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                log(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB")
    os.replace(tmp, dest)


def ensure_model_files(spec: ModelSpec, models_dir: Path, log: Callable[[str], None] = print) -> Path:
    """Download the model on first use (the only network access the app makes by default)."""
    root = models_dir / spec.name
    for rel in (spec.onnx_file, _TOKENIZER_FILE):
        dest = root / rel
        if not dest.exists():
            _download(f"https://huggingface.co/{spec.repo}/resolve/main/{rel}", dest, log)
    return root


class OnnxEmbedder:
    def __init__(self, spec: ModelSpec, root: Path, max_len: int = 256, threads: int = 1):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.spec = spec
        self.dim = spec.dim
        self.model_id = spec.model_id
        self.tokenizer = Tokenizer.from_file(str(root / _TOKENIZER_FILE))
        self.tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer.enable_padding()  # pad to the longest item in each batch
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        # Spinning workers burn CPU and collapse throughput when threads outnumber free cores
        # (laptops, containers); measured ~2x faster with it off.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(root / spec.onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self.session.get_inputs()}

    def _run(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]
        if self.spec.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            m = mask[:, :, None].astype(hidden.dtype)
            pooled = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        pooled = pooled.astype(np.float32)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int = 64) -> np.ndarray:
        """L2-normalised float32 embeddings, one row per text, in input order."""
        if is_query and self.spec.query_prefix:
            texts = [self.spec.query_prefix + t for t in texts]
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        # Length-sorted batches keep padding (and therefore compute) to a minimum.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            out[idx] = self._run([texts[i] for i in idx])
        return out


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder: shared words => similar vectors."""

    def __init__(self, spec: ModelSpec = HASH_MODEL):
        self.spec = spec
        self.dim = spec.dim
        self.model_id = spec.model_id

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


def load_embedder(cfg: Config, log: Callable[[str], None] = print):
    if cfg.model == HASH_MODEL.name:
        return HashEmbedder()
    spec = MODELS.get(cfg.model)
    if spec is None:
        raise ValueError(f"unknown model {cfg.model!r}; choose one of {sorted(MODELS) + [HASH_MODEL.name]}")
    root = ensure_model_files(spec, cfg.models_dir, log)
    return OnnxEmbedder(spec, root, cfg.max_seq_len, cfg.resolved_embed_threads())
