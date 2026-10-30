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
