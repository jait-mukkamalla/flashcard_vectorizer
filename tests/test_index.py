import numpy as np
import pytest

from flashcard_vectorizer.search import BruteForceIndex, fts_match_expr, rrf


def unit(rng, n, dim=16):
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_search_matches_naive(dtype):
    rng = np.random.default_rng(0)
    vecs = unit(rng, 300)
    cards = np.repeat(np.arange(100), 3)  # three rows per card
    decks = cards % 4
    idx = BruteForceIndex(16, dtype)
    idx.add(cards, decks, np.tile([0, 1, 2], 100), vecs)
    q = unit(rng, 1)[0]

    naive = {}
    for c, s in zip(cards, vecs @ q):
        naive[c] = max(naive.get(c, -9), s)
    expected = sorted(naive, key=lambda c: -naive[c])[:10]
    got = [c for c, _ in idx.search(q, 10)]
    if dtype == "float32":
        assert got == expected
    else:
        assert len(set(got) & set(expected)) >= 9  # half precision may swap near-ties


def test_results_are_distinct_cards_best_first():
    idx = BruteForceIndex(4)
    e = np.eye(4, dtype=np.float32)
    idx.add([1, 1, 2], [0, 0, 0], [0, 1, 0], np.stack([e[0], e[0], e[1]]))
    res = idx.search(e[0], 5)
    assert [c for c, _ in res] == [1, 2]
    assert res[0][1] == pytest.approx(1.0)


def test_deck_filter_and_remove():
    rng = np.random.default_rng(1)
    vecs = unit(rng, 50)
    idx = BruteForceIndex(16)
    idx.add(np.arange(50), np.arange(50) % 2, np.zeros(50, int), vecs)
    res = idx.search(vecs[4], 50, deck_ids=[1])
    assert all(c % 2 == 1 for c, _ in res) and len(res) == 25  # filter never returns other decks
    idx.remove([4, 5])
    assert len(idx) == 48
    assert 4 not in [c for c, _ in idx.search(vecs[4], 48)]


def test_growth_beyond_initial_capacity():
    rng = np.random.default_rng(2)
    idx = BruteForceIndex(8)
    for start in range(0, 5000, 700):
        n = min(700, 5000 - start)
        idx.add(np.arange(start, start + n), np.zeros(n, int), np.zeros(n, int), unit(rng, n, 8))
    assert len(idx) == 5000


def test_empty_index():
    assert BruteForceIndex(4).search(np.ones(4, dtype=np.float32), 5) == []


def test_fts_expression_is_safe():
    assert fts_match_expr('what is "the" AND mitochondria* NEAR(x)') == '"mitochondria" OR "near" OR "x"'
    assert fts_match_expr("the of") == '"the" OR "of"'  # all stop words: keep them rather than match nothing
    assert fts_match_expr("!!!") is None


def test_rrf_rewards_agreement():
    fused = [i for i, _ in rrf([[1, 2, 3], [3, 4, 1]])]
    assert fused[:2] == [1, 3]
    assert set(fused) == {1, 2, 3, 4}
