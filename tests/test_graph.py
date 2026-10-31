import numpy as np
import pytest
from fastapi.testclient import TestClient

from flashcard_vectorizer import graph as graph_mod
from flashcard_vectorizer.api import create_app
from flashcard_vectorizer.importer import import_cards
from flashcard_vectorizer.layout import force_layout, separate_circles
from flashcard_vectorizer.louvain import connected_split, louvain
from flashcard_vectorizer.search import KIND_COMBINED, BruteForceIndex
from flashcard_vectorizer.synthetic import generate_cards

GROUPS = 5


# ---------------------------------------------------------------------------- helpers

class Planted:
    """Cards whose stored vectors form ``GROUPS`` tight blobs, bypassing the embedder (and its network)."""

    def __init__(self, lib, seed: int = 0):
        self.lib = lib
        self.rng = np.random.default_rng(seed)
        self.centres = self.rng.normal(size=(GROUPS, lib.spec.dim))
        self.centres /= np.linalg.norm(self.centres, axis=1, keepdims=True)
        self.vec: dict[int, np.ndarray] = {}  # card id -> the vector as stored (float16 precision)
        self.group: dict[int, int] = {}
        self.batches = 0

    def vector(self, group: int) -> np.ndarray:
        v = self.centres[group] + self.rng.normal(scale=0.04, size=self.centres.shape[1])
        return (v / np.linalg.norm(v)).astype(np.float16).astype(np.float32)

    def add(self, n: int, decks: int = 3) -> list[int]:
        conn = self.lib.conn()
        before = {r[0] for r in conn.execute("SELECT id FROM cards")}
        import_cards(conn, generate_cards(n, decks, seed=self.batches + 1))
        self.batches += 1
        new = [r[0] for r in conn.execute("SELECT id FROM cards ORDER BY id") if r[0] not in before]
        for i, cid in enumerate(new):
            self.group[cid] = i % GROUPS
            self.set_vector(cid, self.vector(self.group[cid]))
        conn.execute("UPDATE cards SET embedded_hash = content_hash, embedded_model = ?", (self.lib.model_id,))
        conn.commit()
        self.lib._index = None  # reload from the database on next use
        return new

    def set_vector(self, cid: int, v: np.ndarray) -> None:
        self.vec[cid] = v
        conn = self.lib.conn()
        conn.execute("DELETE FROM vectors WHERE card_id = ?", (cid,))
        conn.executemany(
            "INSERT INTO vectors(card_id, kind, model, vec) VALUES (?,?,?,?)",
            [(cid, kind, self.lib.model_id, v.astype(np.float16).tobytes()) for kind in range(3)],
        )

    def expected(self, k: int) -> dict[int, set[int]]:
        """Exact top-k neighbours of every card, by brute force."""
        ids = sorted(self.vec)
        m = np.stack([self.vec[i] for i in ids])
        sims = m @ m.T
        np.fill_diagonal(sims, -np.inf)
        top = np.argsort(-sims, axis=1)[:, :k]
        return {cid: {ids[j] for j in top[r]} for r, cid in enumerate(ids)}


def stored(lib) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for src, dst in lib.conn().execute("SELECT src, dst FROM knn"):
        out.setdefault(src, set()).add(dst)
    return out


@pytest.fixture
def planted(lib):
    p = Planted(lib)
    p.add(120)
    return p


# ---------------------------------------------------------------------------- algorithms

def planted_partition(sizes, p_in, p_out, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(len(sizes)), sizes)
    n = len(truth)
    iu = np.triu_indices(n, 1)
    prob = np.where(truth[iu[0]] == truth[iu[1]], p_in, p_out)
    keep = rng.random(len(prob)) < prob
    return n, truth, iu[0][keep], iu[1][keep]


def test_louvain_recovers_planted_communities():
    n, truth, src, dst = planted_partition([60, 50, 40, 30], 0.3, 0.01)
    labels = louvain(n, src, dst, np.ones(len(src)))
    assert len(set(labels)) == 4
    for c in set(labels):  # every found community lies inside one true group
        assert len(set(truth[labels == c])) == 1


def test_louvain_handles_edge_cases():
    assert len(louvain(0, [], [], [])) == 0
    assert list(louvain(3, [], [], [])) == [0, 1, 2]  # no edges: every node alone
    labels = louvain(4, [0, 2], [1, 3], [1.0, 1.0])
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]


def test_connected_split_separates_disconnected_pieces():
    labels = np.array([0, 0, 0, 0])
    assert len(set(connected_split(labels, np.array([0, 2]), np.array([1, 3])))) == 2  # {0,1} and {2,3}
    assert len(set(connected_split(labels, np.array([0, 1, 2]), np.array([1, 2, 3])))) == 1


def test_force_layout_keeps_neighbours_close_and_fits_the_unit_disc():
    n, truth, src, dst = planted_partition([40, 40, 40], 0.2, 0.0)
    pos = force_layout(n, src, dst, np.ones(len(src)), truth, epochs=100)
    assert pos.shape == (n, 2) and np.isfinite(pos).all()
    assert (np.hypot(pos[:, 0], pos[:, 1]) <= 1.0 + 1e-5).all()
    linked = np.hypot(*(pos[src] - pos[dst]).T).mean()
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, 500), rng.integers(0, n, 500)
    same = truth[a] == truth[b]
    random_pairs = np.hypot(*(pos[a[same]] - pos[b[same]]).T).mean()
    assert linked < random_pairs * 0.8


def test_separate_circles_removes_overlap():
    pos = np.zeros((20, 2))
    radius = np.full(20, 2.0)
    out = separate_circles(pos, radius)
    d = np.hypot(out[:, None, 0] - out[None, :, 0], out[:, None, 1] - out[None, :, 1]) + np.eye(20) * 99
    assert d.min() >= 4.0 - 1e-6


def test_topk_rows_matches_brute_force(monkeypatch):
    monkeypatch.setattr(graph_mod, "CHUNK", 37)  # force several candidate chunks
    rng = np.random.default_rng(1)
    v = rng.normal(size=(150, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    index = BruteForceIndex(16)
    index.add(list(range(150)), [0] * 150, [KIND_COMBINED] * 150, v)
    rows = index.kind_rows(KIND_COMBINED)
    qpos = np.array([0, 5, 149, 70])
    sims, nbrs = graph_mod.topk_rows(index, rows, qpos, 6)
    full = v @ v.T
    np.fill_diagonal(full, -np.inf)
    for r, q in enumerate(qpos):
        want = np.argsort(-full[q])[:6]
        assert list(nbrs[r]) == list(want) and np.allclose(sims[r], full[q][want], atol=1e-6)
    # fewer candidates than k: missing slots are marked
    tiny = BruteForceIndex(16)
    tiny.add([1, 2], [0, 0], [0, 0], v[:2])
    s, i = graph_mod.topk_rows(tiny, tiny.kind_rows(0), np.array([0]), 4)
    assert list(i[0]) == [1, -1, -1, -1] and np.isneginf(s[0, 1:]).all()


# ---------------------------------------------------------------------------- building

def test_build_finds_the_planted_clusters(lib, planted):
    result = lib.graph.build()
    assert result["complete"] and result["ready"] and result["nodes"] == 120 and result["clusters"] == GROUPS
    rows = lib.conn().execute("SELECT card_id, cluster_id, x, y FROM graph_nodes").fetchall()
    by_cluster: dict[int, set[int]] = {}
    for cid, cl, x, y in rows:
        by_cluster.setdefault(cl, set()).add(planted.group[cid])
        assert x * x + y * y <= 1.0 + 1e-6
    assert all(len(groups) == 1 for groups in by_cluster.values())  # clusters are pure
    clusters = lib.conn().execute("SELECT * FROM graph_clusters ORDER BY id").fetchall()
    assert [c["size"] for c in clusters] == [24] * GROUPS and not any(c["loose"] for c in clusters)
    assert all(c["label"] for c in clusters)
    centres = np.array([[c["x"], c["y"]] for c in clusters])
    radii = np.array([c["radius"] for c in clusters])
    gaps = np.hypot(centres[:, None, 0] - centres[None, :, 0], centres[:, None, 1] - centres[None, :, 1])
    assert (gaps + np.eye(GROUPS) * 1e9 >= radii[:, None] + radii[None, :] - 1e-6).all()  # circles do not overlap
    per_deck = lib.conn().execute("SELECT sum(n) FROM graph_cluster_decks").fetchone()[0]
    assert per_deck == 120


def test_stored_neighbours_are_exact(lib, planted):
    lib.graph.build()
    assert stored(lib) == planted.expected(lib.cfg.graph_k)


def test_adding_cards_updates_incrementally_and_matches_a_full_build(lib, planted):
    lib.graph.build()
    before = lib.conn().execute("SELECT count(*) FROM knn").fetchone()[0]
    planted.add(60)
    assert lib.graph.needs_build()
    lib.graph.build()
    assert not lib.graph.needs_build()
    assert stored(lib) == planted.expected(lib.cfg.graph_k)  # old cards learned about the new ones
    assert lib.conn().execute("SELECT count(*) FROM knn").fetchone()[0] > before
    assert lib.graph.status()["nodes"] == 180 and lib.graph.status()["clusters"] == GROUPS


def test_changed_card_gets_fresh_neighbours_and_others_drop_it(lib, planted):
    lib.graph.build()
    victim = next(c for c, g in planted.group.items() if g == 0)
    old_friends = {s for s, d in lib.conn().execute("SELECT src, dst FROM knn") if d == victim}
    assert old_friends
    planted.set_vector(victim, planted.vector(3))  # the card now belongs to another blob
    planted.group[victim] = 3
    lib.conn().execute("UPDATE cards SET content_hash = 'edited', embedded_hash = 'edited' WHERE id = ?", (victim,))
    lib.conn().commit()
    lib._index = None
    lib.graph.build()
    assert stored(lib) == planted.expected(lib.cfg.graph_k)
    now_friends = {s for s, d in lib.conn().execute("SELECT src, dst FROM knn") if d == victim}
    assert not old_friends & now_friends  # nobody from its old blob lists it any more
    cluster_of = dict(lib.conn().execute("SELECT card_id, cluster_id FROM graph_nodes"))
    same = [c for c, g in planted.group.items() if g == 3 and c != victim]
    assert cluster_of[victim] == cluster_of[same[0]]


def test_interrupted_build_resumes_where_it_stopped(lib, planted, monkeypatch):
    monkeypatch.setattr(graph_mod, "BLOCK", 16)
    calls = []

    def stop_after_two_blocks():
        calls.append(1)
        return len(calls) > 2

    partial = lib.graph.build(should_stop=stop_after_two_blocks)
    assert partial == {"complete": False}
    done = lib.conn().execute("SELECT count(*) FROM graph_nodes WHERE hash IS NOT NULL").fetchone()[0]
    assert done == 32 and not lib.graph.status()["ready"]
    assert lib.graph.build()["complete"]
    assert stored(lib) == planted.expected(lib.cfg.graph_k)


def test_resuming_after_adding_cards_with_a_half_finished_build(lib, planted, monkeypatch):
    lib.graph.build()
    planted.add(50)
    monkeypatch.setattr(graph_mod, "BLOCK", 8)
    n = []
    lib.graph.build(should_stop=lambda: n.append(1) or len(n) > 3)  # stops after 3 blocks
    assert lib.graph.needs_build()  # clusters are stale until the neighbours are finished
    lib.graph.build()
    assert stored(lib) == planted.expected(lib.cfg.graph_k)
    assert lib.graph.status()["nodes"] == 170 and not lib.graph.needs_build()


def test_threshold_change_reclusters_without_recomputing_neighbours(lib, planted):
    lib.graph.build()
    count = lib.conn().execute("SELECT count(*) FROM knn").fetchone()[0]
    lib.cfg.graph_threshold = 0.99  # nothing is that similar
    lib.graph.build()
    assert lib.conn().execute("SELECT count(*) FROM knn").fetchone()[0] == count
    clusters = lib.conn().execute("SELECT loose, size FROM graph_clusters").fetchall()
    assert [tuple(c) for c in clusters] == [(1, 120)]  # every card is loose
    lib.cfg.graph_threshold = None
    lib.graph.build()
    assert lib.graph.status()["clusters"] == GROUPS


def test_changing_k_rebuilds_neighbour_lists(lib, planted):
    lib.graph.build()
    lib.cfg.graph_k = 3
    lib.graph.build()
    assert stored(lib) == planted.expected(3)


def test_small_and_empty_libraries(lib):
    assert lib.graph.build()["clusters"] == 0 and not lib.graph.needs_build()
    p = Planted(lib)
    p.add(2)  # fewer cards than k
    assert lib.graph.build()["complete"]
    assert lib.graph.status()["nodes"] == 2


def test_reset_and_rebuild(lib, planted):
    lib.graph.build()
    lib.graph.reset()
    assert lib.graph.status() == {**lib.graph.status(), "ready": False, "nodes": 0, "clusters": 0}
    assert lib.graph.needs_build()
    lib.graph.build()
    assert lib.graph.status()["clusters"] == GROUPS


# ---------------------------------------------------------------------------- reading

def test_overview_cluster_and_deck_filter(lib, planted):
    lib.graph.build()
    ov = lib.graph.overview()
    assert ov["ready"] and ov["cards"] == 120 and len(ov["clusters"]) == GROUPS
    assert sum(c["size"] for c in ov["clusters"]) == 120
    one_deck = lib.conn().execute("SELECT id FROM decks ORDER BY id LIMIT 1").fetchone()[0]
    filtered = lib.graph.overview([one_deck])
    assert filtered["cards"] == 40 and all(c["total"] == 24 for c in filtered["clusters"])
    assert all(d[0] == one_deck for c in filtered["clusters"] for d in c["decks"])

    cl = lib.graph.cluster(ov["clusters"][0]["id"])
    assert len(cl["nodes"]) == 24 and cl["edges"] and not cl["truncated"]
    centre = ov["clusters"][0]
    for n in cl["nodes"]:  # cards sit inside their cluster's circle
        assert np.hypot(n["x"] - centre["x"], n["y"] - centre["y"]) <= centre["radius"] + 1e-6
    ids = {n["id"] for n in cl["nodes"]}
    assert all(e["a"] in ids and e["b"] in ids and e["a"] < e["b"] for e in cl["edges"])
    assert len(lib.graph.cluster(centre["id"], [one_deck])["nodes"]) == 8
    assert lib.graph.cluster(centre["id"], limit=10)["truncated"]
    with pytest.raises(LookupError):
        lib.graph.cluster(999)


def test_neighbourhood_hops_limit_and_layout(lib, planted):
    lib.graph.build()
    seed = next(iter(planted.group))
    one = lib.graph.neighbourhood([seed], hops=1)
    assert one["seeds"] == [seed]
    hops = {n["id"]: n["hop"] for n in one["nodes"]}
    assert hops[seed] == 0 and set(hops.values()) == {0, 1}
    assert {planted.group[i] for i in hops} == {planted.group[seed]}  # neighbours stay in the blob
    assert all(np.isfinite([n["x"], n["y"]]).all() for n in one["nodes"])
    assert all(e["a"] in hops and e["b"] in hops for e in one["edges"])
    two = lib.graph.neighbourhood([seed], hops=2)
    assert len(two["nodes"]) >= len(one["nodes"])
    capped = lib.graph.neighbourhood([seed], hops=2, limit=6)
    assert len(capped["nodes"]) == 6
    decks = {r[0]: r[1] for r in lib.conn().execute("SELECT id, deck_id FROM cards")}
    only = lib.graph.neighbourhood([seed], hops=1, decks=[decks[seed]])
    assert {decks[n["id"]] for n in only["nodes"]} == {decks[seed]}
    with pytest.raises(LookupError):
        lib.graph.neighbourhood([987654])


# ---------------------------------------------------------------------------- API and sessions

@pytest.fixture
def client(lib):
    with TestClient(create_app(library=lib, warm=False), base_url="http://127.0.0.1:8765") as c:
        yield c


def test_graph_api(client, lib, planted):
    assert client.get("/api/graph/overview").json()["ready"] is False  # not built yet
    assert client.get("/api/graph/clusters/0").status_code == 404
    assert client.post("/api/graph/rebuild").status_code == 202
    assert lib.jobs.wait(30)
    status = client.get("/api/status").json()
    assert status["graph"]["ready"] and status["graph"]["clusters"] == GROUPS and status["job"]["phase"] == "idle"

    ov = client.get("/api/graph/overview").json()
    assert len(ov["clusters"]) == GROUPS
    cid = ov["clusters"][0]["id"]
    assert len(client.get(f"/api/graph/clusters/{cid}").json()["nodes"]) == 24
    card = client.get(f"/api/graph/clusters/{cid}").json()["nodes"][0]["id"]

    hood = client.get("/api/graph/neighborhood", params={"card": card, "hops": 2}).json()
    assert hood["seeds"] == [card] and len(hood["nodes"]) > 1
    assert client.get("/api/graph/neighborhood", params={"card": 987654}).status_code == 404
    assert client.get("/api/graph/neighborhood").status_code == 400
    assert client.get("/api/graph/neighborhood", params={"card": card, "q": "x"}).status_code == 400
    assert client.get("/api/graph/neighborhood", params={"card": card, "hops": 3}).status_code == 422
    # a query's top hits are the seeds (keyword hits work without a real embedder)
    word = lib.conn().execute("SELECT question FROM cards WHERE id = ?", (card,)).fetchone()[0].split()[2]
    by_query = client.get("/api/graph/neighborhood", params={"q": word, "seeds": 3}).json()
    assert 1 <= len(by_query["seeds"]) <= 3

    # foreign origins are refused here as everywhere
    assert client.post("/api/graph/rebuild", headers={"origin": "https://evil.example"}).status_code == 403


def test_study_session_from_a_cluster_or_a_set_of_cards(client, lib, planted):
    lib.graph.build()
    cid = lib.graph.overview()["clusters"][0]["id"]
    members = set(lib.graph.cluster_card_ids(cid))
    r = client.post("/api/sessions", json={"cluster": cid, "size": 10})
    assert r.status_code == 201 and r.json()["cards"] == 10 and r.json()["title"].startswith("Cluster: ")
    sid = r.json()["id"]
    seen = set()
    while (card := client.get(f"/api/sessions/{sid}/next").json()["card"]) is not None:
        seen.add(card["id"])
        client.post(f"/api/sessions/{sid}/answer", json={"card_id": card["id"], "rating": 3})
    assert len(seen) == 10 and seen <= members

    one_deck = lib.conn().execute("SELECT id FROM decks ORDER BY id LIMIT 1").fetchone()[0]
    scoped = client.post("/api/sessions", json={"cluster": cid, "decks": [one_deck], "size": 100}).json()
    assert scoped["cards"] == 8

    # the answered cards are no longer new/due-first candidates; a due-only session skips what was just reviewed
    due = client.post("/api/sessions", json={"cluster": cid, "scope": "due", "size": 100}).json()
    assert due["cards"] == 24 - 10

    chosen = sorted(members)[:5]
    r = client.post("/api/sessions", json={"card_ids": chosen, "size": 3})
    assert r.status_code == 201 and r.json()["cards"] == 3 and r.json()["title"] == "Selected cards"

    assert client.post("/api/sessions", json={"cluster": 999}).status_code == 404
    assert client.post("/api/sessions", json={"cluster": cid, "query": "x"}).status_code == 400
    assert client.post("/api/sessions", json={"cluster": cid, "card_ids": chosen}).status_code == 400
    assert client.post("/api/sessions", json={"card_ids": [987654]}).status_code == 400  # nothing matched


def test_embedding_job_builds_the_graph_afterwards(lib, tmp_path):
    from flashcard_vectorizer.importer import import_apkg
    from flashcard_vectorizer.synthetic import build_apkg

    cards = [("Bio", f"What does protein number {i} do in the cell?", "It folds and works") for i in range(12)]
    cards += [("Chem", f"What is the pH of acid number {i}?", "Below seven in water") for i in range(12)]
    import_apkg(lib.conn(), build_apkg(tmp_path / "g.apkg", cards))
    lib.jobs.submit_embed()
    assert lib.jobs.wait(30)
    s = lib.stats()
    assert s["pending"] == 0 and s["graph"]["nodes"] == 24 and s["graph"]["ready"]
    assert s["job"]["phase"] == "idle" and s["job"]["graph"] is None


# ---------------------------------------------------------------------------- cluster names

def assign_decks(lib, planted, name_of_group):
    """Put every card into a deck named for its planted group (or leave it where it is when None)."""
    conn = lib.conn()
    deck_ids = {}
    for card, group in planted.group.items():
        name = name_of_group(group)
        if name is None:
            continue
        if name not in deck_ids:
            deck_ids[name] = conn.execute("INSERT INTO decks(name) VALUES (?)", (name,)).lastrowid
        conn.execute("UPDATE cards SET deck_id = ? WHERE id = ?", (deck_ids[name], card))
    conn.commit()


def labels_by_group(lib, planted):
    group_of_cluster = {}
    for cid, cl in lib.conn().execute("SELECT card_id, cluster_id FROM graph_nodes"):
        group_of_cluster[cl] = planted.group[cid]
    return {group_of_cluster[i]: label for i, label in lib.conn().execute("SELECT id, label FROM graph_clusters")}


def test_clusters_are_named_after_their_deck_at_the_most_specific_level(lib, planted):
    assign_decks(lib, planted, lambda g: f"Science::Topic{g}")
    lib.graph.build()
    assert labels_by_group(lib, planted) == {g: f"Topic{g}" for g in range(GROUPS)}


def test_clusters_sharing_a_subject_are_told_apart_by_keywords(lib, planted):
    assign_decks(lib, planted, lambda g: "Chemistry" if g < 2 else f"Other{g}")
    lib.graph.build()
    labels = labels_by_group(lib, planted)
    assert labels[2] == "Other2"
    assert labels[0].startswith("Chemistry: ") and labels[1].startswith("Chemistry: ") and labels[0] != labels[1]


def test_tags_name_a_cluster_when_decks_do_not(lib, planted):
    assign_decks(lib, planted, lambda g: "Default")  # a generic deck name says nothing
    for card, group in planted.group.items():
        lib.conn().execute("UPDATE cards SET tags = ? WHERE id = ?", (f"leech topic_{group}", card))
    lib.conn().commit()
    lib.graph.build()
    assert labels_by_group(lib, planted) == {g: f"topic {g}" for g in range(GROUPS)}


def test_mixed_decks_fall_back_to_keywords(lib, planted):
    lib.graph.build()  # cards are spread over 3 decks regardless of group, so no deck reaches 60%
    labels = labels_by_group(lib, planted)
    assert all(label and not label.startswith("Deck ") for label in labels.values())
