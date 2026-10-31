"""Louvain community detection on a weighted, undirected graph.

Written against NumPy only: the usual fast implementations (python-igraph, leidenalg) are GPL, and
this project stays permissively licensed. The local-moving loop is plain Python over CSR lists, which
handles a 100k-node, ~600k-edge kNN graph in tens of seconds, inside a background job.

Louvain can leave a community internally disconnected, so ``louvain`` finishes by splitting every
community into its connected components (the guarantee Leiden adds).
"""
from __future__ import annotations

import numpy as np


def _csr(n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray, loop: np.ndarray):
    """Symmetric CSR adjacency plus weighted degrees (a node's self-loop weight counts in its degree)."""
    a = np.concatenate([src, dst]).astype(np.int32)
    order = np.argsort(a, kind="stable")
    b = np.concatenate([dst, src]).astype(np.int32)[order]
    ww = np.concatenate([w, w]).astype(np.float32)[order]  # float32: halves the biggest per-edge array
    a = a[order]
    del order
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(a, minlength=n), out=indptr[1:])
    deg = np.bincount(a, weights=ww, minlength=n) + loop
    return indptr, b, ww, deg


def _local_moving(indptr, nbr, wt, deg, resolution: float, rng, max_passes: int = 30) -> np.ndarray:
    """Greedily move nodes to the neighbouring community with the best modularity gain."""
    n = len(deg)
    m2 = float(deg.sum())  # twice the total edge weight
    if m2 <= 0:
        return np.arange(n)
    gam = resolution / m2
    # Per-node lists are small; the per-edge arrays are read through memoryviews, which avoids ~100 MB of
    # Python float/int objects at 100k nodes and 600k edges.
    ip, nb, wl, degl = indptr.tolist(), memoryview(nbr), memoryview(wt), deg.tolist()
    comm = list(range(n))
    tot = list(degl)  # total degree of each community
    nodes = np.arange(n)
    for _ in range(max_passes):
        rng.shuffle(nodes)
        moved = 0
        for i in nodes.tolist():
            s, e = ip[i], ip[i + 1]
            if s == e:
                continue
            ci, ki = comm[i], degl[i]
            links: dict[int, float] = {}
            for p in range(s, e):
                c = comm[nb[p]]
                links[c] = links.get(c, 0.0) + wl[p]
            tot[ci] -= ki
            best, best_gain = ci, links.get(ci, 0.0) - gam * tot[ci] * ki
            for c, wc in links.items():
                g = wc - gam * tot[c] * ki
                if g > best_gain + 1e-12:
                    best, best_gain = c, g
            tot[best] += ki
            if best != ci:
                comm[i] = best
                moved += 1
        if moved <= max(0, n // 2000):  # converged, or only a long tail of tiny improvements left
            break
    return np.asarray(comm)


def _aggregate(comm: np.ndarray, src, dst, w, loop):
    """Collapse each community into one node: returns (node -> new node, count, edges, loops)."""
    _, inv = np.unique(comm, return_inverse=True)
    count = int(inv.max()) + 1
    cs, cd = inv[src], inv[dst]
    intra = cs == cd
    new_loop = np.bincount(inv, weights=loop, minlength=count) + 2 * np.bincount(
        cs[intra], weights=w[intra], minlength=count
    )
    lo = np.minimum(cs[~intra], cd[~intra])
    hi = np.maximum(cs[~intra], cd[~intra])
    keys, idx = np.unique(lo * count + hi, return_inverse=True)
    nw = np.bincount(idx, weights=w[~intra], minlength=len(keys))
    return inv, count, keys // count, keys % count, nw, new_loop


def connected_split(labels: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Relabel so that every label is a connected set of nodes (via edges inside the same label)."""
    n = len(labels)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    same = np.flatnonzero(labels[src] == labels[dst])
    for s in range(0, len(same), 100_000):  # in slices: a list of all 600k endpoints would cost ~50 MB
        part = same[s : s + 100_000]
        for a, b in zip(src[part].tolist(), dst[part].tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    return np.unique(roots, return_inverse=True)[1]


def louvain(
    n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray, resolution: float = 1.0, seed: int = 0,
    max_levels: int = 12,
) -> np.ndarray:
    """Community label (0..C-1) for each of ``n`` nodes.

    ``src``/``dst``/``w`` list each undirected edge once (no self-loops, positive weights).
    """
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    src, dst = np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32)
    w = np.asarray(w, dtype=np.float64)
    member = np.arange(n)  # original node -> node of the current (aggregated) graph
    loop = np.zeros(n)
    cur = n
    e_src, e_dst, e_w = src, dst, w
    for _ in range(max_levels):
        indptr, nbr, wt, deg = _csr(cur, e_src, e_dst, e_w, loop)
        comm = _local_moving(indptr, nbr, wt, deg, resolution, rng)
        inv, count, e_src, e_dst, e_w, loop = _aggregate(comm, e_src, e_dst, e_w, loop)
        member = inv[member]
        if count == cur:  # nothing merged at this level
            break
        cur = count
    return connected_split(member, src, dst)
