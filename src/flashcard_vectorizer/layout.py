"""Force-directed layout for the knowledge graph, vectorised with NumPy.

Fruchterman-Reingold with a cooling schedule, except that repulsion is estimated from a few random
nodes of the same group instead of all pairs. That makes an epoch linear in nodes + edges, so the
cards of every cluster of a 100k-card library are laid out together in seconds. Layout runs on the
server at build time and the result is stored; the browser never simulates anything.
"""
from __future__ import annotations

import numpy as np


EDGE_SLICE = 131_072  # edges processed at a time


def force_layout(
    n: int, a: np.ndarray, b: np.ndarray, w: np.ndarray, group: np.ndarray, epochs: int = 120,
    seed: int = 0, negatives: int = 4,
) -> np.ndarray:
    """Positions (n, 2) for ``n`` nodes with edges ``a[i]``-``b[i]`` of weight ``w[i]`` (0..1].

    Nodes only repel others of their own ``group`` (so groups are laid out independently, in one
    pass). Each group is then centred and scaled to fit the unit disc.
    """
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    group = np.asarray(group, dtype=np.int64)
    sizes = np.bincount(group)
    order = np.argsort(group, kind="stable")
    starts = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=starts[1:])
    size_i = sizes[group].astype(np.float64)
    start_i = starts[group]
    root = np.sqrt(size_i)

    ang = rng.uniform(0, 2 * np.pi, n)
    rad = np.sqrt(rng.uniform(size=n)) * root
    x = (rad * np.cos(ang)).astype(np.float32)
    y = (rad * np.sin(ang)).astype(np.float32)

    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    w = np.asarray(w, dtype=np.float32)
    w = w / w.max() if len(w) else w
    # a sampled neighbour stands for (size - 1) / negatives others
    scale = ((size_i - 1) / negatives).astype(np.float32)
    size_f = size_i.astype(np.float32)
    top = (size_i - 1).astype(np.int64)

    # Float32 and one array per axis: at 100k cards and ~500k edges this keeps each epoch's temporaries
    # to a few MB (the process is close to its memory budget while the graph is being built).
    for e in range(epochs):
        cool = 1.0 - e / epochs
        fx, fy = np.zeros(n), np.zeros(n)
        for s in range(0, len(a), EDGE_SLICE):  # in slices, so the per-edge temporaries stay a few MB
            ea, eb, ew = a[s : s + EDGE_SLICE], b[s : s + EDGE_SLICE], w[s : s + EDGE_SLICE]
            dx, dy = x[ea] - x[eb], y[ea] - y[eb]
            pull = np.hypot(dx, dy)
            pull *= ew  # force d^2 along the edge
            dx *= pull
            dy *= pull
            fx += np.bincount(eb, dx, n)
            fx -= np.bincount(ea, dx, n)
            fy += np.bincount(eb, dy, n)
            fy -= np.bincount(ea, dy, n)
        for _ in range(negatives):
            pick = np.minimum((rng.random(n, dtype=np.float32) * size_f).astype(np.int64), top)
            other = order[start_i + pick]
            rx, ry = x - x[other], y - y[other]
            d2 = rx * rx + ry * ry + np.float32(1e-3)
            fx += rx / d2 * scale  # force 1/d away from the others
            fy += ry / d2 * scale
        length = np.hypot(fx, fy) + 1e-12
        step = np.minimum(length, (0.2 * cool + 0.01) * root) / length
        x += (fx * step).astype(np.float32)
        y += (fy * step).astype(np.float32)

    return _fit_groups(np.stack([x, y], axis=1), group, sizes, starts)


def _fit_groups(pos: np.ndarray, group: np.ndarray, sizes: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Centre each group on its mean and scale it so ~98% of its nodes lie inside the unit disc."""
    g = len(sizes)
    centre = np.stack([np.bincount(group, pos[:, k], g) / np.maximum(sizes, 1) for k in (0, 1)], axis=1)
    rel = pos - centre[group]
    dist = np.hypot(rel[:, 0], rel[:, 1])
    by = np.lexsort((dist, group))  # nodes sorted by group, then by distance from the centre
    ranked = dist[by]
    pick = np.minimum(starts[:-1] + (0.98 * np.maximum(sizes - 1, 0)).astype(np.int64), len(ranked) - 1)
    r = np.maximum(ranked[pick], 1e-9)
    out = rel / r[group][:, None]
    norm = np.hypot(out[:, 0], out[:, 1])
    out /= np.maximum(norm, 1.0)[:, None]  # pull the outliers onto the rim
    return out.astype(np.float32)


def separate_circles(pos: np.ndarray, radius: np.ndarray, pad: float = 0.0, iters: int = 60, seed: int = 0) -> np.ndarray:
    """Push circles apart until none overlap (or ``iters`` passes). Works on chunks of rows, so memory stays small."""
    pos = pos.astype(np.float64).copy()
    n = len(pos)
    if n < 2:
        return pos
    pos += np.random.default_rng(seed).normal(scale=1e-6, size=pos.shape)  # break exact coincidences
    for _ in range(iters):
        push = np.zeros_like(pos)
        overlapping = False
        rows = max(16, min(512, 1_000_000 // n))  # bounds the (rows x n) temporaries however many circles
        for s in range(0, n, rows):
            e = min(s + rows, n)
            dx = pos[s:e, None, 0] - pos[None, :, 0]
            dy = pos[s:e, None, 1] - pos[None, :, 1]
            d = np.hypot(dx, dy)
            need = radius[s:e, None] + radius[None, :] + pad - d
            need[np.arange(e - s), np.arange(s, e)] = 0.0
            hit = need > 0
            if not hit.any():
                continue
            overlapping = True
            f = np.where(hit, need / np.maximum(d, 1e-9), 0.0)
            push[s:e, 0] += (f * dx).sum(1)
            push[s:e, 1] += (f * dy).sum(1)
        if not overlapping:
            break
        pos += push * 0.5
    return pos
