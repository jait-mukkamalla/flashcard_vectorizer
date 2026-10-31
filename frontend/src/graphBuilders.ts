import Graph from "graphology";
import { clusterColor, deckColor, mutedColor, withAlpha } from "./colors";
import type { ClusterData, ColorBy, Hood, Overview } from "./types";

// Node keys. Sizes are in world units (the canvas uses itemSizesReference "positions"), so a cluster
// bubble is as large as the cards it contains occupy and zooming in reveals them.
export const clusterKey = (id: number) => `k${id}`;
export const cardKey = (id: number) => `n${id}`;

const CARD_RADIUS = 0.55;
// Edge thickness shares the node unit (world coordinates), so it is a small fraction of a node radius.
// Sigma still draws every edge at least ~1.7 px wide when zoomed far out.
const LINK_WIDTH = [0.08, 0.3] as const; // between clusters (bubbles have radius >= 4)
const CARD_EDGE_WIDTH = [0.04, 0.12] as const; // between cards (radius 0.55)
const HOOD_EDGE_WIDTH = [0.06, 0.3] as const; // neighbourhood view (card radius 1.2-2.6)
const lerp = ([lo, hi]: readonly [number, number], t: number) => lo + (hi - lo) * t;

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** Clusters as bubbles joined by links; an expanded cluster fades to a ring and shows its cards. */
export function buildOverviewGraph(
  overview: Overview,
  expanded: Map<number, ClusterData>,
  colorBy: ColorBy,
  dark: boolean,
): Graph {
  const g = new Graph({ type: "undirected", multi: false });
  const colorOf = (clusterId: number, deck: number | undefined) =>
    colorBy === "deck" ? deckColor(deck ?? 0, dark) : clusterColor(clusterId, dark);

  for (const c of overview.clusters) {
    const open = expanded.has(c.id);
    const base = colorOf(c.id, c.decks[0]?.[0]);
    g.addNode(clusterKey(c.id), {
      kind: "cluster",
      id: c.id,
      cluster: c.id,
      x: c.x,
      y: c.y,
      size: c.radius,
      label: `${c.label} (${c.size.toLocaleString()})`,
      color: c.loose ? mutedColor(dark, open ? 0.08 : 0.3) : withAlpha(base, open ? 0.1 : 0.62),
      zIndex: 0,
    });
  }

  // Links join bubble centres, so one that touches an open cluster would cut straight through its cards
  // (and, being sized in world units, is thick once you have zoomed in on it). Open clusters show their
  // cards instead; the faint ring still marks where they sit.
  const links = overview.links.filter(
    (l) =>
      g.hasNode(clusterKey(l.a)) && g.hasNode(clusterKey(l.b)) && !expanded.has(l.a) && !expanded.has(l.b),
  );
  const strongest = Math.max(1, ...links.map((l) => l.weight));
  for (const l of links) {
    g.addEdge(clusterKey(l.a), clusterKey(l.b), {
      size: lerp(LINK_WIDTH, Math.sqrt(l.weight / strongest)),
      color: mutedColor(dark, 0.28),
      weight: l.weight,
    });
  }

  for (const [clusterId, data] of expanded) {
    if (!g.hasNode(clusterKey(clusterId))) continue; // cluster no longer in the (filtered) overview
    for (const n of data.nodes) {
      g.addNode(cardKey(n.id), {
        kind: "card",
        id: n.id,
        cluster: clusterId,
        x: n.x,
        y: n.y,
        size: CARD_RADIUS,
        label: n.label,
        color: colorOf(clusterId, n.deck),
        zIndex: 2,
      });
    }
    for (const e of data.edges) {
      if (!g.hasNode(cardKey(e.a)) || !g.hasNode(cardKey(e.b)) || g.hasEdge(cardKey(e.a), cardKey(e.b))) continue;
      g.addEdge(cardKey(e.a), cardKey(e.b), {
        size: lerp(CARD_EDGE_WIDTH, clamp01((e.sim - 0.3) / 0.6)),
        color: mutedColor(dark, 0.3),
        weight: e.sim,
      });
    }
  }
  return g;
}

/** The 1-2 hop surroundings of some cards: seeds large, further hops smaller. */
export function buildHoodGraph(hood: Hood, colorBy: ColorBy, dark: boolean): Graph {
  const g = new Graph({ type: "undirected", multi: false });
  for (const n of hood.nodes) {
    const seed = n.hop === 0;
    g.addNode(cardKey(n.id), {
      kind: "card",
      id: n.id,
      cluster: n.cluster,
      x: n.x,
      y: n.y,
      size: seed ? 2.6 : n.hop === 1 ? 1.7 : 1.2,
      label: n.label,
      color: colorBy === "deck" ? deckColor(n.deck, dark) : clusterColor(n.cluster ?? 0, dark),
      zIndex: seed ? 3 : 2,
      forceLabel: seed,
    });
  }
  for (const e of hood.edges) {
    if (!g.hasNode(cardKey(e.a)) || !g.hasNode(cardKey(e.b)) || g.hasEdge(cardKey(e.a), cardKey(e.b))) continue;
    g.addEdge(cardKey(e.a), cardKey(e.b), {
      size: lerp(HOOD_EDGE_WIDTH, clamp01((e.sim - 0.3) / 0.6)),
      color: mutedColor(dark, 0.4),
      weight: e.sim,
    });
  }
  return g;
}
