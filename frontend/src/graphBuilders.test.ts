import { describe, expect, it } from "vitest";
import { buildHoodGraph, buildOverviewGraph, cardKey, clusterKey } from "./graphBuilders";
import type { ClusterData, Hood, Overview } from "./types";

const overview: Overview = {
  ready: true,
  nodes: 10,
  cards: 10,
  k: 8,
  threshold: 0.6,
  clusters: [
    { id: 0, label: "cells, membrane", size: 6, total: 6, x: 0, y: 0, radius: 5, loose: false, decks: [[1, 6]] },
    { id: 1, label: "acids", size: 4, total: 4, x: 20, y: 0, radius: 4, loose: false, decks: [[2, 4]] },
  ],
  links: [
    { a: 0, b: 1, edges: 3, weight: 2.1 },
    { a: 0, b: 9, edges: 1, weight: 0.7 }, // cluster 9 is filtered out of the overview
  ],
};

const opened: ClusterData = {
  id: 0,
  label: "cells, membrane",
  size: 6,
  truncated: false,
  nodes: [
    { id: 11, x: -1, y: 0, deck: 1, label: "What is a cell?" },
    { id: 12, x: 1, y: 0, deck: 1, label: "What is a membrane?" },
  ],
  edges: [
    { a: 11, b: 12, sim: 0.8 },
    { a: 11, b: 99, sim: 0.7 }, // endpoint not shown (e.g. truncated)
  ],
};

const alpha = (rgba: string) => parseFloat(rgba.split(",")[3]);

describe("buildOverviewGraph", () => {
  it("draws each cluster as one bubble sized by its radius, and links between visible clusters", () => {
    const g = buildOverviewGraph(overview, new Map(), "cluster", false);
    expect(g.order).toBe(2);
    expect(g.size).toBe(1);
    expect(g.getNodeAttribute(clusterKey(0), "size")).toBe(5);
    expect(g.getNodeAttribute(clusterKey(0), "label")).toBe("cells, membrane (6)");
  });

  it("adds an expanded cluster's cards and edges, and fades the bubble to a ring", () => {
    const closed = buildOverviewGraph(overview, new Map(), "cluster", false);
    const g = buildOverviewGraph(overview, new Map([[0, opened]]), "cluster", false);
    expect(g.order).toBe(4);
    expect(g.hasNode(cardKey(11)) && g.hasNode(cardKey(12))).toBe(true);
    expect(g.hasEdge(cardKey(11), cardKey(12))).toBe(true);
    expect(g.size).toBe(1); // the one card edge: the link touches the open cluster, and the dangling edge is dropped
    expect(closed.size).toBe(1); // (the same link when both clusters are closed)
    expect(g.getNodeAttribute(cardKey(11), "cluster")).toBe(0);
    expect(alpha(g.getNodeAttribute(clusterKey(0), "color"))).toBeLessThan(
      alpha(closed.getNodeAttribute(clusterKey(0), "color")),
    );
  });

  it("ignores an expanded cluster that is no longer in the overview", () => {
    const g = buildOverviewGraph(overview, new Map([[7, { ...opened, id: 7 }]]), "cluster", false);
    expect(g.order).toBe(2);
  });

  it("colours by deck when asked", () => {
    const byCluster = buildOverviewGraph(overview, new Map(), "cluster", false);
    const byDeck = buildOverviewGraph(overview, new Map(), "deck", false);
    expect(byDeck.getNodeAttribute(clusterKey(0), "color")).not.toBe(
      byCluster.getNodeAttribute(clusterKey(0), "color"),
    );
  });
});

describe("buildHoodGraph", () => {
  const hood: Hood = {
    seeds: [1],
    nodes: [
      { id: 1, x: 0, y: 0, deck: 1, label: "seed", hop: 0, cluster: 0 },
      { id: 2, x: 3, y: 0, deck: 1, label: "near", hop: 1, cluster: 0 },
      { id: 3, x: 6, y: 0, deck: 2, label: "far", hop: 2, cluster: null },
    ],
    edges: [
      { a: 1, b: 2, sim: 0.9 },
      { a: 2, b: 3, sim: 0.65 },
      { a: 2, b: 3, sim: 0.65 }, // duplicates are ignored
    ],
  };

  it("makes seeds biggest and keeps every edge once", () => {
    const g = buildHoodGraph(hood, "cluster", true);
    expect(g.order).toBe(3);
    expect(g.size).toBe(2);
    const size = (id: number) => g.getNodeAttribute(cardKey(id), "size") as number;
    expect(size(1)).toBeGreaterThan(size(2));
    expect(size(2)).toBeGreaterThan(size(3));
    expect(g.getNodeAttribute(cardKey(1), "forceLabel")).toBe(true);
  });
});
