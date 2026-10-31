// Shapes returned by the backend (see graph.py and api.py).

export interface Deck {
  id: number;
  name: string;
  cards: number;
  embedded: number;
}

export interface ClusterSummary {
  id: number;
  label: string;
  size: number; // cards in the selected decks
  total: number; // cards in the cluster overall
  x: number;
  y: number;
  radius: number;
  loose: boolean;
  decks: [deckId: number, cards: number][]; // biggest first
}

export interface ClusterLink {
  a: number;
  b: number;
  edges: number;
  weight: number;
}

export interface Overview {
  ready: boolean;
  nodes: number;
  clusters: ClusterSummary[];
  links: ClusterLink[];
  cards: number;
  k: number;
  threshold: number;
}

export interface CardNode {
  id: number;
  x: number;
  y: number;
  deck: number;
  label: string;
}

export interface CardEdge {
  a: number;
  b: number;
  sim: number;
}

export interface ClusterData {
  id: number;
  label: string;
  size: number;
  truncated: boolean;
  nodes: CardNode[];
  edges: CardEdge[];
}

export interface HoodNode extends CardNode {
  hop: number;
  cluster: number | null;
}

export interface Hood {
  seeds: number[];
  nodes: HoodNode[];
  edges: CardEdge[];
}

export interface CardDetail {
  id: number;
  deck: string;
  question: string;
  answer: string;
  tags: string;
  notetype: string;
}

export interface Status {
  cards: number;
  decks: number;
  embedded: number;
  pending: number;
  graph: { ready: boolean; nodes: number; clusters: number; k: number; threshold: number };
  job: {
    phase: string;
    error: string | null;
    graph: { stage: string; done: number; total: number } | null;
  };
}

export type ColorBy = "cluster" | "deck";

export type Selection = { kind: "cluster"; id: number } | { kind: "card"; id: number };
