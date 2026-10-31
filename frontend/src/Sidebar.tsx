import { useEffect, useState } from "react";
import { api } from "./api";
import { clusterColor, deckColor } from "./colors";
import type { CardDetail, ClusterSummary, ColorBy, Deck, Hood, Selection } from "./types";

interface Props {
  selection: Selection | null;
  clusters: Map<number, ClusterSummary>;
  decks: Map<number, Deck>;
  expanded: Set<number>;
  truncated: Set<number>;
  hood: Hood | null;
  colorBy: ColorBy;
  dark: boolean;
  quizSize: number;
  onQuizSize: (n: number) => void;
  onToggleCluster: (id: number) => void;
  onQuizCluster: (id: number) => void;
  onHoodOfCard: (id: number) => void;
  onQuizHood: () => void;
}

export function Sidebar(p: Props) {
  const quiz = (
    <label className="muted quiz-size">
      cards{" "}
      <input
        type="number"
        min={1}
        max={100}
        value={p.quizSize}
        onChange={(e) => p.onQuizSize(Math.min(100, Math.max(1, Number(e.target.value) || 20)))}
      />
    </label>
  );

  if (p.selection?.kind === "cluster") {
    const c = p.clusters.get(p.selection.id);
    if (!c) return <Hint />;
    const open = p.expanded.has(c.id);
    return (
      <div className="side-body">
        <h2>{c.loose ? "Loose cards" : c.label}</h2>
        <p className="muted">
          {c.size.toLocaleString()} card{c.size === 1 ? "" : "s"}
          {c.size !== c.total && ` in the selected decks (${c.total.toLocaleString()} in total)`}
          {c.loose && " · cards with no close neighbours, so they belong to no cluster"}
        </p>
        <div className="row">
          <button className="primary" onClick={() => p.onQuizCluster(c.id)}>
            Quiz this cluster
          </button>
          {quiz}
          <button onClick={() => p.onToggleCluster(c.id)}>{open ? "Collapse" : "Expand"}</button>
        </div>
        {open && p.truncated.has(c.id) && (
          <p className="muted">This cluster is very large: only part of it is drawn.</p>
        )}
        <h3>Decks</h3>
        <ul className="decks">
          {c.decks.map(([deckId, n]) => (
            <li key={deckId}>
              <i style={{ background: deckColor(deckId, p.dark) }} />
              <span className="grow">{p.decks.get(deckId)?.name ?? `deck ${deckId}`}</span>
              <span className="muted">{n.toLocaleString()}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  if (p.selection?.kind === "card") {
    return (
      <CardPanel
        id={p.selection.id}
        hood={p.hood}
        clusters={p.clusters}
        colorBy={p.colorBy}
        dark={p.dark}
        quiz={quiz}
        onHoodOfCard={p.onHoodOfCard}
        onQuizHood={p.onQuizHood}
      />
    );
  }

  if (p.hood) {
    return (
      <div className="side-body">
        <h2>Neighbourhood</h2>
        <p className="muted">
          {p.hood.nodes.length} cards around {p.hood.seeds.length === 1 ? "one card" : `${p.hood.seeds.length} cards`}.
          Click a card to read it.
        </p>
        <div className="row">
          <button className="primary" onClick={p.onQuizHood}>
            Quiz this neighbourhood
          </button>
          {quiz}
        </div>
      </div>
    );
  }
  return <Hint />;
}

function Hint() {
  return (
    <div className="side-body">
      <h2>Your knowledge graph</h2>
      <p className="muted">
        Each bubble is a cluster of related cards, joined to the clusters it is most similar to. Click a bubble to open it and
        see its cards; scroll to zoom, drag to pan.
      </p>
      <p className="muted">
        Click a card to read it and to see the cards around it. Search above to see where a topic sits in your library.
      </p>
    </div>
  );
}

function CardPanel(p: {
  id: number;
  hood: Hood | null;
  clusters: Map<number, ClusterSummary>;
  colorBy: ColorBy;
  dark: boolean;
  quiz: JSX.Element;
  onHoodOfCard: (id: number) => void;
  onQuizHood: () => void;
}) {
  const [card, setCard] = useState<CardDetail | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    setCard(null);
    setError("");
    api.card(p.id).then(
      (c) => live && setCard(c),
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, [p.id]);

  const node = p.hood?.nodes.find((n) => n.id === p.id);
  const cluster = node?.cluster != null ? p.clusters.get(node.cluster) : undefined;
  return (
    <div className="side-body">
      {error && <p className="err">{error}</p>}
      {!card && !error && <p className="muted">Loading…</p>}
      {card && (
        <>
          <div className="q">{card.question}</div>
          <div className="a">{card.answer}</div>
          <div className="meta">
            <span>{card.deck}</span>
            {cluster && (
              <span>
                <i style={{ background: clusterColor(cluster.id, p.dark) }} /> {cluster.label}
              </span>
            )}
            {card.tags
              .split(" ")
              .filter(Boolean)
              .map((t) => (
                <span key={t} className="tag">
                  {t}
                </span>
              ))}
          </div>
        </>
      )}
      <div className="row">
        <button className="primary" onClick={() => p.onHoodOfCard(p.id)}>
          Show its neighbourhood
        </button>
        {p.hood && (
          <button onClick={p.onQuizHood} title="Study the cards currently shown">
            Quiz these cards
          </button>
        )}
        {p.hood && p.quiz}
      </div>
    </div>
  );
}
