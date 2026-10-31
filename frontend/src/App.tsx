import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type HoodRequest } from "./api";
import { deckColor } from "./colors";
import { GraphCanvas, type FocusRequest } from "./GraphCanvas";
import { buildHoodGraph, buildOverviewGraph, cardKey, clusterKey } from "./graphBuilders";
import { Sidebar } from "./Sidebar";
import type { ClusterData, ColorBy, Deck, Hood, Overview, Selection, Status } from "./types";

function useDarkMode(): boolean {
  const query = "(prefers-color-scheme: dark)";
  const [dark, setDark] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setDark(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return dark;
}

const BUSY = new Set(["importing", "embedding", "graphing", "queued"]);
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function App() {
  const dark = useDarkMode();
  const [status, setStatus] = useState<Status | null>(null);
  const [epoch, setEpoch] = useState(0); // bumps whenever a graph build finishes: cluster ids may have changed
  const [decks, setDecks] = useState<Deck[]>([]);
  const [pickedDecks, setPickedDecks] = useState<number[]>([]);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [expanded, setExpanded] = useState<Map<number, ClusterData>>(new Map());
  const [hood, setHood] = useState<Hood | null>(null);
  const [hoodSource, setHoodSource] = useState<Pick<HoodRequest, "card" | "q"> | null>(null);
  const [hops, setHops] = useState(1);
  const [colorBy, setColorBy] = useState<ColorBy>("cluster");
  const [selection, setSelection] = useState<Selection | null>(null);
  const [focus, setFocus] = useState<FocusRequest | null>(null);
  const [fitToken, setFitToken] = useState(0);
  const [query, setQuery] = useState("");
  const [quizSize, setQuizSize] = useState(20);
  const [error, setError] = useState("");
  const token = useRef(0);
  const expandedRef = useRef(expanded);
  expandedRef.current = expanded;
  const seenEpoch = useRef(-1);

  // -- status polling: quickly while a job runs, slowly otherwise ----------------------------
  useEffect(() => {
    let stop = false;
    let timer: number | undefined;
    let phase: string | undefined;
    let version: string | undefined;
    const tick = async () => {
      let delay = 5000;
      try {
        const s = await api.status();
        if (stop) return;
        // A finished build can be missed between two polls, so also notice the graph itself changing.
        const now = `${s.graph.nodes}:${s.graph.clusters}`;
        const built = phase === "graphing" && s.job.phase !== "graphing";
        if (built || (version !== undefined && version !== now)) setEpoch((e) => e + 1);
        phase = s.job.phase;
        version = now;
        setStatus(s);
        delay = BUSY.has(s.job.phase) ? 1000 : 5000;
      } catch {
        delay = 5000;
      }
      if (!stop) timer = window.setTimeout(tick, delay);
    };
    tick();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    api.decks().then(setDecks, (e) => setError(message(e)));
  }, [status?.decks, status?.cards]);

  // -- overview (and the open clusters) follow the deck filter and the graph version ----------
  const ready = status?.graph.ready ?? false;
  useEffect(() => {
    if (!ready) {
      setOverview(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const ov = await api.overview(pickedDecks);
        if (cancelled) return;
        const rebuilt = seenEpoch.current !== epoch;
        seenEpoch.current = epoch;
        const keep = rebuilt ? [] : [...expandedRef.current.keys()].filter((id) => ov.clusters.some((c) => c.id === id));
        const fresh = await Promise.all(keep.map((id) => api.cluster(id, pickedDecks)));
        if (cancelled) return;
        setOverview(ov);
        setExpanded(new Map(fresh.map((d) => [d.id, d])));
        if (rebuilt) {
          setSelection(null);
          setHood(null);
          setHoodSource(null);
        }
        setFitToken((t) => t + 1);
        setError("");
      } catch (e) {
        if (!cancelled) setError(message(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ready, pickedDecks, epoch]);

  // -- neighbourhood mode ---------------------------------------------------------------------
  const loadHood = useCallback(
    async (source: Pick<HoodRequest, "card" | "q">, hopCount: number) => {
      try {
        const h = await api.neighbourhood({ ...source, hops: hopCount, decks: pickedDecks });
        setHood(h);
        setHoodSource(source);
        setSelection(null);
        setFitToken((t) => t + 1);
        setError("");
      } catch (e) {
        setError(message(e));
      }
    },
    [pickedDecks],
  );

  const leaveHood = () => {
    setHood(null);
    setHoodSource(null);
    setSelection(null);
    setFitToken((t) => t + 1);
  };

  // -- the graph that is drawn ------------------------------------------------------------------
  const graph = useMemo(() => {
    if (hood) return buildHoodGraph(hood, colorBy, dark);
    if (overview) return buildOverviewGraph(overview, expanded, colorBy, dark);
    return buildOverviewGraph({ ready: false, nodes: 0, clusters: [], links: [], cards: 0, k: 0, threshold: 0 }, new Map(), colorBy, dark);
  }, [hood, overview, expanded, colorBy, dark]);

  const clusterById = useMemo(() => new Map((overview?.clusters ?? []).map((c) => [c.id, c])), [overview]);
  const deckById = useMemo(() => new Map(decks.map((d) => [d.id, d])), [decks]);

  const expand = useCallback(
    async (id: number) => {
      try {
        const data = await api.cluster(id, pickedDecks);
        setExpanded((m) => new Map(m).set(id, data));
        setSelection({ kind: "cluster", id });
        const radius = clusterById.get(id)?.radius ?? 10;
        setFocus({ key: clusterKey(id), radius, token: ++token.current });
        setError("");
      } catch (e) {
        setError(message(e));
      }
    },
    [pickedDecks, clusterById],
  );

  const toggleCluster = (id: number) => {
    if (expanded.has(id)) {
      setExpanded((m) => {
        const next = new Map(m);
        next.delete(id);
        return next;
      });
    } else {
      void expand(id);
    }
  };

  const onNodeClick = (key: string) => {
    if (!graph.hasNode(key)) return;
    const kind = graph.getNodeAttribute(key, "kind");
    const id = graph.getNodeAttribute(key, "id") as number;
    if (kind === "cluster") {
      if (expanded.has(id)) setSelection({ kind: "cluster", id });
      else void expand(id);
    } else {
      setSelection({ kind: "card", id });
    }
  };

  const startQuiz = async (body: Record<string, unknown>) => {
    try {
      const s = await api.startSession({ ...body, size: quizSize });
      window.location.assign(`/#session=${s.id}`);
    } catch (e) {
      setError(message(e));
    }
  };

  const selectedKey = selection ? (selection.kind === "cluster" ? clusterKey(selection.id) : cardKey(selection.id)) : null;
  const toggleDeck = (id: number) =>
    setPickedDecks((cur) => (cur.includes(id) ? cur.filter((d) => d !== id) : [...cur, id]));

  const job = status?.job;
  const building = job?.phase === "graphing";
  const progress = job?.graph && job.graph.total ? Math.round((100 * job.graph.done) / job.graph.total) : null;

  return (
    <div className="app">
      <header className="bar">
        <a className="back" href="/" title="Search and study">
          ← Search &amp; study
        </a>
        <strong>Knowledge graph</strong>
        <form
          className="search"
          onSubmit={(e) => {
            e.preventDefault();
            if (query.trim()) void loadHood({ q: query.trim() }, hops);
          }}
        >
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Find a topic: show the cards around it…"
          />
          <button className="primary" type="submit" disabled={!ready}>
            Find
          </button>
        </form>
        <label className="muted">
          colour by{" "}
          <select value={colorBy} onChange={(e) => setColorBy(e.target.value as ColorBy)}>
            <option value="cluster">cluster</option>
            <option value="deck">deck</option>
          </select>
        </label>
        {hood ? (
          <>
            <label className="muted">
              hops{" "}
              <select
                value={hops}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  setHops(n);
                  if (hoodSource) void loadHood(hoodSource, n);
                }}
              >
                <option value={1}>1</option>
                <option value={2}>2</option>
              </select>
            </label>
            <button onClick={leaveHood}>Back to overview</button>
          </>
        ) : (
          <button onClick={() => setExpanded(new Map())} disabled={expanded.size === 0}>
            Collapse all
          </button>
        )}
      </header>

      {decks.length > 1 && (
        <div className="decks-bar">
          <span className="muted">Decks</span>
          {decks.map((d) => (
            <label key={d.id} className={pickedDecks.includes(d.id) ? "on" : ""}>
              <input type="checkbox" checked={pickedDecks.includes(d.id)} onChange={() => toggleDeck(d.id)} />
              {colorBy === "deck" && <i style={{ background: deckColor(d.id, dark) }} />}
              {d.name} ({d.cards.toLocaleString()})
            </label>
          ))}
          {pickedDecks.length > 0 && <button onClick={() => setPickedDecks([])}>All decks</button>}
        </div>
      )}

      {(building || error || job?.error) && (
        <div className={`banner ${error || job?.error ? "bad" : ""}`}>
          {error || job?.error || (
            <>
              {ready ? "Updating the graph" : "Building the graph"}: {job?.graph?.stage ?? "starting"}
              {progress !== null && ` ${progress}%`}…
            </>
          )}
        </div>
      )}

      <main className="stage">
        <section className="view">
          <GraphCanvas
            graph={graph}
            dark={dark}
            selectedKey={selectedKey}
            focus={focus}
            fitToken={fitToken}
            onNodeClick={onNodeClick}
            onStageClick={() => setSelection(null)}
          />
          {!ready && <EmptyState status={status} onBuild={() => api.rebuild().catch((e) => setError(message(e)))} />}
          {ready && !hood && overview && overview.clusters.length === 0 && (
            <div className="empty">
              <p>No cards in the selected decks are in the graph yet.</p>
            </div>
          )}
          {ready && overview && (
            <div className="legend muted">
              {hood
                ? `${hood.nodes.length} cards · line thickness = similarity`
                : `${overview.clusters.length.toLocaleString()} clusters · ${overview.cards.toLocaleString()} cards · similarity ≥ ${overview.threshold}`}
            </div>
          )}
        </section>
        <aside className="side">
          <Sidebar
            selection={selection}
            clusters={clusterById}
            decks={deckById}
            expanded={new Set(expanded.keys())}
            truncated={new Set([...expanded].filter(([, d]) => d.truncated).map(([id]) => id))}
            hood={hood}
            colorBy={colorBy}
            dark={dark}
            quizSize={quizSize}
            onQuizSize={setQuizSize}
            onToggleCluster={toggleCluster}
            onQuizCluster={(id) => startQuiz({ cluster: id, decks: pickedDecks.length ? pickedDecks : null })}
            onHoodOfCard={(id) => loadHood({ card: id }, hops)}
            onQuizHood={() => hood && startQuiz({ card_ids: hood.nodes.map((n) => n.id), title: "Neighbourhood quiz" })}
          />
        </aside>
      </main>
    </div>
  );
}

function EmptyState({ status, onBuild }: { status: Status | null; onBuild: () => void }) {
  let body: JSX.Element;
  if (!status) body = <p>Loading…</p>;
  else if (status.cards === 0)
    body = (
      <p>
        There are no cards yet. <a href="/">Import an Anki deck</a> first, and the graph is built once its cards are embedded.
      </p>
    );
  else if (BUSY.has(status.job.phase))
    body = (
      <p>
        {status.job.phase === "graphing" ? "Building the graph…" : "Waiting for the cards to be embedded…"}
        <br />
        <span className="muted">
          {status.embedded.toLocaleString()} of {status.cards.toLocaleString()} cards embedded
        </span>
      </p>
    );
  else if (status.pending > 0)
    body = (
      <p>
        {status.pending.toLocaleString()} cards are not embedded yet. The graph is built from embeddings, so it appears when
        the embedding job finishes.
      </p>
    );
  else
    body = (
      <>
        <p>The graph has not been built yet.</p>
        <button className="primary" onClick={onBuild}>
          Build the graph
        </button>
      </>
    );
  return <div className="empty">{body}</div>;
}
