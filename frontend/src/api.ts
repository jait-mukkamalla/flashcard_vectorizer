import type { CardDetail, ClusterData, Deck, Hood, Overview, Status } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : r.statusText);
  }
  return (r.status === 204 ? undefined : await r.json()) as T;
}

/** Query string with repeated keys for arrays (deck=1&deck=2); empty values are dropped. */
function qs(params: Record<string, string | number | number[] | undefined>): string {
  const out = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === "") continue;
    for (const v of Array.isArray(value) ? value : [value]) out.append(key, String(v));
  }
  const s = out.toString();
  return s ? `?${s}` : "";
}

export interface HoodRequest {
  card?: number;
  q?: string;
  hops: number;
  decks: number[];
}

export const api = {
  status: () => request<Status>("/api/status"),
  decks: () => request<Deck[]>("/api/decks"),
  overview: (decks: number[]) => request<Overview>(`/api/graph/overview${qs({ deck: decks })}`),
  cluster: (id: number, decks: number[]) =>
    request<ClusterData>(`/api/graph/clusters/${id}${qs({ deck: decks })}`),
  neighbourhood: ({ card, q, hops, decks }: HoodRequest) =>
    request<Hood>(`/api/graph/neighborhood${qs({ card, q, hops, deck: decks })}`),
  card: (id: number) => request<CardDetail>(`/api/cards/${id}`),
  rebuild: () => request<{ queued: boolean }>("/api/graph/rebuild", { method: "POST" }),
  startSession: (body: Record<string, unknown>) =>
    request<{ id: number; cards: number; title: string }>("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
