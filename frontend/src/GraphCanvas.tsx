import { useEffect, useRef } from "react";
import Graph from "graphology";
import Sigma from "sigma";
import { mutedColor } from "./colors";

/** Ask the canvas to fly to a node and frame a circle of the given radius (world units). */
export interface FocusRequest {
  key: string;
  radius: number;
  token: number;
}

interface Props {
  graph: Graph;
  dark: boolean;
  selectedKey: string | null;
  focus: FocusRequest | null;
  fitToken: number; // change to re-frame the whole graph
  onNodeClick: (key: string) => void;
  onStageClick: () => void;
}

const MAX_HOVER_CHARS = 90;

/** Sigma (WebGL) renderer. All layout is precomputed on the server; this only draws and navigates. */
export function GraphCanvas(props: Props) {
  const box = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const hovered = useRef<string | null>(null);
  const live = useRef(props); // the latest props, read by long-lived Sigma callbacks
  live.current = props;

  useEffect(() => {
    const sigma = new Sigma(new Graph(), box.current!, {
      itemSizesReference: "positions",
      zoomToSizeRatioFunction: (ratio) => ratio,
      renderEdgeLabels: false,
      zIndex: true,
      labelDensity: 0.6,
      labelGridCellSize: 170,
      labelRenderedSizeThreshold: 9,
      labelFont: "system-ui, -apple-system, 'Segoe UI', sans-serif",
      labelColor: { color: props.dark ? "#ecebe6" : "#1d1d1b" },
      minCameraRatio: 0.0005,
      maxCameraRatio: 3,
      defaultDrawNodeHover: (context, data) => {
        const { dark } = live.current;
        const text = (data.label ?? "").length > MAX_HOVER_CHARS ? `${data.label!.slice(0, MAX_HOVER_CHARS)}…` : data.label ?? "";
        if (!text) return;
        context.font = "13px system-ui, sans-serif";
        const w = context.measureText(text).width + 14;
        const x = data.x + 8;
        const y = data.y - 12;
        context.fillStyle = dark ? "#2a2a27" : "#ffffff";
        context.strokeStyle = dark ? "#55544d" : "#cfcdc4";
        context.beginPath();
        context.roundRect(x, y, w, 24, 6);
        context.fill();
        context.stroke();
        context.fillStyle = dark ? "#ecebe6" : "#1d1d1b";
        context.fillText(text, x + 7, y + 16);
      },
      nodeReducer: (key, data) => {
        const { selectedKey, dark } = live.current;
        const g = sigma.getGraph();
        const out = { ...data };
        // Highlight the selected card. Never lift a cluster bubble: it would then sit above its own cards
        // and swallow their clicks.
        if (key === selectedKey && data.kind === "card") {
          out.highlighted = true;
          out.zIndex = 5;
        }
        const focus = hovered.current ?? selectedKey;
        // Looking at a card: fade every other card that is not one of its neighbours.
        if (focus && focus !== key && g.hasNode(focus) && g.getNodeAttribute(focus, "kind") === "card") {
          if (data.kind === "card" && !g.areNeighbors(focus, key)) {
            out.color = mutedColor(dark, 0.14);
            out.label = null;
          }
        }
        return out;
      },
      edgeReducer: (edge, data) => {
        const { selectedKey, dark } = live.current;
        const g = sigma.getGraph();
        const focus = hovered.current ?? selectedKey;
        if (!focus || !g.hasNode(focus) || g.getNodeAttribute(focus, "kind") !== "card") return data;
        const [a, b] = g.extremities(edge);
        if (a === focus || b === focus) return { ...data, color: mutedColor(dark, 0.85), size: data.size * 2.2, zIndex: 1 };
        return { ...data, color: mutedColor(dark, 0.05) };
      },
    });
    sigmaRef.current = sigma;
    sigma.on("clickNode", ({ node }) => live.current.onNodeClick(node));
    sigma.on("clickStage", () => live.current.onStageClick());
    sigma.on("enterNode", ({ node }) => {
      hovered.current = node;
      if (box.current) box.current.style.cursor = "pointer";
      sigma.refresh({ skipIndexation: true });
    });
    sigma.on("leaveNode", () => {
      hovered.current = null;
      if (box.current) box.current.style.cursor = "";
      sigma.refresh({ skipIndexation: true });
    });
    return () => {
      sigma.kill();
      sigmaRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    hovered.current = null;
    sigmaRef.current?.setGraph(props.graph);
  }, [props.graph]);

  useEffect(() => {
    const sigma = sigmaRef.current;
    if (!sigma) return;
    sigma.setSetting("labelColor", { color: props.dark ? "#ecebe6" : "#1d1d1b" });
    sigma.refresh();
  }, [props.dark]);

  useEffect(() => {
    sigmaRef.current?.refresh({ skipIndexation: true });
  }, [props.selectedKey]);

  useEffect(() => {
    sigmaRef.current?.getCamera().animatedReset({ duration: 400 });
  }, [props.fitToken]);

  useEffect(() => {
    const sigma = sigmaRef.current;
    const focus = props.focus;
    if (!sigma || !focus || !sigma.getGraph().hasNode(focus.key)) return;
    const at = sigma.getNodeDisplayData(focus.key);
    if (!at) return;
    const { width, height } = sigma.getGraphDimensions();
    const extent = Math.max(width, height, 1e-9);
    const ratio = Math.min(1, (2.6 * focus.radius) / extent);
    sigma.getCamera().animate({ x: at.x, y: at.y, ratio }, { duration: 500 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.focus?.token]);

  return <div ref={box} className="canvas" />;
}
