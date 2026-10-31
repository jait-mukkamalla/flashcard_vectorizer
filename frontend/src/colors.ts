// Colours: hues spread by the golden angle so neighbouring ids look different, in both themes.

function hslToHex(h: number, s: number, l: number): string {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    const c = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(255 * c)
      .toString(16)
      .padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

export function clusterColor(id: number, dark: boolean): string {
  return hslToHex((id * 137.508) % 360, 0.62, dark ? 0.62 : 0.5);
}

/** A different saturation from cluster colours, so the two colouring modes are not confused. */
export function deckColor(id: number, dark: boolean): string {
  return hslToHex((id * 137.508 + 60) % 360, 0.78, dark ? 0.6 : 0.46);
}

// Sigma 3 blends with (ONE, ONE_MINUS_SRC_ALPHA) but passes colours through unpremultiplied, so a plain
// rgba(r, g, b, a) renders additively (washed out). Translucent colours must be premultiplied by hand.
function premultiplied(r: number, g: number, b: number, alpha: number): string {
  return `rgba(${Math.round(r * alpha)}, ${Math.round(g * alpha)}, ${Math.round(b * alpha)}, ${alpha})`;
}

export function withAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return premultiplied((n >> 16) & 255, (n >> 8) & 255, n & 255, alpha);
}

export function mutedColor(dark: boolean, alpha: number): string {
  return dark ? premultiplied(200, 200, 190, alpha) : premultiplied(60, 60, 55, alpha);
}
