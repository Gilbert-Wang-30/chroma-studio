/**
 * Small pure helpers: colour math for swatches, formatting, timing.
 * No DOM access here so everything is trivially testable.
 */

/** @returns {[number, number, number] | null} 0..255 components, or null for a bad hex. */
export function hexToRgb(hex) {
  if (typeof hex !== 'string') return null;
  let h = hex.trim().replace(/^#/, '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  if (!/^[0-9a-f]{6}$/i.test(h)) return null;
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** 0..255 components -> "#rrggbb". Values are clamped and rounded. */
export function rgbToHex(r, g, b) {
  const c = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
  return `#${c(r)}${c(g)}${c(b)}`;
}

/** Normalises user-entered colour strings to lowercase "#rrggbb"; null when invalid. */
export function normalizeHex(hex) {
  const rgb = hexToRgb(hex);
  return rgb ? rgbToHex(...rgb) : null;
}

/** Relative luminance (WCAG) of an sRGB hex, 0..1. */
export function luminance(hex) {
  const rgb = hexToRgb(hex) || [0, 0, 0];
  const lin = rgb.map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
}

/** WCAG contrast ratio between two hexes (>= 1). */
export function contrastRatio(a, b) {
  const la = luminance(a);
  const lb = luminance(b);
  const [hi, lo] = la > lb ? [la, lb] : [lb, la];
  return (hi + 0.05) / (lo + 0.05);
}

/** Black or white, whichever reads better on the given swatch colour. */
export function textColorOn(hex) {
  return contrastRatio(hex, '#111111') >= contrastRatio(hex, '#ffffff') ? '#111111' : '#ffffff';
}

/** Mix two hexes; t=0 gives a, t=1 gives b. */
export function mixHex(a, b, t) {
  const ra = hexToRgb(a) || [0, 0, 0];
  const rb = hexToRgb(b) || [0, 0, 0];
  return rgbToHex(...ra.map((v, i) => v + (rb[i] - v) * t));
}

/** Hue in degrees (0..360) and saturation (0..1) of a hex, for sorting swatches. */
export function hexToHsl(hex) {
  const [r, g, b] = (hexToRgb(hex) || [0, 0, 0]).map((v) => v / 255);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  if (d === 0) return { h: 0, s: 0, l };
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = ((g - b) / d) % 6;
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  h = (h * 60 + 360) % 360;
  return { h, s, l };
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const lerp = (a, b, t) => a + (b - a) * t;

/** Percentage string with sensible precision: 0.0034 -> "0.3%", 0.42 -> "42%". */
export function formatPct(frac) {
  const p = frac * 100;
  if (p >= 10) return `${Math.round(p)}%`;
  if (p >= 1) return `${p.toFixed(1)}%`;
  return `${p.toFixed(2)}%`;
}

/** Milliseconds -> "42 ms" / "1.3 s". */
export function formatMs(ms) {
  if (!Number.isFinite(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
}

/** Seconds -> "0.4 s" / "1m 12s". */
export function formatSeconds(s) {
  if (!Number.isFinite(s)) return '—';
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

/** Bytes -> "1.2 MB". */
export function formatBytes(n) {
  if (!Number.isFinite(n)) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${i === 0 ? n : n.toFixed(1)} ${units[i]}`;
}

/** "2 min ago", "yesterday" for a unix timestamp in seconds. */
export function relativeTime(unixSeconds) {
  const diff = Date.now() / 1000 - unixSeconds;
  if (diff < 45) return 'just now';
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  if (diff < 172800) return 'yesterday';
  if (diff < 86400 * 30) return `${Math.round(diff / 86400)} days ago`;
  return new Date(unixSeconds * 1000).toLocaleDateString();
}

/** "3 840 × 2 162" style dimensions. */
export function formatDims(w, h) {
  const f = (n) => Number(n).toLocaleString();
  return `${f(w)} × ${f(h)}`;
}

/** Title-cases a stage or layer key: "shading" -> "Shading". */
export function titleCase(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : '';
}

/** Trailing debounce; the returned function has .cancel() and .flush(). */
export function debounce(fn, wait) {
  let t = null;
  let lastArgs = null;
  const run = () => { t = null; const a = lastArgs; lastArgs = null; fn(...a); };
  const d = (...args) => { lastArgs = args; clearTimeout(t); t = setTimeout(run, wait); };
  d.cancel = () => { clearTimeout(t); t = null; lastArgs = null; };
  d.flush = () => { if (t) { clearTimeout(t); run(); } };
  return d;
}

let uidCounter = 0;
/** Unique DOM-safe id per page load. */
export function uid(prefix = 'id') {
  uidCounter += 1;
  return `${prefix}-${uidCounter.toString(36)}`;
}

/** Shallow equality for flat objects (used to skip no-op state writes). */
export function shallowEqual(a, b) {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => a[k] === b[k]);
}

/** Awaitable pause. */
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Copy text to the clipboard; resolves false when the browser refuses. */
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

/** Reads an image File into {url, width, height}; revokes nothing (caller owns the URL). */
export function readImageFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ url, width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Not a readable image')); };
    img.src = url;
  });
}

/** True when the event target is a text-entry element (shortcuts must not fire). */
export function isTypingTarget(el) {
  if (!el || !(el instanceof Element)) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}
