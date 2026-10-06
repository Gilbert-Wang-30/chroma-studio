/**
 * Canvas viewer for the Studio: layer display, before/after wipe, pan/zoom (wheel,
 * drag, pinch) and region/group highlighting driven by the decoded id maps.
 *
 * Coordinate spaces: "image space" is the working-resolution pixel grid (W × H, the
 * size of `ids/regions.png`). Every layer bitmap is drawn stretched onto that grid, so a
 * preview-resolution render and a working-resolution id map line up exactly. The view
 * transform is `screen = image * s + (tx, ty)` in CSS pixels; devicePixelRatio is folded
 * into the canvas transform so drawing stays crisp on HiDPI screens.
 *
 * Highlights are sprites (small canvases covering a region's bounding box) built from
 * the Uint32Array of region ids once per region/colour and cached; drawing them with a
 * canvas shadow gives the glow. Nothing here touches application state: the viewer
 * reports hover/selection through callbacks and is told what to highlight.
 *
 * Select part (`setTool('segment')`): a click reports a positive point, a Shift-click or a
 * right-click a negative one and a left drag a box (`onToolPoint`, `onToolBox`, image space);
 * wheel and pinch zoom as always, a right or middle drag pans. The viewer draws what it is
 * given: the live mask (`setSegmentMask`, a sprite from `maskSprite`), the points and the box
 * (`setSegmentPrompts`) and the Find part candidates (`setCandidates`: a click inside one
 * reports `onCandidate`). An Alt-click outside the tool reports `onAltClick`.
 */
import { h, icon } from './dom.js';
import { spring, reducedMotion } from './motion.js';
import { clamp } from './util.js';

const HOVER_COLOR = [255, 255, 255];
const SELECT_COLOR = [122, 162, 255];
const REGION_COLOR = [255, 209, 102];
const GROUP_HOVER_COLOR = [122, 162, 255];
/** Select part's live mask (the Part badge's green) and the Find part candidates (amber). */
export const SEGMENT_COLOR = [61, 220, 151];
export const CANDIDATE_COLOR = [255, 181, 71];
const SPRITE_CACHE_LIMIT = 96;
//: Total backing store the sprite cache may hold; a full-frame group sprite is w*h*4.
const SPRITE_CACHE_BYTES = 192 * 1024 * 1024;
const FIT_PADDING = 24;

/**
 * Decodes an id PNG bitmap into typed arrays.
 * @returns {{regions: Uint32Array, w: number, h: number}} for kind "regions" (id = R + 256 G + 65536 B)
 *          or {{groups: Uint8Array}} for kind "groups" (id = R).
 */
export function decodeIds(bitmap, kind) {
  const w = bitmap.width;
  const h = bitmap.height;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true, alpha: false });
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, w, h);
  const n = w * h;
  if (kind === 'groups') {
    const groups = new Uint8Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) groups[i] = data[p];
    return { groups, w, h };
  }
  const regions = new Uint32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) regions[i] = data[p] | (data[p + 1] << 8) | (data[p + 2] << 16);
  return { regions, w, h };
}

/** Bounding boxes per region id: Int32Array [x0, y0, x1, y1] × (maxId + 1), x1/y1 exclusive. */
function computeBoxes(regions, w, h) {
  let maxId = 0;
  for (let i = 0; i < regions.length; i++) if (regions[i] > maxId) maxId = regions[i];
  const boxes = new Int32Array((maxId + 1) * 4);
  for (let i = 0; i < boxes.length; i += 4) { boxes[i] = w; boxes[i + 1] = h; boxes[i + 2] = 0; boxes[i + 3] = 0; }
  for (let y = 0, i = 0; y < h; y++) {
    for (let x = 0; x < w; x++, i++) {
      const b = regions[i] * 4;
      if (x < boxes[b]) boxes[b] = x;
      if (y < boxes[b + 1]) boxes[b + 1] = y;
      if (x + 1 > boxes[b + 2]) boxes[b + 2] = x + 1;
      if (y + 1 > boxes[b + 3]) boxes[b + 3] = y + 1;
    }
  }
  return { boxes, maxId };
}

/**
 * A sprite for a mask the server sent (`{png: 1-bit PNG data URL of its bounding box, bbox:
 * [x0, y0, x1, y1]}`): the 2 px glowing edge and a faint fill in `rgb`, plus the decoded
 * bitmap (`inside`, 1 per pixel of the box) for hit tests. Resolves to null for an empty mask.
 */
export async function maskSprite(entry, rgb = SEGMENT_COLOR, { strength = 1, fill = null } = {}) {
  if (!entry?.png || !entry.bbox) return null;
  // a large mask gets a lighter fill, so the photo under it stays readable
  const frac = Number(entry.area_frac) || 0;
  if (fill === null) fill = frac > 0.2 ? 0.35 : frac > 0.05 ? 0.8 : 1.6;
  const [x0, y0, x1, y1] = entry.bbox;
  const w = x1 - x0;
  const hh = y1 - y0;
  if (w <= 0 || hh <= 0) return null;
  const img = new Image();
  img.src = entry.png;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = w;
  c.height = hh;
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.drawImage(img, 0, 0);
  const src = cx.getImageData(0, 0, w, hh).data;
  const inside = new Uint8Array(w * hh);
  for (let i = 0, p = 0; i < inside.length; i++, p += 4) inside[i] = src[p] > 127 ? 1 : 0;
  const image = new ImageData(w, hh);
  paintBitmap(image.data, inside, w, hh, rgb, strength, fill);
  cx.putImageData(image, 0, 0);
  return { canvas: c, x: x0, y: y0, w, h: hh, inside, area: entry.area || 0, anchor: maskAnchor(inside, w, hh, x0, y0) };
}

/** Where a mask's number marker goes: the middle of its top-most row of pixels (image space), so the
 *  marker sits on the part itself (the top-left corner of the bounding box of a diagonal part lay on
 *  whatever was next to it: Find "spring"'s exhaust candidate was numbered on the seat). */
function maskAnchor(inside, w, hh, x0, y0) {
  for (let y = 0; y < hh; y++) {
    let first = -1;
    let last = -1;
    for (let x = 0; x < w; x++) {                                  // the row's first run of pixels
      if (inside[y * w + x]) { if (first < 0) first = x; last = x; } else if (first >= 0) break;
    }
    if (first >= 0) return { x: x0 + (first + last + 1) / 2, y: y0 + y };
  }
  return { x: x0, y: y0 };
}

/** RGBA of a 0/1 bitmap: pixels with an outside pixel within 2 px get the edge alpha, the rest
 *  a faint fill (outside the bitmap counts as outside, so the box's own edge is drawn). */
function paintBitmap(d, inside, w, hh, rgb, strength, fillScale) {
  const edgeA = Math.round(255 * strength);
  const fillA = Math.round(38 * strength * fillScale);
  const at = (x, y) => (x >= 0 && y >= 0 && x < w && y < hh ? inside[y * w + x] : 0);
  for (let y = 0; y < hh; y++) {
    for (let x = 0; x < w; x++) {
      if (!inside[y * w + x]) continue;
      const edge = !at(x - 1, y) || !at(x + 1, y) || !at(x, y - 1) || !at(x, y + 1)
        || !at(x - 2, y) || !at(x + 2, y) || !at(x, y - 2) || !at(x, y + 2);
      const p = (y * w + x) * 4;
      d[p] = rgb[0]; d[p + 1] = rgb[1]; d[p + 2] = rgb[2];
      d[p + 3] = edge ? edgeA : fillA;
    }
  }
}

/**
 * Creates the viewer inside `host`.
 * @param {HTMLElement} host
 * @param {{onHover?: (regionId: number|null) => void,
 *          onSelect?: (info: {regionId: number|null, groupId: number|null, shift: boolean, meta: boolean, x: number, y: number}) => void,
 *          onZoom?: (scale: number) => void, onWipe?: (frac: number) => void,
 *          onToolPoint?: (p: {x: number, y: number, positive: boolean}) => void,
 *          onToolBox?: (b: {x0: number, y0: number, x1: number, y1: number}) => void,
 *          onCandidate?: (index: number) => void, onCandidateHover?: (index: number) => void,
 *          onAltClick?: (p: {x: number, y: number}) => void}} callbacks
 */
export function createViewer(host, callbacks = {}) {
  const canvas = h('canvas.viewer-canvas', { aria: { hidden: true } });
  const overlay = h('canvas.viewer-overlay', { aria: { hidden: true } });
  const wipeHandle = h('div.wipe-handle', { hidden: true, role: 'slider', tabindex: 0, aria: { label: 'Before / after divider', valuemin: 0, valuemax: 100, valuenow: 50 } },
    h('div.wipe-line'),
    h('div.wipe-grip', icon('compare', { size: 16 })),
  );
  const wipeLabels = h('div.wipe-labels', { hidden: true }, h('span.wipe-label', 'Original'), h('span.wipe-label', 'Result'));
  const stage = h('div.viewer-stage', { tabindex: 0, aria: { label: 'Image viewer' } }, canvas, overlay, wipeHandle, wipeLabels);
  host.appendChild(stage);

  const ctx = canvas.getContext('2d', { alpha: false });
  const octx = overlay.getContext('2d');

  const state = {
    W: 0, H: 0,
    layers: new Map(),
    layer: 'result',
    peek: false,
    compare: false,
    wipe: 0,               // divider position as a fraction of image width (0 = all result)
    s: 1, tx: 0, ty: 0,
    fitted: true,
    cw: 0, ch: 0, dpr: window.devicePixelRatio || 1,
    ids: null,             // {regions, groups, w, h, boxes, maxId}
    hover: null,
    hoverGroup: null,
    selection: { groupId: null, groupIds: [], regionIds: [] },
    sprites: new Map(),
    dirty: false, overlayDirty: false, raf: 0,
    pointers: new Map(),
    drag: null,
    pinch: null,
    cancelSpring: null,
    // Select part: the tool, the live mask sprite, the prompts, the rubber band while a box is
    // being drawn, the Find part candidates and the one under the pointer
    tool: null,
    seg: { mask: null, points: [], box: null, rubber: null, candidates: [], hover: -1 },
    toolDrag: null,
  };

  // ------------------------------------------------------------------ sizing

  const ro = new ResizeObserver(() => resize());
  ro.observe(stage);
  // The canvas backdrop is painted from --viewer-bg: repaint when the theme changes (a hover
  // only redraws the overlay, so the old theme's backdrop stayed until the next pan or zoom).
  const themeObserver = new MutationObserver(() => invalidate());
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  function resize() {
    const rect = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (rect.width === state.cw && rect.height === state.ch && dpr === state.dpr) return;
    state.cw = rect.width;
    state.ch = rect.height;
    state.dpr = dpr;
    for (const c of [canvas, overlay]) {
      c.width = Math.max(1, Math.round(rect.width * dpr));
      c.height = Math.max(1, Math.round(rect.height * dpr));
      c.style.width = `${rect.width}px`;
      c.style.height = `${rect.height}px`;
    }
    if (state.fitted) fit({ animate: false });
    else invalidate();
    // resizing cleared both canvases: draw them now, not in the next frame (a frame of an empty
    // canvas showed as a black flash whenever the layout moved)
    if (state.dirty) drawMain();
    if (state.overlayDirty) drawOverlay();
  }

  function fitScale() {
    if (!state.W || !state.cw) return 1;
    return Math.min((state.cw - FIT_PADDING * 2) / state.W, (state.ch - FIT_PADDING * 2) / state.H);
  }

  function fit({ animate = true } = {}) {
    if (!state.W) return;
    const s = fitScale();
    setTransform(s, (state.cw - state.W * s) / 2, (state.ch - state.H * s) / 2, { animate });
    state.fitted = true;
  }

  function oneToOne() {
    if (!state.W) return;
    const s = 1 / state.dpr;
    const cx = state.cw / 2;
    const cy = state.ch / 2;
    const ix = (cx - state.tx) / state.s;
    const iy = (cy - state.ty) / state.s;
    setTransform(s, cx - ix * s, cy - iy * s, { animate: true });
    state.fitted = false;
  }

  function setTransform(s, tx, ty, { animate = false } = {}) {
    state.cancelSpring?.();
    if (!animate || reducedMotion()) {
      Object.assign(state, { s, tx, ty });
      invalidate();
      callbacks.onZoom?.(state.s);
      return;
    }
    const from = { s: state.s, tx: state.tx, ty: state.ty };
    state.cancelSpring = spring({
      from: 0, to: 1, stiffness: 220, damping: 26,
      onUpdate: (t) => {
        state.s = from.s + (s - from.s) * t;
        state.tx = from.tx + (tx - from.tx) * t;
        state.ty = from.ty + (ty - from.ty) * t;
        invalidate();
        callbacks.onZoom?.(state.s);
      },
    });
  }

  function zoomAt(px, py, factor) {
    const minS = fitScale() * 0.25;
    const maxS = 24 / state.dpr;
    const s = clamp(state.s * factor, minS, maxS);
    const f = s / state.s;
    state.tx = px - (px - state.tx) * f;
    state.ty = py - (py - state.ty) * f;
    state.s = s;
    state.fitted = false;
    invalidate();
    callbacks.onZoom?.(state.s);
  }

  function screenToImage(px, py) {
    return { x: (px - state.tx) / state.s, y: (py - state.ty) / state.s };
  }

  function regionAt(px, py) {
    if (!state.ids) return null;
    const { x, y } = screenToImage(px, py);
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    if (ix < 0 || iy < 0 || ix >= state.ids.w || iy >= state.ids.h) return null;
    return state.ids.regions[iy * state.ids.w + ix];
  }

  function groupAt(px, py) {
    if (!state.ids?.groups) return null;
    const { x, y } = screenToImage(px, py);
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    if (ix < 0 || iy < 0 || ix >= state.ids.w || iy >= state.ids.h) return null;
    return state.ids.groups[iy * state.ids.w + ix];
  }

  // ------------------------------------------------------------------ drawing

  function invalidate() {
    state.dirty = true;
    state.overlayDirty = true;
    schedule();
  }
  function invalidateOverlay() {
    state.overlayDirty = true;
    schedule();
  }
  function schedule() {
    if (state.raf) return;
    state.raf = requestAnimationFrame(() => {
      state.raf = 0;
      if (state.dirty) drawMain();
      if (state.overlayDirty) drawOverlay();
      positionWipe();
    });
  }

  function currentLayerName() {
    if (state.peek) return 'original';
    return state.layer;
  }

  function bitmapFor(name) {
    return state.layers.get(name) || (name === 'result' ? state.layers.get('original') : null) || null;
  }

  function drawMain() {
    state.dirty = false;
    const { dpr, s, tx, ty, W, H } = state;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = getComputedStyle(stage).getPropertyValue('--viewer-bg').trim() || '#0b0d12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!W) return;
    ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * tx, dpr * ty);
    const name = currentLayerName();
    const img = bitmapFor(name);
    if (!img) return;
    const flat = name === 'regions' || name === 'groups';
    ctx.imageSmoothingEnabled = !(flat && s * dpr > 1.5);
    ctx.imageSmoothingQuality = 'high';
    try {
      paintLayers(name, img);
    } catch (err) {
      // a bitmap closed under us (replaced by a newer render): draw the original instead
      const fallback = state.layers.get('original');
      if (fallback && fallback !== img) {
        try { ctx.drawImage(fallback, 0, 0, W, H); } catch { /* nothing to draw yet */ }
      }
    }
  }

  function paintLayers(name, img) {
    const { W, H } = state;
    const original = state.layers.get('original');
    const showWipe = state.compare && !state.peek && state.layer === 'result' && original && state.layers.get('result');
    if (showWipe && state.wipe > 0) {
      ctx.drawImage(original, 0, 0, W, H);
      const wx = state.wipe * W;
      ctx.save();
      try {
        ctx.beginPath();
        ctx.rect(wx, 0, W - wx, H);
        ctx.clip();
        ctx.drawImage(img, 0, 0, W, H);
      } finally {
        ctx.restore();
      }
    } else {
      ctx.drawImage(img, 0, 0, W, H);
    }
  }

  function drawOverlay() {
    state.overlayDirty = false;
    const { dpr, s, tx, ty } = state;
    octx.setTransform(1, 0, 0, 1, 0, 0);
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (!state.ids || !state.W) return;
    octx.setTransform(dpr * s, 0, 0, dpr * s, dpr * tx, dpr * ty);
    octx.imageSmoothingEnabled = s * dpr < 1;
    const sel = state.selection;
    const selGroups = sel.groupIds.length ? sel.groupIds : (sel.groupId === null ? [] : [sel.groupId]);
    if (state.hoverGroup !== null && !selGroups.includes(state.hoverGroup)) {
      drawSprite(groupSprite(state.hoverGroup, GROUP_HOVER_COLOR, 0.35), 0, 0, GROUP_HOVER_COLOR, 10);
    }
    if (selGroups.length) {
      // One sprite for the whole selection, then the primary group again on top so the
      // last group picked reads a little stronger than its companions.
      const others = selGroups.filter((g) => g !== sel.groupId);
      // A wide selection covers most of the photo, and a tinted fill over all of it hides
      // the very result the user is judging: past a few groups, show outlines only.
      const fill = selGroups.length <= 3 ? 1 : selGroups.length <= 6 ? 0.45 : 0;
      if (others.length) drawSprite(groupsSprite(others, SELECT_COLOR, 0.7, fill), 0, 0, SELECT_COLOR, 10);
      if (sel.groupId !== null) drawSprite(groupsSprite([sel.groupId], SELECT_COLOR, 1, fill), 0, 0, SELECT_COLOR, 14);
    }
    for (const rid of sel.regionIds) {
      const sp = regionSprite(rid, REGION_COLOR, 1);
      if (sp) drawSprite(sp.canvas, sp.x, sp.y, REGION_COLOR, 12);
    }
    if (state.hover !== null && !state.peek && state.layer !== 'original' && !state.tool) {
      const sp = regionSprite(state.hover, HOVER_COLOR, 0.9);
      if (sp) drawSprite(sp.canvas, sp.x, sp.y, HOVER_COLOR, 16);
    }
    drawSegment();
  }

  /** Select part's layer: the candidates, the live mask, then (in screen space) the box, the
   *  rubber band, the candidates' numbers and the points. */
  function drawSegment() {
    const seg = state.seg;
    if (!state.tool && !seg.candidates.length) return;
    seg.candidates.forEach((c, i) => {
      if (!c.sprite) return;
      octx.globalAlpha = i === seg.hover ? 1 : 0.62;
      drawSprite(c.sprite.canvas, c.sprite.x, c.sprite.y, CANDIDATE_COLOR, i === seg.hover ? 14 : 6);
      octx.globalAlpha = 1;
    });
    if (seg.mask) drawSprite(seg.mask.canvas, seg.mask.x, seg.mask.y, SEGMENT_COLOR, 14);
    const { dpr } = state;
    const toScreen = (x, y) => ({ x: state.tx + x * state.s, y: state.ty + y * state.s });
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const rect = (b, color, dash) => {
      const a = toScreen(b.x0, b.y0);
      const z = toScreen(b.x1, b.y1);
      octx.save();
      octx.lineWidth = 1.5;
      octx.setLineDash(dash);
      octx.strokeStyle = 'rgba(0,0,0,0.55)';
      octx.strokeRect(a.x + 0.5, a.y + 0.5, z.x - a.x, z.y - a.y);
      octx.lineDashOffset = dash.length ? dash[0] : 0;
      octx.strokeStyle = color;
      octx.strokeRect(a.x + 0.5, a.y + 0.5, z.x - a.x, z.y - a.y);
      octx.restore();
    };
    if (seg.box) rect(seg.box, 'rgba(61,220,151,0.95)', [6, 4]);
    if (seg.rubber) rect(seg.rubber, 'rgba(255,255,255,0.95)', [5, 4]);
    octx.font = '600 11px Inter, system-ui, sans-serif';
    octx.textBaseline = 'middle';
    const placed = [];
    seg.candidates.forEach((c, i) => {
      const a = c.sprite?.anchor || (c.bbox ? { x: c.bbox[0], y: c.bbox[1] } : null);
      if (!a) return;
      // on the part: centred over the middle of its top-most pixels, just above them; a number that
      // would cover an earlier one (parts sharing a top edge) steps aside along the row
      const p = toScreen(a.x, a.y);
      let cx = clamp(p.x, 12, state.cw - 12);
      const cy = clamp(p.y - 11, 12, state.ch - 12);
      for (let k = 0; k < 8 && placed.some((q) => Math.hypot(q.x - cx, q.y - cy) < 19); k++) cx = clamp(cx + 20, 12, state.cw - 12);
      placed.push({ x: cx, y: cy });
      octx.fillStyle = i === seg.hover ? '#ffb547' : 'rgba(255,181,71,0.9)';
      octx.shadowColor = 'rgba(0,0,0,0.45)';
      octx.shadowBlur = 3;
      octx.beginPath();
      octx.arc(cx, cy, 9, 0, Math.PI * 2);
      octx.fill();
      octx.shadowBlur = 0;
      octx.fillStyle = '#1a1206';
      octx.textAlign = 'center';
      octx.fillText(String(i + 1), cx, cy + 0.5);
    });
    for (const [px, py, lab] of seg.points) {
      const p = toScreen(px + 0.5, py + 0.5);
      octx.beginPath();
      octx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      octx.fillStyle = lab ? '#3ddc97' : '#ff6b6b';
      octx.shadowColor = 'rgba(0,0,0,0.5)';
      octx.shadowBlur = 4;
      octx.fill();
      octx.shadowBlur = 0;
      octx.lineWidth = 2;
      octx.strokeStyle = '#ffffff';
      octx.stroke();
      octx.beginPath();
      octx.lineWidth = 1.8;
      octx.strokeStyle = lab ? '#05261a' : '#2a0707';
      octx.moveTo(p.x - 3.2, p.y);
      octx.lineTo(p.x + 3.2, p.y);
      if (lab) { octx.moveTo(p.x, p.y - 3.2); octx.lineTo(p.x, p.y + 3.2); }
      octx.stroke();
    }
  }

  /** Index of the smallest Find part candidate under image point (x, y), or -1. */
  function candidateAt(x, y) {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    let best = -1;
    let bestArea = Infinity;
    state.seg.candidates.forEach((c, i) => {
      const sp = c.sprite;
      if (!sp || ix < sp.x || iy < sp.y || ix >= sp.x + sp.w || iy >= sp.y + sp.h) return;
      if (!sp.inside[(iy - sp.y) * sp.w + (ix - sp.x)]) return;
      if (sp.area < bestArea) { best = i; bestArea = sp.area; }
    });
    return best;
  }

  function drawSprite(sprite, x, y, rgb, blur) {
    if (!sprite) return;
    octx.shadowColor = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.9)`;
    octx.shadowBlur = reducedMotion() ? 0 : blur;
    octx.drawImage(sprite, x, y);
    octx.shadowBlur = 0;
    octx.drawImage(sprite, x, y);
  }

  // ------------------------------------------------------------------ sprites

  function cacheGet(key) {
    const v = state.sprites.get(key);
    if (v) { state.sprites.delete(key); state.sprites.set(key, v); }   // LRU touch
    return v;
  }
  function cacheSet(key, value) {
    state.sprites.set(key, value);
    // Region sprites are small (a bounding box), but a group sprite is full-frame, so a
    // count-only limit is not a memory limit: evict on both. 192 MB of backing store is
    // roughly thirty full-frame sprites at 4 MP.
    let bytes = 0;
    for (const c of state.sprites.values()) bytes += c.width * c.height * 4;
    while (state.sprites.size > SPRITE_CACHE_LIMIT || (bytes > SPRITE_CACHE_BYTES && state.sprites.size > 2)) {
      const oldest = state.sprites.keys().next().value;
      const c = state.sprites.get(oldest);
      bytes -= c.width * c.height * 4;
      state.sprites.delete(oldest);
    }
    return value;
  }

  /** Sprite covering one region's bbox: 2 px glowing edge plus a faint fill. */
  function regionSprite(rid, rgb, strength) {
    const key = `r:${rid}:${rgb.join(',')}:${strength}`;
    const cached = cacheGet(key);
    if (cached) return cached;
    const ids = state.ids;
    if (rid < 0 || rid > ids.maxId) return null;
    const b = rid * 4;
    const x0 = ids.boxes[b]; const y0 = ids.boxes[b + 1]; const x1 = ids.boxes[b + 2]; const y1 = ids.boxes[b + 3];
    if (x1 <= x0 || y1 <= y0) return null;
    const w = x1 - x0;
    const hh = y1 - y0;
    const image = new ImageData(w, hh);
    paintMask(image, ids.regions, ids.w, ids.h, x0, y0, w, hh, (v) => v === rid, rgb, strength);
    const c = document.createElement('canvas');
    c.width = w;
    c.height = hh;
    c.getContext('2d').putImageData(image, 0, 0);
    return cacheSet(key, { canvas: c, x: x0, y: y0 });
  }

  /** Sprite covering the whole image for one group (built once per group + colour). */
  /**
   * One full-frame sprite covering every group in `gids`. Built for the whole set at
   * once rather than one sprite per group: a group sprite is a full-size canvas, so
   * selecting a dozen groups used to allocate a dozen of them and could take the tab
   * down on a large photo.
   */
  function groupsSprite(gids, rgb, strength, fillScale = 1) {
    const ids = state.ids;
    if (!ids || !ids.groups || !gids.length) return null;
    const sorted = [...gids].sort((a, b) => a - b);
    const key = `g:${sorted.join('.')}:${rgb.join(',')}:${strength}:${fillScale}`;
    const cached = cacheGet(key);
    if (cached) return cached;
    const want = sorted.length === 1
      ? ((v) => v === sorted[0])
      : ((set) => (v) => set.has(v))(new Set(sorted));
    const image = new ImageData(ids.w, ids.h);
    paintMask(image, ids.groups, ids.w, ids.h, 0, 0, ids.w, ids.h, want, rgb, strength, fillScale);
    const c = document.createElement('canvas');
    c.width = ids.w;
    c.height = ids.h;
    c.getContext('2d').putImageData(image, 0, 0);
    return cacheSet(key, c);
  }

  function groupSprite(gid, rgb, strength) {
    return groupsSprite([gid], rgb, strength);
  }

  /**
   * Paints an RGBA mask for pixels where `test(map[idx])` holds: edge pixels (a
   * neighbour within 2 px fails the test) get full alpha, interior pixels a faint fill.
   */
  function paintMask(image, map, mw, mh, x0, y0, w, hh, test, rgb, strength, fillScale = 1) {
    const d = image.data;
    const edgeA = Math.round(255 * strength);
    const fillA = Math.round(38 * strength * fillScale);
    for (let y = 0; y < hh; y++) {
      const my = y0 + y;
      for (let x = 0; x < w; x++) {
        const mx = x0 + x;
        const i = my * mw + mx;
        if (!test(map[i])) continue;
        let edge = false;
        // 2 px ring: any of the 8 axis-aligned neighbours at distance 1 or 2 outside the mask.
        if (mx > 0 && !test(map[i - 1])) edge = true;
        else if (mx < mw - 1 && !test(map[i + 1])) edge = true;
        else if (my > 0 && !test(map[i - mw])) edge = true;
        else if (my < mh - 1 && !test(map[i + mw])) edge = true;
        else if (mx > 1 && !test(map[i - 2])) edge = true;
        else if (mx < mw - 2 && !test(map[i + 2])) edge = true;
        else if (my > 1 && !test(map[i - 2 * mw])) edge = true;
        else if (my < mh - 2 && !test(map[i + 2 * mw])) edge = true;
        const p = (y * w + x) * 4;
        d[p] = rgb[0]; d[p + 1] = rgb[1]; d[p + 2] = rgb[2];
        d[p + 3] = edge ? edgeA : fillA;
      }
    }
  }

  // ------------------------------------------------------------------ wipe

  function positionWipe() {
    const show = state.compare && state.layer === 'result' && !state.peek && state.layers.get('result') && state.layers.get('original');
    wipeHandle.hidden = !show;
    wipeLabels.hidden = !show;
    if (!show) return;
    const x = state.tx + state.wipe * state.W * state.s;
    wipeHandle.style.transform = `translateX(${x}px)`;
    wipeHandle.setAttribute('aria-valuenow', String(Math.round(state.wipe * 100)));
  }

  function animateWipe(to) {
    state.cancelSpring?.();
    state.cancelSpring = spring({
      from: state.wipe, to, stiffness: 160, damping: 16,
      onUpdate: (v) => { state.wipe = clamp(v, 0, 1); state.dirty = true; schedule(); },
    });
  }

  wipeHandle.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    e.preventDefault();
    state.cancelSpring?.();
    wipeHandle.setPointerCapture(e.pointerId);
    wipeHandle.classList.add('is-dragging');
    const move = (ev) => {
      const rect = stage.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      state.wipe = clamp((x - state.tx) / (state.W * state.s), 0, 1);
      state.dirty = true;
      schedule();
      callbacks.onWipe?.(state.wipe);
    };
    const up = () => {
      wipeHandle.classList.remove('is-dragging');
      wipeHandle.removeEventListener('pointermove', move);
      wipeHandle.removeEventListener('pointerup', up);
      wipeHandle.removeEventListener('pointercancel', up);
    };
    wipeHandle.addEventListener('pointermove', move);
    wipeHandle.addEventListener('pointerup', up);
    wipeHandle.addEventListener('pointercancel', up);
  });
  wipeHandle.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 0.1 : 0.02;
    if (e.key === 'ArrowLeft') { state.wipe = clamp(state.wipe - step, 0, 1); invalidate(); e.preventDefault(); }
    if (e.key === 'ArrowRight') { state.wipe = clamp(state.wipe + step, 0, 1); invalidate(); e.preventDefault(); }
  });

  // ------------------------------------------------------------------ pointer input

  stage.addEventListener('wheel', (e) => {
    if (!state.W) return;
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0016));
    zoomAt(e.clientX - rect.left, e.clientY - rect.top, factor);
  }, { passive: false });

  stage.addEventListener('pointerdown', (e) => {
    if (!state.W) return;
    stage.focus({ preventScroll: true });
    const rect = stage.getBoundingClientRect();
    state.pointers.set(e.pointerId, { x: e.clientX - rect.left, y: e.clientY - rect.top });
    stage.setPointerCapture(e.pointerId);
    if (state.pointers.size === 2) {
      const [a, b] = [...state.pointers.values()];
      state.pinch = { dist: Math.hypot(a.x - b.x, a.y - b.y), mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } };
      state.drag = null;
      state.toolDrag = null;
      state.seg.rubber = null;
      return;
    }
    if (state.tool && e.button === 0) {
      // Select part: a click is a point, a drag a box (pan with a right or middle drag)
      state.toolDrag = { x0: e.clientX - rect.left, y0: e.clientY - rect.top, moved: false, shift: e.shiftKey };
      e.preventDefault();
      return;
    }
    if (e.button !== 0 && e.button !== 1 && !(state.tool && e.button === 2)) return;
    state.drag = { x0: e.clientX, y0: e.clientY, tx: state.tx, ty: state.ty, moved: false, shift: e.shiftKey, meta: e.metaKey || e.ctrlKey, alt: e.altKey, button: e.button };
  });
  stage.addEventListener('contextmenu', (e) => { if (state.tool) e.preventDefault(); });

  function rubberBox(d, px, py) {
    const a = screenToImage(d.x0, d.y0);
    const b = screenToImage(px, py);
    return {
      x0: clamp(Math.min(a.x, b.x), 0, state.W), y0: clamp(Math.min(a.y, b.y), 0, state.H),
      x1: clamp(Math.max(a.x, b.x), 0, state.W), y1: clamp(Math.max(a.y, b.y), 0, state.H),
    };
  }

  stage.addEventListener('pointermove', (e) => {
    const rect = stage.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    if (state.pointers.has(e.pointerId)) state.pointers.set(e.pointerId, { x: px, y: py });

    if (state.pinch && state.pointers.size === 2) {
      const [a, b] = [...state.pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      state.tx += mid.x - state.pinch.mid.x;
      state.ty += mid.y - state.pinch.mid.y;
      if (state.pinch.dist > 0) zoomAt(mid.x, mid.y, dist / state.pinch.dist);
      state.pinch = { dist, mid };
      return;
    }
    if (state.toolDrag) {
      const d = state.toolDrag;
      if (!d.moved && Math.hypot(px - d.x0, py - d.y0) > 4) d.moved = true;
      if (d.moved) {
        state.seg.rubber = rubberBox(d, px, py);
        invalidateOverlay();
      }
      return;
    }
    if (state.drag) {
      const dx = e.clientX - state.drag.x0;
      const dy = e.clientY - state.drag.y0;
      if (!state.drag.moved && Math.hypot(dx, dy) > 4) {
        state.drag.moved = true;
        stage.classList.add('is-panning');
      }
      if (state.drag.moved) {
        state.tx = state.drag.tx + dx;
        state.ty = state.drag.ty + dy;
        state.fitted = false;
        invalidate();
      }
      return;
    }
    if (state.tool || state.seg.candidates.length) {
      if (e.pointerType === 'mouse' || e.pointerType === 'pen') {
        const { x, y } = screenToImage(px, py);
        const ci = candidateAt(x, y);
        if (ci !== state.seg.hover) {
          state.seg.hover = ci;
          stage.classList.toggle('has-candidate', ci >= 0);
          invalidateOverlay();
          callbacks.onCandidateHover?.(ci);
        }
      }
      if (state.tool) return;
    }
    if (e.pointerType === 'mouse' || e.pointerType === 'pen') setHover(regionAt(px, py));
  });

  function endPointer(e) {
    const rect = stage.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    state.pointers.delete(e.pointerId);
    if (state.pointers.size < 2) state.pinch = null;
    const tdrag = state.toolDrag;
    state.toolDrag = null;
    if (tdrag) {
      const box = state.seg.rubber;
      state.seg.rubber = null;
      invalidateOverlay();
      if (e.type !== 'pointerup' || !state.tool) return;
      if (tdrag.moved) {
        if (box && box.x1 - box.x0 >= 3 && box.y1 - box.y0 >= 3) callbacks.onToolBox?.(box);
        return;
      }
      const { x, y } = screenToImage(px, py);
      if (x < 0 || y < 0 || x >= state.W || y >= state.H) return;
      const ci = tdrag.shift ? -1 : candidateAt(x, y);
      if (ci >= 0) callbacks.onCandidate?.(ci);
      else callbacks.onToolPoint?.({ x, y, positive: !(tdrag.shift || e.shiftKey) });
      return;
    }
    const drag = state.drag;
    state.drag = null;
    stage.classList.remove('is-panning');
    if (drag && !drag.moved && e.type === 'pointerup') {
      const { x, y } = screenToImage(px, py);
      const onImage = x >= 0 && y >= 0 && x < state.W && y < state.H;
      if (state.tool && drag.button === 2) {
        if (onImage) callbacks.onToolPoint?.({ x, y, positive: false });
        return;
      }
      if (drag.button !== 0) return;
      if (!state.tool && state.seg.candidates.length) {
        const ci = onImage ? candidateAt(x, y) : -1;
        if (ci >= 0) { callbacks.onCandidate?.(ci); return; }
      }
      if ((drag.alt || e.altKey) && onImage && callbacks.onAltClick) {
        callbacks.onAltClick({ x, y });
        return;
      }
      const regionId = regionAt(px, py);
      const groupId = groupAt(px, py);
      callbacks.onSelect?.({ regionId, groupId, shift: drag.shift || e.shiftKey, meta: drag.meta || e.metaKey || e.ctrlKey, x: px, y: py });
    }
  }
  stage.addEventListener('pointerup', endPointer);
  stage.addEventListener('pointercancel', endPointer);
  stage.addEventListener('pointerleave', () => setHover(null));
  stage.addEventListener('dblclick', (e) => {
    // in Select part two quick clicks are two points, not a zoom
    if (state.tool) return;
    const rect = stage.getBoundingClientRect();
    if (state.fitted) zoomAt(e.clientX - rect.left, e.clientY - rect.top, 2.5);
    else fit();
  });

  function setHover(rid) {
    if (rid === state.hover) return;
    state.hover = rid;
    stage.classList.toggle('has-region', rid !== null);
    invalidateOverlay();
    callbacks.onHover?.(rid);
  }

  // ------------------------------------------------------------------ public API

  const viewer = {
    el: stage,
    /** Declare the image-space size (working resolution) and fit the view. */
    setImageSize(W, H) {
      if (W === state.W && H === state.H) return;
      state.W = W;
      state.H = H;
      state.sprites.clear();
      resize();
      fit({ animate: false });
    },
    /** Store a bitmap for a layer name; the current layer redraws if it matches. */
    setLayerBitmap(name, bitmap) {
      const old = state.layers.get(name);
      if (old && old !== bitmap && typeof old.close === 'function') old.close();
      state.layers.set(name, bitmap);
      if (!state.W && bitmap) this.setImageSize(bitmap.width, bitmap.height);
      invalidate();
    },
    hasLayer: (name) => state.layers.has(name),
    setLayer(name) {
      if (name === state.layer) return;
      state.layer = name;
      invalidate();
    },
    getLayer: () => state.layer,
    setPeek(on) {
      if (on === state.peek) return;
      state.peek = on;
      invalidate();
    },
    setCompare(on) {
      if (on === state.compare) return;
      state.compare = on;
      animateWipe(on ? 0.5 : 0);
      positionWipe();
    },
    isCompare: () => state.compare,
    /** Provide decoded ids; pass null to clear. */
    setIds(ids) {
      state.sprites.clear();
      if (!ids) { state.ids = null; invalidateOverlay(); return; }
      const { boxes, maxId } = computeBoxes(ids.regions, ids.w, ids.h);
      state.ids = { ...ids, boxes, maxId };
      if (!state.W) this.setImageSize(ids.w, ids.h);
      invalidateOverlay();
    },
    hasIds: () => Boolean(state.ids),
    setSelection(sel) {
      state.selection = {
        groupId: sel?.groupId ?? null,
        groupIds: [...(sel?.groupIds || [])],
        regionIds: [...(sel?.regionIds || [])],
      };
      invalidateOverlay();
    },
    setHoverGroup(gid) {
      if (gid === state.hoverGroup) return;
      state.hoverGroup = gid;
      invalidateOverlay();
    },
    fit: () => fit({ animate: true }),
    oneToOne,
    zoomIn: () => zoomAt(state.cw / 2, state.ch / 2, 1.5),
    zoomOut: () => zoomAt(state.cw / 2, state.ch / 2, 1 / 1.5),
    getZoom: () => state.s * state.dpr,
    isFitted: () => state.fitted,
    /** Bounding box of a region in image space or null. */
    regionBox(rid) {
      if (!state.ids || rid === null || rid > state.ids.maxId) return null;
      const b = rid * 4;
      return { x0: state.ids.boxes[b], y0: state.ids.boxes[b + 1], x1: state.ids.boxes[b + 2], y1: state.ids.boxes[b + 3] };
    },
    imageToScreen(x, y) { return { x: state.tx + x * state.s, y: state.ty + y * state.s }; },
    size: () => ({ cw: state.cw, ch: state.ch, W: state.W, H: state.H }),
    /** Select part on (`'segment'`) or off (null): clicks become prompts, the cursor a crosshair. */
    setTool(tool) {
      const next = tool || null;
      if (next === state.tool) return;
      state.tool = next;
      state.toolDrag = null;
      state.seg.rubber = null;
      stage.classList.toggle('is-tool', Boolean(next));
      if (next) setHover(null);
      invalidateOverlay();
    },
    /** How many pixels of a `maskSprite` lie in each group (a Map gid -> count; empty without ids). */
    groupCounts(sprite) {
      const out = new Map();
      const ids = state.ids;
      if (!sprite?.inside || !ids?.groups) return out;
      const { x: x0, y: y0, w, h: hh, inside } = sprite;
      for (let y = 0; y < hh; y++) {
        const iy = y0 + y;
        if (iy < 0 || iy >= ids.h) continue;
        for (let x = 0; x < w; x++) {
          const ix = x0 + x;
          if (!inside[y * w + x] || ix < 0 || ix >= ids.w) continue;
          const g = ids.groups[iy * ids.w + ix];
          out.set(g, (out.get(g) || 0) + 1);
        }
      }
      return out;
    },
    /** `sprite` (a `maskSprite`) grown by every pixel of the groups `gids` (the pill's "Take all"):
     *  a new sprite of the union, in `rgb`; `sprite` itself without ids or groups. */
    spriteWithGroups(sprite, gids, rgb = SEGMENT_COLOR) {
      const ids = state.ids;
      if (!sprite?.inside || !ids?.groups || !gids?.length) return sprite;
      const want = new Set(gids);
      const { w: W, h: H, groups } = ids;
      let x0 = sprite.x;
      let y0 = sprite.y;
      let x1 = sprite.x + sprite.w;
      let y1 = sprite.y + sprite.h;
      for (let i = 0; i < groups.length; i++) {
        if (!want.has(groups[i])) continue;
        const x = i % W;
        const y = (i - x) / W;
        if (x < x0) x0 = x;
        if (x >= x1) x1 = x + 1;
        if (y < y0) y0 = y;
        if (y >= y1) y1 = y + 1;
      }
      x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(W, x1); y1 = Math.min(H, y1);
      const w = x1 - x0;
      const hh = y1 - y0;
      if (w <= 0 || hh <= 0) return sprite;
      const inside = new Uint8Array(w * hh);
      let area = 0;
      for (let y = 0; y < hh; y++) {
        const iy = y0 + y;
        for (let x = 0; x < w; x++) {
          const ix = x0 + x;
          const sx = ix - sprite.x;
          const sy = iy - sprite.y;
          const inMask = sx >= 0 && sy >= 0 && sx < sprite.w && sy < sprite.h && sprite.inside[sy * sprite.w + sx];
          if (inMask || want.has(groups[iy * W + ix])) { inside[y * w + x] = 1; area++; }
        }
      }
      const c = document.createElement('canvas');
      c.width = w;
      c.height = hh;
      const image = new ImageData(w, hh);
      const frac = area / Math.max(1, W * H);
      paintBitmap(image.data, inside, w, hh, rgb, 1, frac > 0.2 ? 0.35 : frac > 0.05 ? 0.8 : 1.6);
      c.getContext('2d').putImageData(image, 0, 0);
      return { canvas: c, x: x0, y: y0, w, h: hh, inside, area, anchor: maskAnchor(inside, w, hh, x0, y0) };
    },
    /** The live mask (a `maskSprite`), or null. */
    setSegmentMask(sprite) {
      state.seg.mask = sprite || null;
      invalidateOverlay();
    },
    /** The prompt drawn over the image: points `[x, y, 1|0]` and a box `{x0, y0, x1, y1}` or null. */
    setSegmentPrompts({ points = [], box = null } = {}) {
      state.seg.points = points.map((p) => [p[0], p[1], p[2]]);
      state.seg.box = box ? { ...box } : null;
      invalidateOverlay();
    },
    /** Find part candidates `[{sprite, bbox}]` (numbered 1..n on the image); [] clears them. */
    setCandidates(list) {
      state.seg.candidates = [...(list || [])];
      state.seg.hover = -1;
      stage.classList.remove('has-candidate');
      invalidateOverlay();
    },
    setCandidateHover(i) {
      if (i === state.seg.hover) return;
      state.seg.hover = i;
      invalidateOverlay();
    },
    /** Pan (no zoom change) so the image box `[x0, y0, x1, y1]` is in view, when it is not. */
    reveal(bbox) {
      if (!bbox || !state.W) return;
      const a = this.imageToScreen(bbox[0], bbox[1]);
      const b = this.imageToScreen(bbox[2], bbox[3]);
      const m = 24;
      if (a.x >= m && a.y >= m && b.x <= state.cw - m && b.y <= state.ch - m) return;
      const cx = (bbox[0] + bbox[2]) / 2;
      const cy = (bbox[1] + bbox[3]) / 2;
      setTransform(state.s, state.cw / 2 - cx * state.s, state.ch / 2 - cy * state.s, { animate: true });
      state.fitted = false;
    },
    invalidate,
    destroy() {
      ro.disconnect();
      themeObserver.disconnect();
      state.cancelSpring?.();
      cancelAnimationFrame(state.raf);
      for (const bmp of state.layers.values()) bmp?.close?.();
      state.layers.clear();
      state.sprites.clear();
      stage.remove();
    },
  };
  return viewer;
}
