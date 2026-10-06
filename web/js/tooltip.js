/**
 * One floating tooltip for every element with `data-tip` (set by `tip()` in dom.js, or
 * directly on `dataset.tip` / `dataset.tipPos`). It lives in <body>, so a scrolling panel
 * never clips it: the pseudo-element tooltips of the inspector were cut to fragments at the
 * group list's edges and behind the viewer. Shown after a short hover delay or at once on
 * keyboard focus, placed on the requested side (top / bottom / left / right) or flipped to
 * the side that has room, kept inside the viewport and wrapped at a readable width (CSS
 * `.tooltip`). Hidden on leave, blur, any pointer press, Escape, scroll and resize.
 */
const SHOW_DELAY_MS = 350;
const GAP = 8;         // px between the element and the tooltip
const MARGIN = 8;      // px kept from the viewport's edges
const FLIP = { top: 'bottom', bottom: 'top', left: 'right', right: 'left' };

let box = null;
let target = null;
let timer = 0;

function element() {
  if (!box) {
    box = document.createElement('div');
    box.className = 'tooltip';
    box.setAttribute('role', 'tooltip');
    box.hidden = true;
    document.body.append(box);
  }
  return box;
}

function place(el, text, want) {
  const b = element();
  b.textContent = text;
  b.classList.remove('is-visible');
  b.hidden = false;
  const r = el.getBoundingClientRect();
  const w = b.offsetWidth;
  const h = b.offsetHeight;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const room = {
    top: r.top - GAP - h >= MARGIN,
    bottom: r.bottom + GAP + h <= vh - MARGIN,
    left: r.left - GAP - w >= MARGIN,
    right: r.right + GAP + w <= vw - MARGIN,
  };
  let side = want in room ? want : 'top';
  if (!room[side] && room[FLIP[side]]) side = FLIP[side];
  let x;
  let y;
  if (side === 'top' || side === 'bottom') {
    x = r.left + r.width / 2 - w / 2;
    y = side === 'top' ? r.top - GAP - h : r.bottom + GAP;
  } else {
    y = r.top + r.height / 2 - h / 2;
    x = side === 'left' ? r.left - GAP - w : r.right + GAP;
  }
  x = Math.max(MARGIN, Math.min(vw - w - MARGIN, x));
  y = Math.max(MARGIN, Math.min(vh - h - MARGIN, y));
  b.style.left = `${Math.round(x)}px`;
  b.style.top = `${Math.round(y)}px`;
  b.dataset.side = side;
  requestAnimationFrame(() => { if (target === el && !b.hidden) b.classList.add('is-visible'); });
}

function show(el, immediate = false) {
  clearTimeout(timer);
  target = el;
  const go = () => {
    if (target !== el || !el.isConnected) return;
    const text = el.dataset.tip;
    if (text) place(el, text, el.dataset.tipPos || 'top');
  };
  if (immediate) go();
  else timer = setTimeout(go, SHOW_DELAY_MS);
}

/** Hides the tooltip (if any) and forgets its element. */
export function hideTooltip() {
  clearTimeout(timer);
  target = null;
  if (box) {
    box.classList.remove('is-visible');
    box.hidden = true;
  }
}

/** Installs the document-level listeners once (app boot). */
export function initTooltips() {
  document.addEventListener('mouseover', (e) => {
    const el = e.target instanceof Element ? e.target.closest('[data-tip]') : null;
    if (!el) { if (target) hideTooltip(); return; }
    if (el !== target) { hideTooltip(); show(el); }
  });
  document.addEventListener('mouseout', (e) => {
    if (target && !(e.relatedTarget instanceof Node && target.contains(e.relatedTarget))) hideTooltip();
  });
  document.addEventListener('focusin', (e) => {
    const el = e.target instanceof Element ? e.target.closest('[data-tip]') : null;
    // keyboard focus only (a click focuses buttons too, and has just hidden the tooltip)
    if (el && el.matches(':focus-visible')) { hideTooltip(); show(el, true); }
  });
  document.addEventListener('focusout', () => hideTooltip());
  document.addEventListener('pointerdown', () => hideTooltip(), true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideTooltip(); });
  // a scroll or resize moves the element: a shown tooltip goes, a pending one is left to
  // place itself from the element's final position (hovering right after a scroll, or a
  // control that scrolls into view under the pointer, still gets its tooltip)
  document.addEventListener('scroll', () => hideShown(), true);
  window.addEventListener('resize', () => hideShown());
}

function hideShown() {
  if (box && !box.hidden) hideTooltip();
}
