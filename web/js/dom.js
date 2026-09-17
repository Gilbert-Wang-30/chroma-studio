/**
 * Tiny DOM template helpers and the inline icon set.
 *
 *   h('button.btn.btn-primary', { onClick, disabled }, 'Analyze', icon('spark'))
 *
 * Props: `class`/`className` merge with the selector classes; `on<Event>` attaches a
 * listener; `dataset`/`style`/`aria` objects spread; `ref(el)` receives the element;
 * everything else is set as an attribute (booleans toggle, null/undefined skip).
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Builds an element from a "tag.class#id" selector, props and children. */
export function h(selector, props, ...children) {
  if (props !== null && typeof props === 'object' && !(props instanceof Node) && !Array.isArray(props)) {
    // props is a real props object
  } else {
    children.unshift(props);
    props = {};
  }
  const [tag, ...parts] = selector.split(/(?=[.#])/);
  const el = document.createElement(tag || 'div');
  const classes = [];
  for (const p of parts) {
    if (p[0] === '.') classes.push(p.slice(1));
    else if (p[0] === '#') el.id = p.slice(1);
  }
  applyProps(el, props || {}, classes);
  append(el, children);
  return el;
}

function applyProps(el, props, classes) {
  const cls = [...classes];
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) {
      if (key !== 'class' && key !== 'className') continue;
    }
    if (key === 'class' || key === 'className') {
      if (typeof value === 'string') cls.push(...value.split(/\s+/).filter(Boolean));
      else if (Array.isArray(value)) cls.push(...value.filter(Boolean));
      else if (value && typeof value === 'object') cls.push(...Object.keys(value).filter((k) => value[k]));
    } else if (key === 'ref') {
      value(el);
    } else if (key === 'style' && typeof value === 'object') {
      for (const [k, v] of Object.entries(value)) {
        if (k.startsWith('--')) el.style.setProperty(k, v);
        else el.style[k] = v;
      }
    } else if (key === 'dataset' && typeof value === 'object') {
      Object.assign(el.dataset, value);
    } else if (key === 'aria' && typeof value === 'object') {
      for (const [k, v] of Object.entries(value)) el.setAttribute(`aria-${k}`, String(v));
    } else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'html') {
      el.innerHTML = value;
    } else if (key === 'text') {
      el.textContent = value;
    } else if (key === 'value' || key === 'checked' || key === 'selected' || key === 'indeterminate') {
      el[key] = value;
    } else if (value === true) {
      el.setAttribute(key, '');
    } else {
      el.setAttribute(key, String(value));
    }
  }
  if (cls.length) el.className = cls.join(' ');
}

/** Appends strings, nodes, arrays or nulls to an element. */
export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false || c === true) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** Removes every child. */
export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

/** Replaces the children of `el` with `children`. */
export function replace(el, ...children) {
  clear(el);
  return append(el, children);
}

/** A document fragment from children. */
export function frag(...children) {
  return append(document.createDocumentFragment(), children);
}

/** The inline SVG icon set (24 px grid, stroked, `currentColor`). */
const ICONS = {
  upload: '<path d="M12 16V4m0 0-4 4m4-4 4 4"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
  spark: '<path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1M5.6 18.4l2.1-2.1m8.6-8.6 2.1-2.1"/><circle cx="12" cy="12" r="3"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.5"/><path d="m21 16-5-5-8 8"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/>',
  lock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  unlock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.5-2"/>',
  split: '<path d="M4 6h5l6 12h5"/><path d="M4 18h5l6-12h5"/><path d="m17 4 3 2-3 2M17 16l3 2-3 2"/>',
  merge: '<path d="M4 6h4l5 6-5 6H4"/><path d="M20 12h-7"/><path d="m17 9 3 3-3 3"/>',
  regroup: '<path d="M20 11a8 8 0 0 0-14.5-4M4 13a8 8 0 0 0 14.5 4"/><path d="M4 4v4h4M20 20v-4h-4"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  check: '<path d="m5 12 5 5L20 7"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-4"/>',
  arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>',
  wand: '<path d="m15 4 5 5L8 21l-5-5L15 4Z"/><path d="M13 6l5 5"/><path d="M6 3v2M3 6h2M19 15v2M17 19h2"/>',
  download: '<path d="M12 4v12m0 0 4-4m-4 4-4-4"/><path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6"/><path d="M6 7l1 13h10l1-13"/><path d="M9 7V4h6v3"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M4.9 19.1l1.4-1.4m11.4-11.4 1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z"/>',
  keyboard: '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h.01M11 10h.01M15 10h.01M7 14h10"/>',
  fit: '<path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/>',
  one: '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M11 9h2v7"/>',
  zoomIn: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4M11 8v6M8 11h6"/>',
  zoomOut: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4M8 11h6"/>',
  compare: '<path d="M12 3v18"/><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10v4M17 10v4"/>',
  chevron: '<path d="m9 6 6 6-6 6"/>',
  chevronDown: '<path d="m6 9 6 6 6-6"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  alert: '<path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v4M12 17h.01"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  palette: '<path d="M12 3a9 9 0 0 0 0 18h1.5a2 2 0 0 0 0-4H12a2 2 0 0 1 0-4h5a4 4 0 0 0 0-8h-.5"/><circle cx="7.5" cy="10.5" r="1"/><circle cx="12" cy="7" r="1"/><circle cx="16.5" cy="10.5" r="1"/>',
  drag: '<circle cx="9" cy="6" r="1.2"/><circle cx="15" cy="6" r="1.2"/><circle cx="9" cy="12" r="1.2"/><circle cx="15" cy="12" r="1.2"/><circle cx="9" cy="18" r="1.2"/><circle cx="15" cy="18" r="1.2"/>',
  edit: '<path d="M4 20h4l10.5-10.5a2 2 0 0 0-4-4L4 16v4Z"/><path d="m13 7 4 4"/>',
  pin: '<path d="M12 21s-6-5.3-6-10a6 6 0 0 1 12 0c0 4.7-6 10-6 10Z"/><circle cx="12" cy="11" r="2"/>',
  scan: '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><path d="M4 12h16"/>',
  sliders: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  bolt: '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5"/>',
  home: '<path d="m3 11 9-8 9 8"/><path d="M5 10v10h5v-6h4v6h5V10"/>',
  eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  shine: '<path d="M12 4v4M12 16v4M4 12h4M16 12h4"/><path d="m7 7 2.5 2.5M14.5 14.5 17 17M7 17l2.5-2.5M14.5 9.5 17 7"/>',
  dot: '<circle cx="12" cy="12" r="4"/>',
};

/** An inline SVG icon element. Decorative by default (aria-hidden). */
export function icon(name, { size = 18, label = null, className = '' } = {}) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', String(size));
  svg.setAttribute('height', String(size));
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.8');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('class', `icon icon-${name}${className ? ' ' + className : ''}`);
  if (label) {
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', label);
  } else {
    svg.setAttribute('aria-hidden', 'true');
  }
  svg.innerHTML = ICONS[name] || ICONS.dot;
  return svg;
}

/** Simple event delegation: on(el, 'click', '.row', (event, matchedEl) => …). */
export function on(el, type, selector, handler, options) {
  const listener = (event) => {
    const target = event.target instanceof Element ? event.target.closest(selector) : null;
    if (target && el.contains(target)) handler(event, target);
  };
  el.addEventListener(type, listener, options);
  return () => el.removeEventListener(type, listener, options);
}

/** A labelled native control wrapper: <label class="field"><span>Label</span>control</label>. */
export function field(label, control, { className = '', hint = null } = {}) {
  return h(`label.field${className ? '.' + className : ''}`, h('span.field-label', label), control,
    hint ? h('span.field-hint', hint) : null);
}

/** Segmented control: options [{value, label, title?}], returns element with .value and change events. */
export function segmented(options, value, onChange, { ariaLabel = 'Options', size = '' } = {}) {
  const root = h(`div.segmented${size ? '.segmented-' + size : ''}`, { role: 'radiogroup', aria: { label: ariaLabel } });
  const buttons = options.map((o) =>
    h('button.segment', {
      type: 'button', role: 'radio', aria: { checked: o.value === value }, title: o.title || null,
      dataset: { value: o.value },
      onClick: () => { setValue(o.value); onChange?.(o.value); },
    }, o.label));
  append(root, buttons);
  function setValue(v) {
    root.dataset.value = v;
    buttons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value === String(v))));
  }
  setValue(value);
  root.setValue = setValue;
  Object.defineProperty(root, 'value', { get: () => root.dataset.value });
  return root;
}

/** Accessible tooltip via data-tip; styling lives in CSS. */
export function tip(el, text, position = 'top') {
  el.dataset.tip = text;
  el.dataset.tipPos = position;
  if (!el.getAttribute('aria-label') && !el.textContent.trim()) el.setAttribute('aria-label', text);
  return el;
}

/** Skeleton placeholder block. */
export function skeleton(className = '', style = {}) {
  return h(`div.skeleton${className ? '.' + className : ''}`, { style, aria: { hidden: true } });
}

/** A <kbd> chip. */
export function kbd(text) {
  return h('kbd', text);
}
