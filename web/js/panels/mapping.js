/**
 * Mapping panel: one row per group `group swatch → target swatch`. Targets come from
 * a strategy suggestion, a palette swatch dropped on the row, or the row's own colour
 * picker. Clearing a row keeps the group's original colour. All changes go through the
 * store's undoable `setTarget`/`commitMapping`.
 */
import { h, icon, tip, replace } from '../dom.js';
import { flip, pop } from '../motion.js';
import { normalizeHex, textColorOn } from '../util.js';
import { COLOR_DRAG_TYPE } from './palette.js';
import { panel, iconButton } from './panel.js';

export const STRATEGIES = [
  { value: 'balanced', label: 'Balanced', title: 'Area, lightness and hue together' },
  { value: 'area', label: 'Area', title: 'Largest group gets the dominant colour' },
  { value: 'luminance', label: 'Luminance', title: 'Keep the light/dark structure' },
  { value: 'hue', label: 'Hue', title: 'Closest hue wins' },
  { value: 'contrast', label: 'Contrast', title: 'Preserve light/dark relationships' },
];

/**
 * @param store Studio store
 * @param {{suggest: (strategy: string) => Promise}} actions
 */
export function createMappingPanel(store, actions) {
  const strategy = h('select.select.select-sm', { aria: { label: 'Mapping strategy' } },
    STRATEGIES.map((s) => h('option', { value: s.value, title: s.title }, s.label)));
  strategy.value = store.get().strategy;
  strategy.addEventListener('change', () => store.set({ strategy: strategy.value }));
  const suggestBtn = h('button.btn.btn-primary.btn-sm', { type: 'button', onClick: () => onSuggest() }, icon('spark', { size: 14 }), 'Suggest');
  tip(suggestBtn, 'Assign palette colours to groups automatically');

  const list = h('ul.map-list', { role: 'list' });
  const empty = h('div.panel-empty', { hidden: true }, icon('arrow', { size: 22 }), h('p', 'Groups appear here once the analysis is done.'));
  const hint = h('p.field-hint.map-hint', 'Drop a palette swatch on a row, or click a target to pick any colour. Empty targets keep the original paint.');

  const undoBtn = iconButton('undo', 'Undo (⌘Z)', () => store.undoMapping(), { size: 14 });
  const redoBtn = iconButton('undo', 'Redo (⇧⌘Z)', () => store.redoMapping(), { size: 14, className: 'is-flipped' });
  const clearBtn = h('button.btn.btn-sm.btn-ghost', { type: 'button', onClick: () => store.clearMapping() }, 'Clear all');
  const footer = h('div.panel-footer', undoBtn, redoBtn, h('span.spacer'), clearBtn);

  const p = panel({ id: 'mapping', title: 'Mapping', icon: 'arrow', actions: [strategy, suggestBtn] });
  p.body.append(empty, hint, list, footer);

  const rows = new Map();

  async function onSuggest() {
    if (!store.get().palette?.colors?.length) {
      p.el.dispatchEvent(new CustomEvent('chroma:need-palette', { bubbles: true }));
      return;
    }
    suggestBtn.disabled = true;
    suggestBtn.classList.add('is-loading');
    try { await actions.suggest(strategy.value); } finally {
      suggestBtn.disabled = false;
      suggestBtn.classList.remove('is-loading');
    }
  }

  function buildRow(g) {
    const picker = h('input.map-picker', { type: 'color', value: g.albedo_hex, aria: { label: `Target colour for ${g.name}` } });
    picker.addEventListener('input', () => setTargetLive(g.id, picker.value));
    picker.addEventListener('change', () => store.setTarget(g.id, normalizeHex(picker.value)));
    const targetFace = h('span.map-target-face');
    const targetLabel = h('span.map-target-label', 'Original');
    const targetBtn = h('button.map-target', { type: 'button', aria: { label: `Pick target colour for ${g.name}` }, onClick: () => picker.click() }, targetFace, targetLabel, picker);
    const clear = h('button.btn-icon.map-clear', { type: 'button', aria: { label: 'Keep original colour' }, onClick: (e) => { e.stopPropagation(); store.setTarget(g.id, null); } }, icon('x', { size: 12 }));
    tip(clear, 'Keep original');
    const name = h('span.map-name', g.name);
    const flags = h('span.map-flags');
    const li = h('li.map-row', {
      dataset: { key: String(g.id), gid: String(g.id) }, role: 'listitem',
      onMouseenter: () => store.set({ hoverGroup: g.id }),
      onMouseleave: () => { if (store.get().hoverGroup === g.id) store.set({ hoverGroup: null }); },
      onClick: (e) => { if (!(e.target instanceof Element && e.target.closest('button, input'))) store.select(g.id); },
      onDragover: (e) => {
        if (!e.dataTransfer.types.includes(COLOR_DRAG_TYPE) || store.groupById(g.id)?.locked) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        li.classList.add('is-drop-target');
      },
      onDragleave: () => li.classList.remove('is-drop-target'),
      onDrop: (e) => {
        li.classList.remove('is-drop-target');
        const hex = normalizeHex(e.dataTransfer.getData(COLOR_DRAG_TYPE));
        if (!hex) return;
        e.preventDefault();
        store.setTarget(g.id, hex);
        pop(targetBtn);
      },
    },
      h('span.map-source', { style: { '--swatch': g.albedo_hex } }),
      h('div.map-main', h('div.map-title', name, flags), h('span.map-sub.mono', g.albedo_hex)),
      h('span.map-arrow', icon('arrow', { size: 16 })),
      targetBtn, clear);
    return { li, name, flags, targetFace, targetLabel, targetBtn, picker, clear };
  }

  function setTargetLive(gid, hex) {
    // Preview while the native picker is dragged; the store commit happens on `change`.
    const row = rows.get(gid);
    if (row) paintTarget(row, normalizeHex(hex), store.groupById(gid));
  }

  function paintTarget(row, hex, g) {
    const has = Boolean(hex);
    row.li.classList.toggle('is-mapped', has);
    row.targetFace.style.setProperty('--swatch', has ? hex : 'transparent');
    row.targetFace.style.color = has ? textColorOn(hex) : '';
    row.targetLabel.textContent = has ? hex : 'Original';
    row.targetLabel.classList.toggle('mono', has);
    row.clear.hidden = !has;
    if (has) row.picker.value = hex;
    else if (g) row.picker.value = g.albedo_hex;
  }

  function updateRow(row, g, hex) {
    row.name.textContent = g.name;
    row.li.classList.toggle('is-locked', g.locked);
    row.li.querySelector('.map-source').style.setProperty('--swatch', g.albedo_hex);
    row.li.querySelector('.map-sub').textContent = g.albedo_hex;
    replace(row.flags,
      g.locked ? tip(h('span.badge.badge-lock', icon('lock', { size: 10 }), 'Locked'), 'Locked groups are never recolored') : null,
      g.is_background ? h('span.badge.badge-bg', 'Background') : null);
    row.targetBtn.disabled = g.locked;
    paintTarget(row, g.locked ? null : hex, g);
  }

  function render(state) {
    const { groups, mapping } = state;
    empty.hidden = groups.length > 0;
    hint.hidden = groups.length === 0;
    footer.hidden = groups.length === 0;
    const seen = new Set();
    flip(list, () => {
      for (const g of groups) {
        seen.add(g.id);
        let row = rows.get(g.id);
        if (!row) { row = buildRow(g); rows.set(g.id, row); }
        updateRow(row, g, mapping[String(g.id)] || null);
        list.appendChild(row.li);
      }
      for (const [gid, row] of rows) if (!seen.has(gid)) { row.li.remove(); rows.delete(gid); }
    });
    const mapped = Object.values(mapping).filter(Boolean).length;
    p.setBadge(mapped ? `${mapped}/${groups.length}` : String(groups.length));
    undoBtn.disabled = !store.canUndo();
    redoBtn.disabled = !store.canRedo();
    clearBtn.disabled = mapped === 0;
  }

  function renderSelection(sel) {
    for (const [gid, row] of rows) row.li.classList.toggle('is-selected', sel.groupId === gid);
  }

  const offs = [
    store.watch(render, ['groups', 'mapping']),
    store.subscribe((s) => renderSelection(s.selection), ['selection']),
    store.subscribe((s) => { strategy.value = s.strategy; }, ['strategy']),
  ];

  return {
    el: p.el,
    panel: p,
    /** Open the colour picker of a group's row (used by the selection pill). */
    pickFor(gid) { rows.get(gid)?.picker.click(); },
    destroy() { offs.forEach((f) => f()); },
  };
}
