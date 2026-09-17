/**
 * Groups panel: every colour group as a row (albedo swatch, name, area bar, region
 * count), lock toggle, background badge, inline rename, and the three ways to change
 * how colours are grouped: pick several rows and Merge them, Split one in two, or
 * Auto-regroup everything with a group-count slider. Dragging one row onto another is
 * kept as a shortcut for merging a pair.
 *
 * Server mutations go through `actions` so the Studio view owns the API calls and
 * error handling; this module only renders state and emits intent.
 */
import { h, icon, tip } from '../dom.js';
import { flip, pop } from '../motion.js';
import { formatPct, textColorOn } from '../util.js';
import { panel, iconButton } from './panel.js';

const DRAG_TYPE = 'application/x-chroma-group';

/**
 * @param store Studio store
 * @param {{merge: (ids: number[]) => Promise, split: (gid: number, k: number) => Promise,
 *          regroup: (opts: {max_groups: number|null}) => Promise, patch: (gid: number, patch: object) => Promise}} actions
 */
export function createGroupsPanel(store, actions) {
  const list = h('ul.group-list', { role: 'list' });
  const empty = h('div.panel-empty', { hidden: true }, icon('layers', { size: 22 }), h('p', 'No groups yet — they appear once the analysis finishes.'));

  const mergeBtn = h('button.btn.btn-sm.btn-primary', { type: 'button', disabled: true, onClick: () => onMerge() }, icon('layers', { size: 14 }), 'Merge');
  tip(mergeBtn, 'Merge the selected groups into one colour');
  const splitBtn = h('button.btn.btn-sm', { type: 'button', disabled: true, onClick: () => onSplit() }, icon('split', { size: 14 }), 'Split');
  tip(splitBtn, 'Split the selected group in two by colour');
  const hint = h('p.panel-hint', 'Click a colour to select it. ',
    h('kbd', navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'),
    '-click to pick several, then paint them together or Merge them into one.');
  const regroupToggle = h('button.btn.btn-sm', { type: 'button', aria: { expanded: false }, onClick: () => toggleRegroup() }, icon('regroup', { size: 14 }), 'Auto-regroup');
  const countInput = h('input.range', { type: 'range', min: 2, max: 17, step: 1, value: 17, id: 'regroup-count', aria: { label: 'Number of groups' } });
  const countValue = h('span.mono.range-value', 'Auto');
  const regroupApply = h('button.btn.btn-primary.btn-sm', { type: 'button', onClick: () => onRegroup() }, 'Regroup');
  const regroupForm = h('div.regroup-form', { hidden: true },
    h('label.range-row', { for: 'regroup-count' }, h('span.range-label', 'Groups'), countInput, countValue),
    h('p.field-hint', 'Auto lets the ΔE threshold decide. A smaller number merges the closest colours first.'),
    regroupApply);
  countInput.addEventListener('input', () => {
    const v = Number(countInput.value);
    countValue.textContent = v >= 17 ? 'Auto' : String(v);
    countInput.style.setProperty('--p', `${((v - 2) / 15) * 100}%`);
  });
  countInput.style.setProperty('--p', '100%');

  const footer = h('div.panel-footer', mergeBtn, splitBtn, regroupToggle);
  const p = panel({ id: 'groups', title: 'Groups', icon: 'layers', badge: '0' });
  p.body.append(empty, list, hint, footer, regroupForm);

  let busy = false;
  const rowEls = new Map();   // gid -> {li, name, bar, count, lock}

  function toggleRegroup() {
    const open = regroupForm.hidden;
    regroupForm.hidden = !open;
    regroupToggle.setAttribute('aria-expanded', String(open));
    regroupToggle.classList.toggle('is-active', open);
  }

  async function run(fn) {
    if (busy) return;
    busy = true;
    p.el.classList.add('is-busy');
    try { await fn(); } finally { busy = false; p.el.classList.remove('is-busy'); }
  }

  function onSplit() {
    const gid = store.get().selection.groupId;
    if (gid === null) return;
    run(() => actions.split(gid, 2));
  }
  function onMerge() {
    const ids = store.get().selection.groupIds;
    if (ids.length < 2) return;
    run(() => actions.merge([...ids]));
  }
  function onRegroup() {
    const v = Number(countInput.value);
    run(() => actions.regroup({ max_groups: v >= 17 ? null : v }));
  }

  // ---------------------------------------------------------------- rows

  function buildRow(g) {
    const swatch = h('span.group-swatch', { style: { background: g.albedo_hex, color: textColorOn(g.albedo_hex) } });
    const name = h('span.group-name', g.name);
    const editBtn = iconButton('edit', 'Rename', (e) => { e.stopPropagation(); startRename(g.id); }, { size: 13, className: 'group-edit' });
    const bar = h('span.group-bar-fill', { style: { width: `${Math.max(2, g.area_frac * 100)}%` } });
    const pct = h('span.group-pct.mono', formatPct(g.area_frac));
    const count = h('span.group-count', `${g.region_ids.length} ${g.region_ids.length === 1 ? 'region' : 'regions'}`);
    const bgBadge = h('button.badge.badge-bg', {
      type: 'button', hidden: !g.is_background, aria: { label: 'Background group. Click to unmark' },
      onClick: (e) => { e.stopPropagation(); run(() => actions.patch(g.id, { is_background: false })); },
    }, 'Background');
    const lock = iconButton(g.locked ? 'lock' : 'unlock', g.locked ? 'Unlock group' : 'Lock group (never recolored)',
      (e) => { e.stopPropagation(); run(() => actions.patch(g.id, { locked: !store.groupById(g.id)?.locked })); },
      { size: 14, className: 'group-lock', pressed: g.locked });
    const li = h('li.group-row', {
      dataset: { key: String(g.id), gid: String(g.id) }, draggable: true, tabindex: 0, role: 'listitem',
      aria: { label: `${g.name}, ${formatPct(g.area_frac)} of the image` },
      onClick: (e) => { if (e.metaKey || e.ctrlKey || e.shiftKey) store.toggleGroup(g.id); else store.select(g.id); },
      onDblclick: (e) => { e.preventDefault(); startRename(g.id); },
      onKeydown: (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          if (e.metaKey || e.ctrlKey || e.shiftKey) store.toggleGroup(g.id); else store.select(g.id);
        }
        if (e.key === 'F2') { e.preventDefault(); startRename(g.id); }
      },
      onMouseenter: () => store.set({ hoverGroup: g.id }),
      onMouseleave: () => { if (store.get().hoverGroup === g.id) store.set({ hoverGroup: null }); },
      onDragstart: (e) => {
        e.dataTransfer.setData(DRAG_TYPE, String(g.id));
        e.dataTransfer.effectAllowed = 'move';
        li.classList.add('is-dragging');
      },
      onDragend: () => li.classList.remove('is-dragging'),
      onDragover: (e) => {
        if (!e.dataTransfer.types.includes(DRAG_TYPE)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        li.classList.add('is-drop-target');
      },
      onDragleave: () => li.classList.remove('is-drop-target'),
      onDrop: (e) => {
        li.classList.remove('is-drop-target');
        const src = Number(e.dataTransfer.getData(DRAG_TYPE));
        if (!Number.isFinite(src) || src === g.id) return;
        e.preventDefault();
        pop(li);
        // Dragging a row that is part of a multi-selection brings the whole selection.
        const selected = store.get().selection.groupIds;
        const sources = selected.includes(src) ? selected.filter((id) => id !== g.id) : [src];
        run(() => actions.merge([g.id, ...sources]));
      },
    },
      h('span.group-drag', icon('drag', { size: 14 })),
      swatch,
      h('div.group-main',
        h('div.group-title', name, editBtn, bgBadge),
        h('div.group-meta', h('span.group-bar', bar), pct, count)),
      lock);
    if (g.locked) li.classList.add('is-locked');
    return { li, name, bar, pct, count, lock, bgBadge, swatch };
  }

  function updateRow(row, g) {
    row.name.textContent = g.name;
    row.bar.style.width = `${Math.max(2, g.area_frac * 100)}%`;
    row.pct.textContent = formatPct(g.area_frac);
    row.count.textContent = `${g.region_ids.length} ${g.region_ids.length === 1 ? 'region' : 'regions'}`;
    row.bgBadge.hidden = !g.is_background;
    row.swatch.style.background = g.albedo_hex;
    row.li.classList.toggle('is-locked', g.locked);
    row.li.setAttribute('aria-label', `${g.name}, ${formatPct(g.area_frac)} of the image`);
    row.lock.replaceChildren(icon(g.locked ? 'lock' : 'unlock', { size: 14 }));
    row.lock.setAttribute('aria-pressed', String(g.locked));
    row.lock.dataset.tip = g.locked ? 'Unlock group' : 'Lock group (never recolored)';
    row.lock.setAttribute('aria-label', row.lock.dataset.tip);
  }

  function startRename(gid) {
    const row = rowEls.get(gid);
    const g = store.groupById(gid);
    if (!row || !g || row.li.querySelector('input')) return;
    const input = h('input.group-rename', { type: 'text', value: g.name, maxlength: 40, aria: { label: 'Group name' } });
    row.name.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = (commit) => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      input.replaceWith(row.name);
      if (commit && value && value !== g.name) {
        row.name.textContent = value;
        run(() => actions.patch(gid, { name: value }));
      }
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      if (e.key === 'Escape') { e.preventDefault(); finish(false); }
      e.stopPropagation();
    });
    input.addEventListener('blur', () => finish(true));
    input.addEventListener('click', (e) => e.stopPropagation());
  }

  function renderGroups(groups) {
    p.setBadge(String(groups.length));
    empty.hidden = groups.length > 0;
    footer.hidden = groups.length === 0;
    const seen = new Set();
    flip(list, () => {
      for (const g of groups) {
        seen.add(g.id);
        let row = rowEls.get(g.id);
        if (!row) {
          row = buildRow(g);
          rowEls.set(g.id, row);
        } else {
          updateRow(row, g);
        }
        list.appendChild(row.li);   // appending in order == reordering
      }
      for (const [gid, row] of rowEls) {
        if (!seen.has(gid)) { row.li.remove(); rowEls.delete(gid); }
      }
    });
    renderSelection(store.get().selection);
  }

  function renderSelection(sel) {
    const ids = sel.groupIds || [];
    for (const [gid, row] of rowEls) {
      row.li.classList.toggle('is-selected', ids.includes(gid));
      row.li.classList.toggle('is-primary', sel.groupId === gid && ids.length > 1);
      row.li.setAttribute('aria-selected', String(ids.includes(gid)));
    }
    mergeBtn.disabled = ids.length < 2;
    mergeBtn.querySelector('.merge-count')?.remove();
    if (ids.length >= 2) mergeBtn.append(h('span.merge-count.mono', String(ids.length)));
    splitBtn.disabled = ids.length !== 1 || (store.groupById(sel.groupId)?.region_ids.length || 0) < 1;
    if (sel.groupId !== null) rowEls.get(sel.groupId)?.li.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  function renderHover(rid) {
    const g = rid === null ? null : store.groupForRegion(rid);
    for (const [gid, row] of rowEls) row.li.classList.toggle('is-hovered', g?.id === gid);
  }

  const offs = [
    store.watch((s) => renderGroups(s.groups), ['groups']),
    store.subscribe((s) => renderSelection(s.selection), ['selection']),
    store.subscribe((s) => renderHover(s.hoverRegion), ['hoverRegion']),
  ];

  return {
    el: p.el,
    panel: p,
    destroy() { offs.forEach((f) => f()); },
  };
}
