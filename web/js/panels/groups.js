/**
 * Groups panel: every group as a row (albedo swatch, name, area bar, region count), lock
 * toggle, background toggle, finish badge (shiny / chrome, advisory), inline rename, the
 * "Ignore background" switch and the three ways to change how colours are grouped: pick
 * several rows and Merge them, Split one, or Auto-regroup everything with a group-count
 * slider. Dragging one row onto another is kept as a shortcut for merging a pair.
 *
 * The rows come in four sections, in this order:
 *   Parts       the detected parts people personalise (a shock spring, the wheel rims, a
 *               grille) and the parts drawn with Select part, each its own group whatever its
 *               colour, named after the part and marked with a Part badge; Split on a part with
 *               several instances splits it into its instances ("Mirror (left)" / "(right)"),
 *               Merge joins them again, and a drawn part's Remove puts its pixels back where
 *               they came from (a detected part it swallowed is that part again).
 *   Colours     the paint colours, largest first.
 *   Minor       tiny groups that are the group next to them under other light (a shadow
 *               sliver, a reflection: within a small colour difference of it), collapsed under
 *               a divider, one click away; each names that group and can be merged into it in
 *               one click. A tiny group of a colour of its own stays among the Colours (a small
 *               part a user may well paint). Never hidden from painting.
 *   Background  last; collapsed while the background is ignored.
 * The sections are a view: selection, painting, Merge and the canvas work across them, and a
 * group selected on the canvas opens its section.
 *
 * While the background is ignored every background group is shown dimmed and locked (the
 * server treats it as locked for rendering, export and suggestions); any group can still be
 * marked or unmarked as background from its row's toggle (the Background badge itself is
 * only a label: as a button it sat at the row's centre and a click meant to select the row
 * unmarked it).
 *
 * Find part (the search row above the list): a phrase ("spring") asks the server for up to
 * five candidates, which the studio outlines on the image; the chips here list them, and a
 * click on one (here or on the image) takes it into Select part, where Enter makes it a group.
 *
 * Server mutations go through `actions` so the Studio view owns the API calls and
 * error handling; this module only renders state and emits intent.
 */
import { h, icon, tip, replace } from '../dom.js';
import { flip, pop } from '../motion.js';
import { formatPct, textColorOn } from '../util.js';
import { panel, iconButton } from './panel.js';

/** A part drawn with Select part or Find part (its kind is `user_<n>`). */
export function isUserPart(g) {
  return Boolean(g && /^user_\d+$/.test(g.part || ''));
}

const DRAG_TYPE = 'application/x-chroma-group';

/** The panel section of a group: 'parts' | 'colours' | 'minor' | 'background'. */
export function sectionOf(g) {
  if (g.is_background) return 'background';
  if (g.part) return 'parts';
  if (g.minor) return 'minor';
  return 'colours';
}

/** A part group with several instances splits by instance; everything else by colour. */
export function splitsByInstance(g) {
  return Boolean(g && g.part && (g.part_instances || 0) > 1);
}

const SECTIONS = [
  { key: 'parts', title: 'Parts', icon: 'scan', collapsible: false,
    tip: 'Detected parts and the parts you selected: each is its own group whatever its colour' },
  { key: 'colours', title: 'Colours', icon: 'palette', collapsible: false, tip: 'The paint colours, largest first' },
  { key: 'minor', title: 'Minor', icon: 'grid', collapsible: true,
    tip: 'Tiny groups in the colour of the group next to them, most of them a shadow or a reflection of it. Paint them like any group, or merge one into its neighbour' },
  { key: 'background', title: 'Background', icon: 'image', collapsible: true,
    tip: 'Backdrop, floor and wall groups' },
];

/**
 * @param store Studio store
 * @param {{merge: (ids: number[]) => Promise, split: (gid: number, k: number, mode?: string) => Promise,
 *          foldInto: (gid: number, into: number) => Promise,
 *          regroup: (opts: {max_groups: number|null}) => Promise, patch: (gid: number, patch: object) => Promise,
 *          setIgnoreBackground: (on: boolean) => Promise, removePart: (gid: number) => Promise,
 *          findParts: (text: string) => Promise, chooseCandidate: (i: number) => void, clearFind: () => void,
 *          startSelect: () => void}} actions
 */
export function createGroupsPanel(store, actions) {
  const list = h('ul.group-list', { role: 'list' });
  const empty = h('div.panel-empty', { hidden: true }, icon('layers', { size: 22 }), h('p', 'No groups yet — they appear once the analysis finishes.'));

  const mergeBtn = h('button.btn.btn-sm.btn-primary', { type: 'button', disabled: true, onClick: () => onMerge() }, icon('layers', { size: 14 }), 'Merge');
  tip(mergeBtn, 'Merge the selected groups into one colour');
  // The label stays "Split" (a part with several instances splits into them, which its tooltip
  // says): a wider label pushed Auto-regroup onto a second line whenever such a part was selected.
  const splitBtn = h('button.btn.btn-sm', { type: 'button', disabled: true, onClick: () => onSplit() }, icon('split', { size: 14 }), 'Split');
  tip(splitBtn, 'Split the selected group in two by colour');
  const hint = h('p.panel-hint', 'Click a group to select it. ',
    h('kbd', navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'),
    '-click to pick several, then paint them together or Merge them into one.');
  const regroupToggle = h('button.btn.btn-sm', { type: 'button', aria: { expanded: false }, onClick: () => toggleRegroup() }, icon('regroup', { size: 14 }), 'Auto-regroup');
  const countInput = h('input.range', { type: 'range', min: 2, max: 17, step: 1, value: 17, id: 'regroup-count', aria: { label: 'Number of groups' } });
  const countValue = h('span.mono.range-value', 'Auto');
  const regroupApply = h('button.btn.btn-primary.btn-sm', { type: 'button', onClick: () => onRegroup() }, 'Regroup');
  const regroupForm = h('div.regroup-form', { hidden: true },
    h('label.range-row', { for: 'regroup-count' }, h('span.range-label', 'Colours'), countInput, countValue),
    h('p.field-hint', 'Auto lets the ΔE threshold decide. A smaller number merges the closest colours first; a detected part, a part locked as another material and a decal keep a group of their own.'),
    regroupApply);
  countInput.addEventListener('input', () => {
    const v = Number(countInput.value);
    countValue.textContent = v >= 17 ? 'Auto' : String(v);
    countInput.style.setProperty('--p', `${((v - 2) / 15) * 100}%`);
  });
  countInput.style.setProperty('--p', '100%');

  const ignoreInput = h('input', { type: 'checkbox', id: 'ignore-background', role: 'switch' });
  ignoreInput.addEventListener('change', () => run(() => actions.setIgnoreBackground(ignoreInput.checked)));
  const ignoreCount = h('span.ignore-count.mono', '');
  const ignoreRow = h('label.switch-row.ignore-row', { for: 'ignore-background' },
    h('span.ignore-label', h('span', 'Ignore background'), ignoreCount),
    h('span.switch', ignoreInput, h('span.switch-knob')));
  tip(ignoreRow, 'Background groups are locked, dimmed and left out of suggestions while this is on', 'bottom');

  // Find part: a phrase, the candidates as chips (and outlined on the image by the studio)
  const findInput = h('input.input.find-input', { type: 'search', maxlength: 60, placeholder: 'Find a part: spring, caliper…',
    autocomplete: 'off', spellcheck: false, enterkeyhint: 'search', aria: { label: 'Find a part by name' } });
  const findBtn = h('button.btn.btn-sm.find-go', { type: 'submit' }, 'Find');
  const findForm = h('form.find-row', { role: 'search', onSubmit: (e) => {
    e.preventDefault();
    const text = findInput.value.trim();
    if (text) actions.findParts(text);
  } }, h('label.find-field', icon('scan', { size: 14 }), findInput), findBtn);
  tip(findBtn, 'OWLv2 looks for it, SAM 2 draws each candidate; or press S and click the part', 'bottom');
  const findResults = h('div.find-results', { hidden: true, aria: { live: 'polite' } });

  const footer = h('div.panel-footer', mergeBtn, splitBtn, regroupToggle);
  const p = panel({ id: 'groups', title: 'Groups', icon: 'layers', badge: '0' });
  p.body.append(ignoreRow, findForm, findResults, empty, list, hint, footer, regroupForm);

  let busy = false;
  const rowEls = new Map();   // gid -> {li, name, bar, count, lock, ...}
  const heads = new Map();    // section key -> {li, count, toggle}
  // Collapsed state per section: Minor starts closed; Background follows the ignore switch
  // until the user opens or closes it by hand.
  const collapsed = { minor: true, background: null };

  function isCollapsed(key) {
    if (key === 'background') return collapsed.background ?? Boolean(store.get().ignoreBackground);
    return Boolean(collapsed[key]);
  }

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
    const g = store.groupById(gid);
    run(() => actions.split(gid, 2, splitsByInstance(g) ? 'instances' : 'colour'));
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

  // ---------------------------------------------------------------- section heads

  function buildHead(sec) {
    const count = h('span.group-section-count.mono', '0');
    const chevron = sec.collapsible ? icon('chevronDown', { size: 14, className: 'group-section-chevron' }) : null;
    const label = h('span.group-section-title', icon(sec.icon, { size: 13 }), h('span', sec.title), count);
    const toggle = sec.collapsible
      ? h('button.group-section-toggle', { type: 'button', aria: { expanded: true }, onClick: () => {
        collapsed[sec.key] = !isCollapsed(sec.key);
        renderGroups(store.get().groups);
      } }, label, chevron)
      : h('div.group-section-toggle.is-static', label);
    tip(toggle, sec.tip, 'left');
    const li = h('li.group-section-head', { dataset: { key: `section:${sec.key}`, section: sec.key }, role: 'presentation' }, toggle);
    return { li, count, toggle };
  }

  // ---------------------------------------------------------------- rows

  function buildRow(g) {
    const swatch = h('span.group-swatch', { style: { background: g.albedo_hex, color: textColorOn(g.albedo_hex) } });
    const name = h('span.group-name', g.name);
    const editBtn = iconButton('edit', 'Rename', (e) => { e.stopPropagation(); startRename(g.id); }, { size: 13, className: 'group-edit' });
    const bar = h('span.group-bar-fill', { style: { width: `${Math.max(2, g.area_frac * 100)}%` } });
    const pct = h('span.group-pct.mono', formatPct(g.area_frac));
    const count = h('span.group-count', '');
    const partBadge = h('span.badge.badge-part', { hidden: !g.part, role: 'img', aria: { label: isUserPart(g) ? 'Your part' : 'Detected part' } },
      icon(isUserPart(g) ? 'wand' : 'scan', { size: 10 }), h('span.badge-text', 'Part'));
    const bgBadge = h('span.badge.badge-bg', { hidden: !g.is_background }, 'Background');
    tip(bgBadge, 'Marked as background: the toggle on the right unmarks it');
    const finish = h('span.badge.badge-finish', { hidden: !g.finish, role: 'img', aria: { label: 'Finish' } }, g.finish || '');
    // The title line wraps: a long name keeps its width and the badges move to a second
    // line, instead of squeezing the name to "Ligh..." (or to nothing) on a background row.
    // Folds a minor row into the group next to it: its regions move there and that group keeps
    // its own lock and background flags (a merge would carry a locked sliver's lock over).
    const mergeParent = iconButton('merge', 'Merge into the group next to it',
      (e) => {
        e.stopPropagation();
        const cur = store.groupById(g.id);
        if (cur && cur.parent !== undefined && cur.parent >= 0) run(() => actions.foldInto(cur.id, cur.parent));
      },
      { size: 14, className: 'group-merge-parent' });
    // a drawn part goes back into the group around it (a merge into that group)
    const removePart = iconButton('trash', 'Remove this part: its pixels go back to the group around it',
      (e) => { e.stopPropagation(); run(() => actions.removePart(g.id)); },
      { size: 14, className: 'group-remove' });
    const bgToggle = iconButton('image', g.is_background ? 'Unmark as background' : 'Mark as background',
      (e) => { e.stopPropagation(); run(() => actions.patch(g.id, { is_background: !store.groupById(g.id)?.is_background })); },
      { size: 14, className: 'group-bg', pressed: g.is_background });
    const lock = iconButton(g.locked ? 'lock' : 'unlock', g.locked ? 'Unlock group' : 'Lock group (never recolored)',
      (e) => {
        e.stopPropagation();
        if (lock.getAttribute('aria-disabled') === 'true') return;
        run(() => actions.patch(g.id, { locked: !store.groupById(g.id)?.locked }));
      },
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
        // the row dropped onto is the group the others join (a drawn part dropped on a colour is removed)
        run(() => actions.merge([g.id, ...sources], g.id));
      },
    },
      h('span.group-drag', icon('drag', { size: 14 })),
      swatch,
      h('div.group-main',
        h('div.group-title', name, editBtn, partBadge, bgBadge, finish),
        h('div.group-meta', h('span.group-bar', bar), pct, count)),
      h('span.group-actions', mergeParent, removePart, bgToggle, lock));
    if (g.locked) li.classList.add('is-locked');
    const row = { li, name, bar, pct, count, lock, bgBadge, bgToggle, finish, swatch, partBadge, mergeParent, removePart, userPart: null };
    updateRow(row, g);
    return row;
  }

  /** Locked in effect: the user's own lock, or a background group while the background is ignored. */
  function effectiveLock(g) {
    return g.locked || (g.is_background && Boolean(store.get().ignoreBackground));
  }

  function metaText(g) {
    const regions = `${g.region_ids.length} ${g.region_ids.length === 1 ? 'region' : 'regions'}`;
    if (g.part && (g.part_instances || 0) > 1) return `${g.part_instances} instances`;
    if (g.minor && !g.is_background) {
      const parent = g.parent >= 0 ? store.groupById(g.parent) : null;
      return parent ? `next to ${parent.name}` : regions;
    }
    return regions;
  }

  function updateRow(row, g) {
    const ignored = g.is_background && Boolean(store.get().ignoreBackground);
    const locked = effectiveLock(g);
    row.name.textContent = g.name;
    row.bar.style.width = `${Math.max(2, g.area_frac * 100)}%`;
    row.pct.textContent = formatPct(g.area_frac);
    row.count.textContent = metaText(g);
    row.partBadge.hidden = !g.part || g.is_background;
    const mine = isUserPart(g);
    if (row.userPart !== mine) {
      row.userPart = mine;
      row.partBadge.replaceChildren(icon(mine ? 'wand' : 'scan', { size: 10 }), h('span.badge-text', 'Part'));
      row.partBadge.setAttribute('aria-label', mine ? 'Your part' : 'Detected part');
    }
    row.partBadge.dataset.tip = mine
      ? 'Your part (Select part): kept as a group of its own whatever its colour. Remove puts it back into the group around it'
      : g.part
        ? `Detected ${(g.part_instances || 0) > 1 ? `${g.part_instances} ${g.part_plural || 'parts'}` : `a ${(g.part_label || 'part').toLowerCase()}`}: kept as a group of its own whatever its colour${(g.part_instances || 0) > 1 ? '. Split separates the instances' : ''}`
        : '';
    row.removePart.hidden = !mine || g.is_background;
    row.removePart.dataset.tipPos = 'left';
    row.partBadge.dataset.tipPos = 'top';
    row.bgBadge.hidden = !g.is_background;
    row.finish.hidden = !g.finish;
    row.finish.textContent = g.finish || '';
    row.finish.dataset.tip = g.finish === 'chrome' ? 'Reflects its surroundings like chrome or glass (advisory: lock it if it should keep its finish)'
      : `Glossy: ${Math.round((g.glint || 0) * 100)}% of its pixels are glints or clipped highlights (${Math.round((g.shiny || 0) * 100)}% carry a highlight)`;
    row.finish.dataset.tipPos = 'top';
    row.swatch.style.background = g.albedo_hex;
    row.li.classList.toggle('is-locked', locked);
    row.li.classList.toggle('is-ignored', ignored);
    row.li.classList.toggle('is-part', Boolean(g.part) && !g.is_background);
    row.li.classList.toggle('is-minor', sectionOf(g) === 'minor');
    const where = isUserPart(g) ? ', your part' : g.part ? ', detected part' : sectionOf(g) === 'minor' ? ', minor' : '';
    row.li.setAttribute('aria-label', `${g.name}, ${formatPct(g.area_frac)} of the image${where}${g.is_background ? ', background' : ''}${ignored ? ', ignored' : ''}`);
    const parent = sectionOf(g) === 'minor' && g.parent >= 0 ? store.groupById(g.parent) : null;
    row.mergeParent.hidden = !parent;
    if (parent) {
      row.mergeParent.dataset.tip = `Merge into ${parent.name}`;
      row.mergeParent.dataset.tipPos = 'left';
      row.mergeParent.setAttribute('aria-label', `Merge into ${parent.name}`);
    }
    row.lock.replaceChildren(icon(locked ? 'lock' : 'unlock', { size: 14 }));
    row.lock.setAttribute('aria-pressed', String(locked));
    // aria-disabled, not disabled: a disabled button gets no hover, so its tooltip (the
    // reason it is locked) would never show
    row.lock.setAttribute('aria-disabled', String(ignored && !g.locked));
    row.lock.dataset.tip = ignored && !g.locked ? 'Locked while the background is ignored' : g.locked ? 'Unlock group' : 'Lock group (never recolored)';
    row.lock.dataset.tipPos = 'left';            // the lock sits on the panel's right edge
    row.lock.setAttribute('aria-label', row.lock.dataset.tip);
    row.bgToggle.setAttribute('aria-pressed', String(g.is_background));
    row.bgToggle.dataset.tip = g.is_background ? 'Unmark as background' : 'Mark as background';
    row.bgToggle.dataset.tipPos = 'left';
    row.bgToggle.setAttribute('aria-label', row.bgToggle.dataset.tip);
  }

  function startRename(gid) {
    const row = rowEls.get(gid);
    const g = store.groupById(gid);
    if (!row || !g || row.li.querySelector('input')) return;
    const input = h('input.group-rename', { type: 'text', value: g.name, maxlength: 48, aria: { label: 'Group name' } });
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
    ignoreRow.hidden = groups.length === 0;
    const nBg = groups.filter((g) => g.is_background).length;
    ignoreCount.textContent = nBg ? `${nBg} ${nBg === 1 ? 'group' : 'groups'}` : 'none found';
    ignoreInput.checked = Boolean(store.get().ignoreBackground);
    ignoreInput.disabled = nBg === 0;
    const bySection = new Map(SECTIONS.map((s) => [s.key, []]));
    for (const g of groups) bySection.get(sectionOf(g)).push(g);
    // largest first in every section (a split or a merge appends ids, it does not reorder them)
    for (const members of bySection.values()) members.sort((a, b) => (b.area - a.area) || (a.id - b.id));
    // One section only (a job analysed before parts, with nothing minor or ignored): no headers.
    const shown = SECTIONS.filter((s) => bySection.get(s.key).length);
    const plain = shown.length === 1 && shown[0].key === 'colours';
    const seen = new Set();
    // Re-appending the rows in order rebuilds the list; keep the user's scroll position, or
    // a lock click on a row far down snaps the list back to the top.
    const scrollTop = list.scrollTop;
    flip(list, () => {
      let at = 0;
      const place = (el) => {
        if (el !== list.children[at]) list.insertBefore(el, list.children[at] || null);
        at += 1;
      };
      for (const sec of SECTIONS) {
        const members = bySection.get(sec.key);
        let head = heads.get(sec.key);
        if (!members.length || plain) {
          head?.li.remove();
          heads.delete(sec.key);
        } else {
          if (!head) { head = buildHead(sec); heads.set(sec.key, head); }
          const closed = sec.collapsible && isCollapsed(sec.key);
          head.count.textContent = String(members.length);
          head.li.classList.toggle('is-collapsed', closed);
          if (sec.collapsible) head.toggle.setAttribute('aria-expanded', String(!closed));
          place(head.li);
        }
        const closed = !plain && sec.collapsible && isCollapsed(sec.key);
        for (const g of members) {
          seen.add(g.id);
          let row = rowEls.get(g.id);
          if (!row) {
            row = buildRow(g);
            rowEls.set(g.id, row);
          } else {
            updateRow(row, g);
          }
          row.li.hidden = closed;
          place(row.li);
        }
      }
      for (const [gid, row] of rowEls) {
        if (!seen.has(gid)) { row.li.remove(); rowEls.delete(gid); }
      }
    });
    list.scrollTop = scrollTop;
    renderSelection(store.get().selection, { scroll: false });
  }

  function renderSelection(sel, { scroll = true } = {}) {
    const ids = sel.groupIds || [];
    for (const [gid, row] of rowEls) {
      row.li.classList.toggle('is-selected', ids.includes(gid));
      row.li.classList.toggle('is-primary', sel.groupId === gid && ids.length > 1);
      row.li.setAttribute('aria-selected', String(ids.includes(gid)));
    }
    mergeBtn.disabled = ids.length < 2;
    mergeBtn.querySelector('.merge-count')?.remove();
    if (ids.length >= 2) mergeBtn.append(h('span.merge-count.mono', String(ids.length)));
    const g = ids.length === 1 ? store.groupById(sel.groupId) : null;
    splitBtn.disabled = !g || (g.region_ids.length || 0) < 1;
    const byInstance = splitsByInstance(g);
    splitBtn.dataset.tip = byInstance ? `Split ${g.name} into its ${g.part_instances} instances` : 'Split the selected group in two by colour';
    splitBtn.setAttribute('aria-label', byInstance ? `Split ${g.name} into its instances` : 'Split');
    // a group selected on the canvas opens its collapsed section, so the selection shows
    const primary = sel.groupId === null ? null : store.groupById(sel.groupId);
    if (primary && scroll) {
      const key = sectionOf(primary);
      const sec = SECTIONS.find((s) => s.key === key);
      if (sec?.collapsible && isCollapsed(key)) {
        collapsed[key] = false;
        renderGroups(store.get().groups);
      }
    }
    if (scroll && sel.groupId !== null) rowEls.get(sel.groupId)?.li.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  function renderHover(rid) {
    const g = rid === null ? null : store.groupForRegion(rid);
    for (const [gid, row] of rowEls) row.li.classList.toggle('is-hovered', g?.id === gid);
  }

  // The chips are built once per Find answer; a hover or a keyboard focus only moves the highlight
  // (rebuilt on every findHover change, a chip that took the focus was replaced under it: the focus
  // fell back to the page, so Tab and Space could never reach a candidate)
  let chips = [];
  function renderFindHover(hover) {
    chips.forEach((chip, i) => chip.classList.toggle('is-hovered', i === hover));
  }
  function renderFind(f, hover) {
    chips = [];
    const busy = Boolean(f?.busy);
    findBtn.classList.toggle('is-loading', busy);
    findBtn.disabled = busy;
    findResults.hidden = !f;
    if (!f) return;
    if (busy) {
      replace(findResults, h('div.find-head', h('span.find-note', h('span.spinner.spinner-sm'), `Looking for “${f.text}”…`)));
      return;
    }
    const cands = f.candidates || [];
    const clear = h('button.link-btn', { type: 'button', onClick: () => actions.clearFind() }, 'Clear');
    if (!cands.length) {
      replace(findResults, h('div.find-head',
        h('span.find-note', f.error || (f.detector === null ? 'Find part needs OWLv2 or Florence-2 on the server.' : `No “${f.text}” found.`)),
        h('button.link-btn', { type: 'button', onClick: () => actions.startSelect() }, 'Select it by hand'), clear));
      return;
    }
    replace(findResults,
      h('div.find-head', h('span.find-note', `${cands.length} for “${f.text}” · click one here or on the image`), clear),
      h('div.find-list', cands.map((c, i) => {
        // three kinds of chip: the group itself (a check), a new outline over a group of what was
        // asked for (overlaps), and one that is mostly a part of another kind (Find "spring"'s
        // exhaust can: an alert, and it comes last)
        const other = !c.existing && c.matches && c.matches.named === false;
        const what = c.existing ? `the ${c.matches?.name || 'part'} group`
          : other ? `mostly the ${c.matches.name} group, another part` : c.matches ? `overlaps the ${c.matches.name} group` : 'a new outline';
        const chip = h('button.find-chip', {
          type: 'button', class: { 'is-hovered': i === hover, 'is-existing': Boolean(c.existing), 'is-other': Boolean(other) },
          aria: { label: `Candidate ${i + 1}, ${what}, ${formatPct(c.mask.area_frac)} of the image` },
          onClick: () => actions.chooseCandidate(i),
          onMouseenter: () => store.set({ findHover: i }),
          onMouseleave: () => { if (store.get().findHover === i) store.set({ findHover: -1 }); },
          onFocus: () => store.set({ findHover: i }),
          onBlur: () => { if (store.get().findHover === i) store.set({ findHover: -1 }); },
        }, h('span.find-num', String(i + 1)), h('span.find-size', formatPct(c.mask.area_frac)),
        c.existing ? h('span.find-known', icon('check', { size: 11 }), c.matches?.name || 'Part')
          : c.matches ? h(`span.find-known.${other ? 'is-other' : 'is-overlap'}`, icon(other ? 'alert' : 'layers', { size: 11 }), c.matches.name) : null,
        c.score === null || c.score === undefined ? null : h('span.find-score.mono', c.score.toFixed(2)));
        chips.push(chip);
        if (c.existing) return tip(chip, `Already the ${c.matches?.name || 'part'} group · click to select it`);
        const conf = c.score === null || c.score === undefined ? 'Florence-2 grounding (no confidence)'
          : `OWLv2 confidence ${c.score.toFixed(2)} · SAM 2 ${c.mask.score.toFixed(2)}`;
        const iou = c.matches ? Math.round(c.matches.iou * 100) : 0;
        if (other) return tip(chip, `Mostly the ${c.matches.name} group (${iou}%), a part of another kind than “${f.text}” · ${conf}`);
        return tip(chip, c.matches ? `Overlaps the ${c.matches.name} group (${iou}%): a new outline of it · ${conf}` : conf);
      })));
  }

  const offs = [
    store.watch((s) => renderFind(s.find, s.findHover), ['find']),
    store.subscribe((s) => renderFindHover(s.findHover), ['findHover']),
    store.watch((s) => renderGroups(s.groups), ['groups', 'ignoreBackground']),
    store.subscribe((s) => renderSelection(s.selection), ['selection']),
    store.subscribe((s) => renderHover(s.hoverRegion), ['hoverRegion']),
  ];

  return {
    el: p.el,
    panel: p,
    destroy() { offs.forEach((f) => f()); },
  };
}
