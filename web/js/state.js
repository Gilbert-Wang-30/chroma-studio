/**
 * A tiny observable store plus the Studio store with an undo stack for the mapping.
 *
 *   const store = createStore({ a: 1 });
 *   store.subscribe((state, changed) => …, ['a']);   // only when `a` changed
 *   store.set({ a: 2 });
 *
 * Writes are shallow merges; a write that changes nothing (by ===) does not notify.
 */

/** Generic observable store over a flat state object. */
export function createStore(initial) {
  let state = { ...initial };
  const subs = new Set();

  function notify(changed) {
    for (const sub of Array.from(subs)) {
      if (!sub.keys || changed.some((k) => sub.keys.has(k))) {
        try { sub.fn(state, changed); } catch (err) { console.error('[store] subscriber failed', err); }
      }
    }
  }

  return {
    get: () => state,
    /** Merge `patch`; returns the list of keys that actually changed. */
    set(patch) {
      const changed = Object.keys(patch).filter((k) => patch[k] !== state[k]);
      if (!changed.length) return changed;
      state = { ...state, ...patch };
      notify(changed);
      return changed;
    },
    /** `set` with a function of the current state. */
    update(fn) { return this.set(fn(state)); },
    /** Subscribe to all changes, or only to `keys`. Returns an unsubscribe function. */
    subscribe(fn, keys = null) {
      const sub = { fn, keys: keys ? new Set(keys) : null };
      subs.add(sub);
      return () => subs.delete(sub);
    },
    /** Subscribe and immediately call with the current state. */
    watch(fn, keys = null) {
      const off = this.subscribe(fn, keys);
      fn(state, keys || Object.keys(state));
      return off;
    },
    destroy() { subs.clear(); },
  };
}

export const LAYERS = ['result', 'original', 'albedo', 'shading', 'regions', 'groups'];

export const DEFAULT_RENDER_OPTIONS = Object.freeze({
  mode: 'shift',
  texture: 1.0,
  feather_px: 1.5,
  keep_residual: true,
  residual_tint: 0.0,
  shading_strength: 1.0,
  saturation: 1.0,
  sharpen_edges: false,
});

const UNDO_LIMIT = 60;

/**
 * The Studio store: one per open job. Beyond the generic store it owns an undo/redo
 * stack for `mapping` (the only state a user edits in rapid, reversible steps).
 *
 * Mapping values are `{ [groupId: string]: hex | null }` — string keys, like the API.
 */
export function createStudioStore(job) {
  const store = createStore({
    job,                                  // the Job JSON, refreshed on every server reply
    groups: job.groups || [],
    palette: null,                        // Palette JSON or null
    mapping: normalizeMapping(job.mapping),
    renderOptions: { ...DEFAULT_RENDER_OPTIONS, ...(job.render_options || {}) },
    strategy: 'balanced',
    layer: 'result',
    compare: false,
    peek: false,
    selection: { groupId: null, groupIds: [], regionIds: [] },
    hoverRegion: null,                    // region id under the cursor (viewer -> UI)
    hoverGroup: null,                     // group id hovered in the list (UI -> viewer)
    render: { busy: false, ms: null, error: null, count: 0 },
    idsVersion: 0,                        // bumped when ids/* must be re-fetched
    lastExport: null,
    analysisStartedAt: null,
  });

  const undo = [];
  const redo = [];

  function normalizeMapping(m) {
    const out = {};
    for (const [k, v] of Object.entries(m || {})) out[String(k)] = v || null;
    return out;
  }

  /** Replace the mapping and record the previous one for undo. */
  function commitMapping(next, { record = true } = {}) {
    const prev = store.get().mapping;
    const norm = normalizeMapping(next);
    if (sameMapping(prev, norm)) return false;
    if (record) {
      undo.push(prev);
      if (undo.length > UNDO_LIMIT) undo.shift();
      redo.length = 0;
    }
    store.set({ mapping: norm });
    return true;
  }

  return Object.assign(store, {
    commitMapping,
    /** Set one group's target (hex or null). */
    setTarget(gid, hex) {
      commitMapping({ ...store.get().mapping, [String(gid)]: hex || null });
    },
    /** One colour onto many groups, as one undo step. Used by the multi-selection pill so
     *  "make all of this black" is a single action even when one paint reads as several
     *  groups, which is common on a glossy surface. */
    setTargets(gids, hex) {
      const next = { ...store.get().mapping };
      for (const gid of gids) next[String(gid)] = hex || null;
      commitMapping(next);
    },
    clearMapping() { commitMapping({}); },
    undoMapping() {
      if (!undo.length) return false;
      redo.push(store.get().mapping);
      store.set({ mapping: undo.pop() });
      return true;
    },
    redoMapping() {
      if (!redo.length) return false;
      undo.push(store.get().mapping);
      store.set({ mapping: redo.pop() });
      return true;
    },
    canUndo: () => undo.length > 0,
    canRedo: () => redo.length > 0,
    /** Apply a fresh Job JSON from the server (groups may have been renumbered). */
    applyJob(nextJob, { keepMapping = true } = {}) {
      const validIds = new Set((nextJob.groups || []).map((g) => String(g.id)));
      const current = store.get().mapping;
      const mapping = keepMapping
        ? Object.fromEntries(Object.entries(current).filter(([k]) => validIds.has(k)))
        : normalizeMapping(nextJob.mapping);
      const sel = store.get().selection;
      // Group ids are renumbered by merge/split/regroup, so keep only what still exists.
      const keptIds = (sel.groupIds || []).filter((id) => validIds.has(String(id)));
      const keptPrimary = validIds.has(String(sel.groupId)) ? sel.groupId : (keptIds[keptIds.length - 1] ?? null);
      store.set({
        job: nextJob,
        groups: nextJob.groups || [],
        mapping: sameMapping(mapping, current) ? current : mapping,
        selection: keptPrimary === null && !keptIds.length && !sel.regionIds.length
          ? { groupId: null, groupIds: [], regionIds: [] }
          : { groupId: keptPrimary, groupIds: keptIds, regionIds: sel.regionIds },
      });
    },
    setRenderOption(key, value) {
      const ro = store.get().renderOptions;
      if (ro[key] === value) return;
      store.set({ renderOptions: { ...ro, [key]: value } });
    },
    resetRenderOptions() { store.set({ renderOptions: { ...DEFAULT_RENDER_OPTIONS } }); },
    /** Select exactly one group (or none), replacing any multi-selection. */
    select(groupId, regionIds = []) {
      const gid = groupId ?? null;
      store.set({ selection: { groupId: gid, groupIds: gid === null ? [] : [gid], regionIds: [...regionIds] } });
    },
    /**
     * Add or remove one group from the multi-selection, for ⌘/⇧-click. The last group
     * added stays the primary one, so the inspector pill and the viewer keep a focus.
     */
    toggleGroup(groupId) {
      if (groupId === null || groupId === undefined) return;
      const sel = store.get().selection;
      const has = sel.groupIds.includes(groupId);
      const groupIds = has ? sel.groupIds.filter((id) => id !== groupId) : [...sel.groupIds, groupId];
      store.set({ selection: { groupId: groupIds[groupIds.length - 1] ?? null, groupIds, regionIds: [] } });
    },
    selectGroups(groupIds) {
      const ids = [...new Set(groupIds)];
      store.set({ selection: { groupId: ids[ids.length - 1] ?? null, groupIds: ids, regionIds: [] } });
    },
    clearSelection() { store.set({ selection: { groupId: null, groupIds: [], regionIds: [] } }); },
    groupById(gid) { return store.get().groups.find((g) => String(g.id) === String(gid)) || null; },
    groupForRegion(rid) { return store.get().groups.find((g) => g.region_ids.includes(rid)) || null; },
  });
}

function sameMapping(a, b) {
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => (a[k] || null) === (b[k] || null));
}

/** Persistent UI preferences (theme etc.) with safe storage access. */
export const prefs = {
  get(key, fallback = null) {
    try { const v = localStorage.getItem(`chroma:${key}`); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`chroma:${key}`, JSON.stringify(value)); } catch { /* private mode */ }
  },
};
