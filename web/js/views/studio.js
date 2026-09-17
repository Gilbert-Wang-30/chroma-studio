/**
 * Studio view: the workspace for one job. Owns the Studio store, the viewer, the
 * inspector panels, the SSE subscription, the debounced/abortable render loop, state
 * persistence and the keyboard shortcuts. Panels never call the API themselves; they
 * receive `actions` closures defined here so error handling lives in one place.
 */
import { api, layerUrl, idsUrl, subscribeJob, createRenderQueue, ApiError } from '../api.js';
import { h, icon, tip, kbd, skeleton, replace } from '../dom.js';
import { enter, enterChildren, pop } from '../motion.js';
import { createStudioStore, LAYERS } from '../state.js';
import { toast } from '../toast.js';
import { debounce, formatDims, formatMs, formatPct, isTypingTarget, titleCase, textColorOn } from '../util.js';
import { createViewer, decodeIds } from '../viewer.js';
import { createStepper } from '../panels/stepper.js';
import { createGroupsPanel } from '../panels/groups.js';
import { createPalettePanel } from '../panels/palette.js';
import { createMappingPanel } from '../panels/mapping.js';
import { createFinishPanel } from '../panels/finish.js';

const LAYER_API = { result: null, original: 'work', albedo: 'albedo', shading: 'shading', regions: 'regions', groups: 'groups' };
const RENDER_DEBOUNCE_MS = 80;
const PERSIST_DEBOUNCE_MS = 700;

export function studioView(root, { jobId, navigate }) {
  const disposers = [];
  let destroyed = false;

  // ---------------------------------------------------------------- skeleton while the job loads
  const shell = h('div.studio', { aria: { busy: true } },
    h('div.studio-viewer', h('div.viewer-toolbar', skeleton('sk-line', { width: '180px' })), h('div.viewer-host', skeleton('sk-fill'))),
    h('aside.studio-inspector', skeleton('sk-panel'), skeleton('sk-panel'), skeleton('sk-panel')));
  replace(root, shell);

  api.job(jobId).then(boot).catch((err) => {
    if (destroyed) return;
    replace(root, h('div.empty-view',
      icon('alert', { size: 28 }),
      h('h2', err instanceof ApiError && err.status === 404 ? 'This job does not exist any more' : 'Could not open the job'),
      h('p', err?.userMessage || err?.message || ''),
      h('a.btn.btn-primary', { href: '#/' }, icon('home', { size: 15 }), 'Back to Home')));
    enter(root.firstElementChild);
  });

  // ---------------------------------------------------------------- boot
  function boot(job) {
    if (destroyed) return;
    try { sessionStorage.setItem('chroma:lastJob', job.id); } catch { /* ignore */ }
    const store = createStudioStore(job);
    const renderQueue = createRenderQueue(job.id);
    const layerPromises = new Map();
    let idsLoaded = false;
    let assetsRequested = false;
    let closeEvents = null;

    // ---- toolbar
    const nameEl = h('span.toolbar-name', job.name);
    const dimsEl = h('span.chip.mono', formatDims(job.image.width, job.image.height));
    const tabs = LAYERS.map((name, i) => h('button.layer-tab', {
      type: 'button', role: 'tab', dataset: { layer: name }, aria: { selected: name === 'result' }, disabled: true,
      onClick: () => setLayer(name),
    }, h('span', titleCase(name)), kbd(String(i + 1))));
    const tabList = h('div.layer-tabs', { role: 'tablist', aria: { label: 'Layers' } }, tabs);

    const compareBtn = tip(h('button.btn-icon.toolbar-btn', { type: 'button', aria: { label: 'Compare before / after', pressed: false }, onClick: () => store.set({ compare: !store.get().compare }) }, icon('compare')), 'Compare (C)');
    const zoomOut = tip(h('button.btn-icon.toolbar-btn', { type: 'button', aria: { label: 'Zoom out' }, onClick: () => viewer.zoomOut() }, icon('zoomOut')), 'Zoom out (−)');
    const zoomIn = tip(h('button.btn-icon.toolbar-btn', { type: 'button', aria: { label: 'Zoom in' }, onClick: () => viewer.zoomIn() }, icon('zoomIn')), 'Zoom in (+)');
    const zoomPct = h('button.zoom-pct.mono', { type: 'button', aria: { label: 'Zoom level, click to fit' }, onClick: () => viewer.fit() }, '100%');
    const fitBtn = tip(h('button.btn-icon.toolbar-btn', { type: 'button', aria: { label: 'Fit to window' }, onClick: () => viewer.fit() }, icon('fit')), 'Fit (F)');
    const oneBtn = tip(h('button.btn-icon.toolbar-btn', { type: 'button', aria: { label: 'Actual pixels' }, onClick: () => viewer.oneToOne() }, icon('one')), '1:1 (0)');
    const toolbar = h('div.viewer-toolbar',
      h('div.toolbar-left', nameEl, dimsEl),
      tabList,
      h('div.toolbar-right', compareBtn, h('span.toolbar-sep'), zoomOut, zoomPct, zoomIn, fitBtn, oneBtn));

    // ---- viewer host
    const host = h('div.viewer-host');
    const renderBar = h('div.render-bar', { hidden: true, role: 'progressbar', aria: { label: 'Rendering' } });
    const overlayLoading = h('div.viewer-overlay-state', { hidden: true }, skeleton('sk-fill'));
    const analyzing = h('div.viewer-analyzing', { hidden: true },
      h('img.analyzing-img', { alt: '', hidden: true }),
      h('div.analyzing-card', h('span.spinner'), h('div', h('strong', 'Analyzing your photo'), h('p.analyzing-msg', 'Queued'))));
    const pill = h('div.select-pill', { hidden: true, role: 'toolbar', aria: { label: 'Selection' } });
    const viewer = createViewer(host, {
      onHover: (rid) => store.set({ hoverRegion: rid }),
      onSelect: onViewerSelect,
      onZoom: (s) => { zoomPct.textContent = `${Math.round(viewer.getZoom() * 100)}%`; },
    });
    host.append(renderBar, overlayLoading, analyzing, pill);

    // ---- status line
    const hoverInfo = h('span.status-hover', 'Hover a part to see its region');
    const renderInfo = h('span.status-render.mono', '');
    const statusLine = h('div.viewer-status', hoverInfo, h('span.spacer'), renderInfo);

    // ---- panels
    const actions = makeActions();
    const stepper = createStepper(store);
    const groupsPanel = createGroupsPanel(store, actions);
    const palettePanel = createPalettePanel(store, actions);
    const mappingPanel = createMappingPanel(store, actions);
    const finishPanel = createFinishPanel(store, actions);
    const inspector = h('aside.studio-inspector', { aria: { label: 'Inspector' } },
      stepper.el, groupsPanel.el, palettePanel.el, mappingPanel.el, finishPanel.el);
    if (job.status === 'ready') stepper.panel.setOpen(false);

    const view = h('div.studio', h('div.studio-viewer', toolbar, host, statusLine), inspector);
    replace(root, view);
    enter(view.firstElementChild);
    enterChildren(inspector, { step: 50 });

    inspector.addEventListener('chroma:need-palette', () => {
      toast('Generate or add a palette first', { type: 'warning' });
      palettePanel.panel.setOpen(true);
      palettePanel.focus();
    });

    // ---------------------------------------------------------------- actions for panels
    function makeActions() {
      // `reloadIds` is only needed when the edit can change the id maps (merge, split,
      // move, regroup); a rename or lock toggle keeps the maps and just re-renders.
      const guard = (fn, label, { reloadIds: needIds = true } = {}) => async (...args) => {
        try {
          if (needIds) {
            // Structural edits re-key the saved mapping by region on the server, so push
            // the current mapping first and then adopt the server's version.
            const { mapping, renderOptions, palette } = store.get();
            await api.saveState(job.id, { mapping, render_options: renderOptions, palette_id: palette?.id || null }).catch(() => {});
          }
          const nextJob = await fn(...args);
          if (destroyed) return nextJob;
          if (nextJob?.id) {
            store.applyJob(nextJob, { keepMapping: !needIds });
            if (needIds) {
              store.set({ idsVersion: store.get().idsVersion + 1 });
              await reloadIds();
            }
            scheduleRender();
          }
          if (label) toast.success(label(nextJob, ...args));
          return nextJob;
        } catch (err) {
          toast.error(err?.userMessage || err?.message || 'Request failed');
          throw err;
        }
      };
      return {
        merge: guard((ids) => api.mergeGroups(job.id, ids), (j, ids) => `Merged ${ids.length} groups`),
        split: guard((gid, k) => api.splitGroup(job.id, gid, k), () => 'Group split in two'),
        moveRegions: guard((rids, gid) => api.moveRegions(job.id, rids, gid), (j, rids) => `Moved ${rids.length} ${rids.length === 1 ? 'region' : 'regions'}`),
        regroup: guard((opts) => api.regroup(job.id, opts), (j) => `Regrouped into ${j.groups.length} groups`),
        patch: guard((gid, patch) => api.patchGroup(job.id, gid, patch), null, { reloadIds: false }),
        async generate(prompt, n) {
          try {
            const pal = await api.createPalette(prompt, n);
            if (destroyed) return pal;
            store.set({ palette: pal });
            persist();
            toast.success(`Palette ready · ${pal.colors.length} colours`);
            return pal;
          } catch (err) {
            toast.error(err?.userMessage || err?.message || 'Palette failed');
            throw err;
          }
        },
        async suggest(strategy) {
          const colors = (store.get().palette?.colors || []).map((c) => c.hex);
          try {
            const mapping = await api.suggestMapping(job.id, colors, strategy);
            if (destroyed) return;
            store.commitMapping(mapping);
            toast.success(`Mapped with the ${strategy} strategy`, { id: 'suggest' });
          } catch (err) {
            toast.error(err?.userMessage || err?.message || 'Suggestion failed');
          }
        },
        async exportImage(quality, format) {
          const { mapping, renderOptions } = store.get();
          const res = await api.exportImage(job.id, { mapping, options: renderOptions, quality, format });
          if (!destroyed) { store.set({ lastExport: res }); toast.success(`Exported ${formatDims(res.width, res.height)} in ${formatMs(res.ms)}`); }
          return res;
        },
        shareUrl: () => `${location.origin}${location.pathname}#/studio/${job.id}`,
      };
    }

    // ---------------------------------------------------------------- SSE
    closeEvents = subscribeJob(job.id, {
      stage: (ev) => {
        const cur = store.get().job;
        const stages = { ...cur.stages, [ev.stage]: { ...(cur.stages?.[ev.stage] || {}), state: ev.state, progress: ev.progress, message: ev.message, seconds: ev.seconds ?? cur.stages?.[ev.stage]?.seconds ?? 0 } };
        store.set({ job: { ...cur, stages } });
        analyzing.querySelector('.analyzing-msg').textContent = ev.state === 'running' ? `${titleCase(ev.stage)} · ${ev.message || ''}` : `${titleCase(ev.stage)} done`;
        if (ev.stage === 'ingest' && ev.state === 'done') showAnalyzingPreview();
      },
      status: (ev) => {
        if (ev.status === 'deleted') {
          onDeleted();
          return;
        }
        const cur = store.get().job;
        if (cur.status !== ev.status) store.set({ job: { ...cur, status: ev.status } });
        if (ev.status === 'ready') onReady();
      },
      groups: (ev) => {
        const cur = store.get().job;
        store.applyJob({ ...cur, groups: ev.groups });
      },
      done: () => onReady(),
      error: (ev) => {
        if (ev.message === 'Job deleted') { onDeleted(); return; } // older mock shape
        const cur = store.get().job;
        store.set({ job: { ...cur, status: 'error', error: ev.message } });
        toast.error(`Analysis failed: ${ev.message}`, { id: 'analysis-error' });
      },
      disconnect: ({ final }) => { if (final) toast.warn('Lost connection to the server', { id: 'sse' }); },
    });
    disposers.push(() => closeEvents?.());

    function onDeleted() {
      // The server announces deletion as `status: deleted` and then closes the stream;
      // close first so the EventSource close is not reported as a lost connection.
      closeEvents?.();
      closeEvents = null;
      toast.warn('This job was deleted');
      navigate('#/gallery');
    }

    function showAnalyzingPreview() {
      const img = analyzing.querySelector('img');
      if (img.src) return;
      img.onload = () => { img.hidden = false; };
      img.src = layerUrl(job.id, 'preview');
    }

    // ---------------------------------------------------------------- ready → assets
    async function onReady() {
      if (assetsRequested || destroyed) return;
      assetsRequested = true;
      try {
        const fresh = await api.job(job.id);
        if (destroyed) return;
        store.applyJob(fresh, { keepMapping: false });
        store.set({ renderOptions: { ...store.get().renderOptions, ...(fresh.render_options || {}) } });
        if (fresh.palette_id) api.palette(fresh.palette_id).then((pal) => { if (!destroyed) store.set({ palette: pal }); }).catch(() => {});
      } catch (err) {
        toast.error(err?.userMessage || 'Could not refresh the job');
      }
      viewer.setImageSize(job.image.work_width, job.image.work_height);
      tabs.forEach((t) => { t.disabled = false; });
      overlayLoading.hidden = false;
      try {
        await Promise.all([ensureLayer('original'), reloadIds()]);
      } catch (err) {
        toast.error('Could not load the image layers');
      }
      overlayLoading.hidden = true;
      scheduleRender.flush?.();
      scheduleRender();
    }

    function ensureLayer(name) {
      const apiName = LAYER_API[name];
      if (!apiName) return Promise.resolve(null);
      const version = store.get().idsVersion;
      const key = `${name}@${name === 'regions' || name === 'groups' ? version : 0}`;
      if (!layerPromises.has(key)) {
        const promise = api.loadBitmap(layerUrl(job.id, apiName, name === 'regions' || name === 'groups' ? version : 0))
          .then((bmp) => { if (!destroyed) viewer.setLayerBitmap(name, bmp); return bmp; })
          .catch((err) => { layerPromises.delete(key); throw err; });
        layerPromises.set(key, promise);
      }
      return layerPromises.get(key);
    }

    async function reloadIds() {
      const version = store.get().idsVersion;
      const [rb, gb] = await Promise.all([
        api.loadBitmap(idsUrl(job.id, 'regions', version), { exact: true }),
        api.loadBitmap(idsUrl(job.id, 'groups', version), { exact: true }),
      ]);
      if (destroyed) return;
      const r = decodeIds(rb, 'regions');
      const g = decodeIds(gb, 'groups');
      rb.close?.(); gb.close?.();
      viewer.setIds({ regions: r.regions, groups: g.groups, w: r.w, h: r.h });
      idsLoaded = true;
      // Flat layers changed too (regions / groups colours).
      for (const key of [...layerPromises.keys()]) if (key.startsWith('regions@') || key.startsWith('groups@')) layerPromises.delete(key);
      const layer = store.get().layer;
      if (layer === 'regions' || layer === 'groups') ensureLayer(layer).catch(() => {});
    }

    // ---------------------------------------------------------------- rendering
    let renderErrorShown = false;
    const scheduleRender = debounce(async () => {
      if (destroyed || store.get().job.status !== 'ready') return;
      const { mapping, renderOptions } = store.get();
      store.set({ render: { ...store.get().render, busy: true, error: null } });
      renderBar.hidden = false;
      try {
        const res = await renderQueue.render(mapping, renderOptions);
        if (!res || destroyed) return;   // superseded
        const bmp = await createImageBitmap(res.blob);
        if (destroyed) { bmp.close?.(); return; }
        viewer.setLayerBitmap('result', bmp);
        const r = store.get().render;
        store.set({ render: { busy: false, ms: res.ms, error: null, count: r.count + 1 } });
        renderErrorShown = false;
      } catch (err) {
        store.set({ render: { ...store.get().render, busy: false, error: err?.message || 'Render failed' } });
        if (!renderErrorShown) { renderErrorShown = true; toast.error(err?.userMessage || 'Render failed', { id: 'render' }); }
      } finally {
        if (!store.get().render.busy) renderBar.hidden = true;
      }
    }, RENDER_DEBOUNCE_MS);
    disposers.push(() => { scheduleRender.cancel(); renderQueue.cancel(); });

    const persist = debounce(() => {
      if (destroyed) return;
      const { mapping, renderOptions, palette } = store.get();
      api.saveState(job.id, { mapping, render_options: renderOptions, palette_id: palette?.id || null })
        .catch((err) => console.warn('[studio] state not saved', err));
    }, PERSIST_DEBOUNCE_MS);
    disposers.push(() => persist.flush());

    disposers.push(store.subscribe(() => { scheduleRender(); persist(); }, ['mapping', 'renderOptions']));

    // ---------------------------------------------------------------- store → UI
    function setLayer(name) {
      if (!LAYERS.includes(name)) return;
      store.set({ layer: name });
    }
    disposers.push(store.watch((s) => {
      tabs.forEach((t) => t.setAttribute('aria-selected', String(t.dataset.layer === s.layer)));
      viewer.setLayer(s.layer);
      if (s.job.status === 'ready' && LAYER_API[s.layer] && !viewer.hasLayer(s.layer)) {
        overlayLoading.hidden = false;
        ensureLayer(s.layer).catch(() => toast.error(`Could not load the ${s.layer} layer`)).finally(() => { if (store.get().layer === s.layer) overlayLoading.hidden = true; });
      }
    }, ['layer']));
    disposers.push(store.watch((s) => {
      compareBtn.setAttribute('aria-pressed', String(s.compare));
      compareBtn.classList.toggle('is-active', s.compare);
      viewer.setCompare(s.compare);
    }, ['compare']));
    disposers.push(store.watch((s) => viewer.setPeek(s.peek), ['peek']));
    disposers.push(store.watch((s) => { viewer.setSelection(s.selection); renderPill(s); }, ['selection', 'groups', 'mapping']));
    disposers.push(store.watch((s) => viewer.setHoverGroup(s.hoverGroup), ['hoverGroup']));
    disposers.push(store.watch((s) => {
      if (s.hoverRegion === null) { hoverInfo.textContent = idsLoaded ? 'Hover a part to see its region · click to select its group · ⇧click to pick regions' : 'Hover a part to see its region'; hoverInfo.classList.remove('has-region'); return; }
      const g = store.groupForRegion(s.hoverRegion);
      replace(hoverInfo,
        h('span.status-dot', { style: { background: g?.albedo_hex || 'transparent' } }),
        `Region #${s.hoverRegion}`,
        g && [h('span.status-sep', '·'), g.name, h('span.status-sep', '·'), `${formatPct(g.area_frac)} of image`]);
      hoverInfo.classList.add('has-region');
    }, ['hoverRegion', 'groups']));
    disposers.push(store.watch((s) => {
      const r = s.render;
      replace(renderInfo, r.ms !== null ? [icon('bolt', { size: 12 }), ` ${formatMs(r.ms)}`] : null);
      renderInfo.classList.toggle('is-busy', r.busy);
    }, ['render']));
    disposers.push(store.watch((s) => {
      const st = s.job.status;
      analyzing.hidden = !(st === 'queued' || st === 'analyzing');
      view.classList.toggle('is-analyzing', st === 'queued' || st === 'analyzing');
      view.classList.toggle('is-error', st === 'error');
      if (st === 'error') {
        analyzing.hidden = false;
        analyzing.querySelector('.analyzing-card').replaceChildren(icon('alert', { size: 20 }), h('div', h('strong', 'Analysis failed'), h('p.analyzing-msg', s.job.error || 'Unknown error')),
          h('a.btn.btn-sm', { href: '#/' }, 'Try another photo'));
      }
      [groupsPanel, palettePanel, mappingPanel, finishPanel].forEach((pn) => pn.panel.setDisabled(st !== 'ready'));
    }, ['job']));

    // ---------------------------------------------------------------- selection pill
    function onViewerSelect({ regionId, groupId, shift, meta }) {
      if (regionId === null) { store.clearSelection(); return; }
      const sel = store.get().selection;
      if (meta) {
        const g = store.groupById(groupId) || store.groupForRegion(regionId);
        if (g) store.toggleGroup(g.id);
        return;
      }
      if (shift) {
        const ids = sel.regionIds.includes(regionId) ? sel.regionIds.filter((r) => r !== regionId) : [...sel.regionIds, regionId];
        store.select(null, ids);
        return;
      }
      const g = store.groupById(groupId) || store.groupForRegion(regionId);
      if (!g) { store.clearSelection(); return; }
      if (sel.groupId === g.id && !sel.regionIds.length) { store.clearSelection(); return; }
      store.select(g.id);
    }

    /** Paint every selected group the same colour in one step. */
    function paintAllButton(ids) {
      const picker = h('input.pill-picker', { type: 'color', value: '#111111', aria: { label: `Colour for ${ids.length} groups` } });
      picker.addEventListener('input', () => store.setTargets(ids, picker.value));
      const btn = h('button.btn.btn-sm.btn-primary', { type: 'button', onClick: () => picker.click() },
        icon('palette', { size: 14 }), `Paint all ${ids.length}`, picker);
      return tip(btn, 'Give every selected colour the same new paint');
    }

    function renderPill(s) {
      const { selection, groups } = s;
      if (selection.regionIds.length) {
        const n = selection.regionIds.length;
        const select = h('select.select.select-sm', { aria: { label: 'Move regions to group' } },
          h('option', { value: '' }, 'Move to group…'),
          groups.map((g) => h('option', { value: g.id }, `${g.name} · ${formatPct(g.area_frac)}`)));
        select.addEventListener('change', () => {
          if (select.value === '') return;
          actions.moveRegions(selection.regionIds, Number(select.value)).then(() => store.clearSelection()).catch(() => {});
        });
        pill.replaceChildren(
          h('span.pill-swatch.pill-multi', icon('grid', { size: 14 })),
          h('span.pill-title', `${n} ${n === 1 ? 'region' : 'regions'} selected`),
          h('span.pill-sub', '⇧click to add more'),
          select,
          tip(h('button.btn-icon', { type: 'button', aria: { label: 'Clear selection' }, onClick: () => store.clearSelection() }, icon('x', { size: 14 })), 'Clear (Esc)'));
        showPill();
        return;
      }
      if ((selection.groupIds || []).length > 1) {
        const ids = selection.groupIds;
        const picked = ids.map((id) => store.groupById(id)).filter(Boolean);
        const share = picked.reduce((acc, x) => acc + x.area_frac, 0);
        pill.replaceChildren(
          h('span.pill-stack', picked.slice(0, 4).map((x) => h('span.pill-chip', { style: { background: x.albedo_hex } }))),
          h('div.pill-text',
            h('span.pill-title', `${ids.length} colours selected`),
            h('span.pill-sub', `${formatPct(share)} of the image · paint them together, or merge into one group`)),
          h('span.pill-actions',
            paintAllButton(ids),
            tip(h('button.btn.btn-sm', { type: 'button', onClick: () => actions.merge([...ids]) }, icon('layers', { size: 14 }), 'Merge'), 'Make these one colour group'),
            tip(h('button.btn-icon', { type: 'button', aria: { label: 'Clear selection' }, onClick: () => store.clearSelection() }, icon('x', { size: 14 })), 'Clear (Esc)')));
        showPill();
        return;
      }
      const g = selection.groupId === null ? null : store.groupById(selection.groupId);
      if (!g) { pill.hidden = true; return; }
      const target = s.mapping[String(g.id)];
      pill.replaceChildren(
        h('span.pill-swatch', { style: { background: g.albedo_hex } }, target ? h('span.pill-target', { style: { background: target } }) : null),
        h('div.pill-text', h('span.pill-title', g.name), h('span.pill-sub', `${formatPct(g.area_frac)} · ${g.region_ids.length} ${g.region_ids.length === 1 ? 'region' : 'regions'}${g.locked ? ' · locked' : ''}${g.is_background ? ' · background' : ''}`)),
        h('span.pill-actions',
          tip(h('button.btn-icon', { type: 'button', aria: { label: 'Pick a colour' }, disabled: g.locked, onClick: () => mappingPanel.pickFor(g.id) }, icon('palette', { size: 15 })), 'Pick colour'),
          target ? tip(h('button.btn-icon', { type: 'button', aria: { label: 'Keep original colour' }, onClick: () => store.setTarget(g.id, null) }, icon('undo', { size: 15 })), 'Keep original') : null,
          tip(h('button.btn-icon', { type: 'button', aria: { label: g.locked ? 'Unlock' : 'Lock', pressed: g.locked }, onClick: () => actions.patch(g.id, { locked: !g.locked }) }, icon(g.locked ? 'lock' : 'unlock', { size: 15 })), g.locked ? 'Unlock' : 'Lock'),
          tip(h('button.btn-icon', { type: 'button', aria: { label: 'Split group' }, onClick: () => actions.split(g.id, 2) }, icon('split', { size: 15 })), 'Split'),
          tip(h('button.btn-icon', { type: 'button', aria: { label: 'Clear selection' }, onClick: () => store.clearSelection() }, icon('x', { size: 14 })), 'Clear (Esc)')));
      showPill();
    }
    function showPill() {
      if (pill.hidden) { pill.hidden = false; enter(pill, { distance: 10 }); } else pop(pill, { scale: 1.02, duration: 180 });
    }

    // ---------------------------------------------------------------- keyboard
    let spaceHeld = false;
    const onKeyDown = (e) => {
      if (isTypingTarget(e.target)) return;
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        const ok = e.shiftKey ? store.redoMapping() : store.undoMapping();
        if (ok) toast(e.shiftKey ? 'Redo' : 'Undo', { id: 'undo', duration: 1200 });
        return;
      }
      if (mod) return;
      if (e.key >= '1' && e.key <= '6') { setLayer(LAYERS[Number(e.key) - 1]); e.preventDefault(); return; }
      switch (e.key) {
        case ' ': if (!spaceHeld) { spaceHeld = true; store.set({ peek: true }); } e.preventDefault(); break;
        case 'Escape': store.clearSelection(); break;
        case 'c': case 'C': store.set({ compare: !store.get().compare }); break;
        case 'm': case 'M': {
          const ids = store.get().selection.groupIds || [];
          if (ids.length > 1) actions.merge([...ids]);
          else toast('Select two or more colours to merge (⌘-click)', { id: 'merge-hint', duration: 2200 });
          break;
        }
        case 'a': case 'A': {
          const all = store.get().groups.map((g) => g.id);
          if (all.length) store.selectGroups(all);
          break;
        }
        case 'f': case 'F': viewer.fit(); break;
        case '0': viewer.oneToOne(); break;
        case '+': case '=': viewer.zoomIn(); break;
        case '-': case '_': viewer.zoomOut(); break;
        default: return;
      }
    };
    const onKeyUp = (e) => { if (e.key === ' ') { spaceHeld = false; store.set({ peek: false }); } };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onKeyUp);
    disposers.push(() => { document.removeEventListener('keydown', onKeyDown); document.removeEventListener('keyup', onKeyUp); window.removeEventListener('blur', onKeyUp); });

    disposers.push(() => { stepper.destroy(); groupsPanel.destroy(); palettePanel.destroy(); mappingPanel.destroy(); finishPanel.destroy(); viewer.destroy(); store.destroy(); });

    if (job.status === 'ready') onReady();
  }

  return {
    destroy() {
      destroyed = true;
      disposers.splice(0).forEach((fn) => { try { fn(); } catch (err) { console.warn(err); } });
    },
  };
}
