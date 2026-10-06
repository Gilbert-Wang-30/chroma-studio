/**
 * Studio view: the workspace for one job. Owns the Studio store, the viewer, the
 * inspector panels, the SSE subscription, the debounced/abortable render loop, state
 * persistence and the keyboard shortcuts. Panels never call the API themselves; they
 * receive `actions` closures defined here so error handling lives in one place.
 */
import { api, layerUrl, idsUrl, subscribeJob, createRenderQueue, createSegmentQueue, ApiError } from '../api.js';
import { h, icon, tip, kbd, skeleton, replace } from '../dom.js';
import { enter, enterChildren, pop } from '../motion.js';
import { createStudioStore, LAYERS } from '../state.js';
import { toast } from '../toast.js';
import { debounce, formatDims, formatMs, formatPct, isTypingTarget, titleCase, textColorOn } from '../util.js';
import { createViewer, decodeIds, maskSprite, SEGMENT_COLOR, CANDIDATE_COLOR } from '../viewer.js';
import { createStepper } from '../panels/stepper.js';
import { createGroupsPanel, splitsByInstance, isUserPart } from '../panels/groups.js';
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
    // Select part: point at a part and SAM 2 cuts it out as a group of its own
    const selectBtn = tip(h('button.btn-icon.toolbar-btn.tool-select', { type: 'button', disabled: true, aria: { label: 'Select part', pressed: false }, onClick: () => toggleSegment() }, icon('wand')), 'Select part (S): click a part, ⇧-click to exclude, drag a box');
    const zoomOut = tip(h('button.btn-icon.toolbar-btn', { type: 'button', aria: { label: 'Zoom out' }, onClick: () => viewer.zoomOut() }, icon('zoomOut')), 'Zoom out (−)');
    const zoomIn = tip(h('button.btn-icon.toolbar-btn', { type: 'button', aria: { label: 'Zoom in' }, onClick: () => viewer.zoomIn() }, icon('zoomIn')), 'Zoom in (+)');
    const zoomPct = h('button.zoom-pct.mono', { type: 'button', aria: { label: 'Zoom level, click to fit' }, onClick: () => viewer.fit() }, '100%');
    const fitBtn = tip(h('button.btn-icon.toolbar-btn', { type: 'button', aria: { label: 'Fit to window' }, onClick: () => viewer.fit() }, icon('fit')), 'Fit (F)');
    const oneBtn = tip(h('button.btn-icon.toolbar-btn', { type: 'button', aria: { label: 'Actual pixels' }, onClick: () => viewer.oneToOne() }, icon('one')), '1:1 (0)');
    const toolbar = h('div.viewer-toolbar',
      h('div.toolbar-left', nameEl, dimsEl),
      tabList,
      h('div.toolbar-right', selectBtn, compareBtn, h('span.toolbar-sep'), zoomOut, zoomPct, zoomIn, fitBtn, oneBtn));

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
      onToolPoint: (p) => addPoint(p),
      onToolBox: (b) => setBox(b),
      onCandidate: (i) => chooseCandidate(i),
      onCandidateHover: (i) => store.set({ findHover: i }),
      onAltClick: (p) => openSegment({ point: p }),
    });
    host.append(renderBar, overlayLoading, analyzing, pill);

    // ---- status line (the hint shrinks with an ellipsis; the timing chips never wrap)
    const hoverInfo = h('span.status-hover', h('span.status-text', 'Hover a part to see its region'));
    const renderInfo = h('span.status-render.mono', '');
    const segInfo = h('span.status-seg.mono', { hidden: true });
    const statusLine = h('div.viewer-status', hoverInfo, segInfo, renderInfo);

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
            // the current mapping first (when it changed) and then adopt the server's version.
            await saveNow().catch(() => {});
          }
          const nextJob = await fn(...args);
          if (destroyed) return nextJob;
          if (nextJob?.id) {
            store.applyJob(nextJob, { keepMapping: !needIds });
            if (needIds) markSaved();          // the server holds the re-keyed mapping already
            if (needIds) {
              store.set({ idsVersion: store.get().idsVersion + 1 });
              await reloadIds();
            }
            scheduleRender();
          }
          const msg = label ? label(nextJob, ...args) : null;
          if (msg) toast.success(msg);
          return nextJob;
        } catch (err) {
          toast.error(err?.userMessage || err?.message || 'Request failed');
          throw err;
        }
      };
      return {
        // `into` (a row dropped onto) keeps its name; a part drawn with Select part merged into a
        // colour group is removed, merged into a part it becomes one of its instances
        merge: guard((ids, into = null) => api.mergeGroups(job.id, ids, into), (j, ids) => {
          // the merged group keeps every member's background flag: say so when the merge just
          // made a painted group part of the ignored background
          const into = j?.groups?.find((g) => g.id === Math.min(...ids));
          if (into?.is_background && store.get().ignoreBackground) return `Merged ${ids.length} groups into ${into.name}, a background group: ignored while the switch is on`;
          return `Merged ${ids.length} groups`;
        }),
        // Select part's commit: the prompt runs again on the server and becomes a part group
        // (tried again after the server's Retry-After while an analysis holds the GPU)
        addPart: guard(async (prompt) => {
          for (let attempt = 0; ; attempt++) {
            try {
              return await api.addPart(job.id, prompt);
            } catch (err) {
              if (!(err instanceof ApiError) || err.status !== 503 || attempt >= 2 || destroyed) throw err;
              toast('The GPU is busy with an analysis · making the group in a moment', { id: 'seg-busy', duration: 2400 });
              await new Promise((r) => setTimeout(r, Math.max(0.5, err.retryAfter || 1) * 1000));
            }
          }
        }, (j) => {
          // short: the selection pill beside it shows the new group's name and size
          const g = j?.groups?.find((x) => x.id === j.created_group);
          const cp = j?.created_part || {};
          if (!g) return 'Part created';
          if (cp.replaced) return cp.replaced === g.name ? `${g.name} has its new outline` : `${g.name} replaces ${cp.replaced}`;
          const took = (cp.took_in || []).filter(Boolean);
          if (took.length) return `${g.name} is a group of its own · it took in ${took.join(', ')}`;
          return `${g.name} is a group of its own, ready to paint`;
        }),
        // a drawn part goes back into the group its pixels came from (the merge endpoint)
        removePart: (gid) => {
          const g = store.groupById(gid);
          if (!g) return Promise.resolve(null);
          const rid = g.region_ids[0];
          return guard(() => api.removePart(job.id, g.id), (j) => {
            // the server says which groups came back (the ones the part had taken in whole)
            const back = (j?.removed_part?.restored || []).filter(Boolean);
            if (back.length) return `${g.name} removed · ${back.length > 2 ? `${back.slice(0, 2).join(', ')} and ${back.length - 2} more` : back.join(' and ')} ${back.length === 1 ? 'is' : 'are'} back`;
            const host = j?.groups?.find((x) => x.region_ids.includes(rid));
            // a group it had taken in whole is that group again (an earlier outline, a detected
            // part, a colour group it replaced)
            if (host && host.name === g.name) {
              return `${g.name} removed · ${isUserPart(host) ? `the earlier ${host.name}` : host.part ? `the detected ${host.name}` : `the ${host.name} group`} is back`;
            }
            if (host && isUserPart(host)) return `${g.name} removed · ${host.name} is back`;
            return `${g.name} removed · its pixels are back in ${host?.name || 'the group they came from'}`;
          })();
        },
        findParts: (text) => findParts(text),
        chooseCandidate: (i) => chooseCandidate(i),
        clearFind: () => clearFind(),
        startSelect: () => openSegment(),
        split: (gid, k, mode = 'colour') => {
          // a part group with several instances splits into them (no colour clustering); a
          // colour split of a group that is one colour changes nothing, and says so
          const g = store.groupById(gid);
          const before = store.get().groups.length;
          const label = (j) => {
            if (mode === 'instances') return `${g?.name || 'Part'} split into ${g?.part_instances || 'its'} instances`;
            if ((j?.groups?.length || 0) > before) return `${g?.name || 'Group'} split in two`;
            toast(`Nothing to split: ${g?.name || 'this group'} is one colour`, { id: 'split', duration: 2600 });
            return null;
          };
          return guard(() => api.splitGroup(job.id, gid, k, mode), label)();
        },
        moveRegions: guard((rids, gid) => api.moveRegions(job.id, rids, gid), (j, rids) => `Moved ${rids.length} ${rids.length === 1 ? 'region' : 'regions'}`),
        // a Minor row folded into the group next to it: its regions move, that group's flags stay
        foldInto: (gid, into) => {
          const g = store.groupById(gid);
          const host = store.groupById(into);
          if (!g || !host) return Promise.resolve(null);
          return guard(() => api.moveRegions(job.id, g.region_ids, host.id), () => `${g.name} merged into ${host.name}`)();
        },
        regroup: guard((opts) => api.regroup(job.id, opts), (j) => `Regrouped into ${j.groups.length} groups`),
        patch: guard((gid, patch) => api.patchGroup(job.id, gid, patch), null, { reloadIds: false }),
        async setIgnoreBackground(on) {
          const prev = store.get().ignoreBackground;
          store.set({ ignoreBackground: on });
          try {
            const nextJob = await api.saveState(job.id, { ignore_background: on });
            if (destroyed) return;
            if (nextJob?.id) store.set({ job: nextJob });
            scheduleRender();
            toast(on ? 'Background groups are locked and left out of suggestions' : 'Background groups can be painted again', { id: 'ignore-bg', duration: 2200 });
          } catch (err) {
            store.set({ ignoreBackground: prev });
            toast.error(err?.userMessage || err?.message || 'Could not save the setting');
          }
        },
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
        store.set({ renderOptions: { ...store.get().renderOptions, ...(fresh.render_options || {}) },
                    ignoreBackground: fresh.ignore_background !== false });
        markSaved(fresh.palette_id);         // what the server sent: nothing to write back
        if (fresh.palette_id) api.palette(fresh.palette_id).then((pal) => { if (!destroyed) store.set({ palette: pal }); }).catch(() => {});
      } catch (err) {
        toast.error(err?.userMessage || 'Could not refresh the job');
      }
      viewer.setImageSize(job.image.work_width, job.image.work_height);
      tabs.forEach((t) => { t.disabled = false; });
      selectBtn.disabled = false;
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

    // The state the server holds: opening a job (or adopting the server's re-keyed mapping after
    // an edit) must not write it back, or merely viewing an old job rewrote its job.json.
    let savedState = null;
    const statePayload = () => {
      const { mapping, renderOptions, palette } = store.get();
      return { mapping, render_options: renderOptions, palette_id: palette?.id || null };
    };
    function markSaved(paletteId) {
      const payload = statePayload();
      if (paletteId !== undefined) payload.palette_id = paletteId || null;
      savedState = JSON.stringify(payload);
    }
    function saveNow() {
      const payload = statePayload();
      const key = JSON.stringify(payload);
      if (key === savedState) return Promise.resolve(null);
      return api.saveState(job.id, payload).then((res) => { savedState = key; return res; });
    }
    const persist = debounce(() => {
      if (destroyed) return;
      saveNow().catch((err) => console.warn('[studio] state not saved', err));
    }, PERSIST_DEBOUNCE_MS);
    disposers.push(() => persist.flush());

    disposers.push(store.subscribe(() => { scheduleRender(); persist(); }, ['mapping', 'renderOptions']));

    // ---------------------------------------------------------------- Select part (SAM 2 prompts)
    // A click adds a positive point, a Shift-click or right-click a negative one, a drag draws a
    // box; each change asks the server for the mask (debounced, the in-flight prompt aborted) and
    // the viewer outlines it. Enter makes it a group (the server runs the prompt again and carves
    // it), Backspace undoes the last point, N switches to SAM's next shape, Esc cancels. Find part
    // shows its candidates in the same tool; a click on one takes its prompt, or selects the group
    // when the candidate is a part group already.
    const segQueue = createSegmentQueue(job.id);
    // a phone or tablet without a mouse: no keys to hint, and no keyboard to pop up over the photo
    const touchOnly = () => { try { return matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches; } catch { return false; } };
    const seg = {
      active: false, points: [], box: null, pick: null, answer: null, answerKey: null, sprite: null,
      busy: false, committing: false, status: '', seq: 0, candidates: [], findText: '', finding: false,
      // Find's own sequence and request (a find answered after the tool closed must not show), and
      // what the selection would take in: the groups it covers whole (`overlapOf`)
      findSeq: 0, findCtl: null, matchName: '', matchOther: false, takes: [], replaces: null,
      // the prompt's version and the version the outline on screen answers: Enter commits only an
      // outline the user sees (pressed while the next one is on its way, it waits for it)
      want: 0, have: 0, commitPending: false,
      // SAM's own mask (`raw`), the part group it covers most of but not whole (`most`: {g, share})
      // and the group the pill's "Take all" adds to it (`take`); `sprite` is what is outlined
      raw: null, most: null, take: null,
    };
    const SEG_RESET = () => ({ points: [], box: null, pick: null, answer: null, answerKey: null, sprite: null, raw: null, status: '',
      matchName: '', matchOther: false, takes: [], replaces: null, most: null, take: null, busy: false, commitPending: false });
    const W0 = job.image.work_width;
    const H0 = job.image.work_height;
    const segName = h('input.pill-name', { type: 'text', maxlength: 48, spellcheck: false, aria: { label: 'Name of the new part' } });
    segName.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); commitSegment(); } else if (e.key === 'Escape') { e.preventDefault(); segName.blur(); closeSegment(); }
      e.stopPropagation();
    });
    segName.addEventListener('input', () => { if (seg.active) renderPill(store.get()); });
    const segEls = (() => {
      const swatchIcon = h('span.pill-seg-icon', icon('wand', { size: 15 }));
      const swatch = h('span.pill-swatch.pill-seg', swatchIcon, h('span.pill-seg-ring', { aria: { hidden: true } }));
      const title = h('span.pill-title', 'Select part');
      const sub = h('span.pill-sub', '');
      const make = tip(h('button.btn.btn-sm.btn-primary.seg-make', { type: 'button', onClick: () => commitSegment() },
        icon('check', { size: 14 }), h('span', 'Make group'), kbd('↵')), 'Make it a group of its own (Enter)');
      // the part group the selection covers most of: "Take all" makes the rest of it join too
      const takeAll = tip(h('button.btn.btn-sm.seg-take', { type: 'button', hidden: true, aria: { pressed: false }, onClick: () => toggleTake() },
        icon('plus', { size: 13 }), h('span', 'Take all')), 'Take all of it');
      const other = tip(h('button.btn-icon', { type: 'button', aria: { label: 'Other shape' }, onClick: () => cyclePick(1) }, icon('layers', { size: 15 })),
        'SAM 2 offers other shapes for one click: try the next (N)');
      const undo = tip(h('button.btn-icon', { type: 'button', aria: { label: 'Undo the last point' }, onClick: () => undoPrompt() }, icon('undo', { size: 15 })), 'Undo the last point (⌫)');
      const close = tip(h('button.btn-icon', { type: 'button', aria: { label: 'Cancel Select part' }, onClick: () => closeSegment() }, icon('x', { size: 14 })), 'Cancel (Esc)');
      const actionsEl = h('span.pill-actions', takeAll, make, other, undo, close);
      return { swatch, swatchIcon, title, sub, make, takeAll, other, undo, close, nodes: [swatch, h('div.pill-text', title, sub), segName, actionsEl] };
    })();

    function nextPartName() {
      const nums = store.get().groups.map((g) => /^user_(\d+)$/.exec(g.part || '')).filter(Boolean).map((m) => Number(m[1]));
      return `Part ${(nums.length ? Math.max(...nums) : 0) + 1}`;
    }
    function currentPrompt() {
      return {
        points: seg.points.map((p) => [p[0], p[1], p[2]]),
        box: seg.box ? [seg.box.x0, seg.box.y0, seg.box.x1, seg.box.y1] : null,
        multimask: true,
        pick: seg.pick,
      };
    }
    const promptKey = () => JSON.stringify(currentPrompt());
    function syncPrompts() {
      viewer.setSegmentPrompts({ points: seg.points, box: seg.box });
      renderPill(store.get());
    }
    function toggleSegment() { if (seg.active) closeSegment(); else openSegment(); }
    function openSegment({ point = null } = {}) {
      if (store.get().job.status !== 'ready' || destroyed) return;
      if (!seg.active) {
        Object.assign(seg, SEG_RESET(), { active: true });
        seg.want += 1;
        seg.have = seg.want;
        segName.value = '';
        segName.placeholder = nextPartName();
        store.clearSelection();
        viewer.setTool('segment');
        selectBtn.setAttribute('aria-pressed', 'true');
        selectBtn.classList.add('is-active');
        store.set({ tool: 'segment' });
        // the first click should not pay for the image embedding: prepare it now
        api.segment(job.id, {}).catch(() => {});
      }
      if (point) addPoint({ x: point.x, y: point.y, positive: true });
      else syncPrompts();
    }
    function closeSegment() {
      if (!seg.active) return;
      seg.active = false;
      seg.seq += 1;
      seg.want += 1;
      segQueue.cancel();
      requestMask.cancel();
      cancelFind();
      Object.assign(seg, SEG_RESET(), { candidates: [], finding: false });
      viewer.setTool(null);
      viewer.setSegmentMask(null);
      viewer.setSegmentPrompts({});
      viewer.setCandidates([]);
      selectBtn.setAttribute('aria-pressed', 'false');
      selectBtn.classList.remove('is-active');
      segInfo.hidden = true;
      store.set({ tool: null, find: null, findHover: -1 });
      renderPill(store.get());
    }
    /** Drops a Find in flight (its answer is ignored when it comes). */
    function cancelFind() {
      seg.findSeq += 1;
      seg.findCtl?.abort();
      seg.findCtl = null;
      seg.finding = false;
    }
    function clearCandidates() {
      if (seg.finding) cancelFind();
      if (!seg.candidates.length && !store.get().find) return;
      seg.candidates = [];
      viewer.setCandidates([]);
      store.set({ find: null, findHover: -1 });
    }
    /** What Enter would do to the groups the outline covers, said before it is done: the groups it
     *  takes in whole (a part group covered 90 %: the server grows the part over the rest; a colour
     *  group covered whole, up to the carve's specks) and the one it replaces (taken in, most of the
     *  new part). */
    function overlapOf(sprite, area) {
      const takes = [];
      let replaces = null;
      for (const [gid, n] of viewer.groupCounts(sprite)) {
        const g = store.groupById(gid);
        if (!g?.area) continue;
        const whole = g.part ? n >= 0.9 * g.area : n >= g.area - 11 * Math.max(1, g.region_ids.length);
        if (!whole) continue;
        takes.push(g);
        if (g.area >= 0.5 * area && (!replaces || g.area > replaces.area)) replaces = g;
      }
      takes.sort((a, b) => b.area - a.area);
      return { takes, replaces };
    }
    /** The part group SAM's mask covers most of (half or more) without taking it: the one "Take all"
     *  offers (the rest of it would stay a part group of its own beside the new one). */
    function mostOf(sprite) {
      let most = null;
      for (const [gid, n] of viewer.groupCounts(sprite)) {
        const g = store.groupById(gid);
        if (!g?.part || !g.area) continue;
        const share = n / g.area;
        if (share >= 0.5 && share < 0.9 && (!most || share > most.share)) most = { g, share };
      }
      return most;
    }
    /** The outline and the notes after SAM's mask (`seg.raw`) or the "Take all" choice changed. */
    function applyOverlap() {
      const raw = seg.raw;
      seg.most = raw ? mostOf(raw) : null;
      if (seg.take !== null && seg.most?.g.id !== seg.take) seg.take = null;      // it no longer covers most of it
      const shown = raw && seg.take !== null ? viewer.spriteWithGroups(raw, [seg.take]) : raw;
      seg.sprite = shown;
      const { takes, replaces } = shown ? overlapOf(shown, shown.area || 0) : { takes: [], replaces: null };
      seg.takes = takes;
      seg.replaces = replaces;
      segName.placeholder = replaces?.name || nextPartName();
      viewer.setSegmentMask(shown);
    }
    function toggleTake() {
      if (!seg.active || !seg.most || seg.committing) return;
      seg.take = seg.take === seg.most.g.id ? null : seg.most.g.id;
      applyOverlap();
      renderPill(store.get());
    }
    /** The prompt changed: the outline on screen answers an older one until the next answer. */
    function promptChanged() {
      seg.want += 1;
      seg.commitPending = false;
    }
    function busyCommitting() {
      if (!seg.committing) return false;
      toast('Making the group · one moment', { id: 'seg-committing', duration: 1600 });
      return true;
    }
    function addPoint({ x, y, positive }) {
      if (!seg.active) { openSegment(); if (!seg.active) return; }
      if (busyCommitting()) return;
      clearCandidates();
      seg.matchName = '';
      seg.matchOther = false;
      seg.points.push([clampInt(x, W0), clampInt(y, H0), positive ? 1 : 0]);
      if (!seg.points.some((p) => p[2]) && !seg.box) {
        seg.points.pop();
        toast('Click on the part first; ⇧-click then leaves pieces out', { id: 'seg-neg', duration: 2400 });
        return;
      }
      promptChanged();
      syncPrompts();
      requestMask();
    }
    function setBox(b) {
      if (!seg.active || busyCommitting()) return;
      clearCandidates();
      seg.matchName = '';
      seg.matchOther = false;
      seg.box = { x0: Math.round(b.x0), y0: Math.round(b.y0), x1: Math.round(b.x1), y1: Math.round(b.y1) };
      if (!seg.points.length) seg.pick = null;
      promptChanged();
      syncPrompts();
      requestMask();
    }
    function undoPrompt() {
      if (!seg.active || busyCommitting()) return;
      if (seg.points.length) seg.points.pop();
      else if (seg.box) seg.box = null;
      else { closeSegment(); return; }
      promptChanged();
      if (!seg.points.length && !seg.box) {
        // nothing left to ask: an answer already on its way must not bring the outline back
        seg.seq += 1;
        seg.pick = null;
        seg.answer = null;
        seg.raw = null;
        seg.take = null;
        seg.busy = false;
        seg.matchName = '';
        seg.have = seg.want;
        applyOverlap();
        requestMask.cancel();
        segQueue.cancel();
        syncPrompts();
        return;
      }
      syncPrompts();
      requestMask();
    }
    function cyclePick(step) {
      const a = seg.answer;
      if (!seg.active || !a || !(a.alternatives || []).length || busyCommitting()) return;
      const order = [a.mask.index, ...a.alternatives.map((x) => x.index)].sort((p, q) => p - q);
      const at = order.indexOf(a.pick);
      seg.pick = order[(at + step + order.length) % order.length];
      promptChanged();
      // show SAM's coarse alternative at once; the refined one follows
      const alt = a.alternatives.find((x) => x.index === seg.pick);
      if (alt) maskSprite(alt, SEGMENT_COLOR).then((sp) => { if (seg.active && seg.pick === alt.index) viewer.setSegmentMask(sp); }).catch(() => {});
      requestMask();
    }
    const requestMask = debounce(async () => {
      if (!seg.active || destroyed) return;
      const prompt = currentPrompt();
      if (!prompt.points.length && !prompt.box) return;
      const key = JSON.stringify(prompt);
      const want = seg.want;
      const mine = ++seg.seq;
      seg.busy = true;
      renderPill(store.get());
      try {
        const res = await segQueue.run(prompt, { onBusy: () => { seg.status = 'The GPU is busy with an analysis · retrying…'; renderPill(store.get()); } });
        if (!res || mine !== seg.seq || !seg.active) return;
        const sprite = await maskSprite(res.mask, SEGMENT_COLOR);
        if (mine !== seg.seq || !seg.active) return;
        Object.assign(seg, { answer: res, answerKey: key, raw: sprite, pick: res.pick, status: res.mask.area ? '' : 'SAM 2 found nothing there · click on the part' });
        seg.have = want;
        applyOverlap();
        segInfo.hidden = false;
        segInfo.replaceChildren(icon('wand', { size: 12 }), ` SAM 2 ${formatMs(res.roundTripMs)}`);
        segInfo.dataset.tip = `Round trip ${formatMs(res.roundTripMs)} · server ${formatMs(res.ms)}${res.embed === 'computed' ? ' · image embedded now' : ''}${res.mask.refined ? ' · refined on a crop' : ''}`;
      } catch (err) {
        if (mine === seg.seq) {
          seg.status = '';
          seg.commitPending = false;
          toast.error(err?.userMessage || 'Select part failed', { id: 'segment' });
        }
      } finally {
        if (mine === seg.seq) {
          seg.busy = false;
          renderPill(store.get());
          // Enter was pressed while this outline was on its way: now it is on screen, make it
          if (seg.commitPending && seg.active && seg.have === seg.want) {
            seg.commitPending = false;
            commitSegment();
          }
        }
      }
    }, 24);
    disposers.push(() => { requestMask.cancel(); segQueue.cancel(); });

    async function commitSegment() {
      if (!seg.active || seg.committing) return;
      if (seg.busy || seg.have !== seg.want) {
        // the outline of what was clicked last is still on its way (or retried while an analysis
        // holds the GPU): commit it when it shows, never a mask the user has not seen
        seg.commitPending = true;
        renderPill(store.get());
        return;
      }
      const a = seg.answer;
      if (!a?.mask?.area) {
        toast('Click on the part first', { id: 'seg-empty', duration: 2000 });
        return;
      }
      const prompt = currentPrompt();
      // the answer's refinement window, when it answers exactly this prompt: the server replays
      // the same steps (it never takes the mask itself from here)
      if (seg.answerKey === JSON.stringify(prompt) && a.crop) prompt.crop = a.crop;
      const name = segName.value.trim();
      if (name) prompt.name = name;
      if (seg.take !== null) prompt.take = [seg.take];
      seg.committing = true;
      renderPill(store.get());
      try {
        const next = await actions.addPart(prompt);
        if (destroyed) return;
        closeSegment();
        const gid = next?.created_group;
        if (gid !== null && gid !== undefined && store.groupById(gid)) store.select(gid);
      } catch {
        // the guard showed the error; the tool stays open with the prompt
      } finally {
        seg.committing = false;
        if (seg.active) renderPill(store.get());
      }
    }

    // ---- Find part: candidates on the canvas, a click takes one into Select part
    async function findParts(text) {
      openSegment();
      if (!seg.active) return null;
      clearCandidates();
      // this find's token and request: a newer find, a click on the image or closing the tool drop it
      const ctl = new AbortController();
      const mine = ++seg.findSeq;
      seg.findCtl = ctl;
      seg.finding = true;
      seg.findText = text;
      seg.status = '';
      store.set({ find: { text, busy: true, candidates: [] }, findHover: -1 });
      renderPill(store.get());
      const current = () => mine === seg.findSeq && seg.active && !destroyed;
      try {
        const res = await api.findParts(job.id, text, ctl.signal);
        if (!current()) return null;
        const sprites = await Promise.all(res.candidates.map((c) => maskSprite(c.mask, CANDIDATE_COLOR, { strength: 0.9, fill: 1.2 }).catch(() => null)));
        if (!current()) return null;
        seg.candidates = res.candidates.map((c, i) => ({ ...c, sprite: sprites[i], bbox: c.mask.bbox }));
        viewer.setCandidates(seg.candidates);
        store.set({ find: { text, busy: false, candidates: res.candidates, detector: res.detector, ms: res.ms }, findHover: -1 });
        if (!segName.value.trim()) segName.value = titleCase(text);
        if (!res.detector) seg.status = 'Find part needs OWLv2 or Florence-2, which this server lacks · click the part to select it by hand';
        else if (!res.candidates.length) seg.status = `No “${text}” found · click the part to select it by hand`;
        segInfo.hidden = false;
        segInfo.replaceChildren(icon('scan', { size: 12 }), ` Find ${formatMs(res.ms)}`);
        segInfo.dataset.tip = `${res.detector === 'owlv2' ? 'OWLv2' : res.detector === 'florence' ? 'Florence-2' : 'No detector'} ${formatMs(res.detect_ms)} · SAM 2 masks ${formatMs(res.ms - res.detect_ms)}`;
        return res;
      } catch (err) {
        if (err?.name === 'AbortError' || mine !== seg.findSeq) return null;
        store.set({ find: { text, busy: false, candidates: [], error: err?.userMessage || 'Find failed' } });
        toast.error(err?.userMessage || 'Find part failed', { id: 'find' });
        return null;
      } finally {
        if (mine === seg.findSeq) {
          seg.finding = false;
          seg.findCtl = null;
          renderPill(store.get());
        }
      }
    }
    function chooseCandidate(i) {
      const c = seg.candidates[i];
      if (!c || !seg.active) return;
      if (c.existing) {
        // a part group already: taking it selects that group, ready to paint
        const gid = c.group_id;
        closeSegment();
        if (store.groupById(gid)) {
          store.select(gid);
          viewer.reveal(c.mask.bbox);
        }
        return;
      }
      const p = c.prompt || {};
      seg.points = (p.points || []).map((q) => [q[0], q[1], q[2]]);
      seg.box = p.box ? { x0: p.box[0], y0: p.box[1], x1: p.box[2], y1: p.box[3] } : null;
      seg.pick = p.pick ?? null;
      seg.seq += 1;
      segQueue.cancel();
      requestMask.cancel();
      const answer = { mask: c.mask, alternatives: [], pick: seg.pick, crop: p.crop || null };
      promptChanged();
      Object.assign(seg, { answer, answerKey: promptKey(), have: seg.want, busy: false, status: '', raw: null, take: null,
        // the part group the candidate already is: of a kind the phrase names ("overlaps"), or of
        // another kind (Find "spring"'s exhaust can is "mostly Exhausts")
        matchName: c.matches?.name || '', matchOther: Boolean(c.matches) && c.matches.named === false });
      clearCandidates();
      if (!segName.value.trim()) segName.value = titleCase(seg.findText || '');
      applyOverlap();
      maskSprite(c.mask, SEGMENT_COLOR).then((sp) => {
        if (!seg.active || seg.answer !== answer) return;
        seg.raw = sp;
        applyOverlap();
        renderPill(store.get());
      }).catch(() => {});
      viewer.reveal(c.mask.bbox);
      syncPrompts();
      if (!touchOnly()) segName.focus({ preventScroll: true });     // a phone's keyboard would cover the photo
    }
    function clearFind() { clearCandidates(); seg.status = ''; renderPill(store.get()); }
    disposers.push(store.subscribe((st) => viewer.setCandidateHover(st.findHover ?? -1), ['findHover']));

    /** The status bar's hint while Select part is open: what a click does, and once an outline is
     *  there, the keys (the pill's own line keeps the part's facts; at 1024 px it had no room for both). */
    function segmentHint() {
      if (touchOnly()) {
        // no keyboard: the hints name the pill's buttons
        if (seg.candidates.length) return 'Tap a candidate on the image or in the list · or tap the part yourself';
        return seg.answer?.mask?.area ? 'Tap adds a point · ✓ makes it a group · ↶ undoes a point' : 'Tap the part · drag a box around it';
      }
      if (seg.candidates.length) return 'Click a candidate on the image or in the list · or click the part yourself · Esc cancels';
      if (seg.answer?.mask?.area) {
        const shapes = (seg.answer.alternatives || []).length ? ' · N other shape' : '';
        return `Enter makes it a group · ⌫ undoes a point${shapes} · Esc cancels · click adds, ⇧-click leaves out`;
      }
      return 'Click the part · ⇧-click or right-click leaves a piece out · drag a box · right-drag pans';
    }
    function showSegmentHint() {
      const text = segmentHint();
      const cur = hoverInfo.firstChild;
      if (cur?.classList?.contains('status-text') && cur.textContent === text && hoverInfo.childNodes.length === 1) return;
      replace(hoverInfo, h('span.status-text', text));
      hoverInfo.classList.remove('has-region');
    }
    function renderSegmentPill() {
      const els = segEls;
      if (pill.firstChild !== els.nodes[0]) pill.replaceChildren(...els.nodes);
      const a = seg.answer;
      const shownArea = seg.sprite?.area || a?.mask?.area || 0;
      const area = a?.mask?.area ? shownArea : 0;
      const nPos = seg.points.filter((p) => p[2]).length;
      const nNeg = seg.points.length - nPos;
      const bits = [];
      if (nPos) bits.push(`${nPos} ${nPos === 1 ? 'point' : 'points'}`);
      if (nNeg) bits.push(`${nNeg} left out`);
      if (seg.box) bits.push('a box');
      let title = 'Select part';
      let sub = 'Click the part · ⇧-click or right-click leaves a piece out · drag a box';
      const most = !seg.candidates.length && area ? seg.most : null;
      if (seg.finding) { title = `Finding “${seg.findText}”`; sub = 'OWLv2 is looking for it, SAM 2 draws each one'; }
      else if (seg.candidates.length) { title = `${seg.candidates.length} ${seg.candidates.length === 1 ? 'candidate' : 'candidates'} for “${seg.findText}”`; sub = 'Click one on the image to use it, or click the part yourself'; }
      else if (area) {
        title = segName.value.trim() || segName.placeholder || 'New part';
        // what Enter does to the groups it covers, said first and before it is done, never silently
        const others = seg.takes.filter((g) => g !== seg.replaces).map((g) => g.name);
        const cut = most && seg.take === null ? `cuts ${most.g.name} (${Math.round((1 - most.share) * 100)} % stays)` : '';
        const note = seg.replaces ? `replaces ${seg.replaces.name}${others.length ? `, takes in ${others.join(', ')}` : ''}`
          : others.length ? `takes in ${others.join(', ')}`
            : cut || (seg.matchName ? (seg.matchOther ? `mostly ${seg.matchName}` : `overlaps ${seg.matchName}`) : '');
        const pct = formatPct(shownArea / Math.max(1, W0 * H0));
        sub = [note, note ? pct : `${pct} of the image`, ...bits].filter(Boolean).join(' · ');
      }
      if (seg.status) sub = seg.status;
      if (seg.commitPending && !seg.status) sub = 'Making the group as soon as the outline is drawn…';
      els.title.textContent = title;
      els.sub.textContent = sub;
      // a line the pill cuts with an ellipsis (a narrow window) is there in full on hover
      if (sub.length > 48) els.sub.dataset.tip = sub; else delete els.sub.dataset.tip;
      // the busy ring shows only when an answer takes a moment (a prompt is ~25 ms, a find ~1 s)
      els.swatch.classList.toggle('is-busy', seg.busy || seg.finding || seg.committing || seg.commitPending);
      const kindIcon = seg.candidates.length || seg.finding ? 'scan' : 'wand';
      if (els.swatchIcon.dataset.icon !== kindIcon) {
        els.swatchIcon.dataset.icon = kindIcon;
        els.swatchIcon.replaceChildren(icon(kindIcon, { size: 15 }));
      }
      segName.hidden = !area || seg.candidates.length > 0;
      els.make.hidden = !area || seg.candidates.length > 0;
      els.make.disabled = seg.committing || seg.commitPending;
      els.make.classList.toggle('is-loading', seg.committing || seg.commitPending);
      // "Take all": the part group the selection covers most of joins it whole (pressed again: cut)
      els.takeAll.hidden = !most;
      if (most) {
        const on = seg.take === most.g.id;
        els.takeAll.setAttribute('aria-pressed', String(on));
        els.takeAll.disabled = seg.committing;
        const rest = Math.round((1 - most.share) * 100);
        els.takeAll.setAttribute('aria-label', on ? `Take all of ${most.g.name}: on` : `Take all of ${most.g.name}`);
        els.takeAll.dataset.tip = on
          ? `All of ${most.g.name} joins the new part (click to cut it along the outline again)`
          : `The outline covers ${100 - rest} % of ${most.g.name}: take all of it, so no ${rest} % of it stays a group of its own`;
      }
      els.other.hidden = !(a?.alternatives || []).length || seg.candidates.length > 0;
      els.undo.disabled = !(seg.points.length || seg.box);
      showSegmentHint();
      if (pill.hidden) { pill.hidden = false; enter(pill, { distance: 10 }); }
    }
    const clampInt = (v, n) => Math.min(n - 1, Math.max(0, Math.floor(v)));

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
    disposers.push(store.watch((s) => { viewer.setSelection(s.selection); renderPill(s); }, ['selection', 'groups', 'mapping', 'ignoreBackground']));
    disposers.push(store.watch((s) => viewer.setHoverGroup(s.hoverGroup), ['hoverGroup']));
    disposers.push(store.watch((s) => {
      const hint = (text) => { replace(hoverInfo, h('span.status-text', text)); hoverInfo.classList.remove('has-region'); };
      if (s.tool === 'segment') { showSegmentHint(); return; }
      if (s.hoverRegion === null) { hint(idsLoaded ? 'Hover to see a region · click selects its group · ⇧-click picks regions · S: Select part' : 'Hover a part to see its region'); return; }
      const g = store.groupForRegion(s.hoverRegion);
      replace(hoverInfo,
        h('span.status-dot', { style: { background: g?.albedo_hex || 'transparent' } }),
        h('span.status-text', `Region #${s.hoverRegion}`,
          g && [h('span.status-sep', ' · '), g.name, h('span.status-sep', ' · '), `${formatPct(g.area_frac)} of image`]));
      hoverInfo.classList.add('has-region');
    }, ['hoverRegion', 'groups', 'tool']));
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
      return tip(btn, 'Give every selected group the same new paint');
    }

    function renderPill(s) {
      if (seg.active) { renderSegmentPill(); return; }
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
        const nParts = picked.filter((x) => x.part && !x.is_background).length;
        const noun = nParts === picked.length ? 'parts' : nParts ? 'groups' : 'colours';
        pill.replaceChildren(
          h('span.pill-stack', picked.slice(0, 4).map((x) => h('span.pill-chip', { style: { background: x.albedo_hex } }))),
          h('div.pill-text',
            h('span.pill-title', `${ids.length} ${noun} selected`),
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
      const ignored = g.is_background && s.ignoreBackground;
      const lockedNow = g.locked || ignored;
      const byInstance = splitsByInstance(g);
      const what = g.part && !g.is_background
        ? ` · part${(g.part_instances || 0) > 1 ? `, ${g.part_instances} instances` : ''}`
        : g.minor && !g.is_background ? ' · minor' : '';
      pill.replaceChildren(
        h('span.pill-swatch', { style: { background: g.albedo_hex } }, target ? h('span.pill-target', { style: { background: target } }) : null),
        h('div.pill-text', h('span.pill-title', g.name), h('span.pill-sub', `${formatPct(g.area_frac)} · ${g.region_ids.length} ${g.region_ids.length === 1 ? 'region' : 'regions'}${what}${g.locked ? ' · locked' : ''}${g.is_background ? (ignored ? ' · background, ignored' : ' · background') : ''}${g.finish ? ` · ${g.finish}` : ''}`)),
        h('span.pill-actions',
          tip(h('button.btn-icon', { type: 'button', aria: { label: 'Pick a colour' }, disabled: lockedNow, onClick: () => mappingPanel.pickFor(g.id) }, icon('palette', { size: 15 })), 'Pick colour'),
          target ? tip(h('button.btn-icon', { type: 'button', aria: { label: 'Keep original colour' }, onClick: () => store.setTarget(g.id, null) }, icon('undo', { size: 15 })), 'Keep original') : null,
          tip(h('button.btn-icon', { type: 'button', aria: { label: g.locked ? 'Unlock' : 'Lock', pressed: g.locked }, onClick: () => actions.patch(g.id, { locked: !g.locked }) }, icon(g.locked ? 'lock' : 'unlock', { size: 15 })), g.locked ? 'Unlock' : 'Lock'),
          tip(h('button.btn-icon', { type: 'button', aria: { label: byInstance ? 'Split into instances' : 'Split group' }, onClick: () => actions.split(g.id, 2, byInstance ? 'instances' : 'colour') }, icon('split', { size: 15 })), byInstance ? `Split into its ${g.part_instances} instances` : 'Split'),
          tip(h('button.btn-icon', { type: 'button', aria: { label: 'Clear selection' }, onClick: () => store.clearSelection() }, icon('x', { size: 14 })), 'Clear (Esc)')));
      showPill();
    }
    function showPill() {
      if (pill.hidden) { pill.hidden = false; enter(pill, { distance: 10 }); } else pop(pill, { scale: 1.02, duration: 180 });
    }

    // ---------------------------------------------------------------- keyboard
    let spaceHeld = false;
    // A focused control keeps its own keys: Enter and Space press a button (the pill's Undo, a Find
    // chip), so neither commits the part nor peeks there. Tab always moves the focus (SAM's next
    // shape is N: Tab cycling the shapes trapped the keyboard focus in the viewer).
    const CONTROLS = 'button, a[href], select, summary, [role="button"], [role="tab"], [role="switch"], [role="slider"], [role="checkbox"], [role="option"]';
    const onControl = (el) => el instanceof Element && Boolean(el.closest(CONTROLS));
    const onKeyDown = (e) => {
      if (e.defaultPrevented || isTypingTarget(e.target)) return;
      const mod = e.metaKey || e.ctrlKey;
      const control = onControl(e.target);
      if (seg.active && !mod && !e.altKey) {
        if (e.key === 'Escape') { e.preventDefault(); closeSegment(); return; }
        if (e.key === 'Enter' && !control) { e.preventDefault(); commitSegment(); return; }
        if ((e.key === 'Backspace' || e.key === 'Delete') && !control) { e.preventDefault(); undoPrompt(); return; }
        if ((e.key === 'n' || e.key === 'N') && (seg.answer?.alternatives || []).length) { e.preventDefault(); cyclePick(e.shiftKey ? -1 : 1); return; }
      }
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        const ok = e.shiftKey ? store.redoMapping() : store.undoMapping();
        if (ok) toast(e.shiftKey ? 'Redo' : 'Undo', { id: 'undo', duration: 1200 });
        return;
      }
      if (mod) return;
      if (e.key >= '1' && e.key <= '6') { setLayer(LAYERS[Number(e.key) - 1]); e.preventDefault(); return; }
      switch (e.key) {
        case ' ':
          if (control) return;
          if (!spaceHeld) { spaceHeld = true; store.set({ peek: true }); }
          e.preventDefault();
          break;
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
        case 's': case 'S': if (store.get().job.status === 'ready') toggleSegment(); break;
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
