/**
 * Home: hero, drop zone (drag / paste / click) with live preview and detail selector,
 * sample strip and recent jobs.
 */
import { api } from '../api.js';
import { h, icon, segmented, skeleton, replace, tip } from '../dom.js';
import { enter, enterChildren, shake } from '../motion.js';
import { toast } from '../toast.js';
import { formatBytes, formatDims, readImageFile, relativeTime } from '../util.js';
import { prefs } from '../state.js';

const DETAILS = [
  { value: 'fast', label: 'Fast', title: 'Fewer SAM points, ~10 s' },
  { value: 'balanced', label: 'Balanced', title: 'Good default' },
  { value: 'max', label: 'Max', title: 'Dense points and crops for busy scenes' },
];

export function homeView(root, { navigate }) {
  let detail = prefs.get('detail', 'balanced');
  let picked = null;        // {file, url, width, height}
  let creating = false;
  const disposers = [];

  // ---------------------------------------------------------------- drop zone
  const fileInput = h('input', { type: 'file', accept: 'image/*', hidden: true, aria: { label: 'Choose an image' } });
  fileInput.addEventListener('change', () => { if (fileInput.files?.[0]) pick(fileInput.files[0]); fileInput.value = ''; });

  const detailSeg = segmented(DETAILS, detail, (v) => { detail = v; prefs.set('detail', v); }, { ariaLabel: 'Analysis detail' });
  const analyzeBtn = h('button.btn.btn-primary.btn-lg', { type: 'button', onClick: () => analyzePicked() }, icon('spark', { size: 18 }), h('span', 'Analyze'));
  const cancelBtn = h('button.btn.btn-ghost', { type: 'button', onClick: () => clearPick() }, 'Choose another');
  const previewImg = h('img.drop-preview-img', { alt: '' });
  const previewMeta = h('div.drop-preview-meta');
  const previewPane = h('div.drop-preview', { hidden: true },
    h('div.drop-preview-frame', previewImg),
    h('div.drop-preview-side', previewMeta, h('div.field', h('span.field-label', 'Detail'), detailSeg), h('div.drop-actions', analyzeBtn, cancelBtn)));
  const idle = h('div.drop-idle',
    h('div.drop-icon', icon('upload', { size: 28 })),
    h('h3', 'Drop a product photo'),
    h('p', 'or ', h('button.link-btn', { type: 'button', onClick: () => fileInput.click() }, 'browse'), ', or paste from the clipboard'),
    h('p.drop-hint', 'JPEG or PNG · up to 6000 px · cars, bikes, sneakers, model kits, furniture…'));
  const zone = h('div.dropzone', { tabindex: 0, role: 'button', aria: { label: 'Drop or choose an image' },
    onClick: (e) => { if (!picked && !(e.target instanceof Element && e.target.closest('button'))) fileInput.click(); },
    onKeydown: (e) => { if (!picked && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); fileInput.click(); } },
    onDragover: (e) => { e.preventDefault(); zone.classList.add('is-over'); },
    onDragleave: () => zone.classList.remove('is-over'),
    onDrop: (e) => { e.preventDefault(); zone.classList.remove('is-over'); const f = e.dataTransfer?.files?.[0]; if (f) pick(f); },
  }, idle, previewPane, fileInput);

  async function pick(file) {
    if (!file.type.startsWith('image/')) { toast.error('That is not an image file'); shake(zone); return; }
    try {
      const info = await readImageFile(file);
      if (picked) URL.revokeObjectURL(picked.url);
      picked = { file, ...info };
      previewImg.src = info.url;
      previewMeta.replaceChildren(h('div.drop-name', file.name), h('div.drop-dims.mono', `${formatDims(info.width, info.height)} · ${formatBytes(file.size)}`));
      idle.hidden = true;
      previewPane.hidden = false;
      zone.classList.add('has-file');
      enter(previewPane);
      analyzeBtn.focus();
    } catch (err) {
      toast.error(err.message || 'Could not read that image');
      shake(zone);
    }
  }
  function clearPick() {
    if (picked) URL.revokeObjectURL(picked.url);
    picked = null;
    previewImg.removeAttribute('src');
    idle.hidden = false;
    previewPane.hidden = true;
    zone.classList.remove('has-file');
  }
  async function analyzePicked() {
    if (!picked) return;
    await createJob({ file: picked.file }, analyzeBtn);
  }
  async function createJob(payload, button) {
    if (creating) return;
    creating = true;
    button?.classList.add('is-loading');
    if (button) button.disabled = true;
    try {
      const job = await api.createJob({ ...payload, detail });
      navigate(`#/studio/${job.id}`);
    } catch (err) {
      toast.error(err?.userMessage || err?.message || 'Could not start the analysis');
    } finally {
      creating = false;
      button?.classList.remove('is-loading');
      if (button) button.disabled = false;
    }
  }

  const onPaste = (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) { const f = item.getAsFile(); if (f) { e.preventDefault(); pick(new File([f], f.name || 'pasted.png', { type: f.type })); } }
  };
  document.addEventListener('paste', onPaste);
  disposers.push(() => document.removeEventListener('paste', onPaste));
  // Highlight the zone when a file is dragged anywhere on the page.
  let dragDepth = 0;
  const onDocDrag = (e) => { if (e.type === 'dragenter') { dragDepth++; root.classList.add('is-dragging-file'); } else if (e.type === 'dragleave') { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) root.classList.remove('is-dragging-file'); } else { dragDepth = 0; root.classList.remove('is-dragging-file'); } };
  for (const t of ['dragenter', 'dragleave', 'drop']) document.addEventListener(t, onDocDrag);
  disposers.push(() => { for (const t of ['dragenter', 'dragleave', 'drop']) document.removeEventListener(t, onDocDrag); });

  // ---------------------------------------------------------------- samples + recent
  const samplesStrip = h('div.sample-strip', Array.from({ length: 6 }, () => skeleton('sk-sample')));
  const recentGrid = h('div.recent-grid', Array.from({ length: 3 }, () => skeleton('sk-recent')));
  const recentSection = h('section.home-section', h('div.section-head', h('h2', 'Recent'), h('a.link', { href: '#/gallery' }, 'Gallery ', icon('arrow', { size: 14 }))), recentGrid);

  const hero = h('section.hero',
    h('div.hero-glow', { aria: { hidden: true } }),
    h('h1.hero-title', h('span.wordmark-gradient', 'Chroma'), ' Studio'),
    h('p.hero-promise', 'Repaint any product photo. Keep every reflection.'),
    h('p.hero-sub', 'Chroma separates paint from light, finds each part, and recolors only the pigment — so the new finish sits under the same highlights, shadows and grain as the original shot.'));

  const view = h('div.home',
    hero,
    zone,
    h('section.home-section', h('div.section-head', h('h2', 'Try a sample'), h('span.section-hint', 'Analyzed with the detail level above')), samplesStrip),
    recentSection);
  replace(root, view);
  enterChildren(view, { step: 70 });

  api.samples().then((samples) => {
    replace(samplesStrip, samples.map((s) => sampleCard(s)));
    enterChildren(samplesStrip, { step: 30 });
  }).catch(() => replace(samplesStrip, h('div.panel-empty', icon('alert', { size: 20 }), h('p', 'Samples are unavailable — the server is not responding.'))));

  api.jobs().then((jobs) => {
    if (!jobs.length) { replace(recentGrid, h('div.panel-empty', icon('image', { size: 20 }), h('p', 'Nothing analyzed yet. Your jobs will show up here.'))); return; }
    replace(recentGrid, jobs.slice(0, 6).map((j) => recentCard(j)));
    enterChildren(recentGrid, { step: 40 });
  }).catch(() => { recentSection.hidden = true; });

  function sampleCard(s) {
    const btn = h('button.sample-card', { type: 'button', aria: { label: `Analyze sample ${s.title}` },
      onClick: () => createJob({ sample: s.name }, btn) },
      h('span.sample-thumb', { style: { aspectRatio: `${s.width} / ${s.height}` } }, h('img', { src: s.thumb, alt: '', loading: 'lazy', width: s.width, height: s.height })),
      h('span.sample-title', s.title),
      h('span.sample-meta.mono', formatDims(s.width, s.height)),
      h('span.sample-cta', icon('spark', { size: 14 }), 'Analyze'));
    return tip(btn, `${s.license}${s.artist ? ' · ' + s.artist : ''}`);
  }
  function recentCard(j) {
    return h('a.recent-card', { href: `#/studio/${j.id}` },
      h('span.recent-thumb', h('img', { src: j.thumb, alt: '', loading: 'lazy', onError: (e) => e.currentTarget.remove() })),
      h('span.recent-body',
        h('span.recent-name', j.name),
        h('span.recent-meta', h(`span.status-chip.status-${j.status}`, j.status), h('span.mono', `${j.n_groups} groups`), h('span', relativeTime(j.created)))));
  }

  return { destroy() { disposers.forEach((f) => f()); if (picked) URL.revokeObjectURL(picked.url); } };
}
