/**
 * Finish panel: realism sliders (live preview through the store's render options)
 * and export (work/full, png/jpg) with a download button and a share link.
 */
import { h, icon, segmented, tip } from '../dom.js';
import { enter, pop } from '../motion.js';
import { copyText, formatDims, formatMs, uid } from '../util.js';
import { DEFAULT_RENDER_OPTIONS } from '../state.js';
import { panel } from './panel.js';

const SLIDERS = [
  { key: 'texture', label: 'Texture', min: 0, max: 1, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%`, hint: 'How much of the original paint texture survives (0 = flat coat).' },
  { key: 'feather_px', label: 'Feather', min: 0, max: 6, step: 0.1, fmt: (v) => `${v.toFixed(1)} px`, hint: 'Softness of the edge between groups.' },
  { key: 'shading_strength', label: 'Shading', min: 0.5, max: 1.5, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%`, hint: 'Below 100% flattens the light, above deepens it.' },
  { key: 'residual_tint', label: 'Highlight tint', min: 0, max: 1, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%`, hint: 'Tint specular highlights toward the new colour.' },
  { key: 'saturation', label: 'Saturation', min: 0, max: 2, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%`, hint: 'Chroma of the target colours.' },
];

/**
 * @param store Studio store
 * @param {{exportImage: (quality: string, format: string) => Promise<{url, width, height, ms}>, shareUrl: () => string}} actions
 */
export function createFinishPanel(store, actions) {
  const controls = new Map();
  const sliders = h('div.slider-stack', SLIDERS.map((s) => {
    const id = uid('opt');
    const input = h('input.range', { type: 'range', id, min: s.min, max: s.max, step: s.step, aria: { label: s.label } });
    const value = h('output.mono.range-value', { for: id });
    input.addEventListener('input', () => {
      const v = Number(input.value);
      paint(s, input, value, v);
      store.setRenderOption(s.key, v);
    });
    input.addEventListener('dblclick', () => { store.setRenderOption(s.key, DEFAULT_RENDER_OPTIONS[s.key]); });
    controls.set(s.key, { input, value, spec: s });
    const row = h('div.range-row', h('label.range-label', { for: id }, s.label), input, value);
    return tip(row, s.hint, 'left');
  }));

  const keepResidual = h('input', { type: 'checkbox', id: uid('keep'), role: 'switch' });
  keepResidual.addEventListener('change', () => store.setRenderOption('keep_residual', keepResidual.checked));
  const keepRow = h('label.switch-row', { for: keepResidual.id }, h('span', 'Keep highlights & reflections'), h('span.switch', keepResidual, h('span.switch-knob')));

  const resetBtn = h('button.btn.btn-sm.btn-ghost', { type: 'button', onClick: () => store.resetRenderOptions() }, 'Reset to defaults');

  function paint(spec, input, value, v) {
    value.textContent = spec.fmt(v);
    input.style.setProperty('--p', `${((v - spec.min) / (spec.max - spec.min)) * 100}%`);
  }

  // ---------------------------------------------------------------- export

  let quality = 'work';
  let format = 'png';
  const qualitySeg = segmented([
    { value: 'work', label: 'Work', title: 'Working resolution (fast)' },
    { value: 'full', label: 'Full', title: 'Original resolution' },
  ], quality, (v) => { quality = v; }, { ariaLabel: 'Export resolution', size: 'sm' });
  const formatSeg = segmented([
    { value: 'png', label: 'PNG' }, { value: 'jpg', label: 'JPG' },
  ], format, (v) => { format = v; }, { ariaLabel: 'Export format', size: 'sm' });
  const exportBtn = h('button.btn.btn-primary', { type: 'button', onClick: () => onExport() }, icon('download', { size: 15 }), h('span', 'Export'));
  const exportRow = h('div.export-row', qualitySeg, formatSeg, h('span.spacer'), exportBtn);
  const exportResult = h('div.export-result', { hidden: true });
  const shareBtn = h('button.btn.btn-sm', { type: 'button', onClick: () => onShare() }, icon('link', { size: 14 }), 'Copy share link');
  tip(shareBtn, 'Link to this job in the Studio');

  let exporting = false;
  async function onExport() {
    if (exporting) return;
    exporting = true;
    exportBtn.disabled = true;
    exportBtn.classList.add('is-loading');
    exportBtn.querySelector('span').textContent = quality === 'full' ? 'Rendering full…' : 'Rendering…';
    try {
      const res = await actions.exportImage(quality, format);
      renderExport(res);
    } catch (err) {
      exportResult.hidden = false;
      exportResult.replaceChildren(h('div.panel-empty.is-error', icon('alert', { size: 18 }), h('p', err?.userMessage || err?.message || 'Export failed.')));
    } finally {
      exporting = false;
      exportBtn.disabled = false;
      exportBtn.classList.remove('is-loading');
      exportBtn.querySelector('span').textContent = 'Export';
    }
  }

  function renderExport(res) {
    const name = decodeURIComponent(res.url.split('/').pop() || 'export');
    exportResult.hidden = false;
    exportResult.replaceChildren(
      h('div.export-file',
        h('span.export-icon', icon('check', { size: 16 })),
        h('div.export-info',
          h('div.export-name', name),
          h('div.export-meta.mono', `${formatDims(res.width, res.height)} · ${formatMs(res.ms)}`)),
        h('a.btn.btn-sm.btn-primary', { href: res.url, download: name }, icon('download', { size: 14 }), 'Download')),
    );
    enter(exportResult);
    pop(exportResult);
  }

  async function onShare() {
    const ok = await copyText(actions.shareUrl());
    p.el.dispatchEvent(new CustomEvent('chroma:toast', { bubbles: true, detail: ok ? { message: 'Share link copied', type: 'success' } : { message: 'Could not copy — your browser blocked clipboard access', type: 'error' } }));
  }

  const p = panel({ id: 'finish', title: 'Finish', icon: 'sliders' });
  p.body.append(
    h('h4.panel-h', 'Realism'), sliders, keepRow, h('div.panel-footer', resetBtn),
    h('h4.panel-h', 'Export'), exportRow, exportResult, h('div.panel-footer', shareBtn),
  );

  const off = store.watch((s) => {
    for (const [key, c] of controls) {
      const v = Number(s.renderOptions[key] ?? DEFAULT_RENDER_OPTIONS[key]);
      if (Number(c.input.value) !== v) c.input.value = String(v);
      paint(c.spec, c.input, c.value, v);
    }
    keepResidual.checked = Boolean(s.renderOptions.keep_residual);
  }, ['renderOptions']);

  return {
    el: p.el,
    panel: p,
    destroy() { off(); },
  };
}
