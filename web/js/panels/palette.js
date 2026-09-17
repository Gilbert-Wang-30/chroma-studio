/**
 * Palette panel: prompt → generated palette with source thumbnails and editable,
 * reorderable swatches. Swatches are draggable onto mapping rows (see mapping.js) using
 * the shared COLOR_DRAG_TYPE; dragging within the strip reorders.
 */
import { h, icon, tip, skeleton, replace } from '../dom.js';
import { enterChildren, flip, pop, shake } from '../motion.js';
import { normalizeHex, textColorOn, uid } from '../util.js';
import { panel, iconButton } from './panel.js';

export const COLOR_DRAG_TYPE = 'application/x-chroma-color';
export const SUGGESTIONS = ['Hawaii sunset', 'Stealth matte', 'Sakura', 'Racing livery', 'Cyberpunk', 'Desert camo'];
const METHOD_LABEL = {
  images: 'Distilled from reference images',
  theme: 'Curated theme, tuned with reference images',
  parsed: 'Built from the colours you named',
  mixed: 'Your colours, topped up from reference images',
  fallback: 'Fallback set — try a more specific prompt',
};

/**
 * @param store Studio store (reads/writes `palette`)
 * @param {{generate: (prompt: string, n: number) => Promise<object>}} actions
 */
export function createPalettePanel(store, actions) {
  const inputId = uid('prompt');
  const input = h('input.input.prompt-input', { type: 'text', id: inputId, placeholder: 'Describe a look — "Hawaii sunset", "navy and gold"…', autocomplete: 'off', maxlength: 120, aria: { label: 'Palette prompt' } });
  const count = h('select.select.count-select', { aria: { label: 'Number of colours' } },
    [4, 5, 6, 7, 8].map((n) => h('option', { value: n, selected: n === 6 }, `${n} colours`)));
  const generate = h('button.btn.btn-primary', { type: 'button', onClick: () => onGenerate() }, icon('wand', { size: 15 }), h('span', 'Generate'));
  const form = h('form.palette-form', { onSubmit: (e) => { e.preventDefault(); onGenerate(); } }, input, count, generate);
  const chips = h('div.chip-row', { aria: { label: 'Suggestions' } },
    SUGGESTIONS.map((s) => h('button.chip', { type: 'button', onClick: () => { input.value = s; onGenerate(); } }, s)));

  const result = h('div.palette-result');
  const p = panel({ id: 'palette', title: 'Palette', icon: 'palette' });
  p.body.append(form, chips, result);

  let generating = false;

  async function onGenerate() {
    const prompt = input.value.trim();
    if (!prompt) { shake(input); input.focus(); return; }
    if (generating) return;
    generating = true;
    generate.disabled = true;
    generate.classList.add('is-loading');
    renderLoading(Number(count.value));
    try {
      await actions.generate(prompt, Number(count.value));
    } catch (err) {
      renderError(err);
    } finally {
      generating = false;
      generate.disabled = false;
      generate.classList.remove('is-loading');
    }
  }

  // ---------------------------------------------------------------- states

  function renderEmpty() {
    result.replaceChildren(h('div.panel-empty',
      icon('palette', { size: 22 }),
      h('p', 'No palette yet. Describe a mood, a place, a livery — or name the colours you want.')));
  }

  function renderLoading(n) {
    result.replaceChildren(
      h('div.palette-meta', skeleton('sk-line', { width: '60%' })),
      h('div.source-row', Array.from({ length: 4 }, () => skeleton('sk-source'))),
      h('div.swatch-strip', Array.from({ length: n }, () => skeleton('sk-swatch'))),
    );
  }

  function renderError(err) {
    result.replaceChildren(h('div.panel-empty.is-error',
      icon('alert', { size: 22 }),
      h('p', err?.userMessage || err?.message || 'Could not generate a palette.'),
      h('button.btn.btn-sm', { type: 'button', onClick: () => onGenerate() }, 'Try again')));
  }

  function renderPalette(pal) {
    if (!pal) { renderEmpty(); return; }
    const meta = h('div.palette-meta',
      h('span.palette-method', METHOD_LABEL[pal.method] || 'Palette'),
      pal.prompt ? h('span.palette-prompt', `“${pal.prompt}”`) : null);
    const sources = (pal.sources || []).length
      ? h('div.source-row', { aria: { label: 'Reference images' } }, pal.sources.map((s) => sourceThumb(s)))
      : null;
    const strip = h('div.swatch-strip', { role: 'list', aria: { label: 'Palette colours' } });
    const addBtn = h('button.swatch-add', { type: 'button', onClick: () => addColor() }, icon('plus', { size: 16 }), h('span', 'Add'));
    tip(addBtn, 'Add a colour');
    replace(result, meta, sources, strip, addBtn);
    renderSwatches(strip, pal.colors, true);
    if (pal.prompt && input.value.trim() === '') input.value = pal.prompt;
  }

  function sourceThumb(s) {
    const a = h('a.source-thumb', { href: s.url || '#', target: '_blank', rel: 'noopener noreferrer', aria: { label: `${s.title} (${s.license})` } },
      h('img', { src: s.thumb, alt: '', loading: 'lazy', width: 64, height: 48 }));
    return tip(a, `${s.title}${s.license ? ' · ' + s.license : ''}`);
  }

  function renderSwatches(strip, colors, first = false) {
    flip(strip, () => {
      strip.replaceChildren(...colors.map((c, i) => swatchEl(c, i)));
    });
    if (first) enterChildren(strip, { step: 40 });
  }

  function swatchEl(c, index) {
    const fg = textColorOn(c.hex);
    const picker = h('input.swatch-picker', { type: 'color', value: c.hex, aria: { label: `Edit colour ${c.name}` } });
    picker.addEventListener('input', () => updateColor(index, picker.value));
    const remove = h('button.swatch-remove', { type: 'button', aria: { label: `Remove ${c.name}` }, onClick: (e) => { e.stopPropagation(); removeColor(index); } }, icon('x', { size: 12 }));
    const el = h('div.swatch', {
      role: 'listitem', draggable: true, dataset: { key: c.hex + ':' + index, index: String(index) },
      style: { '--swatch': c.hex, color: fg },
      onDragstart: (e) => {
        e.dataTransfer.setData(COLOR_DRAG_TYPE, c.hex);
        e.dataTransfer.setData('text/plain', c.hex);
        e.dataTransfer.setData('application/x-chroma-swatch-index', String(index));
        e.dataTransfer.effectAllowed = 'copyMove';
        el.classList.add('is-dragging');
      },
      onDragend: () => el.classList.remove('is-dragging'),
      onDragover: (e) => {
        if (!e.dataTransfer.types.includes('application/x-chroma-swatch-index')) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        el.classList.add('is-drop-target');
      },
      onDragleave: () => el.classList.remove('is-drop-target'),
      onDrop: (e) => {
        el.classList.remove('is-drop-target');
        const from = Number(e.dataTransfer.getData('application/x-chroma-swatch-index'));
        if (!Number.isFinite(from) || from === index) return;
        e.preventDefault();
        reorder(from, index);
      },
    },
      h('button.swatch-face', { type: 'button', aria: { label: `${c.name} ${c.hex}. Click to edit` }, onClick: () => picker.click() },
        h('span.swatch-weight', { style: { height: `${Math.round(Math.min(1, c.weight * 3) * 100)}%` } }),
        picker),
      remove,
      h('span.swatch-name', c.name),
      h('span.swatch-hex.mono', c.hex));
    return el;
  }

  // ---------------------------------------------------------------- edits

  function currentColors() { return store.get().palette?.colors || []; }
  function setColors(colors, { animate = true } = {}) {
    const pal = store.get().palette || { id: null, prompt: '', method: 'fallback', sources: [], colors: [] };
    store.set({ palette: { ...pal, colors, edited: true } });
    const strip = result.querySelector('.swatch-strip');
    if (strip) renderSwatches(strip, colors);
    if (animate) {
      const last = result.querySelector('.swatch:last-child');
      if (last) pop(last);
    }
  }
  function updateColor(index, hex) {
    const norm = normalizeHex(hex);
    if (!norm) return;
    const colors = currentColors().map((c, i) => (i === index ? { ...c, hex: norm, name: c.name } : c));
    // Live preview while the native picker is open: patch the DOM without a full re-render.
    const el = result.querySelectorAll('.swatch')[index];
    if (el) {
      el.style.setProperty('--swatch', norm);
      el.style.color = textColorOn(norm);
      el.querySelector('.swatch-hex').textContent = norm;
    }
    const pal = store.get().palette;
    store.set({ palette: { ...pal, colors, edited: true } });
  }
  function removeColor(index) {
    const colors = currentColors().filter((_, i) => i !== index);
    setColors(colors, { animate: false });
    if (!colors.length) renderPalette(store.get().palette);
  }
  function addColor() {
    if (!store.get().palette) store.set({ palette: { id: null, prompt: '', method: 'fallback', sources: [], colors: [] } });
    const colors = [...currentColors(), { hex: '#808080', name: 'Gray', weight: 0.1, lab: [54, 0, 0] }];
    if (!result.querySelector('.swatch-strip')) renderPalette({ ...store.get().palette, colors });
    setColors(colors);
    const pickers = result.querySelectorAll('.swatch-picker');
    pickers[pickers.length - 1]?.click();
  }
  function reorder(from, to) {
    const colors = [...currentColors()];
    const [moved] = colors.splice(from, 1);
    colors.splice(to, 0, moved);
    setColors(colors, { animate: false });
  }

  let lastPaletteId = undefined;
  const off = store.watch((s) => {
    const pal = s.palette;
    // Full re-render only when a different palette arrives; edits are patched in place.
    const key = pal ? `${pal.id}:${pal.colors.length}` : null;
    if (key !== lastPaletteId) {
      lastPaletteId = key;
      renderPalette(pal);
    }
  }, ['palette']);

  return {
    el: p.el,
    panel: p,
    focus: () => input.focus(),
    destroy() { off(); },
  };
}
