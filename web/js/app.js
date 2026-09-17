/**
 * Boot: theme, header, hash router and global chrome (toast bridge, shortcut sheet,
 * GPU status chip). Views are functions `(root, ctx) => { destroy() }`.
 */
import { api } from './api.js';
import { h, icon, kbd, tip, replace } from './dom.js';
import { enter } from './motion.js';
import { prefs } from './state.js';
import { toast } from './toast.js';
import { isTypingTarget } from './util.js';
import { homeView } from './views/home.js';
import { studioView } from './views/studio.js';
import { galleryView } from './views/gallery.js';
import { howView } from './views/how.js';

// ------------------------------------------------------------------ theme

function applyTheme(theme) {
  if (theme === 'light') document.documentElement.dataset.theme = 'light';
  else delete document.documentElement.dataset.theme;
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'light' ? '#f6f7fa' : '#0b0d12');
}
function currentTheme() { return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'; }
applyTheme(prefs.get('theme', window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'));

// ------------------------------------------------------------------ header

const themeBtn = tip(h('button.btn-icon.header-btn', { type: 'button', aria: { label: 'Toggle theme' }, onClick: () => {
  const next = currentTheme() === 'light' ? 'dark' : 'light';
  applyTheme(next);
  prefs.set('theme', next);
  paintThemeBtn();
} }), 'Theme');
function paintThemeBtn() { themeBtn.replaceChildren(icon(currentTheme() === 'light' ? 'moon' : 'sun')); }
paintThemeBtn();

const keysBtn = tip(h('button.btn-icon.header-btn', { type: 'button', aria: { label: 'Keyboard shortcuts' }, onClick: () => openShortcuts() }, icon('keyboard')), 'Shortcuts (?)');
const gpuChip = h('a.gpu-chip', { href: '#/how', aria: { label: 'Server status' } }, h('span.gpu-dot'), h('span.gpu-text', 'Connecting…'));
const studioLink = h('a.nav-link', { href: '#/', dataset: { route: 'studio' }, hidden: true }, 'Studio');
const nav = h('nav.header-nav', { aria: { label: 'Primary' } },
  studioLink,
  h('a.nav-link', { href: '#/gallery', dataset: { route: 'gallery' } }, 'Gallery'),
  h('a.nav-link', { href: '#/how', dataset: { route: 'how' } }, 'How it works'));
const header = h('header.app-header',
  h('a.wordmark', { href: '#/', aria: { label: 'Chroma Studio home' } },
    h('span.wordmark-glyph', { aria: { hidden: true } }, h('i'), h('i'), h('i')),
    h('span.wordmark-text', h('span.wordmark-gradient', 'Chroma'), ' Studio')),
  nav,
  h('div.header-right', gpuChip, themeBtn, keysBtn));

const main = h('main.app-main', { id: 'view' });
document.body.prepend(header, main);

// ------------------------------------------------------------------ health chip

async function pollHealth() {
  try {
    const hres = await api.health();
    const ready = hres.models?.sam2 === 'ready' && hres.models?.intrinsic === 'ready';
    const loading = hres.models?.sam2 === 'loading' || hres.models?.intrinsic === 'loading';
    gpuChip.dataset.state = ready ? 'ready' : loading ? 'loading' : 'cold';
    const gb = (mb) => (mb / 1024).toFixed(1);
    const gpuName = (hres.gpu || hres.device || 'GPU').replace(/NVIDIA GeForce /, '').replace(/ \(mock\)/, '');
    gpuChip.querySelector('.gpu-text').textContent = `${gpuName} · ${gb(hres.vram_used_mb)} / ${gb(hres.vram_total_mb)} GB`;
    gpuChip.title = ready ? 'Models ready' : loading ? 'Models loading…' : 'Models cold — the first job will be slower';
  } catch {
    gpuChip.dataset.state = 'offline';
    gpuChip.querySelector('.gpu-text').textContent = 'Server offline';
  }
}
pollHealth();
setInterval(pollHealth, 10000);

// ------------------------------------------------------------------ shortcuts sheet

const SHORTCUTS = [
  ['1 – 6', 'Switch layer: Result, Original, Albedo, Shading, Regions, Groups'],
  ['Space (hold)', 'Peek at the original'],
  ['C', 'Toggle the before / after wipe'],
  ['F  /  0', 'Fit to window  /  actual pixels'],
  ['+  /  −', 'Zoom in / out (or scroll)'],
  ['Click', 'Select the group under the cursor'],
  ['⌘ / Ctrl Click', 'Add a colour to the selection (in the list or on the image)'],
  ['M', 'Merge the selected colours into one'],
  ['A', 'Select every colour'],
  ['⇧ Click', 'Add a region to a multi-selection'],
  ['⌘Z  /  ⇧⌘Z', 'Undo / redo a mapping change'],
  ['Esc', 'Clear the selection'],
  ['?', 'This sheet'],
];
let sheet = null;
function openShortcuts() {
  if (sheet) { closeShortcuts(); return; }
  sheet = h('div.modal-backdrop', { onClick: (e) => { if (e.target === sheet) closeShortcuts(); } },
    h('div.modal', { role: 'dialog', aria: { modal: true, labelledby: 'shortcuts-title' } },
      h('div.modal-head', h('h2#shortcuts-title', 'Keyboard shortcuts'), tip(h('button.btn-icon', { type: 'button', aria: { label: 'Close' }, onClick: closeShortcuts }, icon('x')), 'Close')),
      h('dl.shortcut-list', SHORTCUTS.map(([keys, desc]) => [h('dt', keys.split(/\s+\/\s+/).flatMap((k, i) => [i ? h('span.kbd-sep', '/') : null, kbd(k)])), h('dd', desc)]))));
  document.body.appendChild(sheet);
  enter(sheet.firstElementChild, { distance: 12 });
  sheet.querySelector('button').focus();
}
function closeShortcuts() { sheet?.remove(); sheet = null; }
document.addEventListener('keydown', (e) => {
  if (isTypingTarget(e.target)) return;
  if (e.key === '?' ) { e.preventDefault(); openShortcuts(); }
  if (e.key === 'Escape' && sheet) closeShortcuts();
});

// ------------------------------------------------------------------ toast bridge (panels dispatch chroma:toast)

document.addEventListener('chroma:toast', (e) => { const d = e.detail || {}; toast(d.message, { type: d.type || 'info' }); });

// ------------------------------------------------------------------ router

const ROUTES = [
  { pattern: /^#\/?$/, view: homeView, name: 'home', title: 'Chroma Studio' },
  { pattern: /^#\/studio\/([A-Za-z0-9_-]+)$/, view: studioView, name: 'studio', title: 'Studio · Chroma Studio' },
  { pattern: /^#\/gallery$/, view: galleryView, name: 'gallery', title: 'Gallery · Chroma Studio' },
  { pattern: /^#\/how$/, view: howView, name: 'how', title: 'How it works · Chroma Studio' },
];
let current = null;

function navigate(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

function route() {
  const hash = location.hash || '#/';
  const match = ROUTES.map((r) => ({ r, m: hash.match(r.pattern) })).find((x) => x.m);
  current?.destroy?.();
  current = null;
  main.scrollTop = 0;
  window.scrollTo(0, 0);
  if (!match) {
    replace(main, h('div.empty-view', icon('alert', { size: 28 }), h('h2', 'Page not found'), h('p', `Nothing lives at ${hash}.`), h('a.btn.btn-primary', { href: '#/' }, 'Back to Home')));
    document.title = 'Not found · Chroma Studio';
    setActiveRoute('none');
    return;
  }
  const { r, m } = match;
  document.title = r.title;
  document.body.dataset.route = r.name;
  setActiveRoute(r.name);
  const ctx = { navigate, jobId: m[1] };
  if (r.name === 'studio') { studioLink.href = hash; studioLink.hidden = false; }
  else {
    let last = null;
    try { last = sessionStorage.getItem('chroma:lastJob'); } catch { /* ignore */ }
    studioLink.hidden = !last;
    if (last) studioLink.href = `#/studio/${last}`;
  }
  current = r.view(main, ctx);
}

function setActiveRoute(name) {
  for (const a of nav.querySelectorAll('.nav-link')) a.setAttribute('aria-current', a.dataset.route === name ? 'page' : 'false');
}

window.addEventListener('hashchange', route);
route();

// Mark fonts as ready so the first paint does not shift once Sora/Inter arrive.
document.fonts?.ready.then(() => document.documentElement.classList.add('fonts-ready'));
