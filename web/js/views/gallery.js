/**
 * Gallery: every job as a card with thumbnail, status chip, group count and a
 * two-step delete.
 */
import { api } from '../api.js';
import { h, icon, skeleton, replace, tip } from '../dom.js';
import { enterChildren, exit, flip } from '../motion.js';
import { toast } from '../toast.js';
import { formatDims, relativeTime } from '../util.js';

export function galleryView(root) {
  const grid = h('div.gallery-grid', Array.from({ length: 6 }, () => skeleton('sk-card')));
  const count = h('span.section-hint', '');
  const view = h('div.gallery',
    h('div.section-head.page-head', h('h1', 'Gallery'), count, h('span.spacer'), h('a.btn.btn-primary', { href: '#/' }, icon('plus', { size: 15 }), 'New')),
    grid);
  replace(root, view);

  let jobs = [];
  api.jobs().then((list) => {
    jobs = list;
    render();
    enterChildren(grid, { step: 35 });
  }).catch((err) => replace(grid, h('div.panel-empty', icon('alert', { size: 22 }), h('p', err?.userMessage || 'Could not load jobs.'))));

  function render() {
    count.textContent = jobs.length ? `${jobs.length} ${jobs.length === 1 ? 'job' : 'jobs'}` : '';
    if (!jobs.length) {
      replace(grid, h('div.empty-view.is-inline', icon('image', { size: 28 }), h('h2', 'No jobs yet'), h('p', 'Analyze a photo and it will show up here.'), h('a.btn.btn-primary', { href: '#/' }, 'Go to Home')));
      return;
    }
    replace(grid, jobs.map((j) => card(j)));
  }

  function card(j) {
    const del = tip(h('button.btn-icon.card-delete', { type: 'button', aria: { label: `Delete ${j.name}` }, onClick: (e) => { e.preventDefault(); e.stopPropagation(); askDelete(j, el); } }, icon('trash', { size: 15 })), 'Delete');
    const el = h('a.job-card', { href: `#/studio/${j.id}`, dataset: { key: j.id } },
      h('span.job-thumb', h('img', { src: j.thumb, alt: '', loading: 'lazy', onError: (e) => e.currentTarget.remove() }), h(`span.status-chip.status-${j.status}`, j.status)),
      h('span.job-body',
        h('span.job-name', j.name),
        h('span.job-meta', h('span.mono', formatDims(j.width, j.height)), h('span.mono', `${j.n_groups} groups`), h('span', relativeTime(j.created)))),
      del);
    return el;
  }

  function askDelete(j, el) {
    const confirm = h('div.card-confirm',
      h('span', 'Delete this job?'),
      h('button.btn.btn-sm.btn-danger', { type: 'button', onClick: async (e) => {
        e.preventDefault();
        try {
          await api.deleteJob(j.id);
          jobs = jobs.filter((x) => x.id !== j.id);
          await exit(el);
          flip(grid, () => render());
          toast.success(`Deleted ${j.name}`);
        } catch (err) { toast.error(err?.userMessage || 'Delete failed'); confirm.remove(); }
      } }, 'Delete'),
      h('button.btn.btn-sm.btn-ghost', { type: 'button', onClick: (e) => { e.preventDefault(); confirm.remove(); } }, 'Keep'));
    confirm.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); });
    el.querySelector('.card-confirm')?.remove();
    el.appendChild(confirm);
    confirm.querySelector('.btn-danger').focus();
  }

  return { destroy() {} };
}
