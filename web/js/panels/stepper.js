/**
 * Analysis stepper: the five pipeline stages with live progress bars, SSE messages
 * and an elapsed clock. Once the job is ready it collapses into a summary line with
 * count-up numbers; on error it shows the message and a way back.
 */
import { h, icon } from '../dom.js';
import { countUp, enter } from '../motion.js';
import { formatSeconds } from '../util.js';
import { panel } from './panel.js';

export const STEPS = [
  { key: 'ingest', label: 'Ingest', hint: 'Read and resize' },
  { key: 'intrinsic', label: 'Intrinsic', hint: 'Albedo · shading · residual' },
  { key: 'segment', label: 'Segment', hint: 'SAM 2 mask proposals' },
  { key: 'regions', label: 'Regions', hint: 'Merge, split and fill' },
  { key: 'groups', label: 'Groups', hint: 'Cluster by albedo' },
];

export function createStepper(store) {
  const rows = new Map();
  const list = h('ol.stepper', { aria: { label: 'Analysis progress' } });
  for (const step of STEPS) {
    const bar = h('div.step-bar-fill');
    const msg = h('div.step-msg', step.hint);
    const time = h('span.step-time.mono', '');
    const mark = h('span.step-mark', icon('check', { size: 12 }));
    const li = h('li.step', { dataset: { state: 'idle', key: step.key } },
      mark,
      h('div.step-main',
        h('div.step-head', h('span.step-label', step.label), time),
        msg,
        h('div.step-bar', { role: 'progressbar', aria: { label: `${step.label} progress`, valuemin: 0, valuemax: 100, valuenow: 0 } }, bar),
      ));
    rows.set(step.key, { li, bar, msg, time });
    list.appendChild(li);
  }

  const elapsed = h('span.mono', '0.0 s');
  const statusLine = h('div.stepper-status', h('span.pulse-dot'), h('span.stepper-status-text', 'Queued'), h('span.spacer'), elapsed);
  const summary = h('div.stepper-summary', { hidden: true });
  const errorBox = h('div.stepper-error', { hidden: true });

  const p = panel({ id: 'analysis', title: 'Analysis', icon: 'scan', open: true });
  p.body.append(statusLine, errorBox, list, summary);

  let timer = null;
  let startedAt = null;
  let finished = false;
  let lastSig = null;

  function startClock() {
    if (timer) return;
    startedAt = startedAt || performance.now();
    timer = setInterval(() => {
      elapsed.textContent = formatSeconds((performance.now() - startedAt) / 1000);
    }, 100);
  }
  function stopClock() { clearInterval(timer); timer = null; }

  function render(state) {
    const job = state.job;
    if (!job) return;
    const stages = job.stages || {};
    for (const step of STEPS) {
      const st = stages[step.key] || { state: 'idle', progress: 0, message: '' };
      const row = rows.get(step.key);
      row.li.dataset.state = st.state;
      const pct = Math.round((st.progress || 0) * 100);
      row.bar.style.width = `${st.state === 'done' ? 100 : pct}%`;
      row.bar.parentElement.setAttribute('aria-valuenow', String(pct));
      row.msg.textContent = st.state === 'running' && st.message ? st.message
        : st.state === 'error' ? (st.message || 'Failed')
        : st.state === 'done' ? (st.message || step.hint) : step.hint;
      row.time.textContent = st.state === 'done' && st.seconds ? formatSeconds(st.seconds) : '';
    }
    const text = statusLine.querySelector('.stepper-status-text');
    if (job.status === 'queued') {
      text.textContent = 'Queued — waiting for the GPU';
      statusLine.dataset.state = 'queued';
    } else if (job.status === 'analyzing') {
      const running = STEPS.find((s) => stages[s.key]?.state === 'running');
      text.textContent = running ? `Analyzing · ${running.label}` : 'Analyzing';
      statusLine.dataset.state = 'running';
      startClock();
    } else if (job.status === 'ready') {
      statusLine.dataset.state = 'done';
      text.textContent = 'Ready';
      stopClock();
      // The first 'ready' often arrives via an SSE status event carrying a partial job;
      // rebuild the summary whenever the numbers it shows actually change.
      const sig = `${job.regions_count ?? ''}|${(job.groups || []).length}|${job.timings?.total_s ?? ''}|${job.intrinsic_method ?? ''}`;
      if (sig !== lastSig) {
        showSummary(job, !finished);
        lastSig = sig;
        finished = true;
      }
    } else if (job.status === 'error') {
      statusLine.dataset.state = 'error';
      text.textContent = 'Analysis failed';
      stopClock();
      errorBox.hidden = false;
      errorBox.replaceChildren(icon('alert', { size: 16 }), h('div', h('strong', 'Something went wrong. '), job.error || 'The server reported an error.'));
    }
  }

  function showSummary(job, first = true) {
    const total = job.timings?.total_s || 0;
    elapsed.textContent = formatSeconds(total);
    const regionsN = h('span.stat-value.mono', '0');
    const groupsN = h('span.stat-value.mono', '0');
    const secondsN = h('span.stat-value.mono', '0');
    summary.replaceChildren(
      h('div.stat', regionsN, h('span.stat-label', 'regions')),
      h('div.stat', groupsN, h('span.stat-label', 'groups')),
      h('div.stat', secondsN, h('span.stat-label', 'seconds')),
      h('div.stat.stat-wide', h('span.stat-value', job.intrinsic_method === 'careaga' ? 'Intrinsic v2.1' : 'Heuristic'), h('span.stat-label', 'decomposition')),
    );
    summary.hidden = false;
    if (first) enter(summary);
    countUp(regionsN, job.regions_count || 0);
    countUp(groupsN, (job.groups || []).length);
    countUp(secondsN, total, { format: (n) => n.toFixed(1) });
    p.setBadge('done');
    if (first) setTimeout(() => { if (p.isOpen()) p.setOpen(false); }, 1400);
  }

  const off = store.watch(render, ['job']);
  return {
    el: p.el,
    panel: p,
    destroy() { off(); stopClock(); },
  };
}
