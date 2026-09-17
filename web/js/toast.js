/**
 * Toast notifications, bottom-right. One container per page; toasts stack upward,
 * auto-dismiss, pause on hover, and can carry a single action button.
 */
import { h, icon } from './dom.js';
import { reducedMotion, EASE_OUT } from './motion.js';

let container = null;

function ensureContainer() {
  if (!container) {
    container = h('div.toasts', { role: 'status', aria: { live: 'polite' } });
    document.body.appendChild(container);
  }
  return container;
}

const ICON_FOR = { info: 'info', success: 'check', error: 'alert', warning: 'alert' };

/**
 * Shows a toast. Returns { dismiss() }.
 * @param {string} message
 * @param {{type?: 'info'|'success'|'error'|'warning', action?: {label: string, onClick: () => void},
 *          duration?: number, id?: string}} [opts]
 */
export function toast(message, opts = {}) {
  const { type = 'info', action = null, duration = type === 'error' ? 7000 : 3800, id = null } = opts;
  const root = ensureContainer();
  if (id) root.querySelector(`[data-toast-id="${CSS.escape(id)}"]`)?.remove();

  let timer = null;
  let removed = false;
  const el = h(`div.toast.toast-${type}`, { dataset: id ? { toastId: id } : {} },
    h('span.toast-icon', icon(ICON_FOR[type] || 'info', { size: 16 })),
    h('span.toast-text', message),
    action ? h('button.toast-action', { type: 'button', onClick: () => { action.onClick(); dismiss(); } }, action.label) : null,
    h('button.toast-close', { type: 'button', aria: { label: 'Dismiss' }, onClick: () => dismiss() }, icon('x', { size: 14 })),
  );

  function dismiss() {
    if (removed) return;
    removed = true;
    clearTimeout(timer);
    if (reducedMotion()) { el.remove(); return; }
    el.animate([{ opacity: 1, transform: 'translateY(0)' }, { opacity: 0, transform: 'translateY(6px)' }],
      { duration: 160, easing: 'ease-in', fill: 'forwards' }).finished.catch(() => {}).finally(() => el.remove());
  }
  function arm() { clearTimeout(timer); if (duration > 0) timer = setTimeout(dismiss, duration); }

  el.addEventListener('mouseenter', () => clearTimeout(timer));
  el.addEventListener('mouseleave', arm);
  root.appendChild(el);
  if (!reducedMotion()) {
    el.animate([{ opacity: 0, transform: 'translateY(10px) scale(.98)' }, { opacity: 1, transform: 'translateY(0) scale(1)' }],
      { duration: 240, easing: EASE_OUT });
  }
  arm();
  return { dismiss };
}

toast.success = (m, o) => toast(m, { ...o, type: 'success' });
toast.error = (m, o) => toast(m, { ...o, type: 'error' });
toast.warn = (m, o) => toast(m, { ...o, type: 'warning' });
