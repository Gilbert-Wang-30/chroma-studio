/**
 * Collapsible inspector panel shell shared by every Studio panel.
 *
 *   const p = panel({ id: 'groups', title: 'Groups', icon: 'layers', badge: '8' });
 *   p.body.append(...); p.setBadge('9'); p.setOpen(false);
 */
import { h, icon, tip } from '../dom.js';
import { animateHeight } from '../motion.js';

export function panel({ id, title, icon: iconName = null, badge = null, actions = [], open = true, subtitle = null }) {
  const badgeEl = h('span.panel-badge', { hidden: badge === null }, badge ?? '');
  const chevron = icon('chevronDown', { size: 16, className: 'panel-chevron' });
  const body = h('div.panel-body', { id: `${id}-body`, hidden: !open });
  const toggle = h('button.panel-toggle', {
    type: 'button', aria: { expanded: open, controls: `${id}-body` },
    onClick: () => setOpen(body.hidden),
  }, chevron, iconName ? icon(iconName, { size: 16, className: 'panel-icon' }) : null,
    h('span.panel-title', title), badgeEl);
  const actionsEl = h('div.panel-actions', actions);
  const head = h('div.panel-head', toggle, actionsEl);
  const el = h(`section.panel.panel-${id}`, { aria: { label: title } }, head,
    subtitle ? h('p.panel-subtitle', subtitle) : null, body);

  function setOpen(next) {
    toggle.setAttribute('aria-expanded', String(next));
    el.classList.toggle('is-open', next);
    animateHeight(body, next);
  }
  el.classList.toggle('is-open', open);

  return {
    el, body, actions: actionsEl,
    setOpen,
    isOpen: () => !body.hidden,
    setBadge(text) {
      badgeEl.hidden = text === null || text === undefined || text === '';
      badgeEl.textContent = text ?? '';
    },
    setDisabled(disabled) { el.classList.toggle('is-disabled', disabled); },
  };
}

/** Small icon-only button used in panel headers and rows. */
export function iconButton(name, label, onClick, { className = '', size = 16, pressed = null } = {}) {
  const b = h(`button.btn-icon${className ? '.' + className : ''}`, {
    type: 'button', aria: { label, ...(pressed === null ? {} : { pressed }) }, onClick,
  }, icon(name, { size }));
  return tip(b, label);
}
