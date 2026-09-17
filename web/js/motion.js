/**
 * Motion helpers. Every helper is a no-op (or instant) when the user prefers reduced
 * motion, so callers never need to branch on it themselves.
 */

const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)');

/** True when the OS asks for reduced motion. Re-evaluated on every call. */
export function reducedMotion() {
  return Boolean(mq?.matches);
}

export const EASE_OUT = 'cubic-bezier(.2,.8,.2,1)';
export const DUR = 240;

/**
 * Fade + 8 px rise on mount. `delay` staggers siblings. Returns the element.
 * The element must already be in the DOM for the animation to play.
 */
export function enter(el, { delay = 0, distance = 8, duration = DUR } = {}) {
  if (!el || reducedMotion()) return el;
  el.animate(
    [{ opacity: 0, transform: `translateY(${distance}px)` }, { opacity: 1, transform: 'translateY(0)' }],
    { duration, delay, easing: EASE_OUT, fill: 'backwards' },
  );
  return el;
}

/** Staggered enter for every child of `container`. */
export function enterChildren(container, { step = 28, max = 12, ...opts } = {}) {
  Array.from(container.children).forEach((child, i) => enter(child, { delay: Math.min(i, max) * step, ...opts }));
}

/** Fade out then resolve; the caller removes the element. */
export function exit(el, { duration = 160 } = {}) {
  if (!el || reducedMotion()) return Promise.resolve();
  const anim = el.animate([{ opacity: 1 }, { opacity: 0, transform: 'translateY(-4px)' }],
    { duration, easing: 'ease-in', fill: 'forwards' });
  return anim.finished.catch(() => {});
}

/**
 * FLIP: records the positions of `container`'s children keyed by `data-key`, runs
 * `mutate()` (which reorders / adds / removes children), then animates every surviving
 * child from its old position to the new one. New children get an enter animation.
 */
export function flip(container, mutate, { duration = 260 } = {}) {
  if (reducedMotion()) { mutate(); return; }
  const before = new Map();
  for (const child of container.children) {
    if (child.dataset.key !== undefined) before.set(child.dataset.key, child.getBoundingClientRect());
  }
  mutate();
  for (const child of container.children) {
    const key = child.dataset.key;
    if (key === undefined) continue;
    const prev = before.get(key);
    const next = child.getBoundingClientRect();
    if (!prev) { enter(child, { duration }); continue; }
    const dx = prev.left - next.left;
    const dy = prev.top - next.top;
    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) continue;
    child.animate(
      [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'translate(0, 0)' }],
      { duration, easing: EASE_OUT },
    );
  }
}

/**
 * Damped spring from `from` to `to`, calling `onUpdate(value)` each frame.
 * Returns a cancel function. Instant under reduced motion.
 */
export function spring({ from, to, stiffness = 180, damping = 20, mass = 1, velocity = 0, onUpdate, onDone }) {
  if (reducedMotion() || Math.abs(to - from) < 1e-6) {
    onUpdate?.(to);
    onDone?.();
    return () => {};
  }
  let x = from;
  let v = velocity;
  let raf = 0;
  let last = performance.now();
  const step = (now) => {
    const dt = Math.min(0.064, (now - last) / 1000);
    last = now;
    // Semi-implicit Euler with sub-steps keeps stiff springs stable.
    const sub = 4;
    const h = dt / sub;
    for (let i = 0; i < sub; i++) {
      const a = (-stiffness * (x - to) - damping * v) / mass;
      v += a * h;
      x += v * h;
    }
    if (Math.abs(x - to) < 0.0005 && Math.abs(v) < 0.005) {
      onUpdate?.(to);
      onDone?.();
      return;
    }
    onUpdate?.(x);
    raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
  return () => cancelAnimationFrame(raf);
}

/**
 * Animates a number into `el.textContent` from its current value to `to`.
 * `format(n)` renders the value (default: rounded integer).
 */
export function countUp(el, to, { duration = 700, format = (n) => Math.round(n).toLocaleString(), from = null } = {}) {
  const start = from ?? (parseFloat(String(el.textContent).replace(/[^\d.]/g, '')) || 0);
  if (reducedMotion() || duration <= 0) { el.textContent = format(to); return; }
  const t0 = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - t0) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = format(start + (to - start) * eased);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/**
 * Expands or collapses `el` by animating its height. `el` keeps `hidden` when closed
 * so it is out of the accessibility tree. Returns a promise resolved when done.
 */
export function animateHeight(el, open, { duration = 220 } = {}) {
  if (reducedMotion()) {
    el.hidden = !open;
    el.style.height = '';
    return Promise.resolve();
  }
  el.style.overflow = 'hidden';
  if (open) {
    el.hidden = false;
    const target = el.scrollHeight;
    const anim = el.animate([{ height: '0px', opacity: 0 }, { height: `${target}px`, opacity: 1 }],
      { duration, easing: EASE_OUT });
    return anim.finished.catch(() => {}).finally(() => { el.style.overflow = ''; el.style.height = ''; });
  }
  const current = el.getBoundingClientRect().height;
  const anim = el.animate([{ height: `${current}px`, opacity: 1 }, { height: '0px', opacity: 0 }],
    { duration: duration * 0.8, easing: 'ease-in' });
  return anim.finished.catch(() => {}).finally(() => { el.hidden = true; el.style.overflow = ''; });
}

/** Brief "pop" scale on an element (e.g. a swatch that just received a drop). */
export function pop(el, { scale = 1.08, duration = 260 } = {}) {
  if (!el || reducedMotion()) return;
  el.animate([{ transform: 'scale(1)' }, { transform: `scale(${scale})`, offset: 0.4 }, { transform: 'scale(1)' }],
    { duration, easing: EASE_OUT });
}

/** Horizontal shake for invalid input. */
export function shake(el) {
  if (!el || reducedMotion()) return;
  el.animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-5px)' }, { transform: 'translateX(5px)' },
    { transform: 'translateX(-3px)' }, { transform: 'translateX(0)' }], { duration: 320, easing: 'ease-out' });
}
