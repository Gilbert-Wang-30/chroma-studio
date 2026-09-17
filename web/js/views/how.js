/**
 * How it works: the five-stage pipeline diagram that lights up in sequence, plus one
 * plain-language paragraph per stage.
 */
import { h, icon, replace } from '../dom.js';
import { enterChildren, reducedMotion } from '../motion.js';

const STAGES = [
  { key: 'ingest', icon: 'image', title: 'Ingest', text: 'Your photo is read, oriented and resized into a working copy. The original stays untouched for the final export.' },
  { key: 'intrinsic', icon: 'shine', title: 'Separate paint from light', text: 'An intrinsic decomposition model splits every pixel into albedo (the paint colour), shading (how light falls on the surface) and a residual for reflections and glare.' },
  { key: 'segment', icon: 'scan', title: 'Find the parts', text: 'SAM 2 proposes masks for everything it can see — panels, trim, wheels, laces, background — at a density you choose with Fast, Balanced or Max.' },
  { key: 'regions', icon: 'grid', title: 'Clean regions', text: 'Overlapping proposals are resolved, two-tone parts are split, gaps are filled with superpixels and specks are absorbed, so every pixel belongs to exactly one region.' },
  { key: 'groups', icon: 'layers', title: 'Group by colour', text: 'Regions with the same paint are clustered into colour groups. Each group gets a name and a swatch, and that is what you recolor.' },
];

export function howView(root) {
  const nodes = STAGES.map((s, i) => h('li.pipe-node', { dataset: { key: s.key }, style: { '--i': i } },
    h('span.pipe-icon', icon(s.icon, { size: 20 })),
    h('span.pipe-title', s.title),
    i < STAGES.length - 1 ? h('span.pipe-link', { aria: { hidden: true } }) : null));
  const diagram = h('ol.pipeline', { aria: { label: 'Pipeline stages' } }, nodes);

  const view = h('div.how',
    h('div.page-head', h('h1', 'How it works'), h('p.page-lede', 'Recoloring only the paint layer is what keeps the result photographic. Here is what happens between drop and download.')),
    diagram,
    h('div.how-steps', STAGES.map((s, i) => h('article.how-step', h('span.how-num.mono', String(i + 1).padStart(2, '0')), h('div', h('h3', s.title), h('p', s.text))))),
    h('section.how-callout',
      h('h3', 'Why the result looks real'),
      h('p', 'A new colour is applied to the albedo layer only, then multiplied back with the original shading and topped with the original highlights. Nothing about the lighting is invented, so reflections, shadows, grain and panel gaps all stay exactly where the camera saw them.'),
      h('p', h('strong', 'Formula: '), h('code.mono', 'out = sRGB( albedo′ · shading + residual )'))));
  replace(root, view);
  enterChildren(view, { step: 80 });

  // Light the stages in sequence, forever (unless reduced motion).
  let timer = null;
  let idx = 0;
  const step = () => {
    nodes.forEach((n, i) => { n.classList.toggle('is-lit', i === idx); n.classList.toggle('is-done', i < idx); });
    idx = (idx + 1) % (STAGES.length + 1);
  };
  if (!reducedMotion()) { step(); timer = setInterval(step, 900); } else nodes.forEach((n) => n.classList.add('is-lit'));

  return { destroy() { clearInterval(timer); } };
}
