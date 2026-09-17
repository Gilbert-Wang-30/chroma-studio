#!/usr/bin/env python
"""Generate a palette for one or more prompts and write swatch sheets to scratch/.

Usage:
    .venv/bin/python scripts/dev_palette.py "hawaii sunset" ["cyberpunk" ...] [--n 6]
        [--max-images 6] [--no-cache] [--out scratch]
    .venv/bin/python scripts/dev_palette.py --themes          # sheet of every curated theme

For each prompt the palette is printed (hex, name, Lab, weight, method, sources) and
``scratch/palette_<pid>.png`` is written: a swatch strip on top, source thumbnails below.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recolor import config, imageio  # noqa: E402
from recolor.palette import generate_palette, load_palette, palette_dir  # noqa: E402
from recolor.palette.themes import THEMES  # noqa: E402
from recolor.types import Palette  # noqa: E402

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _text_color(hex_color: str) -> tuple[int, int, int]:
    L = imageio.hex_to_lab(hex_color)[0]
    return (20, 20, 20) if L > 55 else (245, 245, 245)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(round(v * 255)) for v in imageio.hex_to_rgb01(hex_color))


def draw_swatch_strip(hexes: list[str], names: list[str], weights: list[float] | None,
                      width: int = 1200, height: int = 190, title: str = "") -> Image.Image:
    """Swatch strip: equal-width blocks, weight expressed as a bar under each swatch."""
    pad_top = 34 if title else 0
    img = Image.new("RGB", (width, height + pad_top), (250, 250, 248))
    d = ImageDraw.Draw(img)
    if title:
        d.text((12, 8), title, fill=(30, 30, 30), font=_font(18))
    n = max(1, len(hexes))
    w = width / n
    f_big, f_small = _font(15), _font(12)
    for i, hx in enumerate(hexes):
        x0, x1 = int(round(i * w)), int(round((i + 1) * w))
        d.rectangle([x0, pad_top, x1 - 1, pad_top + height - 1], fill=_rgb(hx))
        tc = _text_color(hx)
        d.text((x0 + 10, pad_top + 12), hx, fill=tc, font=f_big)
        d.text((x0 + 10, pad_top + 34), names[i] if i < len(names) else "", fill=tc, font=f_small)
        if weights is not None:
            frac = float(weights[i])
            bar_w = int((x1 - x0 - 20) * min(1.0, frac * n / 2.0))  # 2x the mean weight fills the bar
            y = pad_top + height - 22
            d.rectangle([x0 + 10, y, x0 + 10 + max(bar_w, 2), y + 8], fill=tc)
            d.text((x0 + 10, y - 18), f"{frac * 100:.0f}%", fill=tc, font=f_small)
    return img


def render_palette_sheet(pal: Palette, thumbs: list[np.ndarray]) -> Image.Image:
    """Swatch strip with the source thumbnails tiled underneath."""
    title = f"{pal.prompt}   ·   method={pal.method}   ·   {pal.id[:10]}"
    strip = draw_swatch_strip([c.hex for c in pal.colors], [c.name for c in pal.colors],
                              [c.weight for c in pal.colors], title=title)
    if not thumbs:
        return strip
    per_row = min(6, len(thumbs))
    tw = strip.width // per_row
    th = int(tw * 0.66)
    rows = (len(thumbs) + per_row - 1) // per_row
    sheet = Image.new("RGB", (strip.width, strip.height + rows * th), (250, 250, 248))
    sheet.paste(strip, (0, 0))
    for i, t in enumerate(thumbs):
        im = Image.fromarray(t)
        scale = max(tw / im.width, th / im.height)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        left, top = (im.width - tw) // 2, (im.height - th) // 2
        im = im.crop((left, top, left + tw, top + th))
        sheet.paste(im, ((i % per_row) * tw, strip.height + (i // per_row) * th))
    return sheet


def render_theme_sheet() -> Image.Image:
    strips = [draw_swatch_strip(hexes, [""] * len(hexes), None, width=900, height=70, title=name)
              for name, hexes in THEMES.items()]
    cols = 2
    rows = (len(strips) + cols - 1) // cols
    h = strips[0].height + 6
    sheet = Image.new("RGB", (cols * 910, rows * h), (250, 250, 248))
    for i, s in enumerate(strips):
        sheet.paste(s, ((i % cols) * 910, (i // cols) * h))
    return sheet


def run_prompt(prompt: str, n: int, max_images: int, use_cache: bool, out_dir: str) -> None:
    t0 = time.time()
    msgs: list[str] = []
    pal = generate_palette(prompt, n_colors=n, max_images=max_images, use_cache=use_cache,
                           progress=lambda f, m: msgs.append(f"{f:5.0%} {m}"))
    dt = time.time() - t0
    print(f"\n=== {prompt!r}  ->  {pal.id}  method={pal.method}  {dt:.1f}s")
    for m in msgs:
        print("   ", m)
    for c in pal.colors:
        L, a, b = c.lab
        print(f"  {c.hex}  {c.name:<14s} L={L:5.1f} a={a:6.1f} b={b:6.1f}  w={c.weight:.3f}")
    print(f"  weights sum = {sum(c.weight for c in pal.colors):.4f}")
    for s in pal.sources:
        print(f"  src: {s.title[:60]!r}  [{s.license}]  {s.url}")
    thumbs = []
    for i in range(len(pal.sources)):
        p = os.path.join(palette_dir(pal.id), "sources", f"{i}.jpg")
        if os.path.exists(p):
            thumbs.append(imageio.load_image(p, max_long_side=400))
    sheet = render_palette_sheet(pal, thumbs)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"palette_{pal.id}.png")
    sheet.save(path)
    print(f"  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", nargs="*", help="prompt(s), e.g. 'hawaii sunset'")
    ap.add_argument("--n", type=int, default=6, help="number of colors (default 6)")
    ap.add_argument("--max-images", type=int, default=6, help="reference images to fetch (0 = offline)")
    ap.add_argument("--no-cache", action="store_true", help="ignore the palette cache")
    ap.add_argument("--out", default=os.path.join(config.ROOT, "scratch"), help="output directory")
    ap.add_argument("--themes", action="store_true", help="write a sheet of every curated theme")
    args = ap.parse_args()
    if not args.prompts and not args.themes:
        ap.error("give at least one prompt or --themes")
    config.ensure_dirs()
    os.makedirs(args.out, exist_ok=True)
    if args.themes:
        path = os.path.join(args.out, "palette_themes.png")
        render_theme_sheet().save(path)
        print(f"wrote {path} ({len(THEMES)} themes)")
    for p in args.prompts:
        run_prompt(p, args.n, args.max_images, not args.no_cache, args.out)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
