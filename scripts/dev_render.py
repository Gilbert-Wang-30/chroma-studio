#!/usr/bin/env python
"""Exercise the recoloring engine and write renders to scratch/.

Two modes:

* ``dev_render.py --synthetic samples/car_red_sports_1.jpg``
  No siblings needed. Builds a heuristic intrinsic decomposition (guided-filtered
  illumination -> shading, albedo = lin / shading) and fake color groups (SLIC
  superpixels + k-means on the albedo in Lab), then renders three mappings —
  identity, a flat repaint and a Lab-shift repaint with texture 0.5 — plus a strip
  comparing them side by side.

* ``dev_render.py data/jobs/<id> [--mapping mapping.json|'{"0":"#ff0000"}'] [--options '{...}']``
  Renders an analyzed job directory (layers as written by recolor.pipeline).

Prints per-render timings (native and preview resolution), peak VRAM, and the
identity error against the numpy recomposition. GPU memory is released on exit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from recolor import colornames, config, engine, filters, imageio  # noqa: E402
from recolor.types import ColorGroup, Mapping, RenderOptions, mapping_from_json  # noqa: E402

SCRATCH = os.path.join(ROOT, "scratch")


# ----------------------------------------------------------------- synthetic decomposition

def heuristic_decompose(image_u8: np.ndarray, radius_frac: float = 0.02, eps: float = 1e-3,
                        specular: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cheap intrinsic decomposition for engine validation, not a product feature.

    1. Speculars: sharp, whitish bumps of the per-pixel *min* channel above its heavily
       smoothed version (guided filter, large eps) are removed from the image and kept
       as the positive residual, so ``residual_tint`` has something to act on.
    2. Illumination is the per-pixel max channel of what is left, guided-filtered by
       itself with a small ``eps`` so reflections and shading gradients stay in the
       shading layer while the albedo becomes flat paint (max-RGB assumption: every
       surface's brightest channel is close to 1, so a red car keeps its full chroma).
    3. ``albedo = clip(lin / shading, 0, 1)``, ``residual += lin - albedo * shading``
       so ``albedo * shading + residual == lin`` exactly. Returns float32 linear layers.
    """
    lin = imageio.srgb_to_linear(imageio.to_float(image_u8))
    radius = max(4, int(round(max(lin.shape[:2]) * radius_frac)))
    spec = np.zeros_like(lin)
    if specular:
        minc = lin.min(axis=2)
        base = filters.guided_filter(minc, minc, radius=radius, eps=0.05)
        bump = np.clip(minc - base - 0.02, 0.0, None)
        spec = np.repeat(bump[..., None], 3, axis=2).astype(np.float32)
    diffuse = np.clip(lin - spec, 0.0, 1.0)
    maxc = diffuse.max(axis=2)
    illum = filters.guided_filter(maxc, maxc, radius=radius, eps=eps)
    illum = np.clip(illum, 0.02, None).astype(np.float32)
    # Keep the albedo out of saturation for most pixels: scale so the 99.5th
    # percentile of diffuse/illum lands at 1.
    ratio = np.percentile(maxc / illum, 99.5)
    illum = illum * float(max(ratio, 1e-3))
    shading = np.repeat(illum[..., None], 3, axis=2).astype(np.float32)
    albedo = np.clip(diffuse / shading, 0.0, 1.0).astype(np.float32)
    residual = (lin - albedo * shading).astype(np.float32)
    return albedo, shading, residual


def _merge_close_clusters(labels: np.ndarray, lab: np.ndarray, weight: np.ndarray, max_groups: int,
                          delta_e_thresh: float) -> np.ndarray:
    """Agglomerate cluster labels whose area-weighted mean Lab centroids are within
    ``delta_e_thresh`` (CIEDE2000), and keep merging the closest pair until at most
    ``max_groups`` remain. Returns labels renumbered 0..K-1."""
    labels = labels.copy()
    while True:
        ids = np.unique(labels)
        if len(ids) <= 1:
            break
        cents = np.stack([np.average(lab[labels == i], axis=0, weights=weight[labels == i]) for i in ids])
        best, pair = None, None
        for i in range(len(ids)):
            d = imageio.delta_e(np.repeat(cents[i][None], len(ids) - i - 1, 0), cents[i + 1:])
            if d.size and (best is None or d.min() < best):
                best, pair = float(d.min()), (ids[i], ids[i + 1 + int(np.argmin(d))])
        if best is None or (best >= delta_e_thresh and len(ids) <= max_groups):
            break
        labels[labels == pair[1]] = pair[0]
    _, out = np.unique(labels, return_inverse=True)
    return out.astype(np.int32)


def fake_groups(image_u8: np.ndarray, albedo_lin: np.ndarray, n_groups: int = 7,
                seed: int = 0, merge_delta_e: float = 12.0) -> tuple[np.ndarray, list[ColorGroup]]:
    """SLIC superpixels + k-means on the superpixels' median albedo (Lab), with close
    clusters merged (CIEDE2000 < ``merge_delta_e``) -> a group map and ColorGroup
    records shaped like the segmentation stage's output."""
    from scipy import ndimage
    from skimage.segmentation import slic
    from sklearn.cluster import KMeans

    h, w = albedo_lin.shape[:2]
    n_seg = max(50, (h * w) // 900)
    sp = slic(image_u8, n_segments=n_seg, compactness=12.0, start_label=0).astype(np.int32)
    n_sp = int(sp.max()) + 1
    lab = imageio.linear_to_lab(albedo_lin)
    idx = np.arange(n_sp)
    med = np.stack([ndimage.median(lab[..., c], sp, idx) for c in range(3)], axis=1).astype(np.float32)
    area = np.bincount(sp.ravel(), minlength=n_sp).astype(np.float64)
    feats = med.copy()
    feats[:, 0] *= 0.6                       # lightness matters less than hue/chroma for grouping
    k0 = min(n_groups + 4, n_sp)
    km = KMeans(n_clusters=k0, n_init=8, random_state=seed).fit(feats, sample_weight=area)
    sp_group = km.labels_.astype(np.int32)
    sp_group = _merge_close_clusters(sp_group, med, area, n_groups, merge_delta_e)
    k = int(sp_group.max()) + 1
    group_map = sp_group[sp]

    # ids by area descending
    g_area = np.bincount(group_map.ravel(), minlength=k)
    order = np.argsort(-g_area)
    remap = np.empty(k, np.int32)
    remap[order] = np.arange(k, dtype=np.int32)
    group_map = remap[group_map].astype(np.int32)
    sp_group = remap[sp_group]

    border = np.zeros((h, w), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border_ids = group_map[border]
    border_frac = np.bincount(border_ids, minlength=k) / max(1, border_ids.size)
    bg = int(np.argmax(border_frac)) if border_frac.max() > 0.35 else -1

    groups: list[ColorGroup] = []
    for gid in range(k):
        sel = group_map == gid
        px = lab[sel]
        if px.shape[0] > 200_000:
            px = px[:: px.shape[0] // 200_000 + 1]
        g_lab = tuple(float(v) for v in np.median(px, axis=0))
        groups.append(ColorGroup(
            id=gid, name=colornames.nearest_name(g_lab), albedo_lab=g_lab,
            albedo_hex=imageio.lab_to_hex(g_lab), area=int(sel.sum()),
            area_frac=float(sel.mean()), region_ids=[int(i) for i in np.nonzero(sp_group == gid)[0]],
            hue_family=colornames.hue_family(g_lab), locked=False, is_background=(gid == bg),
        ))
    return group_map, groups


def pick_hero(groups: list[ColorGroup]) -> ColorGroup:
    """The group a demo repaint should target: the largest non-background group with
    noticeable chroma, else the largest non-background group."""
    cands = [g for g in groups if not g.is_background] or list(groups)
    chromatic = [g for g in cands if math.hypot(g.albedo_lab[1], g.albedo_lab[2]) > 12.0]
    pool = chromatic or cands
    return max(pool, key=lambda g: g.area)


# ----------------------------------------------------------------- job loading

def load_job(job_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[ColorGroup], Optional[np.ndarray]]:
    with open(os.path.join(job_dir, "job.json")) as f:
        meta = json.load(f)
    albedo = imageio.load_f16(os.path.join(job_dir, "albedo.npy"))
    shading = imageio.load_f16(os.path.join(job_dir, "shading.npy"))
    residual = imageio.load_f16(os.path.join(job_dir, "residual.npy"))
    group_map = np.load(os.path.join(job_dir, "group_map.npy")).astype(np.int32)
    groups = [ColorGroup.from_dict(g) for g in meta.get("groups", [])]
    work_path = os.path.join(job_dir, "work.png")
    work = imageio.load_image(work_path) if os.path.exists(work_path) else None
    return albedo, shading, residual, group_map, groups, work


def parse_mapping(arg: Optional[str]) -> Mapping:
    if not arg:
        return {}
    if os.path.exists(arg):
        with open(arg) as f:
            return mapping_from_json(json.load(f))
    return mapping_from_json(json.loads(arg))


# ----------------------------------------------------------------- output helpers

def labeled(img: np.ndarray, text: str, long_side: int = 640) -> np.ndarray:
    im = imageio.resize_long_side(img, long_side).copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (16, 16, 16), -1)
    cv2.putText(im, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return im


def strip(panels: list[tuple[str, np.ndarray]], long_side: int = 640) -> np.ndarray:
    tiles = [labeled(im, t, long_side) for t, im in panels]
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        canvas = np.full((h, w, 3), 24, np.uint8)
        canvas[: t.shape[0], : t.shape[1]] = t
        padded.append(canvas)
    return np.concatenate(padded, axis=1)


def group_overlay(group_map: np.ndarray, groups: list[ColorGroup]) -> np.ndarray:
    lut = np.zeros((max(int(group_map.max()) + 1, len(groups)), 3), np.uint8)
    for g in groups:
        lut[g.id] = (imageio.hex_to_rgb01(g.albedo_hex) * 255 + 0.5).astype(np.uint8)
    return lut[group_map]


def vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def timed_renders(r: engine.Renderer, mapping: Mapping, options: RenderOptions, label: str,
                  preview_side: int) -> np.ndarray:
    out = r.render(mapping, options)
    native_ms = r.last_render_ms
    r.render_at(preview_side, mapping, options)              # warm the preview level
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        r.render_at(preview_side, mapping, options)
        t.append((time.perf_counter() - t0) * 1000)
    print(f"  {label:<10s} native {out.shape[1]}x{out.shape[0]}: {native_ms:6.1f} ms | "
          f"preview {preview_side}: {min(t):5.1f} ms (min of 5, incl. GPU->CPU copy)")
    return out


# ----------------------------------------------------------------- main

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="sample image (with --synthetic) or an analyzed job directory")
    ap.add_argument("--synthetic", action="store_true", help="heuristic intrinsic + SLIC/k-means groups on a sample image")
    ap.add_argument("--groups", type=int, default=7, help="number of fake groups (synthetic)")
    ap.add_argument("--hero", type=int, default=None, help="group id to repaint (synthetic; default: auto)")
    ap.add_argument("--flat-color", default="#1f5fd6", help="target for the flat repaint (synthetic)")
    ap.add_argument("--shift-color", default="#f2b705", help="target for the Lab-shift repaint (synthetic)")
    ap.add_argument("--texture", type=float, default=0.5, help="texture for the shift render (synthetic)")
    ap.add_argument("--tint", type=float, default=0.0, help="residual_tint for the shift render")
    ap.add_argument("--shading-strength", type=float, default=1.0, help="shading_strength for the shift render")
    ap.add_argument("--feather", type=float, default=1.5)
    ap.add_argument("--mapping", default=None, help="job mode: mapping JSON file or inline JSON")
    ap.add_argument("--options", default=None, help="job mode: RenderOptions JSON (inline)")
    ap.add_argument("--long-side", type=int, default=config.WORK_LONG_SIDE, help="resize the source image (synthetic)")
    ap.add_argument("--preview-side", type=int, default=config.PREVIEW_LONG_SIDE)
    ap.add_argument("--out-dir", default=SCRATCH)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(f"device: {config.device()}")

    if args.synthetic:
        stem = os.path.splitext(os.path.basename(args.source))[0]
        image = imageio.load_image(args.source, args.long_side)
        print(f"image {stem}: {image.shape[1]}x{image.shape[0]}")
        t0 = time.perf_counter()
        albedo, shading, residual = heuristic_decompose(image)
        print(f"heuristic decomposition: {(time.perf_counter() - t0) * 1000:.0f} ms, "
              f"albedo mean {albedo.mean():.3f}, shading median {np.median(shading):.3f}, "
              f"|residual| mean {np.abs(residual).mean():.4f}")
        t0 = time.perf_counter()
        group_map, groups = fake_groups(image, albedo, args.groups)
        print(f"fake grouping: {(time.perf_counter() - t0) * 1000:.0f} ms, {len(groups)} groups")
        for g in groups:
            print(f"  [{g.id}] {g.name:<12s} {g.albedo_hex} area {g.area_frac * 100:5.1f}% "
                  f"{g.hue_family:<8s}{' background' if g.is_background else ''}")
        hero = groups[args.hero] if args.hero is not None else pick_hero(groups)
        print(f"hero group: [{hero.id}] {hero.name} {hero.albedo_hex} -> flat {args.flat_color}, shift {args.shift_color}")
        imageio.save_image(os.path.join(args.out_dir, f"render_{stem}_groups.png"), group_overlay(group_map, groups))
        imageio.save_image(os.path.join(args.out_dir, f"render_{stem}_albedo.png"),
                           imageio.to_uint8(imageio.linear_to_srgb(albedo)))
        imageio.save_image(os.path.join(args.out_dir, f"render_{stem}_shading.png"),
                           imageio.to_uint8(imageio.linear_to_srgb(shading / np.percentile(shading, 99.5))))
        imageio.save_image(os.path.join(args.out_dir, f"render_{stem}_residual.png"),
                           imageio.to_uint8(imageio.linear_to_srgb(np.abs(residual) * 4.0)))

        r = engine.Renderer(albedo, shading, residual, group_map, groups)
        base = RenderOptions(feather_px=args.feather)
        renders = {
            "identity": ({}, base),
            "flat": ({hero.id: args.flat_color}, RenderOptions(mode="flat", feather_px=args.feather)),
            "shift": ({hero.id: args.shift_color},
                      RenderOptions(mode="shift", texture=args.texture, feather_px=args.feather,
                                    residual_tint=args.tint, shading_strength=args.shading_strength)),
        }
        outputs: dict[str, np.ndarray] = {}
        print("renders:")
        for name, (mapping, opts) in renders.items():
            out = timed_renders(r, mapping, opts, name, args.preview_side)
            outputs[name] = out
            imageio.save_image(os.path.join(args.out_dir, f"render_{stem}_{name}.png"), out)
        ref = engine.recompose(albedo, shading, residual)
        err = np.abs(outputs["identity"].astype(np.int16) - ref.astype(np.int16)).max()
        err_img = np.abs(outputs["identity"].astype(np.int16) - image.astype(np.int16)).max()
        print(f"identity vs numpy recomposition: max abs error {err}/255 ; vs input image {err_img}/255")
        panels = [("original", image), ("identity", outputs["identity"]),
                  (f"flat {args.flat_color}", outputs["flat"]),
                  (f"shift {args.shift_color} tex {args.texture:.2f}", outputs["shift"])]
        strip_path = os.path.join(args.out_dir, f"render_{stem}_strip.png")
        imageio.save_image(strip_path, strip(panels))
        print(f"wrote {args.out_dir}/render_{stem}_{{identity,flat,shift,strip,groups,albedo,shading,residual}}.png")
        r.free()
    else:
        job_dir = args.source
        job_id = os.path.basename(os.path.normpath(job_dir))
        albedo, shading, residual, group_map, groups, work = load_job(job_dir)
        print(f"job {job_id}: {albedo.shape[1]}x{albedo.shape[0]}, {len(groups)} groups")
        mapping = parse_mapping(args.mapping)
        options = RenderOptions.from_dict(json.loads(args.options)) if args.options else RenderOptions(feather_px=args.feather)
        print(f"mapping: {mapping}\noptions: {options.to_dict()}")
        r = engine.Renderer(albedo, shading, residual, group_map, groups)
        print("renders:")
        identity = timed_renders(r, {}, RenderOptions(), "identity", args.preview_side)
        out = timed_renders(r, mapping, options, "mapping", args.preview_side)
        ref = engine.recompose(albedo, shading, residual)
        print(f"identity vs numpy recomposition: max abs error "
              f"{np.abs(identity.astype(np.int16) - ref.astype(np.int16)).max()}/255")
        imageio.save_image(os.path.join(args.out_dir, f"render_{job_id}.png"), out)
        panels = [("original", work if work is not None else identity), ("recolored", out)]
        imageio.save_image(os.path.join(args.out_dir, f"render_{job_id}_strip.png"), strip(panels))
        print(f"wrote {args.out_dir}/render_{job_id}.png and {args.out_dir}/render_{job_id}_strip.png")
        r.free()

    if torch.cuda.is_available():
        print(f"peak VRAM: {vram_mb():.0f} MB")
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
