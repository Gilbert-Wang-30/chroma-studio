#!/usr/bin/env python
"""Run the segmentation stage on a sample image and write inspection PNGs.

    .venv/bin/python scripts/dev_segment.py samples/street_complex_1.jpg --detail max
    .venv/bin/python scripts/dev_segment.py samples/motorcycle_1.jpg --detail all

Writes to scratch/ (or --out):
    <name>_regions.png   random color per region, blended over the image, with edges
    <name>_groups.png    flat group albedo colors with group boundaries
    <name>_edges.png     region boundaries drawn over the image
With `--detail all` the files carry a `_<detail>` suffix and every preset is timed.

Albedo comes from `recolor.intrinsic.decompose` when that module imports and runs;
otherwise (or with --albedo heuristic) a guided-filter luminance normalisation stands in.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from recolor import config, filters, imageio  # noqa: E402
from recolor.segmentation import SamMasker, build_regions, group_regions  # noqa: E402
from recolor.segmentation.labelops import adjacency  # noqa: E402

DETAILS = ("fast", "balanced", "max")


# ---------------------------------------------------------------------- albedo

def heuristic_albedo(image_rgb_u8: np.ndarray) -> np.ndarray:
    """Edge-aware illumination estimate (guided-filtered luminance at a large radius);
    albedo = linear / shading, clipped to [0, 1]. Only for development when the intrinsic
    module is unavailable."""
    lin = imageio.srgb_to_linear(imageio.to_float(image_rgb_u8))
    lum = imageio.luminance(lin)
    r = max(8, max(lum.shape) // 24)
    sh = filters.guided_filter(lum, lum, radius=r, eps=0.02)
    sh = np.clip(sh, 1e-3, None)
    sh = sh / max(1e-3, float(np.percentile(sh, 99.5)))
    return np.clip(lin / np.clip(sh, 0.05, None)[..., None], 0.0, 1.0).astype(np.float32)


def get_albedo(image_rgb_u8: np.ndarray, how: str) -> tuple[np.ndarray, str]:
    if how != "heuristic":
        try:
            from recolor.intrinsic import decompose
            t0 = time.perf_counter()
            res = decompose(image_rgb_u8, method="auto")
            alb = np.clip(np.asarray(res.albedo, np.float32), 0.0, 1.0)
            if alb.shape[:2] != image_rgb_u8.shape[:2]:
                raise RuntimeError(f"intrinsic returned {alb.shape[:2]} for {image_rgb_u8.shape[:2]}")
            return alb, f"intrinsic:{res.method} ({time.perf_counter() - t0:.1f} s)"
        except Exception as e:  # sibling module missing or broken: fall back, say so
            print(f"[albedo] intrinsic module unavailable ({type(e).__name__}: {e}); using heuristic")
    t0 = time.perf_counter()
    return heuristic_albedo(image_rgb_u8), f"heuristic guided-filter ({time.perf_counter() - t0:.2f} s)"


def release_intrinsic() -> str:
    """Drop the intrinsic model from the GPU if the sibling module offers a way to.

    `recolor.intrinsic` keeps the Careaga stages in a lazy singleton with no public
    unload helper at the time of writing, so the returned note says whether the
    model is still resident (the memory is freed when the process exits either way).
    """
    try:
        from recolor.intrinsic import careaga
    except Exception:  # module missing or broken: nothing to release
        return "intrinsic model not loaded"
    if not careaga.is_loaded():
        return "intrinsic model not loaded"
    for name in ("release", "unload"):
        fn = getattr(careaga, name, None)
        if callable(fn):
            try:
                fn()
                return "intrinsic model released"
            except Exception as e:
                return f"intrinsic release failed: {type(e).__name__}"
    return "intrinsic model still resident; SAM released"


# ---------------------------------------------------------------------- drawing

def boundaries(labels: np.ndarray) -> np.ndarray:
    b = np.zeros(labels.shape, bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    return b


def draw_regions(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(labels.max()) + 1
    colors = rng.integers(40, 255, size=(n, 3), dtype=np.uint8)
    gray = imageio.luminance(imageio.to_float(image))[..., None]
    out = colors[labels].astype(np.float32) * (0.35 + 0.65 * gray)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[boundaries(labels)] = 0
    return out


def draw_groups(image: np.ndarray, group_map: np.ndarray, groups) -> np.ndarray:
    lut = np.array([imageio.hex_to_rgb01(g.albedo_hex) for g in groups], np.float32)
    flat = imageio.to_uint8(lut[group_map])
    flat[boundaries(group_map)] = 255
    return flat


def draw_edges(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = image.copy()
    b = boundaries(labels)
    out[b] = (0.25 * out[b] + np.array([190, 60, 255]) * 0.75).astype(np.uint8)
    return out


# ---------------------------------------------------------------------- main

def run_one(image: np.ndarray, albedo: np.ndarray, detail: str, args, stem: str, suffix: str) -> dict:
    masker = SamMasker.instance()
    if not masker.is_loaded():
        t0 = time.perf_counter()
        masker.load()
        print(f"  SAM 2.1 loaded in {time.perf_counter() - t0:.1f} s")
    torch = sys.modules["torch"]
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    masks = masker.generate(image, detail=detail)
    t_sam = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels, info = build_regions(image, albedo, masks, detail=detail)
    t_reg = time.perf_counter() - t0
    t0 = time.perf_counter()
    regions, groups, group_map = group_regions(labels, albedo, info, args.max_groups, args.delta_e)
    t_grp = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    sources = {s: sum(1 for d in info if d["source"] == s) for s in ("sam", "split", "superpixel")}
    areas = np.array([d["area"] for d in info])
    print(f"  [{detail:8s}] SAM {t_sam:6.2f} s ({len(masks):4d} proposals) | regions {t_reg:5.2f} s "
          f"({len(info):4d}: {sources}) | groups {t_grp:5.2f} s ({len(groups):3d}) | "
          f"total {t_sam + t_reg + t_grp:6.2f} s | peak VRAM {vram:.0f} MB")
    print(f"             region area px: min {areas.min()} median {int(np.median(areas))} max {areas.max()}; "
          f"adjacent pairs {len(adjacency(labels, len(info))[0])}")
    for g in groups[:12]:
        flag = " bg" if g.is_background else ""
        print(f"             g{g.id:<3d} {g.name:<14s} {g.albedo_hex} {g.hue_family:<8s} "
              f"{100 * g.area_frac:5.1f}%  {len(g.region_ids):4d} regions{flag}")
    if len(groups) > 12:
        print(f"             ... {len(groups) - 12} more groups")
    out_dir = args.out
    imageio.save_image(os.path.join(out_dir, f"{stem}{suffix}_regions.png"), draw_regions(image, labels))
    imageio.save_image(os.path.join(out_dir, f"{stem}{suffix}_groups.png"), draw_groups(image, group_map, groups))
    imageio.save_image(os.path.join(out_dir, f"{stem}{suffix}_edges.png"), draw_edges(image, labels))
    return {"detail": detail, "sam_s": t_sam, "regions_s": t_reg, "groups_s": t_grp,
            "proposals": len(masks), "regions": len(info), "groups": len(groups), "vram_mb": vram}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="path to an image (e.g. samples/street_complex_1.jpg)")
    ap.add_argument("--detail", default="balanced", choices=DETAILS + ("all",))
    ap.add_argument("--albedo", default="auto", choices=("auto", "heuristic"))
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--delta-e", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(config.ROOT, "scratch"))
    args = ap.parse_args()

    import torch
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    image = imageio.resize_long_side(imageio.load_image(args.image), config.WORK_LONG_SIDE)
    print(f"{args.image}: working size {image.shape[1]}x{image.shape[0]}")
    albedo, how = get_albedo(image, args.albedo)
    print(f"  albedo: {how}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    details = DETAILS if args.detail == "all" else (args.detail,)
    results = []
    try:
        for d in details:
            suffix = f"_{d}" if args.detail == "all" else ""
            results.append(run_one(image, albedo, d, args, stem, suffix))
    finally:
        SamMasker.instance().release()
        intrinsic_note = release_intrinsic()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  CUDA memory after release: {torch.cuda.memory_allocated() / 2**20:.0f} MB allocated"
                  f" ({intrinsic_note})")
    print("summary:")
    for r in results:
        print(f"  {r['detail']:8s} sam {r['sam_s']:5.1f}s regions {r['regions_s']:4.1f}s groups {r['groups_s']:4.1f}s"
              f" | {r['proposals']} proposals -> {r['regions']} regions -> {r['groups']} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
