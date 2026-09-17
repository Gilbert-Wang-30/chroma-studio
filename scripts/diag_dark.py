#!/usr/bin/env python3
"""Diagnose what a bright-to-dark repaint actually produces, term by term.

For one analyzed job it takes the most saturated large group, repaints it to a set of
targets (black first), and reports the group's mean result colour together with the two
terms that make it: albedo' * shading, and the residual. Use it to see which term is
keeping the original hue alive when the target is black.

    .venv/bin/python scripts/diag_dark.py data/jobs/<id> [--targets '#000000,#101010']
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recolor import imageio
from recolor.engine import Renderer
from recolor.types import ColorGroup, RenderOptions


def load_job(job_dir: str):
    meta = json.load(open(os.path.join(job_dir, "job.json"), encoding="utf-8"))
    groups = [ColorGroup.from_dict(g) for g in meta["groups"]]
    alb = imageio.load_f16(os.path.join(job_dir, "albedo.npy"))
    shd = imageio.load_f16(os.path.join(job_dir, "shading.npy"))
    res = imageio.load_f16(os.path.join(job_dir, "residual.npy"))
    gm = np.load(os.path.join(job_dir, "group_map.npy")).astype(np.int32)
    return meta, groups, alb, shd, res, gm


def chroma(lab: np.ndarray) -> float:
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--targets", default="#000000,#141414,#1a2a5a,#2f7a44")
    ap.add_argument("--group", type=int, default=None, help="group id (default: most saturated big group)")
    args = ap.parse_args()

    meta, groups, alb, shd, res, gm = load_job(args.job_dir)
    print(f"{meta['name']}  {alb.shape[1]}x{alb.shape[0]}  {len(groups)} groups")

    if args.group is None:
        cand = [g for g in groups if g.area_frac > 0.05 and not g.is_background]
        gsel = max(cand or groups, key=lambda g: np.hypot(g.albedo_lab[1], g.albedo_lab[2]))
    else:
        gsel = next(g for g in groups if g.id == args.group)
    sel = gm == gsel.id
    n = int(sel.sum())
    print(f"group {gsel.id} {gsel.name} {gsel.albedo_hex} lab={tuple(round(v,1) for v in gsel.albedo_lab)} "
          f"area={gsel.area_frac:.1%} pixels={n}")

    # What the source layers look like inside that group.
    a_lab = imageio.linear_to_lab(alb[sel][None])[0]
    s_lab = imageio.linear_to_lab(np.clip(shd[sel] / max(np.percentile(shd[sel], 99.5), 1e-6), 0, 1)[None])[0]
    pos = np.clip(res[sel], 0, None)
    print(f"  albedo    mean chroma {chroma(a_lab):5.1f}  mean L {a_lab[...,0].mean():5.1f}")
    print(f"  shading   mean chroma {chroma(s_lab):5.1f} (normalized)  "
          f"channel means {shd[sel].mean(axis=0).round(3).tolist()}")
    print(f"  residual+ mean {pos.mean(axis=0).round(4).tolist()}  "
          f"share of linear image {pos.mean() / max((alb[sel]*shd[sel]).mean() + pos.mean(), 1e-6):.1%}")
    prod = alb[sel] * shd[sel]
    print(f"  product   mean {prod.mean(axis=0).round(4).tolist()}")

    r = Renderer(alb, shd, res, gm, groups)
    for hexcol in args.targets.split(","):
        hexcol = hexcol.strip()
        for keep in (True, False):
            out = r.render({gsel.id: hexcol}, RenderOptions(keep_residual=keep))
            lin = imageio.srgb_to_linear(imageio.to_float(out))
            lab = imageio.rgb_to_lab(imageio.to_float(out[sel][None]))[0]
            want = imageio.hex_to_lab(hexcol)
            tag = "with residual" if keep else "no residual  "
            print(f"  -> {hexcol} {tag}: mean sRGB {out[sel].mean(axis=0).round(1).tolist()} "
                  f"L {lab[...,0].mean():5.1f} (target {want[0]:4.1f})  chroma {chroma(lab):5.1f} "
                  f"(target {np.hypot(want[1], want[2]):4.1f})  hexish {imageio.rgb01_to_hex(imageio.to_float(out[sel].mean(axis=0)))}")
    r.free()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
