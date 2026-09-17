#!/usr/bin/env python3
"""Render one analyzed job to a row of dark targets and write a comparison sheet.

    .venv/bin/python scripts/dark_sheet.py data/jobs/<id> [--out scratch/dark_<name>.jpg]

Each column is the same part repainted to a different target, so a colour cast that
survives the repaint is obvious side by side with the original.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recolor import imageio
from recolor.engine import Renderer
from recolor.types import ColorGroup, RenderOptions

TARGETS = ["#000000", "#141414", "#2c3539", "#1a2a5a", "#004225", "#f2f0eb"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--group", type=int, default=None)
    ap.add_argument("--height", type=int, default=520)
    ap.add_argument("--targets", default=",".join(TARGETS))
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.job_dir, "job.json"), encoding="utf-8"))
    groups = [ColorGroup.from_dict(g) for g in meta["groups"]]
    alb = imageio.load_f16(os.path.join(args.job_dir, "albedo.npy"))
    shd = imageio.load_f16(os.path.join(args.job_dir, "shading.npy"))
    res = imageio.load_f16(os.path.join(args.job_dir, "residual.npy"))
    gm = np.load(os.path.join(args.job_dir, "group_map.npy")).astype(np.int32)

    if args.group is None:
        cand = [g for g in groups if g.area_frac > 0.05 and not g.is_background]
        gsel = max(cand or groups, key=lambda g: np.hypot(g.albedo_lab[1], g.albedo_lab[2]))
    else:
        gsel = next(g for g in groups if g.id == args.group)

    r = Renderer(alb, shd, res, gm, groups)
    tiles = []

    def tile(img: np.ndarray, label: str) -> np.ndarray:
        s = args.height / img.shape[0]
        im = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (int(img.shape[1] * s), args.height),
                        interpolation=cv2.INTER_AREA)
        pad = np.full((args.height + 26, im.shape[1], 3), 24, np.uint8)
        pad[:args.height] = im
        cv2.putText(pad, label, (6, args.height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return pad

    tiles.append(tile(r.render({}, RenderOptions()), f"original · {gsel.name} {gsel.albedo_hex}"))
    for hexcol in args.targets.split(","):
        hexcol = hexcol.strip()
        out = r.render({gsel.id: hexcol}, RenderOptions())
        lab = imageio.rgb_to_lab(imageio.to_float(out[gm == gsel.id][None]))[0]
        tiles.append(tile(out, f"{hexcol}  L {lab[..., 0].mean():.0f}  chroma {np.hypot(lab[..., 1], lab[..., 2]).mean():.1f}"))
    r.free()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0)), constant_values=24) for t in tiles]
    per = 4
    rows = [np.hstack(tiles[i:i + per]) for i in range(0, len(tiles), per)]
    wide = max(x.shape[1] for x in rows)
    rows = [np.pad(x, ((0, 0), (0, wide - x.shape[1]), (0, 0)), constant_values=24) for x in rows]
    out_path = args.out or os.path.join("scratch", f"dark_{os.path.splitext(meta['name'])[0]}.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
