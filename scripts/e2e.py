#!/usr/bin/env python3
"""End-to-end drive of a running Chroma Studio server: analyze samples, build a palette,
suggest a mapping, render, export, and write a contact sheet.

    .venv/bin/python scripts/e2e.py --server http://127.0.0.1:8810 --prompt "hawaii sunset" \
        --samples gundam_sazabi_verka_a.jpg gundam_rx78_rg.jpg --detail balanced

Writes scratch/e2e/<sample>_{preview,result,albedo,groups}.jpg and scratch/e2e/sheet.jpg.
Prints per-stage timings so regressions are visible at a glance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "scratch", "e2e")


def call(server: str, method: str, path: str, body=None, raw=False, timeout=600):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(server + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = r.read()
        if raw:
            return payload, dict(r.headers)
        return json.loads(payload) if payload else None


def wait_ready(server: str, jid: str, timeout: float = 900) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        job = call(server, "GET", f"/api/jobs/{jid}")
        if job["status"] in ("ready", "error"):
            return job
        running = [(k, v) for k, v in job["stages"].items() if v["state"] == "running"]
        msg = f"{running[0][0]} {running[0][1]['progress']:.0%} {running[0][1]['message']}" if running else job["status"]
        if msg != last:
            print(f"    {msg}")
            last = msg
        time.sleep(0.5)
    raise TimeoutError(jid)


def save(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8810")
    ap.add_argument("--samples", nargs="+", default=["gundam_sazabi_verka_a.jpg"])
    ap.add_argument("--prompt", default="hawaii sunset")
    ap.add_argument("--n-colors", type=int, default=6)
    ap.add_argument("--detail", default="balanced")
    ap.add_argument("--strategy", default="balanced")
    ap.add_argument("--export", default="work", choices=["work", "full", "none"])
    ap.add_argument("--keep", action="store_true", help="do not delete jobs afterwards")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    health = call(args.server, "GET", "/api/health")
    print("health:", {k: health[k] for k in ("device", "gpu", "models") if k in health})

    t = time.time()
    pal = call(args.server, "POST", "/api/palettes", {"prompt": args.prompt, "n_colors": args.n_colors})
    print(f"palette '{args.prompt}' via {pal['method']} in {time.time() - t:.1f}s:",
          " ".join(f"{c['name']}({c['hex']})" for c in pal["colors"]))
    hexes = [c["hex"] for c in pal["colors"]]

    rows = []
    for name in args.samples:
        stem = os.path.splitext(name)[0]
        print(f"== {name}")
        t0 = time.time()
        job = call(args.server, "POST", "/api/jobs", {"sample": name, "detail": args.detail})
        jid = job["id"]
        job = wait_ready(args.server, jid)
        if job["status"] == "error":
            print("   ERROR:", job.get("error"))
            continue
        stages = {k: round(v["seconds"], 2) for k, v in job["stages"].items()}
        print(f"   analyzed in {time.time() - t0:.1f}s stages={stages} regions={job['regions_count']} "
              f"groups={len(job['groups'])} intrinsic={job['intrinsic_method']}")
        for g in job["groups"][:8]:
            print(f"     g{g['id']:<3} {g['name']:<14} {g['albedo_hex']} {g['area_frac']:6.1%} {len(g['region_ids']):3d} regions"
                  + ("  bg" if g["is_background"] else ""))

        sug = call(args.server, "POST", f"/api/jobs/{jid}/mapping/suggest", {"colors": hexes, "strategy": args.strategy})
        mapping = sug["mapping"]
        t = time.time()
        img, hdr = call(args.server, "POST", f"/api/jobs/{jid}/render", {"mapping": mapping, "options": {}}, raw=True)
        print(f"   render {len(img)//1024} KB in {time.time() - t:.2f}s (server {hdr.get('X-Render-Ms') or hdr.get('x-render-ms')} ms)")
        save(os.path.join(OUT, f"{stem}_result.jpg"), img)
        for layer in ("preview", "albedo", "groups", "regions", "shading"):
            data, _ = call(args.server, "GET", f"/api/jobs/{jid}/layers/{layer}", raw=True)
            save(os.path.join(OUT, f"{stem}_{layer}.jpg" if layer != "regions" and layer != "groups" else f"{stem}_{layer}.png"), data)
        if args.export != "none":
            t = time.time()
            ex = call(args.server, "POST", f"/api/jobs/{jid}/export",
                      {"mapping": mapping, "options": {}, "quality": args.export, "format": "jpg"})
            data, _ = call(args.server, "GET", ex["url"], raw=True)
            save(os.path.join(OUT, f"{stem}_export_{args.export}.jpg"), data)
            print(f"   export {args.export} {ex['width']}x{ex['height']} in {time.time() - t:.1f}s")
        rows.append(stem)
        if not args.keep:
            call(args.server, "DELETE", f"/api/jobs/{jid}")

    if rows:
        import cv2
        import numpy as np
        tiles = []
        for stem in rows:
            a = cv2.imread(os.path.join(OUT, f"{stem}_preview.jpg"))
            b = cv2.imread(os.path.join(OUT, f"{stem}_result.jpg"))
            g = cv2.imread(os.path.join(OUT, f"{stem}_groups.png"))
            if a is None or b is None:
                continue
            h = 420
            def fit(im):
                s = h / im.shape[0]
                return cv2.resize(im, (int(im.shape[1] * s), h), interpolation=cv2.INTER_AREA)
            row = np.hstack([fit(a), fit(g) if g is not None else fit(a), fit(b)])
            cv2.putText(row, stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(row)
        w = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]
        sheet = np.vstack(tiles)
        cv2.imwrite(os.path.join(OUT, "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("sheet:", os.path.join(OUT, "sheet.jpg"), sheet.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
