"""Model-backed smoke test for the intrinsic stage (run manually, not under pytest).

    .venv/bin/python scripts/dev_intrinsic.py samples/motorcycle_1.jpg [--method auto|careaga|heuristic]
                                              [--long-side 1536] [--out scratch]

Writes ``<out>/<name>_<method>_{albedo,shading,residual,recomposed}.png`` plus a
side-by-side ``_layers.jpg`` contact sheet (2400 px wide, JPEG q90), prints timings,
VRAM and the reconstruction error.
With ``--method auto`` (default) both the model and the heuristic are run so the two
can be compared on the same image.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from recolor import config, imageio, intrinsic  # noqa: E402
from recolor.intrinsic import careaga           # noqa: E402


def _vram_mb() -> tuple[float, float]:
    import torch
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 2**20, torch.cuda.max_memory_allocated() / 2**20


def _sheet(image: np.ndarray, layers: dict[str, np.ndarray], recomposed: np.ndarray) -> np.ndarray:
    """2x3 contact sheet: original, albedo, shading / residual, recomposed, |diff|x8."""
    diff = imageio.to_uint8(np.abs(image.astype(np.float32) - recomposed.astype(np.float32)) * 8 / 255.0)
    top = np.concatenate([image, layers["albedo"], layers["shading"]], axis=1)
    bottom = np.concatenate([layers["residual"], recomposed, diff], axis=1)
    return np.concatenate([top, bottom], axis=0)


def run_one(image: np.ndarray, method: str, name: str, out_dir: str) -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    messages: list[str] = []
    t0 = time.perf_counter()
    res = intrinsic.decompose(image, method=method, progress=lambda p, m: messages.append(f"{p:.2f} {m}"))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_dec = time.perf_counter() - t0

    t1 = time.perf_counter()
    layers = intrinsic.layers_for_display(res)
    recomposed = intrinsic.recompose(res)
    t_disp = time.perf_counter() - t1

    err_lin = intrinsic.reconstruction_error(image, res)
    err_u8 = int(np.abs(image.astype(np.int16) - recomposed.astype(np.int16)).max())
    cur, peak = _vram_mb()

    tag = f"{name}_{res.method}"
    for key, arr in layers.items():
        imageio.save_image(os.path.join(out_dir, f"{tag}_{key}.png"), arr)
    imageio.save_image(os.path.join(out_dir, f"{tag}_recomposed.png"), recomposed)
    sheet = _sheet(image, layers, recomposed)
    imageio.save_image(os.path.join(out_dir, f"{tag}_layers.jpg"), imageio.resize_long_side(sheet, 2400), quality=90)

    print(f"[{res.method:9s}] requested={method:9s} {image.shape[1]}x{image.shape[0]}  "
          f"decompose {t_dec*1000:7.1f} ms  display+recompose {t_disp*1000:6.1f} ms")
    print(f"             albedo [{res.albedo.min():.3f}, {res.albedo.max():.3f}]  "
          f"shading [{res.shading.min():.3f}, {res.shading.max():.3f}] p99.5={np.quantile(res.shading, 0.995):.3f}  "
          f"residual [{res.residual.min():+.3f}, {res.residual.max():+.3f}]")
    print(f"             reconstruction: max |lin err| = {err_lin:.2e}   max |uint8 err| = {err_u8}")
    print(f"             VRAM now {cur:.0f} MB, peak {peak:.0f} MB")
    for m in messages:
        print(f"             progress {m}")
    print(f"             wrote {out_dir}/{tag}_{{albedo,shading,residual,recomposed}}.png and _layers.jpg")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="path to a JPEG/PNG (e.g. samples/motorcycle_1.jpg)")
    ap.add_argument("--method", default="auto", choices=list(intrinsic.METHODS))
    ap.add_argument("--long-side", type=int, default=config.WORK_LONG_SIDE,
                    help="resize the image to this long side first (0 = keep original)")
    ap.add_argument("--out", default=os.path.join(config.ROOT, "scratch"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    image = imageio.load_image(args.image)
    if args.long_side > 0:
        image = imageio.resize_long_side(image, args.long_side)
    name = os.path.splitext(os.path.basename(args.image))[0]
    print(f"{args.image}: {image.shape[1]}x{image.shape[0]}  device={config.device()}")

    methods = ["careaga", "heuristic"] if args.method == "auto" else [args.method]
    if "careaga" in methods:
        if not careaga.is_available():
            print("Careaga model not available on this machine; running the heuristic only")
            methods = ["heuristic"]
        else:
            t0 = time.perf_counter()
            intrinsic.warmup("careaga")
            print(f"warmup (load + tiny forward): {time.perf_counter() - t0:.1f} s, "
                  f"resident VRAM {_vram_mb()[0]:.0f} MB")
    for method in methods:
        run_one(image, method, name, args.out)
        if method == "heuristic":
            # Second call shows the steady-state time (first call pays kernel warm-up).
            t0 = time.perf_counter()
            intrinsic.decompose(image, method="heuristic")
            print(f"             heuristic steady-state: {(time.perf_counter() - t0)*1000:.1f} ms")

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
