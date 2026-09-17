"""Image loading, resizing, color-space conversion and encoding.

Conventions (see CLAUDE.md): arrays are RGB, never BGR. Display images are uint8
HxWx3; intrinsic quantities are float32 linear HxWx3. The sRGB <-> linear transform is
a plain 2.2 gamma to match the Intrinsic pipeline exactly (it linearizes with 2.2).
"""
from __future__ import annotations

import io
import os
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage import color as skcolor

from . import config

GAMMA = 2.2


# ------------------------------------------------------------------ load / save

def load_image(src: Union[str, bytes, bytearray], max_long_side: int | None = None) -> np.ndarray:
    """Load an image as uint8 RGB with EXIF orientation applied. Alpha is composited on
    white. Images longer than `max_long_side` (default config.MAX_INGEST_LONG_SIDE) are
    downscaled with area interpolation."""
    if isinstance(src, (bytes, bytearray)):
        im = Image.open(io.BytesIO(bytes(src)))
    else:
        im = Image.open(src)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.uint8).copy()
    limit = max_long_side or config.MAX_INGEST_LONG_SIDE
    if max(arr.shape[:2]) > limit:
        arr = resize_long_side(arr, limit)
    return arr


def save_image(path: str, arr: np.ndarray, quality: int = 95) -> None:
    """Write uint8 RGB to disk; format from the extension."""
    ext = os.path.splitext(path)[1].lower()
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(path, bgr)


def encode_png(arr: np.ndarray) -> bytes:
    """uint8 RGB (or single-channel / RGBA) -> PNG bytes."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def encode_jpeg(arr: np.ndarray, quality: int = 92) -> bytes:
    """uint8 RGB -> JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def save_f16(path: str, arr: np.ndarray) -> None:
    """Store a float layer compactly (float16 .npy)."""
    np.save(path, arr.astype(np.float16))


def load_f16(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


# ------------------------------------------------------------------ resizing

def fit_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """(w, h) scaled so max(w, h) == long_side; never upscales."""
    s = long_side / float(max(width, height))
    if s >= 1.0:
        return width, height
    return max(1, round(width * s)), max(1, round(height * s))


def resize_to(arr: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    """Resize to (w, h). Area interpolation when shrinking, cubic when enlarging, nearest
    for label maps."""
    w, h = size
    if arr.shape[1] == w and arr.shape[0] == h:
        return arr
    if nearest:
        interp = cv2.INTER_NEAREST
    elif w < arr.shape[1]:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    return cv2.resize(arr, (w, h), interpolation=interp)


def resize_long_side(arr: np.ndarray, long_side: int, nearest: bool = False) -> np.ndarray:
    return resize_to(arr, fit_size(arr.shape[1], arr.shape[0], long_side), nearest=nearest)


# ------------------------------------------------------------------ color math

def to_float(arr_u8: np.ndarray) -> np.ndarray:
    return arr_u8.astype(np.float32) / 255.0


def to_uint8(arr01: np.ndarray) -> np.ndarray:
    return (np.clip(arr01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> linear, plain 2.2 gamma (matches the Intrinsic pipeline)."""
    return np.power(np.clip(x, 0.0, 1.0), GAMMA).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0.0, 1.0), 1.0 / GAMMA).astype(np.float32)


def rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB [0,1] (any leading shape, last dim 3) -> CIE Lab (L 0..100)."""
    return skcolor.rgb2lab(np.clip(rgb01, 0, 1).astype(np.float64)).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE Lab -> sRGB [0,1], clipped."""
    return np.clip(skcolor.lab2rgb(np.asarray(lab, dtype=np.float64)), 0, 1).astype(np.float32)


def linear_to_lab(lin: np.ndarray) -> np.ndarray:
    return rgb_to_lab(linear_to_srgb(lin))


def lab_to_linear(lab: np.ndarray) -> np.ndarray:
    return srgb_to_linear(lab_to_rgb(lab))


def delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """CIEDE2000 distance; broadcasts over leading dims."""
    return skcolor.deltaE_ciede2000(np.asarray(lab_a, np.float64), np.asarray(lab_b, np.float64)).astype(np.float32)


def hex_to_rgb01(h: str) -> np.ndarray:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"bad hex color {h!r}")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def rgb01_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(round(float(v) * 255)) for v in np.clip(rgb, 0, 1))
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_lab(h: str) -> tuple[float, float, float]:
    lab = rgb_to_lab(hex_to_rgb01(h)[None, :])[0]
    return float(lab[0]), float(lab[1]), float(lab[2])


def lab_to_hex(lab) -> str:
    return rgb01_to_hex(lab_to_rgb(np.asarray(lab, np.float32)[None, :])[0])


def luminance(lin: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of a linear RGB image (HxWx3 -> HxW)."""
    return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)
