"""Wrapper around the Careaga & Aksoy "Intrinsic" v2.1 pipeline (colorful diffuse shading).

The five network stages are held in a process-wide lazy singleton loaded through
``intrinsic.pipeline.load_models`` (about 20 s and 1.7 GB of VRAM the first time).
Inference runs under ``torch.inference_mode()`` with the library's warnings silenced.

The model only accepts sizes that are multiples of 32. Instead of letting the library
resample the image to such a size (which would misalign the layers with the input by
a few pixels), the input is reflect-padded here and the outputs are cropped back, so
every layer is pixel-aligned with the image it came from.

Nothing in this module imports the third-party ``intrinsic`` package at import time;
tests exercise the padding / cropping logic by monkeypatching ``run_model``.
"""
from __future__ import annotations

import contextlib
import io
import logging
import threading
import time
import warnings
from typing import Any, Callable

import numpy as np

from .. import config, imageio

log = logging.getLogger(__name__)

_MODELS: dict[str, Any] | None = None
_LOAD_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
_LOAD_ERROR: BaseException | None = None

# Size granularity required by the MiDaS-style networks.
MULTIPLE = 32


def _device() -> str:
    return config.device()


def is_available() -> bool:
    """True when the Careaga pipeline can be used on this machine: the ``intrinsic``
    package imports, a CUDA device exists, and no previous load attempt failed."""
    if _LOAD_ERROR is not None:
        return False
    if _MODELS is not None:
        return True
    if _device() != "cuda":
        return False
    try:
        import importlib.util
        return importlib.util.find_spec("intrinsic") is not None
    except (ImportError, ValueError):
        return False


def is_loaded() -> bool:
    """True once the model stages are resident on the GPU."""
    return _MODELS is not None


def release() -> None:
    """Drop the loaded model stages and free the CUDA memory they held.

    Idempotent, and safe to call when nothing is loaded. A remembered load failure
    (`_LOAD_ERROR`) is left in place, so `is_available()` keeps reporting false rather
    than retrying a broken environment on the next job. The next call to
    `load_models()`/`decompose_careaga()` reloads from disk (~20 s)."""
    global _MODELS
    with _LOAD_LOCK:
        if _MODELS is None:
            return
        _MODELS = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_models() -> dict[str, Any]:
    """Load (once) and return the v2.1 model stages. Thread-safe and idempotent; a
    failed load is remembered so later calls fail fast instead of retrying a 20 s
    download or a broken environment."""
    global _MODELS, _LOAD_ERROR
    if _MODELS is not None:
        return _MODELS
    with _LOAD_LOCK:
        if _MODELS is not None:
            return _MODELS
        if _LOAD_ERROR is not None:
            raise RuntimeError("intrinsic model previously failed to load") from _LOAD_ERROR
        t0 = time.perf_counter()
        try:
            # The library prints "loading v2.1 weights" on stdout and torch.hub writes
            # "Using cache found in ..." on stderr for each backbone; keep server logs clean.
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                from intrinsic.pipeline import load_models as _load
                models = _load(config.INTRINSIC_VERSION, device=_device())
        except BaseException as exc:  # noqa: BLE001 - remembered and re-raised
            _LOAD_ERROR = exc
            log.warning("intrinsic model failed to load: %s", exc)
            raise
        _MODELS = models
        log.info("intrinsic %s loaded in %.1f s", config.INTRINSIC_VERSION, time.perf_counter() - t0)
        return models


def warmup() -> None:
    """Load the model onto the GPU and run one tiny image through it so the first real
    job pays neither the load nor the CUDA kernel-compilation cost. Idempotent."""
    models = load_models()
    dummy = np.full((MULTIPLE * 2, MULTIPLE * 2, 3), 0.5, dtype=np.float32)
    run_model(models, dummy)


def pad_to_multiple(img: np.ndarray, multiple: int = MULTIPLE) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad HxWxC on the bottom/right to a multiple of ``multiple``.

    Returns ``(padded, (H, W))`` with the original size, so the result can be cropped
    back with ``crop_to``. Images already aligned are returned unchanged (no copy).
    Reflect padding is used so the padded border carries plausible image content and
    does not create an artificial edge for the network."""
    h, w = img.shape[:2]
    ph = (-h) % multiple
    pw = (-w) % multiple
    if ph == 0 and pw == 0:
        return img, (h, w)
    # np.pad 'reflect' needs the pad to be smaller than the axis; fall back to 'edge'
    # for tiny images.
    mode = "reflect" if (ph < h and pw < w) else "edge"
    pad = ((0, ph), (0, pw)) + ((0, 0),) * (img.ndim - 2)
    return np.pad(img, pad, mode=mode), (h, w)


def crop_to(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Crop the top-left ``size = (H, W)`` window of an HxW... array as a contiguous
    float32 copy."""
    h, w = size
    return np.ascontiguousarray(arr[:h, :w], dtype=np.float32)


def run_model(models: dict[str, Any], img01: np.ndarray) -> dict[str, np.ndarray]:
    """Run the v2.1 pipeline on an sRGB float image in [0, 1] whose H and W are
    multiples of 32, at that exact size. Returns the library's result dict (keys
    ``hr_alb``, ``dif_shd``, ``residual``, ``lin_img`` among others). Serialized with a
    lock because the GPU is shared with the segmentation stage."""
    import torch
    from intrinsic.pipeline import run_pipeline

    if img01.shape[0] % MULTIPLE or img01.shape[1] % MULTIPLE:
        raise ValueError(f"run_model needs a size that is a multiple of {MULTIPLE}, got {img01.shape[:2]}")
    with _RUN_LOCK, torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return run_pipeline(models, img01, resize_conf=None, device=_device())
        finally:
            torch.cuda.empty_cache()


def decompose_careaga(image_rgb_u8: np.ndarray,
                      progress: Callable[[float, str], None] | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a uint8 RGB image with the Careaga v2.1 model.

    Guarantees:
    - Returns ``(albedo, shading, residual)`` float32 HxWx3 with the input's H and W
      (the model's 32-multiple padding is cropped away; layers are pixel-aligned).
    - ``albedo`` in [0, 1], ``shading >= 0`` (three channels, may exceed 1), and
      ``srgb_to_linear(image) == albedo * shading + residual`` exactly in float32,
      because the residual is recomputed from the cropped layers and our own
      linearization (gamma 2.2, the same the model uses internally).
    - Every returned value is finite. The library normalizes its rough albedo by its
      99th percentile, which is 0/0 when >= 99 % of the pixels are pure black (a logo
      or part on a black background), so all layers come back NaN; that is detected
      here and raised as ``RuntimeError`` so the caller falls back instead of storing
      NaN layers.
    - Raises ``torch.cuda.OutOfMemoryError`` (and any other model error) to the caller;
      ``recolor.intrinsic.decompose`` is the place that falls back to the heuristic.
    """
    if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {image_rgb_u8.shape}")
    if image_rgb_u8.shape[0] == 0 or image_rgb_u8.shape[1] == 0:
        raise ValueError(f"empty image, shape {image_rgb_u8.shape}")
    if progress and not is_loaded():
        progress(0.0, "Loading intrinsic model (Careaga v2.1)")
    models = load_models()
    if progress:
        progress(0.15, "Separating albedo and shading")
    img01 = imageio.to_float(image_rgb_u8)
    padded, size = pad_to_multiple(img01)
    out = run_model(models, padded)
    albedo = crop_to(out["hr_alb"], size)
    shading = crop_to(out["dif_shd"], size)
    if albedo.shape != img01.shape or shading.shape != img01.shape:
        raise RuntimeError(f"intrinsic model returned layers of shape {albedo.shape} / "
                           f"{shading.shape} for input {img01.shape}")
    if not (np.isfinite(albedo).all() and np.isfinite(shading).all()):
        raise RuntimeError("intrinsic model returned non-finite layers "
                           "(typically an almost entirely black input)")
    albedo = np.clip(albedo, 0.0, 1.0)
    shading = np.clip(shading, 0.0, None)
    lin = imageio.srgb_to_linear(img01)
    residual = (lin - albedo * shading).astype(np.float32)
    if progress:
        progress(1.0, "Intrinsic layers ready")
    return albedo, shading, residual
