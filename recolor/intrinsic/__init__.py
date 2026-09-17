"""Intrinsic decomposition: image = albedo * shading + residual, in linear RGB.

Two methods share one interface (see docs/ARCHITECTURE.md §3.1):

- ``careaga``: the Careaga & Aksoy v2.1 network (``recolor.intrinsic.careaga``), the
  quality path. Lazy singleton on the GPU.
- ``heuristic``: a gradient-domain (Retinex) estimate (``recolor.intrinsic.heuristic``):
  reflectance edges are removed from the log-luminance gradient field and the rest is
  integrated back with a Poisson solve. Model-free, deterministic, well under 0.5 s
  at working resolution. Used when the model is missing, the image is too large, the
  model runs out of memory, or the model returns invalid (non-finite) layers.

Both methods guarantee ``srgb_to_linear(image) == albedo * shading + residual`` in
float32 and that every layer has the input's H and W.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .. import config, imageio

log = logging.getLogger(__name__)

METHODS = ("auto", "careaga", "heuristic")
Progress = Callable[[float, str], None]


@dataclass
class IntrinsicResult:
    """The three linear layers of one image plus the method that produced them."""
    albedo: np.ndarray    # float32 linear HxWx3 in [0,1]
    shading: np.ndarray   # float32 linear HxWx3, >= 0 (may exceed 1)
    residual: np.ndarray  # float32 HxWx3, may be negative (saturated pixels) or positive (speculars)
    method: str           # 'careaga' | 'heuristic'

    @property
    def shape(self) -> tuple[int, int]:
        """(H, W) of every layer."""
        return int(self.albedo.shape[0]), int(self.albedo.shape[1])


def _validate_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"unknown intrinsic method {method!r}; expected one of {METHODS}")
    return method


def _use_careaga(image_rgb_u8: np.ndarray, method: str) -> bool:
    """Decide whether the model path is attempted for this call."""
    from . import careaga
    if method == "heuristic":
        return False
    available = careaga.is_available()
    if method == "careaga":
        if not available:
            log.warning("Careaga intrinsic requested but not available on this machine "
                        "(no CUDA device, no `intrinsic` package, or a previous load failed); "
                        "using the heuristic")
        return available
    h, w = image_rgb_u8.shape[:2]
    return available and h * w <= config.FULLRES_INTRINSIC_MAX_PIXELS


def decompose(image_rgb_u8: np.ndarray, method: str = "auto",
              progress: Progress | None = None) -> IntrinsicResult:
    """Decompose a uint8 RGB image into albedo, shading and residual.

    Guarantees:
    - Every layer is float32 HxWx3 with the input's H and W.
    - ``imageio.srgb_to_linear(to_float(image)) == albedo * shading + residual`` to
      float32 precision (well within 1e-4), for both methods.
    - ``albedo`` is in [0, 1]; ``shading >= 0``.
    - ``method="auto"`` uses Careaga when the model is available and the image has at
      most ``config.FULLRES_INTRINSIC_MAX_PIXELS`` pixels, else the heuristic.
    - ``method="careaga"`` attempts the model regardless of size when
      ``careaga.is_available()``; a CUDA out-of-memory, non-finite model output or any
      other model failure is logged and the heuristic result is returned, so a job
      never crashes here. When the model is not available at all the heuristic is
      used immediately (with a logged warning) rather than attempting a load.
      ``method="heuristic"`` never touches the model.
    - ``result.method`` reports what actually ran.
    - Raises ``ValueError`` (before any method runs) for anything that is not a
      non-empty uint8 HxWx3 array, or for an unknown ``method``.
    """
    _validate_method(method)
    if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3 RGB, got {image_rgb_u8.dtype} {image_rgb_u8.shape}")
    if image_rgb_u8.shape[0] == 0 or image_rgb_u8.shape[1] == 0:
        raise ValueError(f"empty image, shape {image_rgb_u8.shape}")
    if _use_careaga(image_rgb_u8, method):
        from . import careaga
        try:
            albedo, shading, residual = careaga.decompose_careaga(image_rgb_u8, progress)
            return IntrinsicResult(albedo, shading, residual, "careaga")
        except Exception as exc:  # noqa: BLE001 - deliberate: never crash the job
            _release_cuda()
            log.warning("Careaga intrinsic failed (%s: %s); falling back to the heuristic",
                        type(exc).__name__, exc)
            if progress:
                progress(0.2, "Model unavailable, using gradient-domain (Retinex) estimate")
    if progress:
        progress(0.3, "Estimating illumination (gradient-domain Retinex)")
    from . import heuristic
    albedo, shading, residual = heuristic.decompose_heuristic(image_rgb_u8)
    if progress:
        progress(1.0, "Intrinsic layers ready")
    return IntrinsicResult(albedo, shading, residual, "heuristic")


def _release_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort only
        pass


def recompose(res: IntrinsicResult) -> np.ndarray:
    """``uint8 sRGB`` of ``albedo * shading + residual`` (clipped to [0,1]). For an
    unedited result this reproduces the input image to within rounding."""
    lin = res.albedo * res.shading + res.residual
    return imageio.to_uint8(imageio.linear_to_srgb(np.clip(lin, 0.0, 1.0)))


def reconstruction_error(image_rgb_u8: np.ndarray, res: IntrinsicResult) -> float:
    """Max absolute error of the linear identity ``lin == albedo*shading + residual``
    for the given input (should be ~1e-7 for both methods)."""
    lin = imageio.srgb_to_linear(imageio.to_float(image_rgb_u8))
    return float(np.abs(lin - (res.albedo * res.shading + res.residual)).max())


def warmup(method: str = "careaga") -> None:
    """Load the requested method's models onto the GPU (idempotent). The heuristic has
    nothing to load; ``"auto"`` warms Careaga when it is available."""
    _validate_method(method)
    if method == "heuristic":
        return
    from . import careaga
    if method == "auto" and not careaga.is_available():
        return
    careaga.warmup()


def is_loaded(method: str = "careaga") -> bool:
    """True when ``decompose`` with this method will not have to load anything: always
    for the heuristic; for ``careaga``/``auto`` once the singleton is resident."""
    _validate_method(method)
    if method == "heuristic":
        return True
    from . import careaga
    return careaga.is_loaded()


def release(method: str = "careaga") -> None:
    """Drop the requested method's loaded models and free the CUDA memory they held.
    The heuristic has nothing to release. Idempotent; the next ``decompose`` call
    reloads from disk."""
    _validate_method(method)
    if method == "heuristic":
        return
    from . import careaga
    careaga.release()


SHADING_DISPLAY_QUANTILE = 0.995
RESIDUAL_DISPLAY_GAIN = 4.0


def layers_for_display(res: IntrinsicResult) -> dict[str, np.ndarray]:
    """uint8 sRGB HxWx3 views of the three layers for the UI.

    - ``albedo``: sRGB of the albedo.
    - ``shading``: shading divided by its 99.5th percentile (over all channels) so the
      brightest lit surfaces are white, then sRGB.
    - ``residual``: ``|residual| * 4`` clipped to [0,1], sRGB (positive and negative
      residual are shown alike; the layer is a "where did the model deviate" map).
    """
    shd = res.shading
    q = float(np.quantile(shd, SHADING_DISPLAY_QUANTILE)) if shd.size else 1.0
    q = q if q > 1e-6 else 1.0
    return {
        "albedo": imageio.to_uint8(imageio.linear_to_srgb(res.albedo)),
        "shading": imageio.to_uint8(imageio.linear_to_srgb(shd / q)),
        "residual": imageio.to_uint8(imageio.linear_to_srgb(np.abs(res.residual) * RESIDUAL_DISPLAY_GAIN)),
    }


__all__ = [
    "IntrinsicResult", "METHODS", "decompose", "recompose", "reconstruction_error",
    "warmup", "is_loaded", "release", "layers_for_display",
]
