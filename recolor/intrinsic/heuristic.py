"""Model-free intrinsic decomposition: a gradient-domain (Retinex) illumination estimate.

Classic Retinex assumption: in the log image, small gradients are illumination and
large gradients are reflectance edges. Every pixel pair of the log-luminance gradient
field is soft-weighted by four tests and the kept gradients are integrated back with
a Poisson solve under Neumann boundaries:

1. chromaticity gradient (shading never changes chromaticity, so a chromatic edge is
   reflectance, at a tight threshold);
2. log-luminance gradient (a gray edge must be sharp and strong to count);
3. the same two measured over 4 and 8 px baselines, which catches *defocused*
   reflectance edges whose per-pixel gradient is small but whose total change is far
   larger than shading ever produces over that distance;
4. a trust factor that fades out in near-black regions, where log gradients are
   noise and black-plastic gloss.

The integrated field is high-passed (frame-scale ramps damped, everything smaller
kept) so the global integral cannot drift, then lightly blurred. The result is a
shading that is smooth *and continuous across material edges*: a flat panel keeps a
flat albedo, its edge lands sharply in the albedo with no halo, and local illumination
(soft shadows, curvature) goes into the shading. What stays in the albedo is the
ambiguous part: hard shadow edges and the overall brightness of a dark material.

A large-radius guided filter was tried first and rejected: it either reproduces strong
edges in the shading (which turns a black background into a grey albedo and leaves
only chromaticity) or, with a larger eps, blurs them into halos.

The Poisson solve is one FFT on the even-extended (2H x 2W) domain, equivalent to a
DCT-II solve; the whole decomposition takes ~25 ms at working resolution on the GPU
(~80 ms on the first call). Deterministic, no model files: this is also the fallback
when the Careaga model is unavailable or runs out of memory.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .. import filters, imageio

# Per-pixel gradient magnitudes above which an edge is treated as reflectance.
# Shading never changes chromaticity (c / sum(c), bounded, ~0 for dark pixels), so a
# chromaticity gradient is a reflectance edge at a tight threshold; a purely gray edge
# must be sharp and strong in log luminance (paint/black edges are 1.5x-10x over 1-3
# px, curved glossy surfaces shade at ~1.1x per pixel) before it leaves the shading.
CHROMA_TAU = 0.02
LUMA_TAU = 0.25
# Defocused reflectance edges have a small per-pixel gradient but a large total change
# over a longer baseline, which shading (a few % per pixel at most) never has. Each
# pixel pair is also tested at these baselines (px at a 1536 long side, scaled with
# the image): the luminance change over baseline k must stay below
# LUMA_TAU_COARSE[k] and the chromaticity change below CHROMA_TAU_COARSE.
COARSE_BASELINES = (4, 8)
LUMA_TAU_COARSE = {4: 0.9, 8: 1.3}
CHROMA_TAU_COARSE = 0.06
REFERENCE_LONG_SIDE = 1536
# Added to sum(c) in the chromaticity so near-black pixels read as neutral.
CHROMA_EPS = 3e-3
# High-pass on the integrated shading: illumination variations with a wavelength
# longer than this fraction of the long side are damped toward flat (second-order
# knee: a feature a sixth of the frame wide passes at >99 %, a frame-wide ramp is cut
# to <1 %). A global Retinex integral otherwise drifts by an order of magnitude across
# a frame whose bright and dark materials are joined by gradual gray transitions.
SCREEN_WAVELENGTH_FRAC = 1.0
# Pre-blur of the log image before taking gradients (suppresses JPEG/sensor noise).
GRAD_SIGMA = 0.7
# Post-blur of the shading as a fraction of the long side (6 px at 1536).
SHADING_SMOOTH_FRAC = 1.0 / 256.0
# The albedo is scaled so that this quantile of its max channel, measured on a copy
# blurred by ALBEDO_NORM_SIGMA px (so point speculars and LEDs do not set the white
# point; large bright surfaces do), lands on ALBEDO_TARGET.
ALBEDO_QUANTILE = 0.995
ALBEDO_TARGET = 0.9
ALBEDO_NORM_SIGMA = 4.0
# Floor for luminance before taking logs; ~uint8 value 11 after gamma 2.2. Keeps the
# black background's quantization noise from dominating the log gradients.
LOG_FLOOR = 1e-3
# Gradients measured where the (blurred) luminance is below this linear level (~uint8
# 50) are progressively distrusted: in near-black regions log gradients are dominated
# by noise and by the gloss of black plastic, and integrating them ramps the shading
# by an order of magnitude across the object.
DARK_LUM = 0.03
# torch.quantile refuses more than 16M elements; take a strided subsample above this.
QUANTILE_SAMPLE = 8_000_000


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def poisson_neumann(div: torch.Tensor, screen: float = 0.0) -> torch.Tensor:
    """Solve the Poisson equation ``laplacian(S) = div`` on an HxW grid with zero-flux
    (Neumann) boundaries, optionally high-passed; ``div`` is [1,H,W].

    The five-point Laplacian is inverted spectrally on the half-sample even extension
    of ``div`` (2H x 2W), which enforces the Neumann condition exactly. ``screen`` >= 0
    applies a second-order high-pass ``lam^2 / (lam^2 + screen^2)`` to the solution
    (``lam`` the Laplacian eigenvalue): wavelengths longer than ``2*pi/sqrt(screen)``
    px are damped toward zero with a sharp knee, shorter ones pass unchanged. Returns
    S with zero mean; S is defined up to that constant. Guarantee: with ``screen == 0``
    and ``div`` computed from the forward differences of a field L (zero flux at the
    border), the result equals ``L - mean(L)`` to float precision."""
    h, w = div.shape[-2:]
    ext = torch.cat([div, div.flip(-2)], dim=-2)
    ext = torch.cat([ext, ext.flip(-1)], dim=-1)                    # [1,2H,2W]
    spec = torch.fft.rfft2(ext)                                     # [1,2H,W+1]
    ky = torch.arange(2 * h, device=div.device, dtype=torch.float32)
    kx = torch.arange(w + 1, device=div.device, dtype=torch.float32)
    lam = (2.0 * torch.cos(math.pi * ky / h) - 2.0)[:, None] + (2.0 * torch.cos(math.pi * kx / w) - 2.0)[None, :]
    lam[0, 0] = 1.0                                                 # DC: arbitrary, zeroed below
    if screen > 0:
        spec = spec * (lam / (lam * lam + float(screen) ** 2))
    else:
        spec = spec / lam
    spec[..., 0, 0] = 0.0
    sol = torch.fft.irfft2(spec, s=(2 * h, 2 * w))[..., :h, :w]
    return sol - sol.mean()


def _divergence(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """Backward-difference divergence of forward-difference gradients (``gx`` [1,H,W-1],
    ``gy`` [1,H-1,W]) with zero flux outside the grid -> [1,H,W]."""
    dx = F.pad(gx, (0, 1)) - F.pad(gx, (1, 0))
    dy = F.pad(gy, (0, 0, 0, 1)) - F.pad(gy, (0, 0, 1, 0))
    return dx + dy


def _baseline_change(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute change of ``x`` [C,H,W] over a baseline of ``k`` px centred on each
    pixel pair: returns (dx [C,H,W-1], dy [C,H-1,W]) with replicate padding so the
    border keeps the nearest interior estimate."""
    lo, hi = k // 2, k - k // 2
    xp = F.pad(x[None], (lo, hi, lo, hi), mode="replicate")[0]
    h, w = x.shape[-2:]
    dx = (xp[:, lo:lo + h, k:k + w] - xp[:, lo:lo + h, 0:w]).abs()      # x+hi vs x-lo, per pixel
    dy = (xp[:, k:k + h, lo:lo + w] - xp[:, 0:h, lo:lo + w]).abs()
    # A pair (i, i+1) is inside a coarse edge if either endpoint's window sees it.
    dx = torch.maximum(dx[:, :, 1:], dx[:, :, :-1])
    dy = torch.maximum(dy[:, 1:, :], dy[:, :-1, :])
    return dx, dy


def estimate_log_shading(lin: torch.Tensor, chroma_tau: float = CHROMA_TAU,
                         luma_tau: float = LUMA_TAU,
                         smooth_sigma: float | None = None) -> torch.Tensor:
    """Log of the grayscale illumination of a linear image ``lin`` [3,H,W] (any
    device) -> [1,H,W] on the same device, zero mean.

    Guarantees: smooth (no luminance gradient above ~``luma_tau`` per pixel survives,
    and the field is Gaussian-blurred by ``smooth_sigma`` px, default long side / 256);
    continuous across reflectance edges (chromaticity gradients above ``chroma_tau``,
    gray gradients above ``luma_tau``, or their coarse-baseline counterparts); free
    of variation at wavelengths beyond ``SCREEN_WAVELENGTH_FRAC`` of the long side;
    and equal to the log-luminance up to a constant inside any area brighter than
    ``DARK_LUM`` whose gradients all stay below the thresholds and whose extent is
    well below that wavelength."""
    h, w = lin.shape[-2:]
    if h < 2 or w < 2:
        return torch.zeros((1, h, w), device=lin.device, dtype=torch.float32)
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    log_l = torch.log(lum.clamp_min(LOG_FLOOR))[None]
    chroma = lin / (lin.sum(0, keepdim=True) + CHROMA_EPS)
    if GRAD_SIGMA > 0:
        log_l = filters.gaussian_blur(log_l, GRAD_SIGMA)
        chroma = filters.gaussian_blur(chroma, GRAD_SIGMA)
    glx = log_l[:, :, 1:] - log_l[:, :, :-1]
    gly = log_l[:, 1:, :] - log_l[:, :-1, :]
    cx = (chroma[:, :, 1:] - chroma[:, :, :-1]).abs().amax(0, keepdim=True)
    cy = (chroma[:, 1:, :] - chroma[:, :-1, :]).abs().amax(0, keepdim=True)
    log_wx = -(cx / chroma_tau) ** 2 - (glx / luma_tau) ** 2
    log_wy = -(cy / chroma_tau) ** 2 - (gly / luma_tau) ** 2
    scale = max(h, w) / REFERENCE_LONG_SIDE
    for base in COARSE_BASELINES:
        k = max(2, int(round(base * scale)))
        lt = LUMA_TAU_COARSE[base]
        # Measured on fields smoothed to the baseline's scale so texture and chroma
        # noise do not read as edges.
        dlx, dly = _baseline_change(filters.gaussian_blur(log_l, k / 4.0), k)
        dcx, dcy = _baseline_change(filters.gaussian_blur(chroma, k / 4.0), k)
        log_wx = log_wx - (dlx / lt) ** 2 - (dcx.amax(0, keepdim=True) / CHROMA_TAU_COARSE) ** 2
        log_wy = log_wy - (dly / lt) ** 2 - (dcy.amax(0, keepdim=True) / CHROMA_TAU_COARSE) ** 2
    trust = (torch.exp(log_l) / DARK_LUM).clamp(0.0, 1.0)
    wx = torch.exp(log_wx) * torch.minimum(trust[:, :, 1:], trust[:, :, :-1])
    wy = torch.exp(log_wy) * torch.minimum(trust[:, 1:, :], trust[:, :-1, :])
    gx = wx * glx
    gy = wy * gly
    screen = (2.0 * math.pi / (max(h, w) * SCREEN_WAVELENGTH_FRAC)) ** 2
    log_s = poisson_neumann(_divergence(gx, gy), screen=screen)
    if smooth_sigma is None:
        smooth_sigma = max(h, w) * SHADING_SMOOTH_FRAC
    if smooth_sigma > 0.3:
        log_s = filters.gaussian_blur(log_s, float(smooth_sigma))
    return log_s


def decompose_heuristic(image_rgb_u8: np.ndarray, chroma_tau: float = CHROMA_TAU,
                        luma_tau: float = LUMA_TAU) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a uint8 RGB image into (albedo, shading, residual), all float32 HxWx3.

    Guarantees:
    - ``srgb_to_linear(image) == albedo * shading + residual`` exactly in float32
      (the residual is computed as the difference, so the identity holds even where
      the albedo was clipped).
    - ``albedo`` is in [0, 1], scaled so the 99.5th percentile of its (lightly
      blurred) max channel is 0.9; ``shading`` is grayscale (three identical channels)
      and strictly positive; ``residual`` is zero except where the albedo clipped to
      1 (speculars, blown highlights).
    - Output shape equals the input shape; runs on the GPU when available and takes
      well under 0.5 s for a 1536 px image (~25 ms steady state, ~80 ms first call).
    """
    if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {image_rgb_u8.shape}")
    if image_rgb_u8.shape[0] == 0 or image_rgb_u8.shape[1] == 0:
        raise ValueError(f"empty image, shape {image_rgb_u8.shape}")
    dev = _device()
    lin_np = imageio.srgb_to_linear(imageio.to_float(image_rgb_u8))
    lin = torch.from_numpy(lin_np).to(dev).permute(2, 0, 1).contiguous()
    with torch.inference_mode():
        shd = torch.exp(estimate_log_shading(lin, chroma_tau=chroma_tau, luma_tau=luma_tau))  # [1,H,W]
        alb = lin / shd                                                                        # [3,H,W]
        flat = filters.gaussian_blur(alb.amax(0, keepdim=True), ALBEDO_NORM_SIGMA).flatten()
        if flat.numel() > QUANTILE_SAMPLE:
            # Deterministic strided subsample (a random one made >8 MP results differ
            # run to run by ~1e-3).
            step = -(-flat.numel() // QUANTILE_SAMPLE)
            flat = flat[::step]
        q = torch.quantile(flat, ALBEDO_QUANTILE).clamp_min(1e-6)
        k = ALBEDO_TARGET / q
        shd = shd / k
        alb = (alb * k).clamp(0.0, 1.0)
        albedo = alb.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
        shading = shd.expand(3, -1, -1).permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
    residual = (lin_np - albedo * shading).astype(np.float32)
    return albedo, shading, residual
