"""Edge-aware filtering on the GPU: box filter, guided filter, soft masks, label upsampling.

Used by the segmentation stage (refining label edges at full resolution) and by the
recoloring engine (feathered group weights). Everything accepts numpy or torch and
returns the same kind it was given.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

Array = Union[np.ndarray, torch.Tensor]


def _dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_t(x: Array) -> tuple[torch.Tensor, bool]:
    """-> (tensor float32 [C,H,W] on device, was_numpy)."""
    was_np = isinstance(x, np.ndarray)
    t = torch.from_numpy(np.ascontiguousarray(x)) if was_np else x
    t = t.to(_dev(), dtype=torch.float32)
    if t.ndim == 2:
        t = t[None]
    elif t.ndim == 3 and t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
        t = t.permute(2, 0, 1)
    return t.contiguous(), was_np


def _from_t(t: torch.Tensor, like: Array, was_np: bool) -> Array:
    if isinstance(like, np.ndarray):
        if like.ndim == 2:
            t = t[0]
        else:
            t = t.permute(1, 2, 0)
    if was_np:
        return t.cpu().numpy().astype(np.float32)
    return t


def box_filter(t: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, edge-replicated. t: [C,H,W]."""
    if r <= 0:
        return t
    k = 2 * r + 1
    x = F.pad(t[None], (r, r, r, r), mode="replicate")
    return F.avg_pool2d(x, k, stride=1)[0]


def guided_filter(guide: Array, src: Array, radius: int = 8, eps: float = 1e-3) -> Array:
    """He et al. guided filter. `guide` is HxW or HxWx3 (float 0..1), `src` is HxW or
    HxWxC. Output has src's shape. For a color guide the fast grey-guide variant is used
    on luminance (adequate for label softening and layer upsampling)."""
    g, was_np = _to_t(guide)
    s, _ = _to_t(src)
    if g.shape[0] == 3:
        g = (0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2])[None]
    if s.shape[-2:] != g.shape[-2:]:
        s = F.interpolate(s[None], size=g.shape[-2:], mode="bilinear", align_corners=False)[0]
    mean_g = box_filter(g, radius)
    mean_s = box_filter(s, radius)
    corr_gs = box_filter(g * s, radius)
    corr_gg = box_filter(g * g, radius)
    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    out = box_filter(a, radius) * g + box_filter(b, radius)
    return _from_t(out, src, was_np)


def guided_filter_color(guide: torch.Tensor, src: torch.Tensor, radius: int = 2,
                       eps: float = 1e-5) -> torch.Tensor:
    """He et al. guided filter with a full **colour** guide, solved per pixel.

    ``guide`` is [3,H,W] in [0,1] and ``src`` is [H,W]; the result is [H,W] on the same
    device. :func:`guided_filter` collapses a colour guide to luminance, which cannot
    separate two surfaces of equal brightness but different hue (a red panel against a
    grey one, a blue vent against a black gap). This solves the 3x3 covariance system
    instead, so the output follows the edges a photograph actually has. ``eps`` is in the
    units of the guide's variance.
    """
    r = int(radius)
    if r <= 0:
        return src
    gi = guide
    p = src[None]
    mI = box_filter(gi, r)
    mp = box_filter(p, r)
    cov = box_filter(gi * p, r) - mI * mp
    II = box_filter(torch.stack((gi[0] * gi[0], gi[0] * gi[1], gi[0] * gi[2],
                                 gi[1] * gi[1], gi[1] * gi[2], gi[2] * gi[2])), r)
    rr = II[0] - mI[0] * mI[0] + eps
    rg = II[1] - mI[0] * mI[1]
    rb = II[2] - mI[0] * mI[2]
    gg = II[3] - mI[1] * mI[1] + eps
    gb = II[4] - mI[1] * mI[2]
    bb = II[5] - mI[2] * mI[2] + eps
    # inverse of the symmetric 3x3 covariance through its adjugate
    c0 = gg * bb - gb * gb
    c1 = rb * gb - rg * bb
    c2 = rg * gb - rb * gg
    det = rr * c0 + rg * c1 + rb * c2
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    i11, i12, i13 = c0 / det, c1 / det, c2 / det
    i22 = (rr * bb - rb * rb) / det
    i23 = (rb * rg - rr * gb) / det
    i33 = (rr * gg - rg * rg) / det
    a0 = i11 * cov[0] + i12 * cov[1] + i13 * cov[2]
    a1 = i12 * cov[0] + i22 * cov[1] + i23 * cov[2]
    a2 = i13 * cov[0] + i23 * cov[1] + i33 * cov[2]
    b = mp[0] - (a0 * mI[0] + a1 * mI[1] + a2 * mI[2])
    ab = box_filter(torch.stack((a0, a1, a2, b)), r)
    return ab[0] * gi[0] + ab[1] * gi[1] + ab[2] * gi[2] + ab[3]


def gaussian_blur(x: Array, sigma: float) -> Array:
    """Separable Gaussian blur; x HxW or HxWxC."""
    t, was_np = _to_t(x)
    if sigma <= 0:
        return _from_t(t, x, was_np)
    r = max(1, int(3 * sigma + 0.5))
    ax = torch.arange(-r, r + 1, device=t.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = k / k.sum()
    c = t.shape[0]
    xp = F.pad(t[None], (r, r, r, r), mode="replicate")
    xp = F.conv2d(xp, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    xp = F.conv2d(xp, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return _from_t(xp[0], x, was_np)


def upsample_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample of an int32 label map to (w, h)."""
    import cv2
    w, h = size
    if labels.shape[1] == w and labels.shape[0] == h:
        return labels
    return cv2.resize(labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def refine_labels_with_guide(labels: np.ndarray, guide_rgb01: np.ndarray, radius: int = 4,
                             eps: float = 1e-4) -> np.ndarray:
    """Snap a nearest-upsampled label map to the edges of a higher-resolution guide.

    Each label's indicator is guided-filtered; every pixel takes the label with the
    highest filtered response. Labels must already be at the guide's resolution. For
    large label counts this runs in chunks on the GPU."""
    ids = np.unique(labels)
    if len(ids) <= 1:
        return labels
    g, _ = _to_t(guide_rgb01)
    if g.shape[0] == 3:
        g = (0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2])[None]
    lab_t = torch.from_numpy(labels).to(_dev())
    best = torch.full(labels.shape, -1.0, device=_dev())
    out = torch.zeros(labels.shape, dtype=torch.int32, device=_dev())
    mean_g = box_filter(g, radius)
    var_g = box_filter(g * g, radius) - mean_g * mean_g
    chunk = 32
    for i in range(0, len(ids), chunk):
        sel = torch.from_numpy(ids[i:i + chunk].astype(np.int32)).to(_dev())
        ind = (lab_t[None] == sel[:, None, None]).float()          # [k,H,W]
        mean_s = box_filter(ind, radius)
        cov = box_filter(g * ind, radius) - mean_g * mean_s
        a = cov / (var_g + eps)
        b = mean_s - a * mean_g
        resp = box_filter(a, radius) * g + box_filter(b, radius)  # [k,H,W]
        m, arg = resp.max(0)
        better = m > best
        best = torch.where(better, m, best)
        out = torch.where(better, sel[arg], out)
    return out.cpu().numpy().astype(np.int32)


def soft_group_weights(group_map: np.ndarray, n_groups: int, feather_px: float) -> torch.Tensor:
    """[G,H,W] float32 weights on the GPU, blurred indicators normalized to sum to 1 per
    pixel. feather_px <= 0 gives hard one-hot weights."""
    gm = torch.from_numpy(group_map.astype(np.int64)).to(_dev())
    onehot = F.one_hot(gm.clamp(0, n_groups - 1), n_groups).permute(2, 0, 1).float()
    if feather_px > 0:
        onehot = gaussian_blur(onehot, float(feather_px))
        onehot = onehot / onehot.sum(0, keepdim=True).clamp_min(1e-6)
    return onehot
