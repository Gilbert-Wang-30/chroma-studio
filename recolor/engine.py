"""Recoloring engine: edits the albedo layer of an intrinsic decomposition on the GPU.

The engine never touches the lighting of a photograph. It takes the linear albedo,
shading and residual layers plus a group map, repaints selected groups in CIE Lab,
and recomposes ``sRGB(albedo' * shading' + residual')``. All heavy math is torch on
``config.device()``; the Lab conversions are implemented here in torch so a render
never round-trips through skimage.

Color-space conventions (identical to :mod:`recolor.imageio`):

* linear <-> sRGB is a plain 2.2 gamma (what the Intrinsic pipeline uses);
* sRGB <-> Lab uses the IEC 61966-2-1 piecewise companding, the skimage sRGB matrix
  and the D65/2deg white point, so ``srgb_to_lab_t`` agrees with
  ``imageio.rgb_to_lab`` to float32 precision and ``ColorGroup.albedo_lab`` values
  produced by the segmentation stage are directly usable as source colors.

Algorithm (see docs/ARCHITECTURE.md section 3.5):

1. Coverage: how much of each pixel belongs to a repainted group. Blurring the mapped
   indicator symmetrically, as this used to, pushes coverage below 1 *inside* the
   repainted part, so a band of it keeps the original paint. That is what put yellow
   outlines around every panel of a yellow motorcycle asked to become black. Instead the
   indicator is snapped onto the photograph's own edges with a colour guided filter,
   maxed with the hard label (coverage may only add to what the user selected), then
   given an outward-only ramp, so coverage is full at the label boundary and falls to
   zero a couple of pixels outside it. ``feather_px`` scales that outward ramp; 0 gives
   edge-snapped hard coverage. See :meth:`Renderer._coverage`.
   The per-group colour constants are a separate, narrowly blurred field: the engine
   never materializes a ``[G,H,W]`` tensor, so a mapping costs the same for 3 groups as
   for 300.
2. For a mapped group with target ``T`` and source albedo ``A`` (the group's
   ``albedo_lab``), ``shift`` mode moves the chroma so that ``A`` lands on ``T``:
   ``ab' = T_ab + s * R(theta) * (ab - A_ab)`` with ``s = min(1, C_T / C_A)`` and
   ``R(theta)`` the rotation from the source's hue to the target's. The deviation from
   the group colour is the paint's texture (a little more or less saturated here, a hint
   of hue drift there), expressed in the *source's* a/b frame. Rotated, "more saturated
   red" becomes "more saturated navy" and a white decal (deviation ``-A_ab``) stays
   neutral; carried over unrotated it lands beside the new colour instead of along it,
   which made the lit flank of a red tank painted navy come out mauve and its decal
   cyan. ``s`` carries the chroma spread over proportionally (a red car repainted grey
   does not turn into a rainbow of purples). Lightness uses a map anchored on the group's own
   lightness, ``L' = T_L + slope * (L - A_L)``, with slope 1 above the anchor and
   ``T_L / A_L`` below it: highlights keep their modelling while the darker half is
   squeezed into the room the new colour actually has. A plain additive shift sent
   everything below the anchor past zero, so more than half a red part collapsed into
   featureless black whenever the target was dark. ``flat`` mode replaces the Lab with
   ``T``, and ``texture`` in [0,1] blends between them.
3. Out-of-gamut colors are brought back by compressing chroma (binary search along
   the (a,b) ray at fixed L), never by clipping channels, so bright repaints do not
   blow out and dark ones do not go muddy.
4. ``shading' = pivot * (shading / pivot) ** shading_strength`` per channel, with the
   pivot the per-channel median shading of the image: strength flattens or deepens
   the light without changing the overall exposure or the color of the light.
5. The residual is not pure specular: on a saturated surface it also carries diffuse
   energy the decomposition could not explain, wearing the original paint's colour.
   Added back untouched it survived every repaint, which is what turned a red part asked
   to become black into muddy maroon. So the positive residual is split at its own
   achromatic floor. The coloured excess is rebuilt as a multiple of the repainted
   product, keeping its energy but taking the new colour. The neutral floor is the
   lamp's own light: kept where it is a sharp glint, attenuated with the repaint where
   it is only a faint veil (see ``_Level.spec_weight``). ``residual_tint`` optionally
   pulls what survives toward the target's hue. The negative residual (sensor clipping)
   is scaled by the ratio of the new to the old ``albedo * shading`` so darkening a
   blown-out panel does not leave dark holes in the highlights.
6. The light on a repainted part changes colour with the paint. Diffuse shading should
   carry only the illuminant, but the decomposition leaks part of a saturated surface
   into it, most where the part is concave or shadowed (the red Ducati's tank shadows
   are lit by light five times redder than the scene's). That is bounce off the old
   paint, and lighting the new albedo with it tints the shadows toward the old hue: a
   navy tank went teal. So per pixel, the light's tint beyond the scene's own illuminant
   is split along the old paint's chroma direction; the aligned part is rotated and
   scaled to the new paint exactly like the albedo, the rest (coloured lighting from
   elsewhere) is kept. The illuminant is estimated from the well-lit low-chroma surfaces,
   the way a white balance does (see ``Renderer._retint_shading``).
7. Unmapped and locked groups are reproduced exactly: with an empty mapping the
   output equals ``linear_to_srgb(albedo * shading + residual)`` bit for bit.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from . import config, filters, imageio
from .types import ColorGroup, Mapping, RenderOptions

GAMMA: float = imageio.GAMMA

# skimage's sRGB (D65) matrices and white point, so the two implementations agree.
_XYZ_FROM_RGB = (
    (0.412453, 0.357580, 0.180423),
    (0.212671, 0.715160, 0.072169),
    (0.019334, 0.119193, 0.950227),
)
_RGB_FROM_XYZ = (
    (3.24048134, -1.53715152, -0.49853633),
    (-0.96925495, 1.87599000, 0.04155593),
    (0.05564664, -0.20404134, 1.05731107),
)
_D65 = (0.95047, 1.0, 1.08883)
_LAB_EPS = 0.008856          # (6/29)^3
_LAB_KAPPA = 7.787           # (1/3) * (29/6)^2
_LAB_F_THRESH = 0.2068966    # 6/29
_GAMUT_TOL = 2e-3


# ----------------------------------------------------------------- torch color math

def _mat(m: Sequence[Sequence[float]], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(m, dtype=like.dtype, device=like.device)


def srgb_to_linear_t(x: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] -> linear with the 2.2 gamma of ``imageio.srgb_to_linear``. Clamps."""
    return x.clamp(0.0, 1.0).pow(GAMMA)


def linear_to_srgb_t(x: torch.Tensor) -> torch.Tensor:
    """Linear [0,1] -> sRGB with the 2.2 gamma of ``imageio.linear_to_srgb``. Clamps."""
    return x.clamp(0.0, 1.0).pow(1.0 / GAMMA)


def srgb_to_lab_t(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] (any leading shape, last dim 3) -> CIE Lab (D65, L in 0..100).

    Matches ``skimage.color.rgb2lab`` / ``imageio.rgb_to_lab`` to float32 precision.
    """
    x = rgb.clamp(0.0, 1.0)
    lin = torch.where(x > 0.04045, ((x + 0.055) / 1.055).pow(2.4), x / 12.92)
    xyz = lin @ _mat(_XYZ_FROM_RGB, lin).T
    xyz = xyz / _mat(_D65, xyz)
    f = torch.where(xyz > _LAB_EPS, xyz.clamp_min(1e-12).pow(1.0 / 3.0), _LAB_KAPPA * xyz + 16.0 / 116.0)
    fx, fy, fz = f.unbind(-1)
    return torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), dim=-1)


def _lab_to_rgb_unclipped(lab: torch.Tensor) -> torch.Tensor:
    """Lab -> sRGB-primaries *linear* RGB (2.4 piecewise sense), not clipped, so that a
    value outside [0,1] tells the caller the color is out of gamut."""
    L, a, b = lab.unbind(-1)
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = (fy - b / 200.0).clamp_min(0.0)
    f = torch.stack((fx, fy, fz), dim=-1)
    xyz = torch.where(f > _LAB_F_THRESH, f.pow(3.0), (f - 16.0 / 116.0) / _LAB_KAPPA)
    xyz = xyz * _mat(_D65, xyz)
    return xyz @ _mat(_RGB_FROM_XYZ, xyz).T


def _companding(rgb_lin24: torch.Tensor) -> torch.Tensor:
    x = rgb_lin24.clamp(0.0, 1.0)
    return torch.where(x > 0.0031308, 1.055 * x.pow(1.0 / 2.4) - 0.055, 12.92 * x)


def lab_to_srgb_t(lab: torch.Tensor) -> torch.Tensor:
    """CIE Lab -> sRGB [0,1], clipped per channel (matches ``imageio.lab_to_rgb``)."""
    return _companding(_lab_to_rgb_unclipped(lab))


def linear_to_lab_t(lin: torch.Tensor) -> torch.Tensor:
    """Linear RGB -> Lab through the 2.2 gamma (matches ``imageio.linear_to_lab``)."""
    return srgb_to_lab_t(linear_to_srgb_t(lin))


def lab_to_linear_t(lab: torch.Tensor) -> torch.Tensor:
    """Lab -> linear RGB through the 2.2 gamma (matches ``imageio.lab_to_linear``)."""
    return srgb_to_linear_t(lab_to_srgb_t(lab))


def lab_to_linear_gamut_t(lab: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Lab -> linear RGB [0,1] with *chroma compression* instead of channel clipping.

    L is clamped to [0,100]. For every pixel whose color lies outside the sRGB gamut,
    (a,b) is scaled toward zero by the largest factor (binary search, ``iters``
    halvings) that brings it inside, keeping lightness and hue. In-gamut pixels are
    returned unchanged (identical to :func:`lab_to_linear_t`).
    """
    L = lab[..., :1].clamp(0.0, 100.0)
    ab = lab[..., 1:]
    lab_c = torch.cat((L, ab), dim=-1)
    rgb = _lab_to_rgb_unclipped(lab_c)
    oog = ((rgb < -_GAMUT_TOL) | (rgb > 1.0 + _GAMUT_TOL)).any(-1)
    if bool(oog.any()):
        lo = torch.zeros_like(L[..., 0])
        hi = torch.ones_like(lo)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            test = _lab_to_rgb_unclipped(torch.cat((L, ab * mid[..., None]), dim=-1))
            ok = ((test >= -_GAMUT_TOL) & (test <= 1.0 + _GAMUT_TOL)).all(-1)
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        scale = torch.where(oog, lo, torch.ones_like(lo))
        rgb = _lab_to_rgb_unclipped(torch.cat((L, ab * scale[..., None]), dim=-1))
    return srgb_to_linear_t(_companding(rgb))


def luminance_t(lin: torch.Tensor) -> torch.Tensor:
    """Rec. 709 luminance of a linear RGB tensor [..., 3] -> [...]."""
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


# ----------------------------------------------------------------- helpers

def _as_hwc(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"{name} must be HxWx3, got {a.shape}")
    return np.ascontiguousarray(a, dtype=np.float32)


def normalize_mapping(mapping: Optional[Mapping]) -> dict[int, str]:
    """Group id -> '#rrggbb' for every group that is actually mapped. Accepts string or
    int keys, ``None`` / empty values (dropped) and 3- or 6-digit hex with or without
    '#'. Raises ``ValueError`` (never another exception type) on an unparsable color
    or on a key that is not an integer group id (``None``, ``"abc"``, ``1.5``...)."""
    out: dict[int, str] = {}
    for k, v in (mapping or {}).items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        try:
            gid = int(k)
        except (TypeError, ValueError):
            raise ValueError(f"mapping key {k!r} is not a group id") from None
        if isinstance(k, float) and not float(k).is_integer():
            raise ValueError(f"mapping key {k!r} is not a group id")
        try:
            rgb = imageio.hex_to_rgb01(str(v))
        except Exception as exc:  # keep the documented exception type
            raise ValueError(f"mapping value {v!r} for group {k!r} is not a color") from exc
        out[gid] = imageio.rgb01_to_hex(rgb)
    return out


def _feather_chw(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a ``[C,H,W]`` float tensor with edge replication,
    done directly on ``[1,C,H,W]`` so the channel count is never guessed from the
    shape (``filters.gaussian_blur`` would permute a field whose width is 1, 3 or 4).
    Returns a tensor of exactly the input shape; ``sigma <= 0`` returns the input."""
    if sigma <= 0:
        return field
    c, h, w = field.shape
    r = max(1, int(3 * sigma + 0.5))
    ax = torch.arange(-r, r + 1, device=field.device, dtype=field.dtype)
    k = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = k / k.sum()
    xp = F.pad(field[None], (r, r, r, r), mode="replicate")
    xp = F.conv2d(xp, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    xp = F.conv2d(xp, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    out = xp[0]
    if tuple(out.shape) != (c, h, w):
        raise RuntimeError(f"feather produced shape {tuple(out.shape)}, expected {(c, h, w)}")
    return out


# Coverage tuning (see Renderer._coverage). The guided filter only has to see both sides
# of one edge, so a small window wins: a larger one drags in a third colour and degenerates
# toward a box blur. The ramp bounds put full coverage on the label boundary and zero about
# 1.3 * sigma outside it.
COVERAGE_RADIUS = 2
COVERAGE_EPS = 1e-5
RAMP_LO, RAMP_HI = 0.10, 0.50
#: feather_px that reproduces the tuned outward ramp width RAMP_SIGMA.
RAMP_SIGMA, RAMP_FEATHER_REF = 2.5, 1.5
#: Narrow Gaussian blend of the colour parameters, plus a wide low-weight fill.
PARAM_FEATHER, PARAM_FAR_WEIGHT = 1.2, 0.02
#: Luminance the colour of the light is normalised to before its tint is measured in Lab.
_LIGHT_LUM = 0.2


@dataclass
class _Level:
    """One resolution of the layers, all on the GPU, channels-last float32."""
    albedo: torch.Tensor        # [H,W,3] linear
    shading: torch.Tensor       # [H,W,3] linear, >= 0
    residual: torch.Tensor      # [H,W,3]
    albedo_lab: torch.Tensor    # [H,W,3]
    product: torch.Tensor       # albedo * shading, cached
    group_map: torch.Tensor     # [H,W] int64
    spec: Optional[torch.Tensor] = None   # [H,W,1] 0..1 glint weight, filled on first use

    @property
    def size(self) -> tuple[int, int]:
        return int(self.albedo.shape[1]), int(self.albedo.shape[0])

    def spec_weight(self) -> torch.Tensor:
        """How much of the neutral residual at each pixel is a real specular glint.

        The achromatic floor of the positive residual mixes two things: sharp mirror
        highlights, which belong to the lamp and survive any repaint, and a faint broad
        veil of diffuse energy the decomposition could not explain, which belongs to the
        old paint. They separate by magnitude, so normalise by a high quantile: near 1
        is a glint, near 0 is veil.
        """
        if self.spec is None:
            gray = self.residual.clamp_min(0.0).amin(dim=-1, keepdim=True)
            flat = gray.reshape(-1)
            if flat.numel() > 1_000_000:
                flat = flat[:: flat.numel() // 1_000_000 + 1]
            q = torch.quantile(flat, 0.99) if flat.numel() else torch.zeros((), device=gray.device)
            self.spec = (gray / q.clamp_min(1e-4)).clamp(0.0, 1.0)
        return self.spec


def _hue_confidence(chroma: float) -> float:
    """0 for a neutral colour, 1 from chroma 8 up: how much a hue means at this chroma."""
    t = min(1.0, max(0.0, (chroma - 2.0) / 6.0))
    return t * t * (3.0 - 2.0 * t)


def _hue_turn(src_lab: torch.Tensor, tgt_lab: torch.Tensor) -> float:
    """Angle (radians) turning the source's chroma direction onto the target's, faded to
    zero as either side approaches neutral, where it would only rotate noise."""
    sa, sb = float(src_lab[1]), float(src_lab[2])
    ta, tb = float(tgt_lab[1]), float(tgt_lab[2])
    w = _hue_confidence(math.hypot(sa, sb)) * _hue_confidence(math.hypot(ta, tb))
    if w <= 0.0:
        return 0.0
    theta = math.atan2(tb, ta) - math.atan2(sb, sa)
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    return w * theta


@dataclass
class _GroupParams:
    """Per-group constants of one render, indexed by group id (rows 0..G-1)."""
    table: torch.Tensor         # [G, 19]: mapped, texture, s*R(theta)[4], bounce*R(theta)[4], T_L, A_L, slope_dn, c_a, c_b, T_a, T_b, A_a, A_b
    n_mapped: int


class Renderer:
    """Holds one image's layers on the GPU so successive renders are fast.

    Guarantees:

    * ``render({}, RenderOptions())`` returns exactly
      ``to_uint8(linear_to_srgb(albedo * shading + residual))``.
    * Unmapped groups, groups mapped to ``None`` and ``locked`` groups are left
      untouched (up to feathering at their borders with repainted neighbours).
    * Every output is finite uint8 RGB of the requested size, for every combination
      of :class:`RenderOptions` values.
    * After the first call at a given size, a preview render (1024 long side) takes
      well under 60 ms on the RTX 5090; layers and label maps are cached per size.

    Source colors come from ``ColorGroup.albedo_lab``; groups present in the map but
    missing from ``groups`` get the median Lab albedo of their pixels.
    """

    def __init__(self, albedo_lin: np.ndarray, shading_lin: np.ndarray, residual: np.ndarray,
                 group_map: np.ndarray, groups: Sequence[ColorGroup],
                 device: Optional[str] = None) -> None:
        albedo = _as_hwc(albedo_lin, "albedo_lin")
        shading = _as_hwc(shading_lin, "shading_lin")
        resid = _as_hwc(residual, "residual")
        gm = np.asarray(group_map)
        if gm.ndim != 2:
            raise ValueError(f"group_map must be HxW, got {gm.shape}")
        if not (albedo.shape == shading.shape == resid.shape and albedo.shape[:2] == gm.shape):
            raise ValueError("albedo, shading, residual and group_map must share H, W")
        if gm.size and int(gm.min()) < 0:
            raise ValueError("group_map contains -1 (unassigned pixels)")

        self.device = torch.device(device or config.device())
        self.groups: list[ColorGroup] = list(groups)
        self._group_by_id: dict[int, ColorGroup] = {int(g.id): g for g in self.groups}
        max_id = max([int(g.id) for g in self.groups] + [int(gm.max()) if gm.size else -1])
        self.n_groups: int = max_id + 1

        dev = self.device
        alb_t = torch.from_numpy(albedo).to(dev)
        shd_t = torch.from_numpy(shading).to(dev)
        res_t = torch.from_numpy(resid).to(dev)
        gm_t = torch.from_numpy(np.ascontiguousarray(gm.astype(np.int64))).to(dev)
        self._base = _Level(alb_t, shd_t, res_t, linear_to_lab_t(alb_t), alb_t * shd_t, gm_t)
        self._levels: dict[tuple[int, int], _Level] = {self._base.size: self._base}
        self._source_lab: dict[int, torch.Tensor] = {}
        self._shading_pivot = self._compute_pivot(shd_t)
        self._light_ref: Optional[torch.Tensor] = None
        self._guides: dict[tuple[int, int], torch.Tensor] = {}
        self.last_render_ms: float = 0.0
        self._freed: bool = False

    # ------------------------------------------------------------ public API

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the layers the renderer was built with."""
        return self._base.size

    @property
    def freed(self) -> bool:
        """True once :meth:`free` has been called; every render then raises."""
        return self._freed

    def _check_alive(self) -> None:
        if self._freed:
            raise RuntimeError("Renderer has been freed; build a new one to render again")

    def render(self, mapping: Mapping, options: Optional[RenderOptions] = None) -> np.ndarray:
        """Render at the native layer resolution -> uint8 sRGB HxWx3.
        Raises ``RuntimeError`` after :meth:`free`."""
        self._check_alive()
        return self._finish(self._render_linear(self._base, mapping, options or RenderOptions()))

    def render_at(self, long_side: int, mapping: Mapping, options: Optional[RenderOptions] = None) -> np.ndarray:
        """Render with the layers resized so the long side is ``long_side`` (never
        upscaled beyond the native size). Resized layers are cached per size.
        Raises ``RuntimeError`` after :meth:`free`."""
        self._check_alive()
        return self._finish(self._render_linear(self._level_for(long_side), mapping, options or RenderOptions()))

    def recolor_albedo(self, mapping: Mapping, options: Optional[RenderOptions] = None,
                       long_side: Optional[int] = None) -> np.ndarray:
        """The repainted albedo only (float32 linear HxWx3), without shading or residual.
        Useful for inspecting a mapping and for the UI's albedo layer.
        Raises ``RuntimeError`` after :meth:`free`."""
        self._check_alive()
        level = self._base if long_side is None else self._level_for(long_side)
        opts = options or RenderOptions()
        params = self._group_params(mapping, opts)
        alb, _, _, _ = self._recolored_albedo(level, params, opts)
        return alb.detach().cpu().numpy().astype(np.float32)

    def free(self) -> None:
        """Drop every cached GPU tensor. The renderer must not be used afterwards:
        ``render``, ``render_at`` and ``recolor_albedo`` raise ``RuntimeError``.
        Idempotent."""
        self._freed = True
        self._levels.clear()
        self._source_lab.clear()
        self._guides.clear()
        self._light_ref = None
        for name in ("albedo", "shading", "residual", "albedo_lab", "product", "group_map"):
            setattr(self._base, name, torch.empty(0))
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ levels

    def _level_for(self, long_side: int) -> _Level:
        w0, h0 = self._base.size
        size = imageio.fit_size(w0, h0, int(long_side))
        lvl = self._levels.get(size)
        if lvl is None:
            lvl = self._resize_level(self._base, size)
            self._levels[size] = lvl
        return lvl

    @staticmethod
    def _resize_level(base: _Level, size: tuple[int, int]) -> _Level:
        w, h = size
        shrink = w < base.albedo.shape[1]

        def rs(x: torch.Tensor) -> torch.Tensor:
            t = x.permute(2, 0, 1)[None]
            if shrink:
                t = F.interpolate(t, size=(h, w), mode="area")
            else:
                t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False)
            return t[0].permute(1, 2, 0).contiguous()

        alb = rs(base.albedo).clamp(0.0, 1.0)
        shd = rs(base.shading).clamp_min(0.0)
        res = rs(base.residual)
        gm = F.interpolate(base.group_map[None, None].float(), size=(h, w), mode="nearest")[0, 0].long()
        return _Level(alb, shd, res, linear_to_lab_t(alb), alb * shd, gm)

    @staticmethod
    def _compute_pivot(shading: torch.Tensor) -> torch.Tensor:
        flat = shading.reshape(-1, 3)
        if flat.shape[0] > 1_000_000:
            flat = flat[:: flat.shape[0] // 1_000_000 + 1]
        piv = flat.median(dim=0).values if flat.shape[0] else torch.ones(3, device=shading.device)
        return piv.clamp_min(1e-3)

    # ------------------------------------------------------------ per-group parameters

    def _light_reference(self) -> torch.Tensor:
        """[3] linear RGB colour of the scene's illuminant, at luminance 1.

        Diffuse shading should carry the colour of the illumination, which belongs to the
        scene, not to the part. In practice the decomposition leaks some of a saturated
        surface into its shading: under this Ducati's red fairing the estimated light is
        3 % redder and 10 % bluer-deficient than the image median, under an RX-78's red
        shield 37 % redder, and in the Ducati tank's shadows five times redder than blue.
        Leave that in place and every repaint inherits the old paint through the light.
        :meth:`_retint_shading` needs the illuminant to tell that leak apart from it.
        """
        if self._light_ref is not None:
            return self._light_ref
        base = self._base
        lum = luminance_t(base.shading).clamp_min(1e-6)[..., None]
        chrom = (base.shading / lum).reshape(-1, 3)
        step = max(1, chrom.shape[0] // 400_000)
        # Estimate the scene's light from the least colourful surfaces, the way a white
        # balance does. A plain image-wide median is dragged toward the paint whenever one
        # saturated colour covers much of the frame: on the red Ducati it reports a warm
        # light that the neutral parts of the same photo do not see.
        lab = base.albedo_lab.reshape(-1, 3)[::step]
        c_alb = torch.hypot(lab[:, 1], lab[:, 2])
        sample = chrom[::step]
        # Well-lit neutral surfaces only: in a near-black area the albedo is ~0 and the
        # estimated shading chromaticity there is numerically meaningless.
        neutral = (c_alb <= torch.quantile(c_alb, 0.4).clamp_min(6.0)) & (lab[:, 0] > 25.0)
        ref = sample[neutral] if int(neutral.sum()) >= 256 else sample
        img_med = ref.median(dim=0).values.clamp_min(1e-3)
        self._light_ref = img_med / luminance_t(img_med).clamp_min(1e-6)
        return self._light_ref

    def _source_lab_for(self, gid: int) -> torch.Tensor:
        cached = self._source_lab.get(gid)
        if cached is not None:
            return cached
        g = self._group_by_id.get(gid)
        if g is not None:
            lab = torch.tensor([float(v) for v in g.albedo_lab], dtype=torch.float32, device=self.device)
        else:
            sel = self._base.group_map == gid
            if bool(sel.any()):
                lab = self._base.albedo_lab[sel].median(dim=0).values
            else:
                lab = torch.tensor([50.0, 0.0, 0.0], device=self.device)
        self._source_lab[gid] = lab
        return lab

    def _group_params(self, mapping: Mapping, options: RenderOptions) -> _GroupParams:
        mode = (options.mode or "shift").lower()
        if mode not in ("shift", "flat"):
            raise ValueError(f"RenderOptions.mode must be 'shift' or 'flat', got {options.mode!r}")
        texture = float(np.clip(options.texture, 0.0, 1.0)) if mode == "shift" else 0.0
        saturation = max(0.0, float(options.saturation))
        table = torch.zeros((max(self.n_groups, 1), 19), dtype=torch.float32, device=self.device)
        n_mapped = 0
        for gid, hexcol in normalize_mapping(mapping).items():
            if gid < 0 or gid >= self.n_groups:
                continue
            g = self._group_by_id.get(gid)
            if g is not None and g.locked:
                continue
            t_lab = torch.tensor(imageio.hex_to_lab(hexcol), dtype=torch.float32, device=self.device)
            t_lab = torch.cat((t_lab[:1], t_lab[1:] * saturation))
            a_lab = self._source_lab_for(gid)
            # Chroma: ab' = T_ab + s * R(theta) * (ab - A_ab). The deviation from the
            # group's own colour is the paint's texture, expressed in the source's a/b
            # frame; R(theta) turns it into the target's frame so "a little more saturated
            # red" becomes "a little more saturated navy" instead of "navy plus magenta"
            # (unrotated, the lit flank of a red tank painted navy came out mauve and its
            # white decal cyan). s shrinks the deviations when the target is less saturated
            # than the source, so hue noise is not amplified.
            c_src = float(torch.hypot(a_lab[1], a_lab[2]))
            c_tgt = float(torch.hypot(t_lab[1], t_lab[2]))
            s_raw = 1.0 if c_src < 1e-3 else min(1.0, c_tgt / c_src)
            s_ab = texture * s_raw
            theta = _hue_turn(a_lab, t_lab)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            m00, m01, m10, m11 = s_ab * cos_t, -s_ab * sin_t, s_ab * sin_t, s_ab * cos_t
            # The same turn at full strength recolours the bounce light (_retint_shading):
            # what the paint reflects onto itself does not depend on the texture setting.
            b00, b01, b10, b11 = s_raw * cos_t, -s_raw * sin_t, s_raw * sin_t, s_raw * cos_t
            a_a, a_b = float(a_lab[1]), float(a_lab[2])
            # Lightness: L' = T_L + texture * slope * (L - A_L), anchored on the group's own
            # lightness. Above the anchor the slope is 1, so highlights keep their modelling.
            # Below it the slope is T_L / A_L, which squeezes the darker half into the room
            # the new colour actually has instead of letting it run negative. A plain
            # additive shift crushed 54 % of a red part to featureless black when the target
            # was black, because everything below the anchor clipped at L = 0.
            a_L = float(a_lab[0])
            slope_dn = 1.0 if a_L <= 1e-3 else min(1.0, float(t_lab[0]) / a_L)
            table[gid] = torch.tensor([
                1.0, texture, m00, m01, m10, m11, b00, b01, b10, b11,
                float(t_lab[0]), a_L, slope_dn,
                float(t_lab[1]) - (m00 * a_a + m01 * a_b), float(t_lab[2]) - (m10 * a_a + m11 * a_b),
                float(t_lab[1]), float(t_lab[2]), a_a, a_b,
            ], dtype=torch.float32, device=self.device)
            n_mapped += 1
        return _GroupParams(table, n_mapped)

    # ------------------------------------------------------------ rendering

    def _guide_for(self, level: _Level) -> torch.Tensor:
        """[3,H,W] sRGB in [0,1]: the photograph the coverage is allowed to follow.

        The recomposed image, not the albedo: the albedo has lost the shading that makes
        many part edges visible in the first place.
        """
        g = self._guides.get(level.size)
        if g is None:
            lin = (level.product + level.residual).clamp(0.0, 1.0)
            g = linear_to_srgb_t(lin).permute(2, 0, 1).contiguous()
            self._guides[level.size] = g
        return g

    def _coverage(self, level: _Level, indicator: torch.Tensor, sigma: float) -> torch.Tensor:
        """How much of each pixel belongs to a repainted group, in [0,1].

        The old rule was a Gaussian blur of ``indicator``, which falls below 1 *inside* the
        repainted part, so a band of it kept the original paint: a yellow motorcycle asked
        to become black came back with yellow outlines around every panel and vent. Three
        steps replace it:

        1. Snap the indicator onto the photograph's own edges with a colour guided filter,
           so a pixel one step inside the paint reads as paint even where the label map is
           a pixel off, and a genuinely half-covered pixel reads as half.
        2. Take the maximum with the hard label. Coverage may only ever add to what the
           user selected, never un-paint it.
        3. Spend the remaining soft edge entirely *outside* the paint: blur, then remap
           through a smoothstep whose upper end sits on the boundary, so coverage is full
           at the label edge and reaches zero a couple of pixels out.

        ``sigma`` is the outward ramp's width in pixels at this level's resolution.
        """
        m = indicator
        if COVERAGE_RADIUS > 0:
            m = filters.guided_filter_color(self._guide_for(level), m,
                                            COVERAGE_RADIUS, COVERAGE_EPS).clamp(0.0, 1.0)
            m = torch.maximum(m, indicator)
        if sigma > 0.0:
            g = _feather_chw(m[None], sigma)[0]
            t = ((g - RAMP_LO) / (RAMP_HI - RAMP_LO)).clamp(0.0, 1.0)
            m = t * t * (3.0 - 2.0 * t)
        return m

    def _recolored_albedo(self, level: _Level, params: _GroupParams, options: RenderOptions
                          ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
                                     Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """-> (albedo' [H,W,3] linear, mapped weight m [H,W] or None, target Lab field
        [H,W,3] or None, the (bounce, src_ab) fields for :meth:`_retint_shading` or None).
        ``m`` is the feathered fraction of each pixel that belongs to a repainted group."""
        if params.n_mapped == 0:
            return level.albedo, None, None, None
        sigma = RAMP_SIGMA * max(float(options.feather_px), 0.0) / RAMP_FEATHER_REF
        table = params.table[level.group_map]                       # [H,W,13], hard
        hard = table[..., 0]
        chw = table.permute(2, 0, 1).contiguous()
        # Colour parameters blend with a narrow Gaussian so two adjacent repaints cross-fade,
        # plus a wide low-weight fill so they are still defined just outside a group, where
        # the coverage below now reaches.
        near = _feather_chw(chw, PARAM_FEATHER) if PARAM_FEATHER > 0 else chw
        if PARAM_FAR_WEIGHT > 0:
            far_r = max(2 * COVERAGE_RADIUS, int(2.0 * sigma) + 2, 4)
            field = (near + PARAM_FAR_WEIGHT * filters.box_filter(chw, far_r)).permute(1, 2, 0)
        else:
            field = near.permute(1, 2, 0)
        m = self._coverage(level, hard, sigma)
        inv = (1.0 / field[..., 0].clamp_min(1e-8))
        inv1 = inv[..., None]
        texture = field[..., 1] * inv
        rot = field[..., 2:6] * inv1                                  # s * R(theta), row-major
        bounce = field[..., 6:10] * inv1
        t_L = field[..., 10] * inv
        a_L = field[..., 11] * inv
        slope_dn = field[..., 12] * inv
        c_ab = field[..., 13:15] * inv1
        t_ab = field[..., 15:17] * inv1
        src_ab = field[..., 17:19] * inv1
        # Two-sided lightness map anchored on the group's own L (see _group_params).
        d = level.albedo_lab[..., 0] - a_L
        slope = texture * torch.where(d >= 0, torch.ones_like(slope_dn), slope_dn)
        ab = level.albedo_lab[..., 1:]
        ab_new = torch.stack((rot[..., 0] * ab[..., 0] + rot[..., 1] * ab[..., 1],
                              rot[..., 2] * ab[..., 0] + rot[..., 3] * ab[..., 1]), dim=-1) + c_ab
        lab_new = torch.cat(((t_L + slope * d)[..., None], ab_new), dim=-1)
        target_lab = torch.cat((t_L[..., None], t_ab), dim=-1)
        lin_new = lab_to_linear_gamut_t(lab_new)
        mm = m[..., None]
        albedo = level.albedo + mm * (lin_new - level.albedo)
        return albedo, m, target_lab, (bounce, src_ab)

    def _retint_shading(self, shading: torch.Tensor, m: torch.Tensor, bounce: torch.Tensor,
                        src_ab: torch.Tensor) -> torch.Tensor:
        """Recolour the old paint's bounce light on repainted pixels (rule 6 above).

        ``bounce`` [H,W,4] is the per-pixel ``min(1, C_T/C_A) * R(theta)`` matrix and
        ``src_ab`` [H,W,2] the source paint's chroma, both blended group fields. The
        light's tint beyond the scene's illuminant is measured in Lab at a fixed
        luminance; its component along the old paint's chroma direction (only where it
        points *toward* that paint: light tinted the other way is not its bounce) is
        replaced by the same component mapped to the new paint. Luminance is preserved
        exactly, and a neutral source (no chroma direction) leaves the light untouched.
        """
        lum = luminance_t(shading)
        col = shading / lum.clamp_min(1e-6)[..., None] * _LIGHT_LUM
        lab = linear_to_lab_t(col)
        ref_ab = linear_to_lab_t(self._light_reference() * _LIGHT_LUM)[1:]
        d = lab[..., 1:] - ref_ab
        c_src = torch.linalg.norm(src_ab, dim=-1)
        u = src_ab / c_src.clamp_min(1e-6)[..., None]
        t = ((c_src - 2.0) / 6.0).clamp(0.0, 1.0)
        confidence = t * t * (3.0 - 2.0 * t)                          # as _hue_confidence
        p = ((d * u).sum(-1).clamp_min(0.0) * confidence)[..., None]
        pu = p * u
        turned = torch.stack((bounce[..., 0] * pu[..., 0] + bounce[..., 1] * pu[..., 1],
                              bounce[..., 2] * pu[..., 0] + bounce[..., 3] * pu[..., 1]), dim=-1)
        lab_new = torch.cat((lab[..., :1], lab[..., 1:] - pu + turned), dim=-1)
        col_new = lab_to_linear_gamut_t(lab_new)
        col_new = col_new / luminance_t(col_new).clamp_min(1e-6)[..., None] * lum[..., None]
        w = (m * (lum > 1e-6).to(m.dtype))[..., None]
        return shading + w * (col_new - shading)

    def _render_linear(self, level: _Level, mapping: Mapping, options: RenderOptions) -> torch.Tensor:
        t0 = time.perf_counter()
        params = self._group_params(mapping, options)
        strength = float(options.shading_strength)
        with torch.no_grad():
            albedo, m, target_lab, relight = self._recolored_albedo(level, params, options)
            untouched = m is None

            if abs(strength - 1.0) < 1e-6:
                shading = level.shading
            else:
                piv = self._shading_pivot
                shading = piv * (level.shading / piv).clamp_min(0.0).pow(strength)
                untouched = False
            if relight is not None:
                shading = self._retint_shading(shading, m, *relight)

            if untouched:
                out = level.product + level.residual if options.keep_residual else level.product
            else:
                product = albedo * shading if (m is not None or shading is not level.shading) else level.product
                out = product
                if options.keep_residual:
                    out = out + self._adjusted_residual(level, product, m, target_lab, options)
            out = out.clamp(0.0, 1.0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return out

    def _adjusted_residual(self, level: _Level, product: torch.Tensor, m: Optional[torch.Tensor],
                           target_lab: Optional[torch.Tensor], options: RenderOptions) -> torch.Tensor:
        res = level.residual
        pos = res.clamp_min(0.0)
        neg = res - pos
        # Negative residual encodes sensor clipping of albedo*shading > 1: scale it with
        # the new product so a darker repaint does not punch holes into highlights.
        ratio = (product / level.product.clamp_min(1e-6)).clamp(0.0, 1.0)
        neg = neg * ratio
        if m is not None:
            # The positive residual is not pure specular: on a saturated surface it also
            # carries diffuse energy the decomposition could not explain, in the colour of
            # the original paint. Added back untouched it survives any repaint, which is
            # why a bright red part asked to become black used to land on muddy maroon.
            # Split it: the coloured excess over its own achromatic floor is old paint and
            # must follow the repaint; the neutral floor is the lamp's own light, kept
            # where it is a glint and attenuated where it is only a veil.
            gray = pos.amin(dim=-1, keepdim=True)
            excess = pos - gray
            lum_new = luminance_t(product)[..., None]
            follow = (lum_new / luminance_t(level.product)[..., None].clamp_min(1e-6)).clamp(0.0, 1.0)
            spec = level.spec_weight()
            # The coloured excess keeps its energy but wears the new paint: rebuilt as a
            # multiple of the repainted product it vanishes on a black target and turns
            # neutral on a white one, instead of staying red under every new colour.
            e_lum = luminance_t(excess)[..., None]
            pos_new = gray * (spec + (1.0 - spec) * follow) + product * (e_lum / lum_new.clamp_min(1e-6))
            pos = pos + m[..., None] * (pos_new - pos)
        tint = float(np.clip(options.residual_tint, 0.0, 1.0))
        if tint > 0.0 and m is not None and target_lab is not None:
            # Tint the highlight toward the target's hue/chroma at the highlight's own
            # lightness; gamut compression keeps bright speculars bright.
            pos_lab = linear_to_lab_t(pos)
            tinted_lab = torch.cat((pos_lab[..., :1], target_lab[..., 1:]), dim=-1)
            tinted = lab_to_linear_gamut_t(tinted_lab)
            w = (tint * m)[..., None]
            pos = pos + w * (tinted - pos)
        return pos + neg

    @staticmethod
    def _finish(lin: torch.Tensor) -> np.ndarray:
        u8 = (linear_to_srgb_t(lin) * 255.0 + 0.5).to(torch.uint8)
        return u8.cpu().numpy()


def render_once(albedo_lin: np.ndarray, shading_lin: np.ndarray, residual: np.ndarray,
                group_map: np.ndarray, groups: Sequence[ColorGroup], mapping: Mapping,
                options: Optional[RenderOptions] = None) -> np.ndarray:
    """One-shot render (uint8 sRGB) that builds a :class:`Renderer`, renders at the
    native resolution and frees the GPU memory again."""
    r = Renderer(albedo_lin, shading_lin, residual, group_map, groups)
    try:
        return r.render(mapping, options or RenderOptions())
    finally:
        r.free()


def recompose(albedo_lin: np.ndarray, shading_lin: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """uint8 sRGB of ``albedo * shading + residual`` computed in numpy exactly the way
    the renderer's identity path does it (the reference for tests)."""
    lin = np.clip(albedo_lin.astype(np.float32) * shading_lin.astype(np.float32) + residual.astype(np.float32), 0, 1)
    return imageio.to_uint8(imageio.linear_to_srgb(lin))
