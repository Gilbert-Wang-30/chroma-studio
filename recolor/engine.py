"""Recoloring engine: edits the albedo layer of an intrinsic decomposition on the GPU.

The engine never re-lights a photograph. It takes the linear albedo, shading and
residual layers plus a group map, repaints the selected groups in OKLab, and recomposes
``sRGB(albedo' * shading' + residual')``. Everything per pixel is torch on
``config.device()``; a render never round-trips through skimage.

Colour space. Pixels and group colours live in OKLab (Ottosson 2020), computed from the
intrinsic pipeline's 2.2-gamma linear RGB exactly as the project's metrics measure it.
CIELAB's hue lines bend toward purple on desaturated blues: the shipping CIELAB engine
turned the yellow BMW's navy repaint violet (worst lightness bin 23 deg off the target),
OKLab keeps it on the target hue. ``ColorGroup.albedo_lab`` (CIELAB, from the
segmentation stage) and target hexes are converted with :func:`cielab_to_oklab` /
:func:`hex_to_oklab`. CIELAB chroma thresholds were translated with the measured
near-neutral scale (OK chroma ~ CIELAB chroma / 350, see ``OK_PER_LAB_CHROMA``).

Rules, in render order. Each exists because its absence was a visible defect.

1. **Coverage** (:meth:`Renderer._coverage`): how much of each pixel belongs to a
   repainted group. A symmetric blur of the mapped indicator dips below 1 *inside* the
   part and left a rim of the old paint around every panel (a yellow bike painted black
   came back with yellow outlines). The indicator is snapped to the photograph's own edges
   with a colour guided filter, maxed with the hard label, ramped outward only
   (``feather_px`` scales the ramp) and maxed with the hard label once more, because the
   ramp alone left 1-3 px slivers of a label partly unpainted (a red dot on a navy tank).
   The ramp only enters a pixel to the extent that its albedo still holds the old paint
   (its projection between its own group's colour and the paint's, :meth:`Renderer._ramp_gate`):
   once the analysis snapped the labels onto the real edges, the ungated ramp painted the
   first 1-3 px of every neighbour, a grey contour around the BMW's white tail lens and a
   haze along every silhouette against a white backdrop. Where the photograph's own edge is
   soft (a defocus blur, :meth:`Renderer._soft_edges`) the ramp is let through anyway: the
   albedo is sharper than the photo there, and the gated repaint of the Exia's defocused feet
   read as a hard cut-out. Decal islands carved out of the paint by the analysis stage are
   "do not enter": the ramp never covers an island pixel unless the island's own group is
   mapped. For a neutral source the island pixels that hold the paint in their albedo, connected
   to its labels and no lighter than it in the photo, are repainted with it
   (:meth:`Renderer._island_entry`, as far as the mapping's neutral weight goes): a decal on
   white paint is dark or coloured, so the white in it is the paint's (the leather the analysis
   took into the sneakers' FILA letters stayed pale speckle on navy), while a reflection in a
   mirror's glass is lighter than the paint.
2. **Albedo shift** (:meth:`Renderer._repaint`): ``ab' = T_ab + s R(theta) (ab - A_ab)``
   with ``s = min(1, C_T / C_A)`` and ``R(theta)`` the turn from the source hue to the
   target's. The deviation from the group colour is the paint's texture in the source's
   a/b frame; unrotated it lands beside the new colour (a red tank painted navy came out
   mauve, its white decal cyan). Lightness uses a map anchored on the group's own
   lightness, ``L' = T_L + slope (L - A_L)``, on a CIELAB-like lightness of OK L (cube
   root with CIELAB's linear toe near black): pure OK L crushed black repaints. Below the
   anchor the slope is ``(T - black) / (A - black)``; above it 1, except for dark targets
   (``DARK_SLOPE_REF``), where the toe turned a few L* of albedo texture into grey
   blotches on black. At a sensor-clipped highlight only the radial part of the deviation
   is kept: there the albedo is 27-40 deg magenta-shifted, which the rotation carried into
   navy as a cyan fringe. For a neutral source (see *Neutral sources* below) the slope above
   the anchor is NEUTRAL_UP_K of the room ratio below it (0: flat): nothing on a white paint is
   lighter than the paint, so what lies above its anchor is light the decomposition left in the
   albedo, which the shading applies again.
3. **Gamut** (:func:`oklab_to_linear_gamut_t`): out-of-gamut colours lose chroma at fixed
   OK lightness and hue instead of being clipped per channel, so bright repaints do not
   blow out and dark ones do not go muddy.
4. **Shading strength**: ``pivot * (shading / pivot) ** strength`` per channel, pivot the
   median shading, so contrast changes without changing exposure or the light's colour.
5. **Bounce light** (:meth:`Renderer._retint_shading`): the decomposition leaks part of a
   saturated surface into its shading, most in shadows (the Ducati tank's shadows are lit
   five times redder than blue). Per pixel, the light's tint beyond the scene illuminant
   (a white balance on well-lit neutral surfaces) is split along the old paint's chroma
   direction and the aligned part is turned and scaled like the albedo; removing only a
   group median left a navy tank's shadows teal. At a clipped highlight the light is the
   lamp's own, so it goes to the illuminant.
6. **Residual** (:meth:`Renderer._adjusted_residual`): the positive residual also
   carries diffuse energy in the old paint's colour (red painted black came out maroon).
   It is split at its achromatic floor: the coloured excess is rebuilt as a multiple of the
   repainted product (for a darker target on the repainted labels with only the square root
   of the product's luminance drop taken off its energy: kept whole, the yellow BMW's
   residual lifted a #1b2a57 navy to a medium royal blue), the neutral floor is kept where
   it is a glint and faded where it is a veil. Within 2 px of a group boundary the floor may not exceed the interior's: on mixed
   boundary pixels it is decomposition error, and kept as a glint it drew a light hairline
   along every silhouette of a dark repaint. The negative residual (sensor clipping) is
   scaled with the new product; at a clipped highlight it is dropped, because re-applying
   it where the new paint is brighter than the old tinted the highlight cyan.
   ``residual_tint`` optionally pulls what survives toward the target hue. For a neutral
   source the neutral floor is neither glint nor veil but the white paint's own diffuse energy, and
   the coloured excess is the light's colour: both are rebuilt in the new paint like the product
   (FLOOR_DIFFUSE_GAMMA, NEUTRAL_EXCESS_GAMMA).
7. **Paint envelope** (:meth:`Renderer._paint_envelope`), on the composed repainted
   pixels: (a) a target darker than the darkest real paint gets that paint's reflectance
   (0.24 %) lit by ``S (S / S_med) ** 0.6``, so black keeps its form without a charcoal
   lift; (b) the repaint may not fall below the photo's white specular, split off
   dichromatically against how white the paint itself looks *at that brightness* (a lit
   saturated paint is also whiter: with one ratio per group the BMW's lit fairing read as
   white marble; the instances of a part split by instance share the part's measurement,
   :meth:`Renderer._pools`), cleaned by the edge band, a min-blur and a soft threshold so it never
   grows the photo's highlight footprint; (c) the OK hue is softly held within 4 deg of the
   target (the white of (b) reads purple on navy in OKLab) and chroma is capped at
   ``1.25 C_T``; (d) the shadows of a light target keep a chroma proportional to their
   lightness (pastel shadows went slate). In the outward coverage ramp the envelope is
   weighted by rule 8's permission, so it never pulls a protected neighbour toward the
   target. For a neutral source, (a') before the black floor, the exposure bound: where the
   photo is white no repaint reflects more than a paint of its albedo under the light a white
   paint implies there (photo / WHITE_REF_Y), as far as the group's photo shows white paint (the
   lighter its albedo, the larger the white share it takes, EXPOSURE_*); a small clipped spot on
   paint that is not white is no source of that light (:meth:`Renderer._small_clips`: the
   Unicorn's sun streak sat in a dark pad); in (b) a neutral group's own saturated-paint
   estimate is scaled by 1 - its neutral weight (the light's colour made a warm-lit off-white pass
   as a saturated paint); and (e) its own glints, small sensor-clipped white spots that stand out
   from the lit paint around them and that the shading layer does not explain (GLINT_*,
   :meth:`Renderer._white_glints`), are added back on its labels as the lamp's white light, the
   clipped core at the photo's brightness, each with a weight that is whole for a spot that passes
   the tests' old hard cuts and dims below them, so a repainted white paint keeps its gloss (the
   Alpine's roof streak stays white) and a glint never switches on or off at once.
8. **Reflections** (:meth:`Renderer._recolor_reflections`): a red tank shows in the
   chrome fork, the caliper and the floor; repainting only the tank left those red. Every
   pixel *outside* the repainted groups that still shows the old paint gets the paint's own
   OK hue turn and chroma scale at fixed OK lightness. The weight is a soft gate: OK chroma
   (half weight at CIELAB 12), OK hue within 30 deg of the source, distance to the paint
   (full within 70 px, none beyond 140; a source below ``REFL_SMALL_FRAC`` of the image
   reaches only ``REFL_REACH_K`` x the square root of its area from itself, so painting the
   Ducati's shock spring alone no longer tinted the reflections of its gold frame 86 px
   away; groups repainted from one colour to one target, the instances of a split part, are
   one source), one minus the coverage, darker than the paint's
   p90 lightness (a bright red sticker next to red paint is not a reflection; decal islands
   are exempt), the analysis stage's ``protect`` mask (an object of that colour in its own
   right), and a per-group permission: repainted, locked and unmapped groups with a real
   colour (CIELAB chroma >= 18) never move, nor does an unmapped group (CIELAB L >= 30)
   at least half of whose photo pixels pass this very colour test: it is an object of that
   colour, not a reflection (the Exia's warm backdrop behind its repainted gold parts went
   grey and blue, the Sazabi's sand wall grew blue blotches). The hue window stops 20 deg short of every
   unmapped coloured group's hue (a reflection of gold wheels is not one of the red), and
   closes altogether when such a group has the source's own hue and at least
   ``REFL_SAME_HUE_AREA`` of its size (every pixel of that hue may be that object's
   reflection: the gold frame beside the gold spring; a crumb of the hue does not close it);
   locked groups (another material in the paint's hue, a gold caliper) protect a 24 px
   neighbourhood instead, because as hues they closed the BMW's window completely.
   Candidates within 5 px of the paint get a 60 deg window: a pixel touching the paint is
   a mix with it (a red-to-white decal edge passes through orange). Inside a decal island
   the old paint's excess over the white floor is rebuilt in the target colour. The result
   is held within 8 deg of the target hue (a reflection off the paint's hue landed violet)
   and a partial weight fades the old chroma out before the new one comes in (a linear mix
   of red and navy is purple). A neutral source reflects with weight 1 - its neutral weight: a
   white paint's colour cast is the light's, not a paint hue to follow; and every source reflects
   with its hue confidence (0 below CIELAB chroma 2, 1 from 8): a near-neutral paint's reflections
   come in with its hue instead of switching on at full weight at chroma 2 (a saturated source is
   unchanged).
9. **Identity**: with an empty mapping the output equals
   ``linear_to_srgb(albedo * shading + residual)`` bit for bit; unmapped and locked groups
   are untouched except for rule 8 on unmapped *neutral* groups near the paint and, for a
   neutral source, rule 1's decal-island pixels that hold its paint (never a locked group's).

**Neutral sources** (white, off-white, a light grey; ``NEUTRAL_*``): on a white paint the
decomposition's leftover is achromatic, so the rules above, written for saturated paints, read
every bit of it as a lamp glint or a veil and kept it white: a navy repaint of a white Gundam came
out cornflower on every lit face with the target only in the shadows, black as grey marble, a
pastel blown to white. The leftover is the white paint's own diffuse light, so a neutral source
rebuilds it in the new paint (rules 2, 6, 7a'), casts no reflections (rule 8) and gets back only
its real glints (rule 7e). A source's neutral weight is 1 up to CIELAB chroma NEUTRAL_C0 and 0 from
NEUTRAL_C1 (18, a real colour), times a light-enough ramp (its albedo lightness, or the share of its
photo that is white: a dark grey part's neutral floor is its sheen), times a white-paint gate: a
grey that glints (the share of its pixels that clip or hold a strong neutral residual, as the
analysis stage counts ``Region.glint``, but counted softly around its two cuts) is metal or a gloss,
whose floor is its reflections, and keeps the rules above unless its albedo or its photo is white
enough to be white paint all the same (a group the analysis tagged chrome is never neutral). Rule 8
also brings every source's reflections in with its hue confidence (CIELAB chroma 2 to 8), so a
near-neutral source that is not white paint (a dark or glossy grey) reflects in proportion to the
hue it has; from chroma 8 rule 8 is the old one. The weight is a group parameter, blended per
pixel, and every neutral-source rule is weighted by it, so a saturated source renders bit for bit as
if the neutral rules did not exist; per mapping it also fades out as the target nears the source
colour (NEUTRAL_NEAR_DE*), so a map to a group's own swatch colour renders as the saturated-paint
rules do, near the photo. Every ramp is a smoothstep: the render changes continuously with the
source's colour, white share and gloss and with the target. On white paint the layers cannot tell a
lamp's glint from a lit face in general (both are white: every local-contrast and residual test that
kept the real glints white also kept lit bevels, a leather crease and a rim light white), so rule 7e
keeps only what is unambiguous: a small clipped white spot that stands out from its own lit paint and
is not in the shading layer (a lit bevel is). Broad, unclipped reflections of a white paint (the sky
on a white car's flank) are repainted with it; the glints of a saturated source are kept exactly as
before. A full-resolution export takes the preview's glints and neutral weights
(:meth:`Renderer.white_glints`, :meth:`Renderer.neutral_weights`), so it treats every group as the
preview did.

Pixel distances (falloff, edge bands, rims, the highlight area) are defined at the
reference resolution, the layers' own unless ``reference_long_side`` is given, and scale
with the render size, so a preview, a working-resolution render and a full-resolution
export treat the same part of the photograph the same way.
"""
from __future__ import annotations

import functools
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import config, filters, imageio
from .types import ColorGroup, Mapping, RenderOptions

GAMMA: float = imageio.GAMMA

# ----------------------------------------------------------------- OKLab (Ottosson 2020)

_OK_M1 = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
_OK_M2 = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
_OK_M2_INV = (
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)
_OK_M1_INV = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)
#: Out-of-gamut tolerance of the chroma compression (linear RGB units).
_GAMUT_TOL = 2e-3

#: OK chroma per CIELAB chroma unit near neutral (measured 1/340..1/365 on the two
#: motorcycles' albedo at CIELAB C < 12). Translates the CIELAB thresholds below.
OK_PER_LAB_CHROMA = 1.0 / 350.0
#: Hue confidence ramp: a colour has no hue below CIELAB chroma 2, a full one from 8.
HUE_C0, HUE_C1 = 2.0 * OK_PER_LAB_CHROMA, 8.0 * OK_PER_LAB_CHROMA
#: Illuminant estimate: neutral surfaces (CIELAB chroma <= 6 or the 40th percentile),
#: well lit (CIELAB L > 25, i.e. OK L > 0.35).
ILLUMINANT_C_MIN = 6.0 * OK_PER_LAB_CHROMA
ILLUMINANT_L_MIN = 0.35
#: Luminance the light's colour is normalised to before its tint is measured.
LIGHT_LUM = 0.2

# The anchored lightness map works on f(L) = L above CIELAB's toe and
# (903.3 L^3 + 16) / 116 below it (CIELAB's lightness curve applied to OK L^3), black at 16/116.
_TOE_L0 = 0.2068966          # cbrt(0.008856), where CIELAB's cube root meets its toe
_TOE_K = 903.3
TOE_BLACK = 16.0 / 116.0
#: Dark targets: the slope above the anchor is min(1, (T - black) / DARK_SLOPE_REF).
DARK_SLOPE_REF = 0.2

# ----------------------------------------------------------------- coverage (rule 1)

#: Colour guided filter that snaps the mapped indicator to the photo's edges. A small
#: window only has to see both sides of one edge; a larger one drags in a third colour.
COVERAGE_RADIUS = 2
COVERAGE_EPS = 1e-5
#: Outward ramp: blur, then a smoothstep from RAMP_LO to RAMP_HI (full coverage on the
#: label boundary, zero about 1.3 sigma outside it). RAMP_SIGMA px at feather RAMP_FEATHER_REF.
RAMP_LO, RAMP_HI = 0.10, 0.50
RAMP_SIGMA, RAMP_FEATHER_REF = 2.5, 1.5
#: The ramp is gated by how much of the old paint each pixel's albedo holds: its projection
#: alpha on the line from its own group's colour to the paint's (toe lightness, OK a, b). A
#: pixel that is its own group's colour (alpha <= RAMP_GATE_A0) gets none of the ramp, one
#: holding RAMP_GATE_A1 of the paint or more all of it. Ungated, the ramp painted the first
#: 1-3 px of every neighbour: a grey contour around the BMW's white tail lens and a blue
#: haze along every silhouette against the white backdrop once the labels were snapped.
RAMP_GATE_A0, RAMP_GATE_A1 = 0.03, 0.15
#: ... unless the two colours are too close to tell apart (distance below RAMP_GATE_SEP0 in
#: those units, fading to the gate by RAMP_GATE_SEP1): then the ramp is kept.
RAMP_GATE_SEP0, RAMP_GATE_SEP1 = 0.02, 0.05
#: ... and where the photograph's own edge is soft (a defocus blur), the ramp is let through
#: whatever the albedo says: the decomposition sharpens the albedo there, so the gate cut the
#: ramp and a repainted part read as a hard, stair-stepped cut-out on a blurred background
#: (the Exia's defocused feet). Edge width (px at the reference resolution) = the local
#: range of the photo's sRGB over a (2 SOFT_EDGE_RADIUS + 1)^2 window divided by the largest
#: 1-px step in it, on the channel of largest range; the ramp is open from SOFT_EDGE_W0 to
#: SOFT_EDGE_W1 px, where the range is at least SOFT_EDGE_R0..R1 (an edge at all). Measured
#: at the repaint boundaries: BMW and Ducati studio shots median 1.5 px (90 % below 2.1),
#: Sazabi 2.1, Exia 2.6. Two limits keep this to defocused edges: the width measured over
#: the wider (2 SOFT_EDGE_WIDE_RADIUS + 1)^2 window closes the gate again from SOFT_EDGE_W2
#: to SOFT_EDGE_W3 px (a shading falloff or a chrome specular next to the paint is a smooth
#: gradient far wider than any defocus blur), and the gate only opens within SOFT_EDGE_BAND
#: px of a group boundary, which is where the ramp lives.
SOFT_EDGE_RADIUS = 3
SOFT_EDGE_W0, SOFT_EDGE_W1 = 2.0, 3.0
SOFT_EDGE_R0, SOFT_EDGE_R1 = 0.04, 0.08
SOFT_EDGE_WIDE_RADIUS = 8
SOFT_EDGE_W2, SOFT_EDGE_W3 = 6.0, 9.0
SOFT_EDGE_BAND = 3
#: Colour parameters blend with a narrow Gaussian, plus a wide low-weight fill so they are
#: defined just outside a group, where the ramp reaches.
PARAM_FEATHER, PARAM_FAR_WEIGHT = 1.2, 0.02

# ----------------------------------------------------------------- highlights (rules 2, 5, 6)

#: A photo channel at or above 0.98 sRGB is sensor-clipped.
CLIP_LIN: float = 0.98 ** GAMMA
#: A clipped pixel is a highlight only with a white component of at least this (linear)
#: in a component of at least HL_MIN_AREA px: most of the BMW's clipped pixels are chroma
#: clipping of the yellow itself, whose colour belongs to the paint.
HL_MIN_WHITE = 0.03
HL_MIN_AREA = 20

# ----------------------------------------------------------------- gloss / envelope (rule 7)

#: Groups whose median min/max channel ratio exceeds this are not saturated paints (no gloss floor).
GLOSS_RATIO_MAX = 0.5
#: Noise floor subtracted from the white estimate (linear).
GLOSS_W0 = 0.005
#: How white the paint itself is: this quantile of min/max per group and per brightness bin.
GLOSS_RATIO_Q = 0.75
GLOSS_BINS = 12
GLOSS_BIN_MIN_PX = 50
#: Boundary band (px) where the white estimate and the residual floor may not exceed the interior's.
EDGE_BAND_PX = 2
#: Rule 6: on the repainted labels, the coloured residual excess of a darker target keeps
#: follow ** EXCESS_FOLLOW_GAMMA of its energy (follow = new / old product luminance, <= 1).
EXCESS_FOLLOW_GAMMA = 0.5
#: Cleaning of the white estimate: W = min(W, gauss(W, sigma)), then a soft threshold.
GLOSS_SMOOTH_SIGMA = 1.0
GLOSS_T0, GLOSS_T1 = 0.004, 0.02
#: OK hue deviations up to ENV_HUE_TOL0 deg are kept, larger ones compress toward ENV_HUE_TOL1.
ENV_HUE_TOL0, ENV_HUE_TOL1 = 2.0, 4.0
#: OK chroma cap = ENV_CHROMA_K * C_target + ENV_CHROMA_C0 (soft knee at 80 %).
ENV_CHROMA_K, ENV_CHROMA_C0 = 1.25, 0.003
#: Light targets (OK L 0.5 -> 0.7): a pixel darker than the target keeps K * C_T * L / L_T chroma.
ENV_SHADOW_CHROMA = 0.9
ENV_SHADOW_L0, ENV_SHADOW_L_RAMP = 0.5, 0.2
#: Diffuse floor of a dark target (linear reflectance), lit by S * (S / S_med) ** (gamma - 1).
BLACK_FLOOR = 0.0024
BLACK_FLOOR_GAMMA = 1.6

# ----------------------------------------------------------------- neutral sources (rules 1, 2, 6, 7, 8)

#: A source paint is *neutral* (white, off-white, a light grey) up to CIELAB albedo chroma NEUTRAL_C0
#: and a real colour from NEUTRAL_C1 (the chroma the analysis and rule 8 call a real colour); in
#: between its neutral weight ramps down (smoothstep). Every neutral-source rule below is weighted by
#: it, per pixel and blended like the other group parameters, so a saturated source (the Ducati, the
#: BMW, every group the halo harness paints) renders bit for bit as if the rules did not exist. On a
#: white paint the decomposition's leftover is achromatic, and the rules written for saturated paints
#: read all of it as a lamp glint or a faint veil: a navy repaint of a white Gundam came out cornflower
#: on every lit face (the target only in the shadows), black as grey marble, a pastel blown to white.
NEUTRAL_C0, NEUTRAL_C1 = 10.0, 18.0
#: ... and only for a paint that is light enough: the weight is also multiplied by the larger of a
#: ramp of the albedo lightness (CIELAB, NEUTRAL_L0 to NEUTRAL_L1) and a ramp of the share of the
#: group's photo pixels that are white (the white test of the exposure bound below, NEUTRAL_WHITE_S0
#: to S1). A white paint shows white lit faces even where the decomposition gave it a grey albedo (the
#: Unicorn's sun-and-shade armour, albedo L 43-51, 9-35 % of it white in the photo); a dark grey part
#: shows only its reflections (the Ducati's grey frame parts, L 47, 3 %: their streaks sit in the
#: shading and the floor, and keep their white under the saturated-paint rules), and a black paint's
#: neutral floor is its sheen. A group the analysis tagged chrome (``ColorGroup.finish``: it glints or
#: clips on a quarter of its pixels with a mid-grey albedo, see materials.chrome_regions) is never
#: neutral: its neutral floor is its reflections (the Ducati's sprocket, the BMW's silver, a window).
NEUTRAL_L0, NEUTRAL_L1 = 45.0, 60.0
NEUTRAL_WHITE_S0, NEUTRAL_WHITE_S1 = 0.04, 0.10
#: ... and a glossy grey keeps the saturated-paint rules unless it shows white paint. A mid-grey or silver
#: group that glints (the share of its pixels that are sensor-clipped or hold a neutral residual above half
#: the image's 99th percentile: the analysis stage's ``Region.glint``, measured again on the layers so a job
#: analysed before that cue gets it too, and counted softly, see GLINT_PX_*) from NEUTRAL_GLOSS_G0 to G1 is
#: metal or a gloss whose neutral floor is its reflections: under the white-paint rules the BMW's cast fork
#: leg and brake disc (L 64, 30 % glints) read as matte plastic on red (their top 10 % fell from OK L 0.80 to
#: 0.49), a white-painted concrete wall (L 69, 27-29 %) lost its texture, and every one of them turned into a
#: flat blob on black. That fade is itself undone by evidence of white paint, the larger of a ramp of the
#: albedo lightness (NEUTRAL_PAINT_L0 to L1) and one of the photo's white share (NEUTRAL_PAINT_S0 to S1): the
#: RX-78's white armour glints on 27-35 % of its pixels (L 85, 56 % white) and is white paint all the same.
#: The weight is the base weight above times 1 - (1 - white evidence) x gloss.
NEUTRAL_GLOSS_G0, NEUTRAL_GLOSS_G1 = 0.12, 0.25
#: The glint share counts each pixel softly around the analysis stage's two cuts, as the larger of a ramp of
#: its largest photo channel from GLINT_PX_CLIP0 to GLINT_PX_CLIP1 (sRGB, around CLIP_LIN's 0.98) and one of
#: its neutral residual over the image's 99th percentile from GLINT_PX_SPEC0 to GLINT_PX_SPEC1 (around 1/2):
#: counted with hard cuts, a streak whose residual sat at one value moved the whole share at once (a glossy
#: grey's weight went from 1 to 0, and its render by up to 81 levels, for a 0.005 change of the streak).
GLINT_PX_CLIP0, GLINT_PX_CLIP1 = 0.97, 0.99
GLINT_PX_SPEC0, GLINT_PX_SPEC1 = 0.3, 0.7
NEUTRAL_PAINT_L0, NEUTRAL_PAINT_L1 = 72.0, 82.0
NEUTRAL_PAINT_S0, NEUTRAL_PAINT_S1 = 0.10, 0.45
#: A target near the source colour repaints nothing much, and every neutral-source rule fades out as the
#: target approaches it: the weight of a mapping is the group's times a smoothstep of the CIELAB distance
#: (dE76) from the target to the source colour (the albedo the swatch shows), 0 up to NEUTRAL_NEAR_DE0 and
#: 1 from NEUTRAL_NEAR_DE1. The rules assume a white paint repainted to another colour: mapped to its own
#: swatch colour, the Unicorn's grey-albedo white armour lost 43 L* on its lit faces to the exposure bound
#: and every light grey its texture above the anchor; now such a map renders exactly as the saturated-paint
#: rules do (mean dE 1.4-5.3 from the photo on six real groups, as before the white-paint work).
NEUTRAL_NEAR_DE0, NEUTRAL_NEAR_DE1 = 2.0, 12.0
#: Rule 6 for a neutral source: the neutral floor of the positive residual is the white paint's own
#: diffuse energy (the white its albedo x shading did not explain; on the six white photos it is 2-50 %
#: of the photo, spread over every lit face and following the shading), not a glint: it is rebuilt in
#: the new paint as a multiple of the repainted product with follow ** FLOOR_DIFFUSE_GAMMA of its
#: energy (1: it scales like the diffuse light it is).
FLOOR_DIFFUSE_GAMMA = 1.0
#: ... and so is the coloured excess of its positive residual: on a white paint the excess over the
#: floor is the light's colour (the model ship's warm lamp, the sky on the sneakers), not a paint colour,
#: and on the repainted labels it follows like the product (follow ** NEUTRAL_EXCESS_GAMMA) instead of
#: the saturated paints' square root, which kept the warm-lit hull's navy 1.6x too light.
NEUTRAL_EXCESS_GAMMA = 1.0
#: Rule 2 for a neutral source: the lightness slope above the anchor is NEUTRAL_UP_K x the room ratio
#: (T - black) / (A - black) of the slope below it. Nothing on a white paint is lighter than the paint
#: itself, so what lies above its anchor is the light the decomposition left in the albedo, which the
#: shading then applies again: copied 1:1 (the saturated-paint rule) it lifted a navy repaint's lit
#: faces by 0.1-0.2 OK L and pushed a pastel past white.
NEUTRAL_UP_K = 0.0
#: Rule 7a' for a neutral source, the exposure bound: where the photo is white (OK L from EXPOSURE_L0 to
#: EXPOSURE_L1, OK chroma below EXPOSURE_C0..C1) the paint is a white paint, which reflects about
#: WHITE_REF_Y of its light, so no repaint of albedo A' reflects more than photo x A' / WHITE_REF_Y
#: there (a soft knee from EXPOSURE_KNEE of it). The light is read from the brightest white pixel of the
#: pixel's own group within EXPOSURE_RADIUS px, so a stripe or seam of another colour inside the white
#: is bounded by the white around it. The decomposition sometimes gives a sunlit white paint a grey
#: albedo (Y 0.33 on the Unicorn) under a shading of 1.9, and every light target then blew out to white.
#: A small clipped white spot (8-connected, up to GLINT_AREA2 px, faded out by GLINT_AREA3; with
#: GLINT_RING0 px around it) on paint that is not white in the photo (the mean white weight of its own group
#: GLINT_RING0 to GLINT_RING1 px around it below SMALL_CLIP_WHITE0, fading to kept at SMALL_CLIP_WHITE1, as far
#: as that mean rests on GLINT_STAT_PX0..1 px) is a glint or a speck, not the white paint's exposure, and is no
#: source of that light
#: (:meth:`Renderer._small_clips`): the Unicorn's sun streak on a grey face (photo Y 0.5, not white) switched
#: the bound on 4 px around it with a light of 1.0, and its navy sat in a pad at 0.57 of the face's
#: brightness. The clipping specks scattered over a lit white face (the RX-78's) are that face's own light.
WHITE_REF_Y = 0.8
EXPOSURE_L0, EXPOSURE_L1 = 0.80, 0.92
EXPOSURE_C0, EXPOSURE_C1 = 0.03, 0.06
EXPOSURE_KNEE = 0.85
SMALL_CLIP_WHITE0, SMALL_CLIP_WHITE1 = 0.3, 0.6
#: The bound assumes the paint is a white paint, so it acts only as far as the source's group shows
#: that in the photo: a smoothstep of its white share (see NEUTRAL_WHITE_S*), and the lighter the albedo,
#: the more of it: from EXPOSURE_S0 to EXPOSURE_S1 for an albedo of luminance EXPOSURE_Y0 or less, from
#: EXPOSURE_LIGHT_S0 to EXPOSURE_LIGHT_S1 from EXPOSURE_Y1 up, interpolated (smoothstep) in between. A dark
#: albedo under a white photo contradicts it (the Unicorn's shaded white armour, Y 0.12 and 9 % white, needs
#: the bound); a light grey whose photo whitens in a corner under a strong light may well be that grey
#: (4.8 % white: bounded as white paint it got a dark hole in the corner; as its albedo brightened from Y
#: 0.42 to 0.45 its white share went from 4 to 10 %, the old 0.04-0.10 ramp switched the bound on and its
#: pastel repaint darkened by 26 levels).
EXPOSURE_S0, EXPOSURE_S1 = 0.04, 0.10
EXPOSURE_LIGHT_S0, EXPOSURE_LIGHT_S1 = 0.20, 0.40
EXPOSURE_Y0, EXPOSURE_Y1 = 0.15, 0.45
EXPOSURE_RADIUS = 4
#: ... and how far the bound acts is a smoothstep of the share of white pixels of the pixel's own group
#: within EXPOSURE_RADIUS px, full from EXPOSURE_NEAR: one white pixel set it to full strength over its
#: whole 4 px disc, so every noise pixel of a face at the edge of white (the Unicorn's grey faces sit at OK
#: L 0.79-0.80) punched a dark disc into its repaint; a white face, and a seam or a stripe in it, still has
#: a white neighbourhood (spread 4 px further, the bound flattened the RX-78's partly white faces into
#: specks: 7x its speck count).
EXPOSURE_NEAR = 0.25
#: Rule 7e for a neutral source, its own glints (:meth:`Renderer._find_white_glints`). On white paint the
#: layers cannot tell a lamp's glint from a lit face in general (both white), but a *small sensor-clipped
#: white spot that stands out from its own lit paint and that the shading does not explain* is a glint: a
#: channel >= CLIP_LIN with OK chroma <= GLINT_C, in an 8-connected component of a neutral group. Each such
#: spot gets a weight, the product of these ramps (smoothsteps, so a glint dims instead of switching off as
#: the layers, the groups or the paint change a little; a hard cut rendered two glints 1.49x and 1.51x
#: their ring one white and one navy). Each ramp ends where the hard tests that first kept the real glints
#: cut, or just past the weakest glint they kept (8 px, 1.5x its ring, 1.44x the shading's step), so those
#: glints are kept whole and the ramp lies below them: centred on the cuts, the ramps halved the glints just
#: past them (the Alpine's roof streak, 1.54x its ring and 1.44x its shading's step, read as a pale blue
#: stripe on navy; the model ship's lamp dots dimmed to grey):
#:
#: * its area, from GLINT_AREA0 to GLINT_AREA1 px up and from GLINT_AREA2 to GLINT_AREA3 px down (a face is
#:   larger);
#: * its median luminance over the median of its own group's paint GLINT_RING0 to GLINT_RING1 px around it,
#:   the standout, from GLINT_STANDOUT0 to GLINT_STANDOUT1 up (a peak, not a clipped plateau of the paint:
#:   1.0-1.1x) and from GLINT_STANDOUT2 to GLINT_STANDOUT3 down (the glints measured 1.5-3.4x, the dimmer
#:   lamp dots of the model ship 1.43-1.49x; white lettering on a black backdrop, taken into the white
#:   paint's group, 50-17 000x), with at least GLINT_RING_PX0 to GLINT_RING_PX1 px of that ring;
#: * the standout over the shading's own step there (core over ring), from GLINT_UNEXPLAINED0 to
#:   GLINT_UNEXPLAINED1 (a lit bevel, crease or rim is in the shading layer: 0.7-1.2x on the sneakers'
#:   creases and the RX-78's lit vent rim and panel strip; the glints sit in the albedo or the residual:
#:   1.44-2.2x; the lit edges between, 1.28-1.4x on the Nu's and the Alpine's shoulder, read as pale lines
#:   and blotches when the ramp reached down to 1.2);
#: * the ring's albedo from GLINT_RING_DARKER0 to GLINT_RING_DARKER1 below the group's own lightness (the
#:   dark anti-aliased rim of a letter is another material; the rings of the glints sit within 0.11 below
#:   and 0.16 above it);
#: * the spot over the brightest lit paint around it, the 90th percentile of its group's unclipped pixels in
#:   that ring, from GLINT_LIT0 to GLINT_LIT1 (the RX-78's one kept "glint" was a clipped corner of a lit face
#:   whose ring fell mostly on the darker face below it: 1.10x the lit face's own brightness), as far as the
#:   shading steps up under the spot (its core's shading over its ring's median, GLINT_LIT_SHADE0 to
#:   GLINT_LIT_SHADE1: only then can the ring's median sit on a darker face; on evenly lit paint the bright
#:   pixels around a glint are its own halo, and the test painted the model ship's brightest lamp dot, 1.04x
#:   its halo with a shading 1.06x its ring's; the RX-78's corner sits on 1.36x) and as far as that
#:   percentile rests on GLINT_STAT_PX0 to GLINT_STAT_PX1 such pixels.
#:
#: The lamp's reflection survives any repaint, so it is added back on the repainted labels as white light:
#: the photo's brightness on the clipped core (the sensor's limit, so the whole core: scaling the core by its
#: own profile left its dimmer clipped pixels pale blue), and around it the photo's own profile over the
#: ring, normalised and raised to GLINT_PROFILE_GAMMA (the halo fades faster than the photo's excess: that
#: excess added 1:1 lifted the black next to a bevel glint to 16x the face where the photo was 1.08x), faded
#: out GLINT_FALLOFF_PX px from the clipped core, times the spot's weight. Without it a navy repaint of the
#: Unicorn lost its sun streak and the model ship its row of lamp reflections.
GLINT_C = 0.04
GLINT_AREA0, GLINT_AREA1 = 4.0, 8.0
GLINT_AREA2, GLINT_AREA3 = 120.0, 180.0
GLINT_RING0, GLINT_RING1 = 2.0, 6.0
GLINT_RING_PX0, GLINT_RING_PX1 = 4.0, 8.0
GLINT_STANDOUT0, GLINT_STANDOUT1 = 1.20, 1.50
GLINT_STANDOUT2, GLINT_STANDOUT3 = 3.5, 4.5
GLINT_UNEXPLAINED0, GLINT_UNEXPLAINED1 = 1.28, 1.44
GLINT_RING_DARKER0, GLINT_RING_DARKER1 = 0.26, 0.14
GLINT_LIT0, GLINT_LIT1 = 1.05, 1.20
GLINT_LIT_SHADE0, GLINT_LIT_SHADE1 = 1.08, 1.20
#: A statistic of a spot's ring (the lit test's brightest paint above, and the whiteness of the paint around a
#: small clipped spot, :meth:`Renderer._small_clips`) counts as far as it rests on GLINT_STAT_PX0 to
#: GLINT_STAT_PX1 px (at the reference resolution): with none it is left out, and one or two pixels decided it
#: outright (from 2 to 3 lit pixels the lit test switched on; one ring pixel set a small clip's whiteness).
GLINT_STAT_PX0, GLINT_STAT_PX1 = 2.0, 8.0
GLINT_PROFILE_GAMMA = 2.0
GLINT_FALLOFF_PX = 4.0
#: Rule 1 for a neutral source, decal islands (:meth:`Renderer._island_entry`): the island pixels that hold
#: the paint of the nearest repainted neutral label (their albedo's share of it from ISLAND_ENTRY_A0 to
#: ISLAND_ENTRY_A1, the decal's own colour clearly another), in a connected piece of them that comes within
#: ISLAND_TOUCH_PX px of that label and reaches at most ISLAND_REACH_PX px from it, and whose photo is no
#: lighter than the paint where the piece touches it (OK L, ISLAND_LIGHTER0 to ISLAND_LIGHTER1 above: a
#: white reflection in a mirror's glass next to the Alpine's body is no paint), are repainted with it, as
#: far as the mapping's neutral weight goes. Connectivity, not a distance, decides: the export's
#: edge-snapped labels closed the white paint's specks inside the FILA letters' decal, leaving leather 13 px
#: (working resolution) from the paint.
ISLAND_ENTRY_A0, ISLAND_ENTRY_A1 = 0.3, 0.8
ISLAND_TOUCH_PX = 1.5
ISLAND_REACH_PX = 24.0
ISLAND_LIGHTER0, ISLAND_LIGHTER1 = 0.04, 0.10
# Rule 7b for a neutral source: a neutral group's own saturated-paint white-specular estimate is scaled
# by 1 - its neutral weight (:meth:`Renderer._gloss_neutral`). An off-white under a warm lamp has a photo
# min/max of 0.38 (the light's colour, not the paint's), so it passed as a saturated paint and its
# whitest quarter was kept as "gloss" over the new colour.
# Rule 8 for a neutral source: its reflection weight is 1 - its neutral weight. A white paint's colour
# cast is the light's (sky blue on the sneakers, a warm lamp on the model ship), and taking it for the
# paint's hue recoloured the navy FILA letters and the ship's other hull groups as its "reflections".

# ----------------------------------------------------------------- reflections (rule 8)

#: Chroma gate, CIELAB units: half weight at REFL_CHROMA_CUT, smoothstep +- REFL_CHROMA_RAMP.
REFL_CHROMA_CUT, REFL_CHROMA_RAMP = 12.0, 4.0
#: OK hue window around the source paint (deg) and the width of its soft edge.
REFL_HUE_WINDOW, REFL_HUE_RAMP = 30.0, 10.0
#: The window stops this many degrees short of an unmapped coloured group's hue.
REFL_HUE_MARGIN = 20.0
#: Unmapped groups with at least this CIELAB chroma are coloured objects: never recoloured.
REFL_PROTECT_CHROMA = 18.0
#: Distance to the repainted parts (px): full weight within half, none beyond.
REFL_FALLOFF_PX = 140.0
#: ... and a repainted group below REFL_SMALL_FRAC of the image (a small part, not a panel)
#: reaches only REFL_REACH_K x the square root of its area from itself: a small part's
#: reflections are next to it. Painting the Ducati's shock spring (3.5k px, 0.2 % of the image)
#: alone turned the gold weave of the carbon panel and a frame tube's highlight 86 px away
#: pinkish-grey: reflections of the gold frame, not of the spring. (At the working resolution a
#: group of 1 % reaches 125 px, close to REFL_FALLOFF_PX.)
REFL_REACH_K = 1.0
REFL_SMALL_FRAC = 0.01
#: A repainted group's hue window closes when an unlocked, unmapped coloured group within
#: REFL_HUE_MARGIN of its hue has at least this share of its pixels (see _reflection_permissions).
REFL_SAME_HUE_AREA = 0.5
#: A candidate brighter than this quantile of the paint's own OK L (+ ramp) is not a reflection.
REFL_LIGHT_Q, REFL_LIGHT_RAMP = 0.90, 0.03
#: Locked groups protect this neighbourhood (px) instead of cutting the hue window.
LOCK_PROTECT_PX = 24.0
#: An unmapped group at least this share of whose photo pixels pass rule 8's own colour test
#: (chroma gate above half, inside a repainted paint's hue window) is an object of that
#: colour, not a reflection, and rule 8 leaves it alone: a warm backdrop behind a kit's
#: repainted gold parts went grey and blue (Exia), a sand wall grew blue blotches (Sazabi).
#: Only groups with a CIELAB albedo lightness of at least REFL_OWN_COLOUR_MIN_L: a dark
#: group in the paint's hue is its bounce and shadow (a red part's reflection on a dark
#: floor), which is exactly what rule 8 must recolour.
REFL_OWN_COLOUR_SHARE = 0.5
REFL_OWN_COLOUR_MIN_L = 30.0
#: Candidates within this distance of the paint (px) get the wide window (a mix with the paint).
EDGE_MIX_PX = 5.0
EDGE_MIX_HUE0, EDGE_MIX_HUE1 = 40.0, 60.0
#: The recoloured reflection's OK hue is held within this many degrees of the target.
REFL_TARGET_HUE_TOL = 8.0
#: A target this chromatic (OK) has a hue for that envelope.
REFL_TARGET_MIN_CHROMA = 0.02

#: Per-mapping caches (distance transforms, quantiles) kept per renderer.
_MAPPING_CACHE_SIZE = 8


# ----------------------------------------------------------------- torch colour math

@functools.lru_cache(maxsize=64)
def _mat_on(m: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(m, dtype=dtype, device=device)


def _mat(m: Sequence[Sequence[float]], like: torch.Tensor) -> torch.Tensor:
    """A constant matrix as a tensor on ``like``'s device and dtype (made once per device)."""
    return _mat_on(m, like.dtype, like.device)


def linear_to_srgb_t(x: torch.Tensor) -> torch.Tensor:
    """Linear [0,1] -> sRGB with the 2.2 gamma of ``imageio.linear_to_srgb``. Clamps."""
    return x.clamp(0.0, 1.0).pow(1.0 / GAMMA)


def luminance_t(lin: torch.Tensor) -> torch.Tensor:
    """Rec. 709 luminance of a linear RGB tensor [..., 3] -> [...]."""
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def _cbrt_t(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * x.abs().pow(1.0 / 3.0)


def linear_to_oklab_t(lin: torch.Tensor) -> torch.Tensor:
    """Linear RGB (the engine's 2.2-gamma linear, any leading shape, last dim 3) -> OKLab
    (L in 0..1 for the sRGB gamut, a/b about -0.3..0.3). Values outside the gamut are
    converted without clamping (sign-preserving cube root)."""
    lms = lin @ _mat(_OK_M1, lin).T
    return _cbrt_t(lms) @ _mat(_OK_M2, lms).T


def _oklab_to_linear_unclipped(lab: torch.Tensor) -> torch.Tensor:
    """OKLab -> linear RGB, not clipped, so a value outside [0,1] reveals an out-of-gamut colour."""
    lms_ = lab @ _mat(_OK_M2_INV, lab).T
    lms = lms_ * lms_ * lms_
    return lms @ _mat(_OK_M1_INV, lms).T


def oklab_to_linear_t(lab: torch.Tensor) -> torch.Tensor:
    """OKLab -> linear RGB [0,1], clipped per channel."""
    return _oklab_to_linear_unclipped(lab).clamp(0.0, 1.0)


def oklab_to_linear_gamut_t(lab: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """OKLab -> linear RGB [0,1] with *chroma compression* instead of channel clipping.

    L is clamped to [0,1]. For every pixel outside the sRGB gamut, (a,b) is scaled toward
    zero by the largest factor (binary search, ``iters`` halvings) that brings it inside,
    keeping OK lightness and OK hue exactly. In-gamut pixels are returned unchanged
    (identical to :func:`oklab_to_linear_t`)."""
    L = lab[..., :1].clamp(0.0, 1.0)
    ab = lab[..., 1:]
    rgb = _oklab_to_linear_unclipped(torch.cat((L, ab), dim=-1))
    oog = ((rgb < -_GAMUT_TOL) | (rgb > 1.0 + _GAMUT_TOL)).any(-1)
    if bool(oog.any()):
        lo = torch.zeros_like(L[..., 0])
        hi = torch.ones_like(lo)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            test = _oklab_to_linear_unclipped(torch.cat((L, ab * mid[..., None]), dim=-1))
            ok = ((test >= -_GAMUT_TOL) & (test <= 1.0 + _GAMUT_TOL)).all(-1)
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        scale = torch.where(oog, lo, torch.ones_like(lo))
        rgb = _oklab_to_linear_unclipped(torch.cat((L, ab * scale[..., None]), dim=-1))
    return rgb.clamp(0.0, 1.0)


def oklab_from_linear_np(lin: np.ndarray) -> np.ndarray:
    """numpy twin of :func:`linear_to_oklab_t` for group colours (any leading shape)."""
    lin = np.asarray(lin, np.float64)
    lms = np.clip(lin @ np.asarray(_OK_M1).T, 0.0, None)
    return (np.cbrt(lms) @ np.asarray(_OK_M2).T).astype(np.float32)


def cielab_to_oklab(lab) -> np.ndarray:
    """``ColorGroup.albedo_lab`` (CIELAB of the linear albedo, as the segmentation stage
    computes it with ``imageio.linear_to_lab``) -> OKLab of the same albedo, so the source
    colour sits where the group's pixels sit."""
    rgb = imageio.lab_to_rgb(np.asarray(lab, np.float32)[None, :])[0]
    return oklab_from_linear_np(imageio.srgb_to_linear(rgb))


def hex_to_oklab(hexcol: str) -> np.ndarray:
    """'#rrggbb' -> OKLab through the engine's 2.2-gamma linear."""
    return oklab_from_linear_np(imageio.srgb_to_linear(imageio.hex_to_rgb01(hexcol)))


def _toe(L: float) -> float:
    """OK L -> the lightness the anchored map works in (python scalar)."""
    return L if L >= _TOE_L0 else (_TOE_K * max(L, 0.0) ** 3 + 16.0) / 116.0


def _toe_t(L: torch.Tensor) -> torch.Tensor:
    """Tensor twin of :func:`_toe`."""
    return torch.where(L >= _TOE_L0, L, (_TOE_K * L.clamp_min(0.0).pow(3.0) + 16.0) / 116.0)


def _untoe_t(Lt: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_toe_t`."""
    y = ((116.0 * Lt - 16.0) / _TOE_K).clamp_min(0.0)
    return torch.where(Lt >= _TOE_L0, Lt, y.pow(1.0 / 3.0))


def _smoothstep_t(t: torch.Tensor) -> torch.Tensor:
    """0 below 0, 1 above 1, the cubic smoothstep in between."""
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _soft_compress_t(ad: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
    """|deviation| kept up to ``t0``, then compressed toward the asymptote ``t1``."""
    return torch.where(ad <= t0, ad, t0 + (t1 - t0) * torch.tanh((ad - t0) / (t1 - t0)))


def _hue_confidence(chroma: float) -> float:
    """0 for a neutral colour, 1 from OK chroma HUE_C1 (~CIELAB 8) up."""
    t = min(1.0, max(0.0, (chroma - HUE_C0) / (HUE_C1 - HUE_C0)))
    return t * t * (3.0 - 2.0 * t)


def _hue_confidence_t(c: torch.Tensor) -> torch.Tensor:
    """Tensor twin of :func:`_hue_confidence`."""
    return _smoothstep_t((c - HUE_C0) / (HUE_C1 - HUE_C0))


def _smooth01(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _neutral_weight(c_lab: float, l_lab: float, white_share: float, glint_share: float = 0.0) -> float:
    """A source paint's neutral weight in [0, 1]: 1 for a neutral paint (CIELAB chroma <= NEUTRAL_C0)
    that is light enough (albedo lightness >= NEUTRAL_L1, or a photo white share >= NEUTRAL_WHITE_S1)
    and either matte (a glint share <= NEUTRAL_GLOSS_G0) or white paint (albedo lightness >=
    NEUTRAL_PAINT_L1, or a photo white share >= NEUTRAL_PAINT_S1); 0 for a real colour (chroma >=
    NEUTRAL_C1), a dark paint (below both lightness ramps) or a glossy grey (glint share >=
    NEUTRAL_GLOSS_G1 with no evidence of white paint); smoothsteps in between, so the weight is
    continuous in every argument."""
    w = 1.0 - _smooth01((float(c_lab) - NEUTRAL_C0) / (NEUTRAL_C1 - NEUTRAL_C0))
    light = max(_smooth01((float(l_lab) - NEUTRAL_L0) / (NEUTRAL_L1 - NEUTRAL_L0)),
                _smooth01((float(white_share) - NEUTRAL_WHITE_S0) / (NEUTRAL_WHITE_S1 - NEUTRAL_WHITE_S0)))
    white = max(_smooth01((float(l_lab) - NEUTRAL_PAINT_L0) / (NEUTRAL_PAINT_L1 - NEUTRAL_PAINT_L0)),
                _smooth01((float(white_share) - NEUTRAL_PAINT_S0) / (NEUTRAL_PAINT_S1 - NEUTRAL_PAINT_S0)))
    gloss = _smooth01((float(glint_share) - NEUTRAL_GLOSS_G0) / (NEUTRAL_GLOSS_G1 - NEUTRAL_GLOSS_G0))
    return w * light * (1.0 - (1.0 - white) * gloss)


def _near_source_fade(target_lab, source_lab) -> float:
    """How far a mapping's target is from its source colour, for the neutral-source rules: 0 up to a CIELAB
    distance (dE76) of NEUTRAL_NEAR_DE0, 1 from NEUTRAL_NEAR_DE1, a smoothstep in between."""
    d = math.dist([float(v) for v in target_lab], [float(v) for v in source_lab])
    return _smooth01((d - NEUTRAL_NEAR_DE0) / (NEUTRAL_NEAR_DE1 - NEUTRAL_NEAR_DE0))


def _disk_offsets(r: int) -> list[tuple[int, int]]:
    """Offsets of a digital disk of radius ``r`` (the centre included)."""
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r + r]


def _group_dilate_t(v: torch.Tensor, gid: torch.Tensor, r: int) -> torch.Tensor:
    """Grey dilation (max filter) of ``v`` [H,W] with a disk of radius ``r`` inside each pixel's own
    group (``gid`` [H,W]). >= ``v``."""
    if r <= 0:
        return v
    H, W = v.shape
    gf = gid.to(torch.float32)
    gp = F.pad(gf[None, None], (r, r, r, r), value=-1.0)[0, 0]
    vp = F.pad(v[None, None], (r, r, r, r), value=float("-inf"))[0, 0]
    ninf = torch.full_like(v, float("-inf"))
    out = v.clone()
    for dy, dx in _disk_offsets(r):
        vs = vp[r + dy:r + dy + H, r + dx:r + dx + W]
        gs = gp[r + dy:r + dy + H, r + dx:r + dx + W]
        out = torch.maximum(out, torch.where(gs == gf, vs, ninf))
    return out


def _group_mean_t(v: torch.Tensor, gid: torch.Tensor, r: int) -> torch.Tensor:
    """Mean of ``v`` [H,W] over a disk of radius ``r`` inside each pixel's own group (``gid`` [H,W])."""
    if r <= 0:
        return v
    H, W = v.shape
    gf = gid.to(torch.float32)
    gp = F.pad(gf[None, None], (r, r, r, r), value=-1.0)[0, 0]
    vp = F.pad(v[None, None], (r, r, r, r), value=0.0)[0, 0]
    acc = torch.zeros_like(v)
    cnt = torch.zeros_like(v)
    for dy, dx in _disk_offsets(r):
        same = (gp[r + dy:r + dy + H, r + dx:r + dx + W] == gf).to(v.dtype)
        acc = acc + same * vp[r + dy:r + dy + H, r + dx:r + dx + W]
        cnt = cnt + same
    return acc / cnt.clamp_min(1.0)


def _hue_turn(src_lab, tgt_lab) -> float:
    """Angle (radians) turning the source's OK chroma direction onto the target's, faded to
    zero as either side approaches neutral, where it would only rotate noise."""
    sa, sb = float(src_lab[1]), float(src_lab[2])
    ta, tb = float(tgt_lab[1]), float(tgt_lab[2])
    w = _hue_confidence(math.hypot(sa, sb)) * _hue_confidence(math.hypot(ta, tb))
    if w <= 0.0:
        return 0.0
    theta = math.atan2(tb, ta) - math.atan2(sb, sa)
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    return w * theta


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


def _edt(outside: torch.Tensor) -> torch.Tensor:
    """Euclidean distance (px) of every pixel to the nearest pixel where ``outside`` is
    False, on the CPU (a few ms at working resolution), back on the tensor's device."""
    src = np.ascontiguousarray(outside.to(torch.uint8).cpu().numpy())
    d = cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(d)).to(outside.device)


class _LRU(OrderedDict):
    """A tiny bounded cache: per-mapping tensors must not pile up over a long session."""

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity

    def get_or(self, key, make: Callable[[], object]):
        if key in self:
            self.move_to_end(key)
            return self[key]
        value = make()
        self[key] = value
        while len(self) > self.capacity:
            self.popitem(last=False)
        return value


# Columns of the per-group parameter table (one row per group id, blended per pixel).
_C_MAPPED, _C_TEXTURE = 0, 1
_C_ROT = slice(2, 6)          # texture * s * R(theta), row-major
_C_BOUNCE = slice(6, 10)      # s * R(theta): the bounce retint and rule 8 work at full strength
_C_TL, _C_AL, _C_SLOPE_DN = 10, 11, 12
_C_OFFSET = slice(13, 15)     # T_ab - rot @ A_ab
_C_TAB = slice(15, 17)
_C_SRC = slice(17, 19)        # A_ab
_C_SLOPE_UP = 19
_C_NEUTRAL = 20               # the source's neutral weight (neutral-source rules)
_C_WHITE = 21                 # the source's white-paint weight (the exposure bound)
_N_COLS = 22


@dataclass
class _Level:
    """One resolution of the layers, all on the GPU, channels-last float32, plus the
    per-image fields derived from them (filled on first use)."""
    albedo: torch.Tensor                     # [H,W,3] linear
    shading: torch.Tensor                    # [H,W,3] linear, >= 0
    residual: torch.Tensor                   # [H,W,3]
    albedo_ok: torch.Tensor                  # [H,W,3] OKLab
    product: torch.Tensor                    # albedo * shading
    group_map: torch.Tensor                  # [H,W] int64
    guide: Optional[torch.Tensor] = None     # [3,H,W] sRGB of the photo (coverage snap)
    spec: Optional[torch.Tensor] = None      # [H,W,1] glint weight of the neutral residual
    highlight: Optional[torch.Tensor] = None # [H,W,1] sensor-clipped white highlight mask
    gloss: Optional[torch.Tensor] = None     # [H,W] white specular estimate W (linear)
    band: Optional[torch.Tensor] = None      # [H,W] bool, near a boundary with another group
    band_px: int = 1
    soft_edges: Optional[torch.Tensor] = None  # [H,W] 1 where the photo's own edge is soft
    islands: Optional[torch.Tensor] = None   # [H,W] float, resized from the base mask
    protect: Optional[torch.Tensor] = None   # [H,W] float, resized from the base mask
    locked_dist: Optional[torch.Tensor] = None
    locked_dist_done: bool = False
    own_neutral: Optional[torch.Tensor] = None      # [H,W] the neutral weight of each pixel's own group
    white_highlight: Optional[torch.Tensor] = None  # [H,W,1] the highlight mask of a neutral source's repaint
    white_light: Optional[tuple] = None             # ([H,W], [H,W]) the exposure bound's light and weight
    white_glints: Optional[torch.Tensor] = None     # [H,W] a white paint's own glints (linear luminance)
    photo_L: Optional[torch.Tensor] = None          # [H,W] OK lightness of the photo
    small_clips: Optional[torch.Tensor] = None      # [H,W] small clipped white spots: no source of the bound's light

    @property
    def size(self) -> tuple[int, int]:
        return int(self.albedo.shape[1]), int(self.albedo.shape[0])


@dataclass
class _ReflParams:
    """One repainted group's constants for rule 8 (python scalars and [3] tensors)."""
    hue_deg: float               # OK hue of the source paint
    s: float                     # chroma scale min(1, C_T / C_A)
    cos: float                   # R(theta), the paint's hue turn
    sin: float
    excess: torch.Tensor         # [3] the old paint's colour excess over its own floor
    excess_sq: float             # |excess|^2
    target_offset: torch.Tensor  # [3] target - the old paint's floor (island rebuild)
    target_dir: Optional[tuple[float, float]]   # unit OK a/b of the target, None when neutral
    z_pos: float = REFL_HUE_WINDOW + REFL_HUE_RAMP   # hue-gate zero point on the + side
    z_neg: float = REFL_HUE_WINDOW + REFL_HUE_RAMP   # ... and on the - side
    gids: tuple = ()                                 # the repainted groups (one source colour and target)
    reach: float = REFL_FALLOFF_PX                   # how far their reflections go (px at the reference size)
    weight: float = 1.0                              # 1 - the source's neutral weight


@dataclass
class _GroupParams:
    """Per-group constants of one render."""
    table: torch.Tensor          # [G, _N_COLS]
    mapped_ids: tuple[int, ...]  # groups actually repainted (mapped, unlocked, in range)
    refl: tuple[_ReflParams, ...]
    allow: torch.Tensor          # [G] 1 where rule 8 (and the ramp's envelope) may act


@dataclass
class _Repaint:
    """What rule 2 produced for one render."""
    albedo: torch.Tensor         # [H,W,3] repainted linear albedo
    coverage: torch.Tensor       # [H,W] m, the fraction of each pixel that is repainted
    hard: torch.Tensor           # [H,W] 1 on the mapped groups' own labels
    target: torch.Tensor         # [H,W,3] OKLab target per pixel
    bounce: torch.Tensor         # [H,W,4] s * R(theta)
    src_ab: torch.Tensor         # [H,W,2] the old paint's OK a/b
    neutral: Optional[torch.Tensor] = None     # [H,W] the source's neutral weight (None: all 0)
    highlight: Optional[torch.Tensor] = None   # [H,W,1] the clipped-highlight mask of this render
    white: Optional[torch.Tensor] = None       # [H,W] the source's white-paint weight (exposure bound)


class Renderer:
    """Holds one image's layers on the GPU so successive renders are fast.

    Guarantees:

    * ``render({}, RenderOptions())`` returns exactly
      ``to_uint8(linear_to_srgb(albedo * shading + residual))``.
    * Unmapped groups, groups mapped to ``None`` and ``locked`` groups are left untouched
      (up to feathering at their borders with repainted neighbours), except that rule 8
      recolours old-paint-hued pixels of *unmapped neutral* groups near the repainted
      parts; locked groups, unmapped groups with a real colour of their own and pixels of
      the ``protect`` mask are never touched by it. A neutral source also repaints the
      decal-island pixels connected to it that hold its paint (rule 1), never a locked
      group's or a protected pixel.
    * Every output is finite uint8 RGB of the requested size, for every combination of
      :class:`RenderOptions` values.
    * A source mapped to its own colour (within CIELAB dE NEUTRAL_NEAR_DE0 of its ``albedo_lab``,
      the swatch) renders exactly as under the saturated-paint rules, near the photo; the
      neutral-source rules, rule 7e's glints included, come in continuously with the source's
      chroma, lightness, white share and gloss and with the target's distance from the source.
    * After the first call at a given size, a preview render (1024 long side) takes well
      under 60 ms on the RTX 5090; layers, masks and per-mapping tables are cached.
      :meth:`update_groups` takes new flags (a lock toggle) without losing them.

    ``islands`` (bool HxW): decals the analysis stage carved out of the paint; the coverage
    ramp never enters them unless their own group is mapped (rules 1, 8). ``protect``
    (bool HxW): objects in the old paint's colour in their own right, never recoloured by
    rule 8 or the envelope. Both default to none. ``reference_long_side``: the resolution
    the engine's pixel distances refer to (default: the layers' own). ``glints`` (float, any
    size): the neutral groups' glint field of rule 7e to use instead of finding one, resized to
    the layers (a full-resolution export passes the working-resolution renderer's
    :meth:`white_glints`, so it keeps exactly the glints the preview showed). ``neutral`` (``[G, 2]``,
    :meth:`neutral_weights`): the groups' neutral-source weights to use instead of measuring them on these
    layers (the export passes the preview's too).

    Source colours come from ``ColorGroup.albedo_lab`` (CIELAB, converted to OKLab); groups
    present in the map but missing from ``groups`` get the median OKLab albedo of their pixels.
    """

    def __init__(self, albedo_lin: np.ndarray, shading_lin: np.ndarray, residual: np.ndarray,
                 group_map: np.ndarray, groups: Sequence[ColorGroup], device: Optional[str] = None,
                 islands: Optional[np.ndarray] = None, protect: Optional[np.ndarray] = None,
                 reference_long_side: Optional[int] = None, glints: Optional[np.ndarray] = None,
                 neutral: Optional[np.ndarray] = None) -> None:
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
        self._base = _Level(alb_t, shd_t, res_t, linear_to_oklab_t(alb_t), alb_t * shd_t, gm_t)
        self._base.islands = self._mask_tensor(islands, gm.shape, "islands")
        self._base.protect = self._mask_tensor(protect, gm.shape, "protect")
        if glints is not None:
            self._base.white_glints = self._resized_field(glints, gm.shape, "glints")
        self._levels: dict[tuple[int, int], _Level] = {self._base.size: self._base}
        w0, h0 = self._base.size
        ref = int(reference_long_side) if reference_long_side else max(w0, h0)
        #: base width that the pixel constants are defined at
        self._ref_width = max(1e-6, w0 * ref / max(1, max(w0, h0)))
        self._source_ok: dict[int, np.ndarray] = {}
        self._own_ok: Optional[torch.Tensor] = None
        self._photo_hc: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._share = _LRU(_MAPPING_CACHE_SIZE)          # rule 8's own-colour share per group
        self._pivot = self._compute_pivot(shd_t)
        self._light_ref: Optional[torch.Tensor] = None
        self._ratio_median: Optional[torch.Tensor] = None
        self._ratio_bins: Optional[torch.Tensor] = None
        self._dist = _LRU(_MAPPING_CACHE_SIZE)          # distance to the repainted parts
        self._gdist = _LRU(4 * _MAPPING_CACHE_SIZE)     # windowed distance to one small repainted group
        self._group_px: Optional[np.ndarray] = None     # pixels per group at the base level
        self._paint_light = _LRU(_MAPPING_CACHE_SIZE)   # OK L quantile of the painted pixels
        self._shade_median = _LRU(_MAPPING_CACHE_SIZE)  # median shading of the painted pixels
        self._group_neutral: Optional[np.ndarray] = None  # [G] each group's own neutral weight
        self._group_white: Optional[np.ndarray] = None    # [G] ... times its white-paint evidence
        if neutral is not None:
            self._group_neutral, self._group_white = self._given_weights(neutral)
        self._island_in = _LRU(_MAPPING_CACHE_SIZE)       # the island pixels a neutral source enters
        self.last_render_ms: float = 0.0
        self._freed: bool = False

    def _resized_field(self, field: np.ndarray, shape: tuple[int, int], name: str) -> torch.Tensor:
        """A float HxW field (any size) as a tensor of ``shape`` on the device: area-averaged when it
        shrinks, bilinear when it grows, never negative, 0 where it is not finite."""
        f = np.asarray(field, np.float32)
        if f.ndim != 2 or f.size == 0:
            raise ValueError(f"{name} must be a non-empty HxW array, got {f.shape}")
        t = torch.from_numpy(np.ascontiguousarray(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0))).to(self.device)
        if tuple(t.shape) != tuple(shape):
            mode = "area" if t.shape[1] > shape[1] else "bilinear"
            kw = {} if mode == "area" else {"align_corners": False}
            t = F.interpolate(t[None, None], size=tuple(shape), mode=mode, **kw)[0, 0]
        return t.clamp_min(0.0).contiguous()

    def _given_weights(self, neutral: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(``[G]`` neutral weights, ``[G]`` white-paint weights) from a :meth:`neutral_weights` table (``[N, 2]``,
        any N: rows past this renderer's groups are dropped, missing ones are 0), clipped to [0, 1], 0 where not
        finite."""
        t = np.asarray(neutral, np.float32)
        if t.ndim != 2 or t.shape[1] != 2:
            raise ValueError(f"neutral must be an Nx2 array of weights, got {t.shape}")
        out = np.zeros((max(self.n_groups, 1), 2), np.float32)
        k = min(len(out), len(t))
        out[:k] = np.clip(np.nan_to_num(t[:k], nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return np.ascontiguousarray(out[:, 0]), np.ascontiguousarray(out[:, 1])

    def _mask_tensor(self, mask: Optional[np.ndarray], shape: tuple[int, int], name: str) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        m = np.asarray(mask)
        if m.shape != shape:
            raise ValueError(f"{name} must be HxW like group_map, got {m.shape}")
        if not m.any():
            return None
        return torch.from_numpy(np.ascontiguousarray(m.astype(np.float32))).to(self.device)

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
        with torch.no_grad():
            paint = self._repaint(level, self._group_params(mapping, opts), opts)
        alb = level.albedo if paint is None else paint.albedo
        return alb.detach().cpu().numpy().astype(np.float32)

    def white_glints(self) -> np.ndarray:
        """The neutral groups' own glints (rule 7e) at the layers' resolution: float32 HxW, the white
        light (linear luminance) a neutral source's repaint gets back on its labels, 0 elsewhere. Found
        on first use (or the field given to the constructor). Raises ``RuntimeError`` after :meth:`free`."""
        self._check_alive()
        with torch.no_grad():
            return self._white_glints(self._base).detach().cpu().numpy().astype(np.float32)

    def neutral_weights(self) -> np.ndarray:
        """float32 ``[G, 2]``, one row per group id: the group's own neutral weight (the neutral-source rules,
        before a mapping's near-source fade) and its white-paint weight (the exposure bound's), measured on
        the layers on first use (or the table given to the constructor). A full-resolution export passes the
        working-resolution renderer's table, so it treats every group as the preview did. Raises
        ``RuntimeError`` after :meth:`free`."""
        self._check_alive()
        with torch.no_grad():
            w = self._group_neutral_weights()
        return np.stack((w, self._group_white), axis=1).astype(np.float32)

    def neutral_sources(self, mapping: Mapping, options: Optional[RenderOptions] = None) -> tuple[int, ...]:
        """The group ids ``mapping`` repaints with some neutral-source rule (a neutral weight above 0 after the
        near-source fade); empty when the mapping renders exactly as under the saturated-paint rules. Raises
        ``RuntimeError`` after :meth:`free` and ``ValueError`` on a bad mapping or options."""
        self._check_alive()
        with torch.no_grad():
            params = self._group_params(mapping, options or RenderOptions())
            col = params.table[:, _C_NEUTRAL].cpu().numpy()
        return tuple(int(g) for g in params.mapped_ids if col[g] > 0.0)

    def update_groups(self, groups: Sequence[ColorGroup], protect: Optional[np.ndarray] = None) -> None:
        """Adopt new flags (``locked``, ``is_background``, names) for the same groups on the
        same group map, and optionally a new ``protect`` mask, without rebuilding: the
        uploaded layers and every per-image field are kept, only what depends on the locks
        (the locked-group distance) and on the protect mask is dropped. A lock toggle in
        the UI then costs a normal render instead of a new renderer. Raises ``ValueError``
        when a group's colour differs (a regroup or merge needs a new renderer)."""
        self._check_alive()
        by_id = {int(g.id): g for g in groups}
        for gid, old in self._group_by_id.items():
            new = by_id.get(gid)
            if new is None or tuple(new.albedo_lab) != tuple(old.albedo_lab) or \
                    getattr(new, "ref_lab", None) != getattr(old, "ref_lab", None):
                raise ValueError("update_groups only changes flags; the groups themselves differ")
        if any(gid >= self.n_groups for gid in by_id):
            raise ValueError("update_groups got a group id outside the group map")
        self.groups = list(groups)
        self._group_by_id = by_id
        base_protect = self._base.protect
        if protect is not None:
            base_protect = self._mask_tensor(protect, tuple(self._base.group_map.shape), "protect")
        self._island_in.clear()
        for lvl in self._levels.values():
            lvl.locked_dist, lvl.locked_dist_done = None, False
            if protect is not None:
                if lvl is self._base or base_protect is None:
                    lvl.protect = base_protect
                else:
                    h, w = lvl.group_map.shape
                    lvl.protect = F.interpolate(base_protect[None, None], size=(h, w), mode="nearest")[0, 0]

    def free(self) -> None:
        """Drop every cached GPU tensor. The renderer must not be used afterwards:
        ``render``, ``render_at`` and ``recolor_albedo`` raise ``RuntimeError``.
        Idempotent."""
        self._freed = True
        self._levels.clear()
        self._source_ok.clear()
        self._own_ok = None
        self._photo_hc = None
        self._share.clear()
        self._dist.clear()
        self._gdist.clear()
        self._group_px = None
        self._paint_light.clear()
        self._shade_median.clear()
        self._island_in.clear()
        self._light_ref = self._ratio_median = self._ratio_bins = None
        for name in ("albedo", "shading", "residual", "albedo_ok", "product", "group_map"):
            setattr(self._base, name, torch.empty(0))
        for name in ("guide", "spec", "highlight", "gloss", "band", "soft_edges", "islands", "protect", "locked_dist",
                     "own_neutral", "white_highlight", "white_light", "white_glints", "photo_L",
                     "small_clips"):
            setattr(self._base, name, None)
        self._group_neutral = self._group_white = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ levels and per-image fields

    def _level_for(self, long_side: int) -> _Level:
        w0, h0 = self._base.size
        size = imageio.fit_size(w0, h0, int(long_side))
        lvl = self._levels.get(size)
        if lvl is None:
            lvl = self._resize_level(size)
            self._levels[size] = lvl
        return lvl

    def _resize_level(self, size: tuple[int, int]) -> _Level:
        base = self._base
        w, h = size
        shrink = w < base.albedo.shape[1]

        def rs(x: torch.Tensor) -> torch.Tensor:
            t = x.permute(2, 0, 1)[None]
            if shrink:
                t = F.interpolate(t, size=(h, w), mode="area")
            else:
                t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False)
            return t[0].permute(1, 2, 0).contiguous()

        def nearest(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is None:
                return None
            return F.interpolate(x[None, None].float(), size=(h, w), mode="nearest")[0, 0]

        alb = rs(base.albedo).clamp(0.0, 1.0)
        shd = rs(base.shading).clamp_min(0.0)
        res = rs(base.residual)
        gm = nearest(base.group_map).long()
        lvl = _Level(alb, shd, res, linear_to_oklab_t(alb), alb * shd, gm)
        lvl.islands = nearest(base.islands)
        lvl.protect = nearest(base.protect)
        return lvl

    def _px(self, level: _Level) -> float:
        """Scale from the engine's pixel constants to pixels at ``level``."""
        return level.size[0] / self._ref_width

    @staticmethod
    def _compute_pivot(shading: torch.Tensor) -> torch.Tensor:
        flat = shading.reshape(-1, 3)
        if flat.shape[0] > 1_000_000:
            flat = flat[:: flat.shape[0] // 1_000_000 + 1]
        piv = flat.median(dim=0).values if flat.shape[0] else torch.ones(3, device=shading.device)
        return piv.clamp_min(1e-3)

    def _guide(self, level: _Level) -> torch.Tensor:
        """[3,H,W] sRGB of the photograph, which the coverage is allowed to follow (the
        albedo has lost the shading that makes many part edges visible)."""
        if level.guide is None:
            lin = (level.product + level.residual).clamp(0.0, 1.0)
            level.guide = linear_to_srgb_t(lin).permute(2, 0, 1).contiguous()
        return level.guide

    def _spec_weight(self, level: _Level) -> torch.Tensor:
        """[H,W,1] how much of the neutral residual at each pixel is a real glint.

        The achromatic floor of the positive residual mixes sharp mirror highlights (the
        lamp's, they survive any repaint) with a faint broad veil of unexplained diffuse
        energy (the old paint's). They separate by magnitude: normalised by a high
        quantile, near 1 is a glint, near 0 is veil."""
        if level.spec is None:
            gray = level.residual.clamp_min(0.0).amin(dim=-1, keepdim=True)
            flat = gray.reshape(-1)
            if flat.numel() > 1_000_000:
                flat = flat[:: flat.numel() // 1_000_000 + 1]
            q = torch.quantile(flat, 0.99) if flat.numel() else torch.zeros((), device=gray.device)
            level.spec = (gray / q.clamp_min(1e-4)).clamp(0.0, 1.0)
        return level.spec

    def _highlight(self, level: _Level) -> torch.Tensor:
        """[H,W,1] sensor-clipped *white* highlights of the photograph: a channel of
        ``albedo * shading + residual`` at or above 0.98 sRGB with a white component of at
        least HL_MIN_WHITE, in components of at least HL_MIN_AREA px."""
        if level.highlight is None:
            photo = level.product + level.residual
            clipped = (photo.amax(dim=-1) >= CLIP_LIN) & (self._gloss(level) >= HL_MIN_WHITE)
            min_area = max(1, int(round(HL_MIN_AREA * self._px(level) ** 2)))
            cm = np.ascontiguousarray(clipped.to(torch.uint8).cpu().numpy())
            _, lab, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
            keep = stats[:, cv2.CC_STAT_AREA] >= min_area
            keep[0] = False
            level.highlight = torch.from_numpy(keep[lab]).to(self.device).to(torch.float32)[..., None]
        return level.highlight

    def _pools(self) -> np.ndarray:
        """[G] the group whose pixels stand for each group in the per-group photo statistics:
        itself, or for the instances of a part split by instance (same ``part`` and
        ``ref_lab``) the lowest id among them, so the instances keep the part's statistics."""
        pool = np.arange(max(self.n_groups, 1), dtype=np.int64)
        first: dict[tuple, int] = {}
        for g in sorted(self.groups, key=lambda g: int(g.id)):
            ref = getattr(g, "ref_lab", None)
            gid = int(g.id)
            if ref is None or not 0 <= gid < len(pool):
                continue
            key = (getattr(g, "part", None), tuple(round(float(v), 4) for v in ref))
            pool[gid] = first.setdefault(key, gid)
        return pool

    def _ratios(self) -> tuple[torch.Tensor, torch.Tensor]:
        """How white each paint is: ([G] median min/max channel ratio of the photo over the
        group's exposed pixels, [G, GLOSS_BINS] its GLOSS_RATIO_Q quantile per bin of the
        max channel, monotone non-decreasing in brightness). The instances of a part split by
        instance share the part's (:meth:`_pools`). Base level, computed once."""
        if self._ratio_median is None:
            base = self._base
            photo = (base.product + base.residual).clamp(0.0, 1.0)
            mx_t = photo.amax(dim=-1)
            ratio_t = photo.amin(dim=-1) / mx_t.clamp_min(1e-4)
            med = torch.ones(max(self.n_groups, 1), dtype=torch.float32, device=self.device)
            pool = self._pools()
            gm = base.group_map
            if bool((pool != np.arange(len(pool))).any()):
                gm = torch.from_numpy(pool).to(self.device)[gm]
            for gid in torch.unique(gm).tolist():
                v = ratio_t[(gm == gid) & (mx_t > 0.02)]
                if v.numel() > 400_000:
                    v = v[:: v.numel() // 400_000 + 1]
                if v.numel() >= 16:
                    med[gid] = torch.quantile(v, 0.5)
            nb = GLOSS_BINS
            mx = mx_t.cpu().numpy().ravel()
            ratio = ratio_t.cpu().numpy().ravel()
            gmn = gm.cpu().numpy().ravel()
            lut = np.repeat(med.cpu().numpy()[:, None], nb, axis=1).astype(np.float32)
            edges = np.linspace(0.02, 1.0, nb + 1)
            order = np.argsort(gmn, kind="stable")
            gs = gmn[order]
            starts = np.searchsorted(gs, np.arange(lut.shape[0]), side="left")
            ends = np.searchsorted(gs, np.arange(lut.shape[0]), side="right")
            for gid in range(lut.shape[0]):
                ii = order[starts[gid]:ends[gid]]
                if ii.size < 16:
                    continue
                x, v = mx[ii], ratio[ii]
                keep = x > 0.02
                x, v = x[keep], v[keep]
                if x.size < 16:
                    continue
                b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
                rs = np.full(nb, np.nan)
                for i in range(nb):
                    vv = v[b == i]
                    if vv.size >= GLOSS_BIN_MIN_PX:
                        rs[i] = np.quantile(vv, GLOSS_RATIO_Q)
                filled = ~np.isnan(rs)
                if not filled.any():
                    continue
                rs = np.interp(np.arange(nb), np.flatnonzero(filled), rs[filled])
                lut[gid] = np.maximum.accumulate(rs).astype(np.float32)
            # every instance takes its pool's row (the pool's pixels are all counted under its id)
            self._ratio_median = med[torch.from_numpy(pool).to(self.device)]
            self._ratio_bins = torch.from_numpy(np.ascontiguousarray(lut[pool])).to(self.device)
        return self._ratio_median, self._ratio_bins

    def _gloss(self, level: _Level) -> torch.Tensor:
        """[H,W] the photo's white specular W (linear, >= 0): the dichromatic split of each
        pixel against how white its own group looks at that brightness, 0 in groups that
        are not saturated paints, then limited in the boundary band, min-blurred and softly
        thresholded (it may shrink the photo's highlight footprint, never grow it)."""
        if level.gloss is None:
            med, bins = self._ratios()
            photo = (level.product + level.residual).clamp(0.0, 1.0)
            mn, mx = photo.amin(dim=-1), photo.amax(dim=-1)
            nb = bins.shape[1]
            pos = ((mx - 0.02) / 0.98 * nb - 0.5).clamp(0.0, float(nb - 1))
            i0 = pos.floor().long().clamp(0, nb - 1)
            i1 = (i0 + 1).clamp(max=nb - 1)
            f = pos - i0.to(pos.dtype)
            flat = bins.reshape(-1)
            gi = level.group_map * nb
            r = flat[gi + i0] * (1.0 - f) + flat[gi + i1] * f
            w = ((mn - r * mx) / (1.0 - r).clamp_min(1e-3) - GLOSS_W0).clamp_min(0.0)
            w = torch.where(med[level.group_map] <= GLOSS_RATIO_MAX, w, torch.zeros_like(w))
            w = self._edge_limit(level, w)
            w = torch.minimum(w, _feather_chw(w[None], max(0.3, GLOSS_SMOOTH_SIGMA * self._px(level)))[0])
            level.gloss = w * _smoothstep_t((w - GLOSS_T0) / (GLOSS_T1 - GLOSS_T0))
        return level.gloss

    # ------------------------------------------------------------ neutral sources (per-image fields)

    def _white_photo(self, level: _Level) -> torch.Tensor:
        """[H,W] in [0,1]: how white each photo pixel is (OK L from EXPOSURE_L0 to EXPOSURE_L1, OK
        chroma below EXPOSURE_C0..C1): the white paint the neutral-source rules recognise."""
        photo = (level.product + level.residual).clamp(0.0, 1.0)
        ok = linear_to_oklab_t(photo)
        c = torch.hypot(ok[..., 1], ok[..., 2])
        return (_smoothstep_t((ok[..., 0] - EXPOSURE_L0) / (EXPOSURE_L1 - EXPOSURE_L0))
                * (1.0 - _smoothstep_t((c - EXPOSURE_C0) / (EXPOSURE_C1 - EXPOSURE_C0))))

    def _group_neutral_weights(self) -> np.ndarray:
        """[G] every group's own neutral weight (:func:`_neutral_weight` of the CIELAB chroma and
        lightness of its ``ref_lab`` or ``albedo_lab``, of the share of its photo pixels that are white,
        :meth:`_white_photo`, and of the share that glint, :meth:`_glint_px`, the instances of a split part
        sharing the part's; a group the map has but ``groups`` lacks: of its median OK albedo, translated),
        or the table given to the constructor. Base level, computed once."""
        if self._group_neutral is None:
            w = np.zeros(max(self.n_groups, 1), np.float32)
            # (numpy's float64 bincount: a weighted bincount on the GPU sums in no fixed order, and a
            # share differing in the 7th digit moved a render by a level)
            gm = self._pool_map(self._base).reshape(-1).cpu().numpy()
            wp = self._white_photo(self._base).reshape(-1).cpu().numpy().astype(np.float64)
            gp = self._glint_px(self._base).reshape(-1).cpu().numpy().astype(np.float64)
            n = np.maximum(np.bincount(gm, minlength=len(w)).astype(np.float64), 1.0)
            pools = self._pools()
            share = (np.bincount(gm, weights=wp, minlength=len(w)) / n)[pools].astype(np.float32)
            glint = (np.bincount(gm, weights=gp, minlength=len(w)) / n)[pools].astype(np.float32)
            for gid in range(len(w)):
                g = self._group_by_id.get(gid)
                if g is not None:
                    if getattr(g, "finish", "") == "chrome":
                        continue            # its neutral floor is its reflections
                    lab = self._source_lab_for(gid)
                    c, l = math.hypot(lab[1], lab[2]), lab[0]
                else:
                    ok = self._source_ok_for(gid)
                    c = math.hypot(float(ok[1]), float(ok[2])) / OK_PER_LAB_CHROMA
                    y = float(luminance_t(oklab_to_linear_t(torch.tensor(ok, dtype=torch.float32))))
                    l = 116.0 * y ** (1.0 / 3.0) - 16.0 if y > 0.008856 else 903.3 * y
                w[gid] = _neutral_weight(c, l, float(share[gid]), float(glint[gid]))
            self._group_neutral = w
            white = np.zeros_like(w)
            for gid in np.flatnonzero(w > 0.0).tolist():
                L = self._source_lab_for(gid)[0] if gid in self._group_by_id else None
                if L is None:
                    y = float(luminance_t(oklab_to_linear_t(torch.tensor(self._source_ok_for(gid), dtype=torch.float32))))
                else:
                    y = ((L + 16.0) / 116.0) ** 3 if L > 8.0 else L / 903.3
                k = _smooth01((y - EXPOSURE_Y0) / (EXPOSURE_Y1 - EXPOSURE_Y0))
                s0 = EXPOSURE_S0 + k * (EXPOSURE_LIGHT_S0 - EXPOSURE_S0)
                s1 = EXPOSURE_S1 + k * (EXPOSURE_LIGHT_S1 - EXPOSURE_S1)
                white[gid] = w[gid] * _smooth01((float(share[gid]) - s0) / (s1 - s0))
            self._group_white = white.astype(np.float32)
        return self._group_neutral

    def _glint_px(self, level: _Level) -> torch.Tensor:
        """[H,W] in [0, 1]: how much each photo pixel glints, as the analysis stage counts it for ``Region.glint``
        (materials.shine_features: a channel at or above CLIP_LIN, or a neutral residual above half its 99th
        percentile, :meth:`_spec_weight` > 1/2), but softly around both cuts (GLINT_PX_*), so the share of a group
        changes continuously with the photo."""
        photo = (level.product + level.residual).clamp(0.0, 1.0)
        c0, c1 = GLINT_PX_CLIP0 ** GAMMA, GLINT_PX_CLIP1 ** GAMMA
        clip = _smoothstep_t((photo.amax(dim=-1) - c0) / (c1 - c0))
        spec = _smoothstep_t((self._spec_weight(level)[..., 0] - GLINT_PX_SPEC0) / (GLINT_PX_SPEC1 - GLINT_PX_SPEC0))
        return torch.maximum(clip, spec)

    def _source_lab_for(self, gid: int) -> tuple[float, float, float]:
        """CIELAB of a group's source colour: its ``ref_lab`` or ``albedo_lab`` (what its swatch shows), or for a
        group the map has but ``groups`` lacks, its median OK albedo (:meth:`_source_ok_for`) translated."""
        g = self._group_by_id.get(gid)
        if g is not None:
            ref = getattr(g, "ref_lab", None)
            lab = ref if ref is not None else g.albedo_lab
            return float(lab[0]), float(lab[1]), float(lab[2])
        ok = self._source_ok_for(gid)
        lin = oklab_to_linear_t(torch.tensor(ok, dtype=torch.float32)).numpy()[None, :]
        lab = imageio.linear_to_lab(lin)[0]
        return float(lab[0]), float(lab[1]), float(lab[2])

    def _own_neutral(self, level: _Level) -> torch.Tensor:
        """[H,W] the neutral weight of each pixel's own group (0 in saturated groups)."""
        if level.own_neutral is None:
            level.own_neutral = torch.from_numpy(self._group_neutral_weights()).to(self.device)[level.group_map]
        return level.own_neutral

    def _pool_map(self, level: _Level) -> torch.Tensor:
        """[H,W] the group map with the instances of a split part read as one part (:meth:`_pools`)."""
        pool = self._pools()
        gm = level.group_map
        if bool((pool != np.arange(len(pool))).any()):
            gm = torch.from_numpy(pool).to(self.device)[gm]
        return gm

    def _gloss_neutral(self, level: _Level) -> torch.Tensor:
        """[H,W] the white specular a *neutral source's* repaint may not fall below: the
        saturated-paint estimate (:meth:`_gloss`), in the neutral groups scaled by 1 - their own
        neutral weight (a warm-lit off-white passed as a saturated paint)."""
        return self._gloss(level) * (1.0 - self._own_neutral(level))

    def _highlight_neutral(self, level: _Level) -> torch.Tensor:
        """[H,W,1] rules 2, 5 and 6's clipped-highlight mask for a neutral source's repaint: the test of
        :meth:`_highlight` on :meth:`_gloss_neutral`, so a white paint's clipped face is never a
        highlight because it is white. Per level, once."""
        if level.white_highlight is None:
            photo = level.product + level.residual
            clipped = (photo.amax(dim=-1) >= CLIP_LIN) & (self._gloss_neutral(level) >= HL_MIN_WHITE)
            min_area = max(1, int(round(HL_MIN_AREA * self._px(level) ** 2)))
            cm = np.ascontiguousarray(clipped.to(torch.uint8).cpu().numpy())
            _, lab, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
            keep = stats[:, cv2.CC_STAT_AREA] >= min_area
            keep[0] = False
            level.white_highlight = torch.from_numpy(keep[lab]).to(self.device).to(torch.float32)[..., None]
        return level.white_highlight

    def _white_light(self, level: _Level) -> tuple[torch.Tensor, torch.Tensor]:
        """([H,W] the photo luminance of the brightest white pixel of each pixel's own group within
        EXPOSURE_RADIUS px, [H,W] how far the bound acts there, 0..1: a smoothstep of the share of white
        pixels of that neighbourhood, full from EXPOSURE_NEAR, so a white face is bounded and a lone white
        pixel is not): the exposure bound's light (rule 7a'). A small clipped spot is no white pixel here
        (:meth:`_small_clips`). Per level, once."""
        if level.white_light is None:
            w = self._white_photo(level) * (1.0 - self._small_clips(level))
            Yw = luminance_t((level.product + level.residual).clamp(0.0, 1.0)) * w
            r = int(round(EXPOSURE_RADIUS * self._px(level)))
            near = _smoothstep_t(w / EXPOSURE_NEAR)
            if r > 0:
                gm = self._pool_map(level)
                near = _smoothstep_t(_group_mean_t(w, gm, r) / EXPOSURE_NEAR)
                Yw, w = _group_dilate_t(Yw, gm, r), _group_dilate_t(w, gm, r)
            level.white_light = (Yw / w.clamp_min(1e-6), near)
        return level.white_light

    def _small_clips(self, level: _Level) -> torch.Tensor:
        """[H,W] in [0,1]: the small sensor-clipped white spots of the neutral groups that sit on paint that is
        not white in the photo, and GLINT_RING0 px around them: a glint or a speck there says nothing about the
        light a white paint is under (the candidates of rule 7e: a channel >= CLIP_LIN, OK chroma <= GLINT_C,
        8-connected components of up to GLINT_AREA2 px fading out by GLINT_AREA3; "not white": the mean
        :meth:`_white_photo` of the unclipped pixels of the spot's own group GLINT_RING0 to GLINT_RING1 px from
        its pixels, from SMALL_CLIP_WHITE0 (excluded) to SMALL_CLIP_WHITE1 (kept: the clipping specks scattered
        over a lit white face are that face's light), as far as it rests on GLINT_STAT_PX0..1 such pixels (a
        spot with none around it is kept). Deterministic: the per-spot sums are float64 on the CPU. Per level,
        once (CPU components, GPU rings)."""
        if level.small_clips is None:
            px = self._px(level)
            photo = (level.product + level.residual).clamp(0.0, 1.0)
            ok = linear_to_oklab_t(photo)
            cand = ((photo.amax(dim=-1) >= CLIP_LIN) & (torch.hypot(ok[..., 1], ok[..., 2]) <= GLINT_C)
                    & (self._own_neutral(level) > 0.0))
            out = torch.zeros(tuple(cand.shape), dtype=torch.float32, device=self.device)
            if bool(cand.any()):
                H, W = cand.shape
                n, comp, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(cand.to(torch.uint8).cpu().numpy()),
                                                                    connectivity=8)
                a = stats[:, cv2.CC_STAT_AREA].astype(np.float64) / max(px * px, 1e-9)
                t = np.clip((a - GLINT_AREA2) / (GLINT_AREA3 - GLINT_AREA2), 0.0, 1.0)
                small = torch.from_numpy((1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)).to(self.device)
                small[0] = 0.0
                gm = self._pool_map(level)
                white = self._white_photo(level)
                k0 = max(1, int(round(GLINT_RING0 * px)))
                near_spot = F.max_pool2d(cand.to(torch.float32)[None, None], 2 * k0 + 1, 1, k0)[0, 0] > 0.0
                ys, xs = cand.nonzero(as_tuple=True)
                g_c = gm[ys, xs]
                acc = torch.zeros(ys.shape[0], dtype=torch.float32, device=self.device)
                cnt = torch.zeros_like(acc)
                r0, r1 = GLINT_RING0 * px, GLINT_RING1 * px
                R = int(math.ceil(r1))
                for dy in range(-R, R + 1):
                    for dx in range(-R, R + 1):
                        d2 = dy * dy + dx * dx
                        if d2 <= r0 * r0 or d2 > r1 * r1:
                            continue
                        qy, qx = (ys + dy).clamp(0, H - 1), (xs + dx).clamp(0, W - 1)
                        v = ((ys + dy >= 0) & (ys + dy < H) & (xs + dx >= 0) & (xs + dx < W)
                             & (gm[qy, qx] == g_c) & ~near_spot[qy, qx])
                        acc = acc + torch.where(v, white[qy, qx], torch.zeros_like(acc))
                        cnt = cnt + v.to(torch.float32)
                # per spot, summed in float64 on the CPU (a weighted bincount on the GPU sums in no fixed order, and
                # the preview and the export must agree); the ring's whiteness counts as far as it rests on
                # GLINT_STAT_PX0..1 px (the most that one of the spot's pixels has around it), and a spot without
                # such pixels around it is the white face's own light
                c_np = comp[ys.cpu().numpy(), xs.cpu().numpy()].astype(np.int64)
                c_id = torch.from_numpy(c_np).to(self.device)
                num = np.bincount(c_np, weights=acc.cpu().numpy().astype(np.float64), minlength=n)
                den = np.bincount(c_np, weights=cnt.cpu().numpy().astype(np.float64), minlength=n)
                most = torch.zeros(n, dtype=cnt.dtype, device=self.device).scatter_reduce(
                    0, c_id, cnt, reduce="amax", include_self=True).cpu().numpy().astype(np.float64)   # (a max: exact)
                trust = np.clip((most / max(px * px, 1e-9) - GLINT_STAT_PX0) / (GLINT_STAT_PX1 - GLINT_STAT_PX0), 0.0, 1.0)
                trust = trust * trust * (3.0 - 2.0 * trust)
                ring = 1.0 + trust * (num / np.maximum(den, 1.0) - 1.0)
                t = np.clip((ring - SMALL_CLIP_WHITE0) / (SMALL_CLIP_WHITE1 - SMALL_CLIP_WHITE0), 0.0, 1.0)
                e = small * torch.from_numpy((1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)).to(self.device)
                out[ys, xs] = e[c_id]
                out = _group_dilate_t(out, gm, k0)
            level.small_clips = out.contiguous()
        return level.small_clips

    def _photo_lightness(self, level: _Level) -> torch.Tensor:
        """[H,W] OK lightness of the photograph (``albedo * shading + residual``, clipped). Per level, once."""
        if level.photo_L is None:
            level.photo_L = linear_to_oklab_t((level.product + level.residual).clamp(0.0, 1.0))[..., 0].contiguous()
        return level.photo_L

    def _white_glints(self, level: _Level) -> torch.Tensor:
        """[H,W] >= 0: the neutral groups' own glints (rule 7e, GLINT_*) as the white light (linear
        luminance) a neutral source's repaint gets back on its labels; 0 everywhere else. Found once at the
        base level and resized for the others, so a preview keeps exactly the glints of the full render."""
        if level.white_glints is None:
            if level is self._base:
                level.white_glints = self._find_white_glints(level)
            else:
                base = self._white_glints(self._base)[None, None]
                h, w = level.group_map.shape
                if w < base.shape[-1]:
                    t = F.interpolate(base, size=(h, w), mode="area")
                else:
                    t = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
                level.white_glints = t[0, 0].clamp_min(0.0)
        return level.white_glints

    def _find_white_glints(self, level: _Level) -> torch.Tensor:
        """The glint field of :meth:`_white_glints` at ``level``. A glint is judged within its own group
        (the instances of a split part read as one, :meth:`_pools`): its ring is that group's paint around it
        (a backdrop beyond a lit edge is not the paint the edge stands out from) and it is added back only
        on that group's pixels, times the spot's weight (GLINT_*: a product of smoothsteps, never a hard
        cut). CPU components, a few ms."""
        px = self._px(level)
        photo = (level.product + level.residual).clamp(0.0, 1.0)
        ok = linear_to_oklab_t(photo)
        clipped = photo.amax(dim=-1) >= CLIP_LIN
        cand = clipped & (torch.hypot(ok[..., 1], ok[..., 2]) <= GLINT_C) & (self._own_neutral(level) > 0.0)
        out = np.zeros(tuple(cand.shape), np.float32)
        if not bool(cand.any()):
            return torch.from_numpy(out).to(self.device)
        n, cc, st, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(cand.to(torch.uint8).cpu().numpy()),
                                                        connectivity=8)
        px2 = max(px * px, 1e-9)
        # a component that runs over two groups is two glints, one per group
        gm = self._pool_map(level).cpu().numpy()
        ng = int(gm.max()) + 1
        on_c = cc > 0
        pair = cc[on_c].astype(np.int64) * ng + gm[on_c]
        keys, counts = np.unique(pair, return_counts=True)
        area = counts / px2
        keep = (area > GLINT_AREA0) & (area < GLINT_AREA3)
        if not keep.any():
            return torch.from_numpy(out).to(self.device)
        Y = luminance_t(photo).cpu().numpy()
        S = luminance_t(level.shading).cpu().numpy()
        clip_np = clipped.cpu().numpy()
        albedo_L = _toe_t(level.albedo_ok[..., 0]).cpu().numpy()
        group_L = self._own_colors()[:, 0].cpu().numpy()
        r0, r1, fall = GLINT_RING0 * px, GLINT_RING1 * px, max(GLINT_FALLOFF_PX * px, 1.0)
        pad = int(math.ceil(max(r1, fall))) + 2
        H, W = out.shape

        def up(v: float, lo: float, hi: float) -> float:
            return _smooth01((v - lo) / (hi - lo))

        for key, a in zip(keys[keep].tolist(), area[keep].tolist()):
            i, g = divmod(int(key), ng)
            weight = up(a, GLINT_AREA0, GLINT_AREA1) * (1.0 - up(a, GLINT_AREA2, GLINT_AREA3))
            x, y, w, h = (int(v) for v in st[i, :4])
            y0, y1, x0, x1 = max(0, y - pad), min(H, y + h + pad), max(0, x - pad), min(W, x + w + pad)
            own = gm[y0:y1, x0:x1] == g
            core = (cc[y0:y1, x0:x1] == i) & own
            d = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
            ring = (d > r0) & (d <= r1) & own
            n_ring = int(ring.sum())
            weight *= up(n_ring / px2, GLINT_RING_PX0, GLINT_RING_PX1)
            if weight <= 0.0:
                continue
            Yc, Sc = Y[y0:y1, x0:x1], S[y0:y1, x0:x1]
            ring_y, top = max(float(np.median(Yc[ring])), 1e-6), float(np.median(Yc[core]))
            shade = float(np.median(Sc[core])) / max(float(np.median(Sc[ring])), 1e-6)
            standout = top / ring_y
            weight *= (up(standout, GLINT_STANDOUT0, GLINT_STANDOUT1) * (1.0 - up(standout, GLINT_STANDOUT2, GLINT_STANDOUT3))
                       * up(standout / max(shade, 1e-6), GLINT_UNEXPLAINED0, GLINT_UNEXPLAINED1))
            if weight <= 0.0:
                continue          # a clipped plateau of the paint, a lit face the shading explains, lettering
            darker = float(group_L[g]) - float(np.median(albedo_L[y0:y1, x0:x1][ring]))
            weight *= 1.0 - up(darker, GLINT_RING_DARKER1, GLINT_RING_DARKER0)
            lit = ring & ~clip_np[y0:y1, x0:x1]
            n_lit = int(lit.sum())
            if n_lit > 0:
                # as far as the shading steps up under the spot (its ring's median may then sit on a darker
                # face) and as far as the percentile rests on enough pixels
                gate = up(shade, GLINT_LIT_SHADE0, GLINT_LIT_SHADE1) * up(n_lit / px2, GLINT_STAT_PX0, GLINT_STAT_PX1)
                lit_f = up(top / max(float(np.quantile(Yc[lit], 0.9)), 1e-6), GLINT_LIT0, GLINT_LIT1)
                weight *= 1.0 + gate * (lit_f - 1.0)
            if weight <= 0.0:
                continue          # not lit white paint around it, or a corner of a face as bright as it
            # the clipped core is at the sensor's limit: all of it gets the photo's brightness, the halo its profile
            prof = np.where(core, 1.0, np.clip((Yc - ring_y) / max(top - ring_y, 1e-6), 0.0, 1.0))
            spec = top * prof ** GLINT_PROFILE_GAMMA
            t = np.clip(d / fall, 0.0, 1.0)
            s = (spec * (1.0 - t * t * (3.0 - 2.0 * t)) * own * weight).astype(np.float32)
            np.maximum(out[y0:y1, x0:x1], s, out=out[y0:y1, x0:x1])
        return torch.from_numpy(out).to(self.device)

    def _edge_band(self, level: _Level) -> tuple[torch.Tensor, int]:
        """([H,W] bool, width): pixels within EDGE_BAND_PX of a pixel of another group.
        Decal islands neither count as another group nor belong to the band (the white at a
        letter's rim is the letter's), and neither does another instance of the same split part
        (:meth:`_pools`: the robot's two feet touch through their rim, and their seam, a boundary
        only after the split, changed 43 px of the render when the part was split)."""
        if level.band is None:
            e = max(1, int(round(EDGE_BAND_PX * self._px(level))))
            k = 2 * e + 1
            pool = self._pools()
            gm = level.group_map
            if bool((pool != np.arange(len(pool))).any()):
                gm = torch.from_numpy(pool).to(self.device)[gm]
            gmf = gm.to(torch.float32)
            isl = level.islands > 0.5 if level.islands is not None else None
            hi = gmf if isl is None else torch.where(isl, torch.full_like(gmf, -1e9), gmf)
            lo = gmf if isl is None else torch.where(isl, torch.full_like(gmf, 1e9), gmf)
            mxn = F.max_pool2d(hi[None, None], k, 1, e)[0, 0]
            mnn = -F.max_pool2d(-lo[None, None], k, 1, e)[0, 0]
            band = (mxn > gmf) | (mnn < gmf)
            if isl is not None:
                band = band & ~isl
            level.band, level.band_px = band, e
        return level.band, level.band_px

    def _edge_limit(self, level: _Level, v: torch.Tensor) -> torch.Tensor:
        """``v`` [H,W] limited, inside the boundary band, to the largest value of the
        interior pixels next to it: a highlight that runs into the edge continues, a
        boundary without one gets none."""
        band, e = self._edge_band(level)
        interior = torch.where(band, torch.zeros_like(v), v)
        lim = F.max_pool2d(interior[None, None], 2 * e + 3, 1, e + 1)[0, 0]
        return torch.where(band, torch.minimum(v, lim), v)

    def _locked_distance(self, level: _Level) -> Optional[torch.Tensor]:
        """[H,W] distance (px at this level) to the nearest locked group, None without one."""
        if not level.locked_dist_done:
            locked = [int(g.id) for g in self.groups if g.locked and 0 <= int(g.id) < self.n_groups]
            if locked:
                hard = torch.isin(level.group_map, torch.tensor(locked, dtype=torch.int64, device=self.device))
                if bool(hard.any()):
                    level.locked_dist = _edt(~hard)
            level.locked_dist_done = True
        return level.locked_dist

    def _light_reference(self) -> torch.Tensor:
        """[3] linear RGB colour of the scene's illuminant at luminance 1, estimated from
        the least colourful well-lit surfaces the way a white balance does (an image-wide
        median is dragged toward the paint when one colour fills the frame)."""
        if self._light_ref is None:
            base = self._base
            lum = luminance_t(base.shading).clamp_min(1e-6)[..., None]
            chrom = (base.shading / lum).reshape(-1, 3)
            step = max(1, chrom.shape[0] // 400_000)
            lab = base.albedo_ok.reshape(-1, 3)[::step]
            c_alb = torch.hypot(lab[:, 1], lab[:, 2])
            sample = chrom[::step]
            neutral = (c_alb <= torch.quantile(c_alb, 0.4).clamp_min(ILLUMINANT_C_MIN)) & (lab[:, 0] > ILLUMINANT_L_MIN)
            ref = sample[neutral] if int(neutral.sum()) >= 256 else sample
            img_med = ref.median(dim=0).values.clamp_min(1e-3)
            self._light_ref = img_med / luminance_t(img_med).clamp_min(1e-6)
        return self._light_ref

    # ------------------------------------------------------------ per-mapping tables

    @staticmethod
    def _key(ids: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted(int(i) for i in ids))

    def _painted(self, level: _Level, ids: Sequence[int]) -> torch.Tensor:
        sel = torch.tensor(list(ids) or [-1], dtype=torch.int64, device=self.device)
        return torch.isin(level.group_map, sel)

    def _mapped_distance(self, level: _Level, ids: Sequence[int]) -> torch.Tensor:
        """[H,W] distance (px at this level) to the nearest repainted pixel."""
        key = (level.size, self._key(ids))

        def make() -> torch.Tensor:
            hard = self._painted(level, key[1])
            if bool(hard.any()):
                return _edt(~hard)
            return torch.full(tuple(hard.shape), 1e9, dtype=torch.float32, device=self.device)

        return self._dist.get_or(key, make)

    def _paint_light_q(self, ids: Sequence[int]) -> float:
        """REFL_LIGHT_Q quantile of the photo's OK lightness inside the repainted labels: the
        ceiling of what a reflection of that paint can be."""
        key = self._key(ids)

        def make() -> float:
            base = self._base
            sel = self._painted(base, key)
            if int(sel.sum()) < 2:
                return 2.0
            L = linear_to_oklab_t((base.product + base.residual).clamp(0.0, 1.0)[sel])[:, 0]
            k = min(L.numel(), max(1, int(round(REFL_LIGHT_Q * (L.numel() - 1))) + 1))
            return float(L.kthvalue(k).values)

        return self._paint_light.get_or(key, make)

    def _shading_median(self, ids: Sequence[int]) -> float:
        """Median shading luminance of the repainted pixels: the pivot of the black floor."""
        key = self._key(ids)

        def make() -> float:
            base = self._base
            s = luminance_t(base.shading)[self._painted(base, key)]
            if s.numel() > 400_000:
                s = s[:: s.numel() // 400_000 + 1]
            return max(float(s.median()) if s.numel() else 1.0, 1e-3)

        return self._shade_median.get_or(key, make)

    def _source_ok_for(self, gid: int) -> np.ndarray:
        """[3] OKLab source colour of a group: ``ColorGroup.ref_lab`` (an instance of a split
        part: the part's albedo) or else ``albedo_lab``, converted from CIELAB; the median OKLab
        albedo of its pixels when the group is unknown."""
        cached = self._source_ok.get(gid)
        if cached is not None:
            return cached
        g = self._group_by_id.get(gid)
        if g is not None:
            # a part split by instance paints every instance from the part's albedo (ref_lab)
            ref = getattr(g, "ref_lab", None)
            lab = cielab_to_oklab(ref if ref is not None else g.albedo_lab)
        else:
            sel = self._base.group_map == gid
            if bool(sel.any()):
                lab = self._base.albedo_ok[sel].median(dim=0).values.cpu().numpy().astype(np.float32)
            else:
                lab = cielab_to_oklab((50.0, 0.0, 0.0))
        self._source_ok[gid] = lab
        return lab

    def _group_params(self, mapping: Mapping, options: RenderOptions) -> _GroupParams:
        mode = (options.mode or "shift").lower()
        if mode not in ("shift", "flat"):
            raise ValueError(f"RenderOptions.mode must be 'shift' or 'flat', got {options.mode!r}")
        texture = float(np.clip(options.texture, 0.0, 1.0)) if mode == "shift" else 0.0
        saturation = max(0.0, float(options.saturation))
        table = np.zeros((max(self.n_groups, 1), _N_COLS), np.float32)
        mapped_ids: list[int] = []
        refl: list[_ReflParams] = []
        refl_of: dict[tuple, _ReflParams] = {}
        for gid, hexcol in normalize_mapping(mapping).items():
            if gid < 0 or gid >= self.n_groups:
                continue
            g = self._group_by_id.get(gid)
            if g is not None and g.locked:
                continue
            t_ok = hex_to_oklab(hexcol)
            t_ok = np.concatenate((t_ok[:1], t_ok[1:] * np.float32(saturation))).astype(np.float32)
            a_ok = self._source_ok_for(gid)
            # Chroma: ab' = T_ab + s R(theta) (ab - A_ab); s shrinks the texture when the target
            # is less saturated than the source, so hue noise is not amplified.
            c_src = float(np.hypot(a_ok[1], a_ok[2]))
            c_tgt = float(np.hypot(t_ok[1], t_ok[2]))
            s_raw = 1.0 if c_src < 1e-6 else min(1.0, c_tgt / c_src)
            s_ab = texture * s_raw
            theta = _hue_turn(a_ok, t_ok)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            m00, m01, m10, m11 = s_ab * cos_t, -s_ab * sin_t, s_ab * sin_t, s_ab * cos_t
            a_a, a_b = float(a_ok[1]), float(a_ok[2])
            # Lightness on the toe axis, anchored on the group's own lightness and measured
            # from black: below the anchor the darker half is squeezed into the room the new
            # colour has (a plain shift crushed 54 % of a red part to black on a dark target);
            # above it the texture is kept, attenuated for dark targets.
            a_L, t_L = _toe(float(a_ok[0])), _toe(float(t_ok[0]))
            k = TOE_BLACK
            slope_dn = 1.0 if a_L - k <= 1e-5 else min(1.0, max(0.0, (t_L - k) / (a_L - k)))
            slope_up = max(slope_dn, min(1.0, (t_L - k) / DARK_SLOPE_REF))
            # a neutral source (a white paint): the neutral-source rules, weighted by how neutral it is and
            # by how far the target is from the source colour (a map to the swatch's own colour is none)
            w_n = float(self._group_neutral_weights()[gid])
            w_white = float(self._group_white[gid])
            if w_n > 0.0:
                t_lab = imageio.hex_to_lab(hexcol)
                fade = _near_source_fade((t_lab[0], t_lab[1] * saturation, t_lab[2] * saturation),
                                         self._source_lab_for(gid))
                w_n, w_white = w_n * fade, w_white * fade
            if w_n > 0.0:
                # above the anchor a white albedo's deviations are light the decomposition left in
                # it: NEUTRAL_UP_K of the room ratio below the anchor (0: flat)
                slope_up = slope_up + w_n * (NEUTRAL_UP_K * slope_dn - slope_up)
            table[gid] = (1.0, texture, m00, m01, m10, m11,
                          s_raw * cos_t, -s_raw * sin_t, s_raw * sin_t, s_raw * cos_t,
                          t_L, a_L, slope_dn,
                          float(t_ok[1]) - (m00 * a_a + m01 * a_b), float(t_ok[2]) - (m10 * a_a + m11 * a_b),
                          float(t_ok[1]), float(t_ok[2]), a_a, a_b, slope_up, w_n, w_white)
            mapped_ids.append(gid)
            # a neutral source's colour cast is the light's, not a paint hue to follow; and a source with
            # little hue has little of it to follow (a smoothstep, so its reflections do not switch on at once)
            refl_w = (1.0 - w_n) * _hue_confidence(c_src)
            if refl_w > 0.0:
                # groups repainted from one colour to one target reflect as one source (the
                # instances of a part split by instance, which share the part's albedo)
                key = (tuple(np.round(a_ok, 6).tolist()), tuple(np.round(t_ok, 6).tolist()))
                if key in refl_of:
                    refl_of[key].gids += (gid,)
                else:
                    rp = self._refl_params(a_ok, t_ok, s_raw, cos_t, sin_t)
                    rp.gids = (gid,)
                    rp.weight = refl_w
                    refl_of[key] = rp
                    refl.append(rp)
        for rp in refl:
            rp.reach = self._reach(rp.gids)
        allow = self._reflection_permissions(mapped_ids, refl)
        return _GroupParams(torch.from_numpy(table).to(self.device), tuple(mapped_ids), tuple(refl), allow)

    def _pixels(self, gid: int) -> float:
        """Pixels of group ``gid`` at the base level."""
        if self._group_px is None:
            gm = self._base.group_map.reshape(-1)
            self._group_px = torch.bincount(gm, minlength=max(self.n_groups, 1)).cpu().numpy()
        return float(self._group_px[gid]) if 0 <= gid < len(self._group_px) else 0.0

    def _reach(self, gids: Sequence[int]) -> float:
        """How far (px at the reference size) rule 8 looks for the reflections of the groups
        ``gids`` (one source): REFL_FALLOFF_PX, or for a source below REFL_SMALL_FRAC of the
        image REFL_REACH_K x the square root of its area (at most REFL_FALLOFF_PX)."""
        a = sum(self._pixels(g) for g in gids)
        if a >= REFL_SMALL_FRAC * float(self._base.group_map.numel()):
            return REFL_FALLOFF_PX
        px = self._px(self._base)
        return float(min(REFL_FALLOFF_PX, REFL_REACH_K * math.sqrt(max(a, 0.0)) / max(px, 1e-6)))

    def _group_distance(self, level: _Level, gids: Sequence[int], reach_px: float) -> Optional[tuple[torch.Tensor, tuple]]:
        """([h, w] distance, px at this level, to the pixels of the groups ``gids``, (y0, y1, x0,
        x1)) in the window of ``reach_px`` around them, or None when they have no pixel here.
        Cached per level and group set."""
        key = (level.size, self._key(gids), round(float(reach_px), 2))

        def make():
            hard = self._painted(level, key[1])
            nz = hard.nonzero()
            if nz.numel() == 0:
                return None
            H, W = hard.shape
            r = int(math.ceil(reach_px)) + 2
            y0, y1 = max(0, int(nz[:, 0].min()) - r), min(H, int(nz[:, 0].max()) + r + 1)
            x0, x1 = max(0, int(nz[:, 1].min()) - r), min(W, int(nz[:, 1].max()) + r + 1)
            return _edt(~hard[y0:y1, x0:x1]), (y0, y1, x0, x1)

        return self._gdist.get_or(key, make)

    def _refl_params(self, a_ok: np.ndarray, t_ok: np.ndarray, s: float, cos_t: float, sin_t: float) -> _ReflParams:
        dev = self.device
        a_lin = oklab_to_linear_t(torch.tensor(a_ok, dtype=torch.float32, device=dev))
        t_lin = oklab_to_linear_t(torch.tensor(t_ok, dtype=torch.float32, device=dev))
        excess = a_lin - a_lin.min()
        t_back = linear_to_oklab_t(t_lin[None])[0]
        t_c = float(torch.linalg.norm(t_back[1:]))
        target_dir = None
        if t_c > REFL_TARGET_MIN_CHROMA:
            tn = t_back[1:] / torch.linalg.norm(t_back[1:]).clamp_min(1e-6)
            target_dir = (float(tn[0]), float(tn[1]))
        return _ReflParams(hue_deg=math.degrees(math.atan2(float(a_ok[2]), float(a_ok[1]))), s=s, cos=cos_t, sin=sin_t,
                           excess=excess, excess_sq=max(1e-9, float((excess * excess).sum())),
                           target_offset=t_lin - a_lin.min(), target_dir=target_dir)

    def _reflection_permissions(self, mapped_ids: Sequence[int], refl: Sequence[_ReflParams]) -> torch.Tensor:
        """[G] rule 8's permission: 0 in the repainted groups, in locked groups and in
        unmapped groups with a real colour of their own (gold wheels, a brown seat), 1
        elsewhere. Also stops every repainted group's hue window REFL_HUE_MARGIN short of
        each unlocked coloured group's hue (in place on ``refl``). Locked groups do not cut
        the window: they protect their neighbourhood spatially (LOCK_PROTECT_PX)."""
        allow = np.ones(max(self.n_groups, 1), np.float32)
        mapped = set(int(i) for i in mapped_ids)
        allow[list(mapped)] = 0.0
        protected_hues: list[tuple[float, float]] = []           # (OK hue, pixels) of unlocked coloured groups
        for g in self.groups:
            gid = int(g.id)
            if gid < 0 or gid >= self.n_groups or gid in mapped:
                continue
            c_lab = math.hypot(float(g.albedo_lab[1]), float(g.albedo_lab[2]))
            if g.locked or c_lab >= REFL_PROTECT_CHROMA:
                allow[gid] = 0.0
                if c_lab >= REFL_PROTECT_CHROMA and not g.locked:
                    ok = self._source_ok_for(gid)
                    protected_hues.append((math.degrees(math.atan2(float(ok[2]), float(ok[1]))), self._pixels(gid)))
        z_max = REFL_HUE_WINDOW + REFL_HUE_RAMP
        same_hue = []
        for rp in refl:
            lim_pos = lim_neg = 180.0
            closed = False
            for hc, area in protected_hues:
                d = (hc - rp.hue_deg + 180.0) % 360.0 - 180.0
                if abs(d) < REFL_HUE_MARGIN and area >= REFL_SAME_HUE_AREA * sum(self._pixels(g) for g in rp.gids):
                    closed = True
                if d >= 0.0:
                    lim_pos = min(lim_pos, d - REFL_HUE_MARGIN)
                else:
                    lim_neg = min(lim_neg, -d - REFL_HUE_MARGIN)
            rp.z_pos = max(0.0, min(z_max, lim_pos))
            rp.z_neg = max(0.0, min(z_max, lim_neg))
            same_hue.append(closed)
        if refl:
            # (the own-colour test below asks which groups are objects of the paint's colour: it
            # reads the window as cut, before a same-hue source closes its window altogether)
            share = self._own_colour_share(mapped_ids, refl)
            for g in self.groups:
                gid = int(g.id)
                if (0 <= gid < self.n_groups and allow[gid] > 0.0 and share[gid] >= REFL_OWN_COLOUR_SHARE
                        and float(g.albedo_lab[0]) >= REFL_OWN_COLOUR_MIN_L):
                    allow[gid] = 0.0
        for rp, closed in zip(refl, same_hue):
            if closed:
                # an unmapped object of this source's own colour (within REFL_HUE_MARGIN) at least
                # REFL_SAME_HUE_AREA of its size: every pixel of that hue may be its reflection
                # (the Ducati's gold frame beside its repainted spring), so the window closes on
                # both sides; a crumb of the paint's hue does not close the paint's
                rp.z_pos = rp.z_neg = 0.0
        return torch.from_numpy(allow).to(self.device)

    def _photo_colour(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Base level, flattened: the photo's OK hue (deg), rule 8's chroma gate > 1/2
        (bool) and the group id of every pixel. Computed once per renderer."""
        if self._photo_hc is None:
            base = self._base
            ok = linear_to_oklab_t((base.product + base.residual).clamp(0.0, 1.0)).reshape(-1, 3)
            c = torch.hypot(ok[:, 1], ok[:, 2])
            c0 = (REFL_CHROMA_CUT - REFL_CHROMA_RAMP) * OK_PER_LAB_CHROMA
            c1 = (REFL_CHROMA_CUT + REFL_CHROMA_RAMP) * OK_PER_LAB_CHROMA
            gate = _smoothstep_t((c - c0) / (c1 - c0)) > 0.5
            hue = torch.rad2deg(torch.atan2(ok[:, 2], ok[:, 1]))
            self._photo_hc = (hue, gate, base.group_map.reshape(-1))
        return self._photo_hc

    def _own_colour_share(self, mapped_ids: Sequence[int], refl: Sequence[_ReflParams]) -> np.ndarray:
        """[G] the share of each group's photo pixels that pass rule 8's colour test for any
        repainted source (chroma gate and the hue window as cut for this mapping)."""
        key = (self._key(mapped_ids), tuple((round(rp.hue_deg, 4), round(rp.z_pos, 4), round(rp.z_neg, 4)) for rp in refl))

        def make() -> np.ndarray:
            hue, gate, gm = self._photo_colour()
            inwin = torch.zeros_like(gate)
            for rp in refl:
                dh = (hue - rp.hue_deg + 180.0) % 360.0 - 180.0
                z = torch.where(dh >= 0.0, torch.full_like(dh, rp.z_pos), torch.full_like(dh, rp.z_neg))
                inwin |= dh.abs() < (z - REFL_HUE_RAMP).clamp_min(0.0)
            n = torch.bincount(gm, minlength=max(self.n_groups, 1)).float()
            hit = torch.bincount(gm[gate & inwin], minlength=max(self.n_groups, 1)).float()
            return (hit / n.clamp_min(1.0)).cpu().numpy()

        return self._share.get_or(key, make)

    # ------------------------------------------------------------ rules 1-2: coverage and albedo

    def _coverage(self, level: _Level, hard: torch.Tensor, sigma: float,
                  gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rule 1: how much of each pixel belongs to a repainted group, in [0,1].

        Snap the mapped indicator onto the photograph's edges (colour guided filter), take
        the max with the hard label (coverage may only add to what the user selected), spend
        the soft edge entirely *outside* the paint (blur + smoothstep whose top sits on the
        boundary) and keep the hard label again. ``sigma`` is the ramp width in pixels at
        this level. ``gate`` [H,W] (see :meth:`_ramp_gate`) limits the ramp to the pixels
        that still hold some of the old paint; where it is 0 the coverage is the snapped
        indicator itself (never more than the ramp). An island pixel is covered only when
        its own group is mapped."""
        m = filters.guided_filter_color(self._guide(level), hard, COVERAGE_RADIUS, COVERAGE_EPS).clamp(0.0, 1.0)
        m = torch.maximum(m, hard)
        if sigma > 0.0:
            g = _feather_chw(m[None], sigma)[0]
            ramp = _smoothstep_t((g - RAMP_LO) / (RAMP_HI - RAMP_LO))
            if gate is None:
                m = torch.maximum(ramp, hard)
            else:
                m = torch.maximum(torch.maximum(ramp * gate, torch.minimum(ramp, m)), hard)
        if level.islands is not None:
            m = m * (1.0 - level.islands * (1.0 - hard))
        return m

    def _own_colors(self) -> torch.Tensor:
        """[G,3] every group's own colour on the ramp gate's axes (toe lightness, OK a, b)."""
        if self._own_ok is None:
            rows = []
            for gid in range(max(self.n_groups, 1)):
                ok = self._source_ok_for(gid)
                rows.append((_toe(float(ok[0])), float(ok[1]), float(ok[2])))
            self._own_ok = torch.tensor(rows, dtype=torch.float32, device=self.device)
        return self._own_ok

    def _soft_edges(self, level: _Level) -> torch.Tensor:
        """[H,W] in [0,1]: 1 where the photograph's own edge is soft (SOFT_EDGE_*) at a group
        boundary, 0 at a sharp edge, away from every boundary, on a wide gradient or where
        there is no edge. Computed once per level."""
        if level.soft_edges is None:
            g = self._guide(level)
            px = self._px(level)
            step = torch.zeros_like(g)
            step[:, :, :-1] = (g[:, :, 1:] - g[:, :, :-1]).abs()
            step[:, :-1, :] = torch.maximum(step[:, :-1, :], (g[:, 1:, :] - g[:, :-1, :]).abs())

            def width_at(radius: int) -> tuple[torch.Tensor, torch.Tensor]:
                r = max(1, int(round(radius * px)))
                k = 2 * r + 1
                rng = F.max_pool2d(g[None], k, 1, r)[0] + F.max_pool2d(-g[None], k, 1, r)[0]
                st = F.max_pool2d(step[None], k, 1, r)[0]
                best = rng.argmax(dim=0, keepdim=True)
                return torch.gather(rng / st.clamp_min(1e-4), 0, best)[0] / max(px, 1e-6), rng.amax(dim=0)

            width, strength = width_at(SOFT_EDGE_RADIUS)
            wide, _ = width_at(SOFT_EDGE_WIDE_RADIUS)
            gm = level.group_map
            edge = torch.zeros(gm.shape, dtype=torch.bool, device=gm.device)
            edge[:, :-1] |= gm[:, 1:] != gm[:, :-1]
            edge[:, 1:] |= gm[:, 1:] != gm[:, :-1]
            edge[:-1, :] |= gm[1:, :] != gm[:-1, :]
            edge[1:, :] |= gm[1:, :] != gm[:-1, :]
            rb = max(1, int(round(SOFT_EDGE_BAND * px)))
            near = F.max_pool2d(edge.to(g.dtype)[None, None], 2 * rb + 1, 1, rb)[0, 0]
            level.soft_edges = (_smoothstep_t((width - SOFT_EDGE_W0) / (SOFT_EDGE_W1 - SOFT_EDGE_W0))
                                * (1.0 - _smoothstep_t((wide - SOFT_EDGE_W2) / (SOFT_EDGE_W3 - SOFT_EDGE_W2)))
                                * _smoothstep_t((strength - SOFT_EDGE_R0) / (SOFT_EDGE_R1 - SOFT_EDGE_R0))
                                * near)
        return level.soft_edges

    def _ramp_gate(self, level: _Level, a_L: torch.Tensor, src_ab: torch.Tensor) -> torch.Tensor:
        """[H,W] rule 1's ramp gate: how much of the repainted paint (``a_L``, ``src_ab``: the
        mapped source colour blended over the neighbourhood) each pixel's albedo holds against
        its own group's colour, smoothstepped from RAMP_GATE_A0 to RAMP_GATE_A1; 1 where the
        two colours are too close to tell (RAMP_GATE_SEP*) and where the photo's own edge is
        soft (:meth:`_soft_edges`)."""
        own = self._own_colors()[level.group_map]
        paint = torch.cat((a_L[..., None], src_ab), dim=-1)
        pix = torch.cat((_toe_t(level.albedo_ok[..., 0])[..., None], level.albedo_ok[..., 1:]), dim=-1)
        axis = paint - own
        sep2 = (axis * axis).sum(-1)
        alpha = (((pix - own) * axis).sum(-1) / sep2.clamp_min(1e-8)).clamp(0.0, 1.0)
        gate = _smoothstep_t((alpha - RAMP_GATE_A0) / (RAMP_GATE_A1 - RAMP_GATE_A0))
        close = 1.0 - _smoothstep_t((sep2.sqrt() - RAMP_GATE_SEP0) / (RAMP_GATE_SEP1 - RAMP_GATE_SEP0))
        return torch.maximum(torch.maximum(gate, close), self._soft_edges(level))

    def _island_entry(self, level: _Level, params: _GroupParams,
                      hard: torch.Tensor) -> Optional[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]]:
        """Rule 1 for a neutral source: the decal-island pixels it enters, as ``(idx, coverage [N], rows
        [N, C])`` (the pixels, how much of each is repainted, the parameter row each is repainted with),
        or None. A decal carved out of a white paint is dark or coloured, and the pixels of it that hold
        the white paint (its anti-aliased rim, and paint the analysis took into the decal: the white
        leather between the sneakers' FILA letters) are paint: kept out of the repaint they stayed pale
        speckle on the new colour. Which pixels (ISLAND_*, :meth:`_find_island_entry`) depends only on
        which groups are mapped, so it is cached per mapped set; the rows follow the targets, and so does
        the coverage, times the mapping's neutral weight of the paint each pixel takes (none for a map to
        the paint's own colour)."""
        key = (level.size, self._key(params.mapped_ids))
        found = self._island_in.get_or(key, lambda: self._find_island_entry(level, hard))
        if found is None:
            return None
        idx, cover, nid = found
        rows = params.table[nid]
        cover = cover * rows[:, _C_NEUTRAL]          # the mapping's neutral weight (0 for a map to its own colour)
        on = cover > 0.0
        if not bool(on.any()):
            return None
        return (idx[0][on], idx[1][on]), cover[on], rows[on]

    def _find_island_entry(self, level: _Level, hard: torch.Tensor):
        """``(idx, coverage, group ids)`` of :meth:`_island_entry` for the mapped labels ``hard``, or None.
        Every island pixel within ISLAND_REACH_PX px of the labels is judged against the paint of the label
        pixel nearest to it (a distance transform with labels, on the CPU over the islands' bounding box):
        how much of that paint its albedo holds against its own group's colour (the projection of
        :meth:`_ramp_gate`, ISLAND_ENTRY_A0..A1), the two colours clearly apart (RAMP_GATE_SEP*), the source
        a neutral group. The pieces of such pixels that touch the label (ISLAND_TOUCH_PX) are entered, as far
        as they are no lighter in the photo than the paint along their contact (ISLAND_LIGHTER*); the
        coverage leaves out the neutral weight, which :meth:`_island_entry` applies per mapping."""
        px = self._px(level)
        mapped_t = hard > 0.5
        isl_t = (level.islands > 0.5) & ~mapped_t
        if level.protect is not None:
            isl_t = isl_t & (level.protect < 0.5)       # an object in its own right
        locked = [int(g.id) for g in self.groups if g.locked and 0 <= int(g.id) < self.n_groups]
        if locked:
            isl_t = isl_t & ~torch.isin(level.group_map, torch.tensor(locked, dtype=torch.int64, device=self.device))
        if not bool(isl_t.any()) or not bool(mapped_t.any()):
            return None
        reach, touch = max(1.0, ISLAND_REACH_PX * px), max(1.0, ISLAND_TOUCH_PX * px)
        H, W = isl_t.shape
        ys, xs = isl_t.nonzero(as_tuple=True)
        m = int(math.ceil(reach)) + 2
        y0, y1 = max(0, int(ys.min()) - m), min(H, int(ys.max()) + m + 1)
        x0, x1 = max(0, int(xs.min()) - m), min(W, int(xs.max()) + m + 1)
        mapped = np.ascontiguousarray(mapped_t[y0:y1, x0:x1].cpu().numpy())
        if not mapped.any():
            return None
        isl = isl_t[y0:y1, x0:x1].cpu().numpy()
        free = (~mapped).astype(np.uint8)
        dist, lab = cv2.distanceTransformWithLabels(free, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        cand = isl & (dist <= reach)
        if not cand.any():
            return None
        gm = level.group_map[y0:y1, x0:x1].cpu().numpy()
        lut = np.zeros(int(lab.max()) + 1, np.int64)
        lut[lab[mapped]] = np.flatnonzero(mapped)
        cy, cx = np.nonzero(cand)
        nid = gm.reshape(-1)[lut[lab[cy, cx]]]
        weights = self._group_neutral_weights()
        w_n = (weights[np.clip(nid, 0, len(weights) - 1)] > 0.0).astype(np.float32)   # (times the mapping's weight later)
        own = self._own_colors().cpu().numpy()
        ok = level.albedo_ok[y0:y1, x0:x1].cpu().numpy()[cy, cx]
        pix = np.stack((np.where(ok[:, 0] >= _TOE_L0, ok[:, 0], (_TOE_K * np.clip(ok[:, 0], 0.0, None) ** 3 + 16.0) / 116.0),
                        ok[:, 1], ok[:, 2]), axis=-1)
        mine, paint = own[gm[cy, cx]], own[nid]
        axis = paint - mine
        sep2 = (axis * axis).sum(-1)
        alpha = np.clip(((pix - mine) * axis).sum(-1) / np.maximum(sep2, 1e-8), 0.0, 1.0)

        def ss(t):
            t = np.clip(t, 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
        gate = (w_n * ss((alpha - ISLAND_ENTRY_A0) / (ISLAND_ENTRY_A1 - ISLAND_ENTRY_A0))
                * ss((np.sqrt(sep2) - RAMP_GATE_SEP0) / (RAMP_GATE_SEP1 - RAMP_GATE_SEP0))
                * (1.0 - ss((dist[cy, cx] - 0.75 * reach) / (0.25 * reach))))
        on = gate > 0.0
        if not on.any():
            return None
        cy, cx, nid, gate = cy[on], cx[on], nid[on], gate[on]
        field = np.zeros(cand.shape, np.uint8)
        field[cy, cx] = 1
        n, cc, st, _ = cv2.connectedComponentsWithStats(field, connectivity=8)
        L = self._photo_lightness(level)[y0:y1, x0:x1].cpu().numpy()
        comp = cc[cy, cx]
        cover = np.zeros(len(cy), np.float32)
        ring_r = int(math.ceil(touch)) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_r + 1, 2 * ring_r + 1))
        for i in range(1, n):
            sel = comp == i
            if float(dist[cy[sel], cx[sel]].min()) > touch:
                continue                        # not a piece of the paint: it does not touch the label
            bx, by, bw, bh = (int(v) for v in st[i, :4])
            a0, a1 = max(0, by - ring_r), min(cc.shape[0], by + bh + ring_r)
            b0, b1 = max(0, bx - ring_r), min(cc.shape[1], bx + bw + ring_r)
            contact = (cv2.dilate((cc[a0:a1, b0:b1] == i).astype(np.uint8), kernel) > 0) & mapped[a0:a1, b0:b1]
            if not contact.any():
                continue
            paint_L = float(np.median(L[a0:a1, b0:b1][contact]))
            lighter = L[cy[sel], cx[sel]] - paint_L
            cover[sel] = gate[sel] * (1.0 - ss((lighter - ISLAND_LIGHTER0) / (ISLAND_LIGHTER1 - ISLAND_LIGHTER0)))
        keep = cover > 0.0
        if not keep.any():
            return None
        dev = self.device
        idx = (torch.from_numpy(cy[keep] + y0).to(dev), torch.from_numpy(cx[keep] + x0).to(dev))
        return idx, torch.from_numpy(cover[keep]).to(dev), torch.from_numpy(nid[keep].astype(np.int64)).to(dev)

    def _repaint(self, level: _Level, params: _GroupParams, options: RenderOptions) -> Optional[_Repaint]:
        """Rule 2 (and 3): the repainted albedo and the fields the later rules need, or
        None when nothing is mapped."""
        if not params.mapped_ids:
            return None
        sigma = RAMP_SIGMA * max(float(options.feather_px), 0.0) / RAMP_FEATHER_REF
        table = params.table[level.group_map]
        hard = table[..., _C_MAPPED]
        chw = table.permute(2, 0, 1).contiguous()
        far_r = max(2 * COVERAGE_RADIUS, int(2.0 * sigma) + 2, 4)
        field = (_feather_chw(chw, PARAM_FEATHER) + PARAM_FAR_WEIGHT * filters.box_filter(chw, far_r)).permute(1, 2, 0)
        has_neutral = bool((params.table[:, _C_NEUTRAL] > 0).any())
        entry = self._island_entry(level, params, hard) if has_neutral and level.islands is not None else None
        if entry is not None:
            field[entry[0]] = entry[2]              # an entered island pixel takes its paint's row
        inv = 1.0 / field[..., _C_MAPPED].clamp_min(1e-8)
        inv1 = inv[..., None]
        a_L = field[..., _C_AL] * inv
        src_ab = field[..., _C_SRC] * inv1
        gate = self._ramp_gate(level, a_L, src_ab) if sigma > 0.0 else None
        m = self._coverage(level, hard, sigma, gate)
        if entry is not None:
            # the paint a neutral source's decal islands hold is repainted like its labels
            m, hard = m.clone(), hard.clone()
            m[entry[0]] = torch.maximum(m[entry[0]], entry[1])
            hard[entry[0]] = torch.maximum(hard[entry[0]], entry[1])
        texture = field[..., _C_TEXTURE] * inv
        rot = field[..., _C_ROT] * inv1
        bounce = field[..., _C_BOUNCE] * inv1
        t_L = field[..., _C_TL] * inv
        slope_dn = field[..., _C_SLOPE_DN] * inv
        offset = field[..., _C_OFFSET] * inv1
        t_ab = field[..., _C_TAB] * inv1
        slope_up = field[..., _C_SLOPE_UP] * inv
        neutral = white = None
        if has_neutral:
            neutral = (field[..., _C_NEUTRAL] * inv).clamp(0.0, 1.0)
            white = (field[..., _C_WHITE] * inv).clamp(0.0, 1.0)
        d = _toe_t(level.albedo_ok[..., 0]) - a_L
        slope = texture * torch.where(d >= 0, slope_up, slope_dn)
        # At a clipped white highlight the albedo's hue offset from the group is glint
        # contamination, not texture: keep only its radial part there. (For a neutral source the
        # mask is the neutral-aware one: a white paint's clipped face is not a highlight.)
        ab = level.albedo_ok[..., 1:]
        hl3 = self._highlight(level)
        if neutral is not None:
            hl3 = hl3 + neutral[..., None] * (self._highlight_neutral(level) - hl3)
        hl = hl3[..., 0]
        dv = ab - src_ab
        u = src_ab / torch.linalg.norm(src_ab, dim=-1, keepdim=True).clamp_min(1e-6)
        radial = (dv * u).sum(-1, keepdim=True) * u
        ab = src_ab + radial + (1.0 - hl)[..., None] * (dv - radial)
        ab_new = torch.stack((rot[..., 0] * ab[..., 0] + rot[..., 1] * ab[..., 1],
                              rot[..., 2] * ab[..., 0] + rot[..., 3] * ab[..., 1]), dim=-1) + offset
        lab_new = torch.cat((_untoe_t(t_L + slope * d)[..., None], ab_new), dim=-1)
        target = torch.cat((_untoe_t(t_L)[..., None], t_ab), dim=-1)
        lin_new = oklab_to_linear_gamut_t(lab_new)
        albedo = level.albedo + m[..., None] * (lin_new - level.albedo)
        return _Repaint(albedo, m, hard, target, bounce, src_ab, neutral, hl3, white)

    # ------------------------------------------------------------ rules 5-7: light, residual, envelope

    def _retint_shading(self, level: _Level, shading: torch.Tensor, paint: _Repaint) -> torch.Tensor:
        """Rule 5: recolour the old paint's bounce light on the repainted pixels.

        The light's tint beyond the scene illuminant is measured in OKLab at a fixed
        luminance; its component along the old paint's chroma direction (only where it
        points *toward* that paint) is replaced by the same component turned to the new
        paint. At a clipped highlight the light goes to the illuminant. Luminance is kept
        exactly; a neutral source leaves the light untouched."""
        bounce, src_ab, m = paint.bounce, paint.src_ab, paint.coverage
        lum = luminance_t(shading)
        col = shading / lum.clamp_min(1e-6)[..., None] * LIGHT_LUM
        lab = linear_to_oklab_t(col)
        ref_ab = linear_to_oklab_t(self._light_reference() * LIGHT_LUM)[1:]
        d = lab[..., 1:] - ref_ab
        c_src = torch.linalg.norm(src_ab, dim=-1)
        u = src_ab / c_src.clamp_min(1e-9)[..., None]
        p = ((d * u).sum(-1).clamp_min(0.0) * _hue_confidence_t(c_src))[..., None]
        pu = p * u
        turned = torch.stack((bounce[..., 0] * pu[..., 0] + bounce[..., 1] * pu[..., 1],
                              bounce[..., 2] * pu[..., 0] + bounce[..., 3] * pu[..., 1]), dim=-1)
        ab_new = lab[..., 1:] - pu + turned
        hl = paint.highlight if paint.highlight is not None else self._highlight(level)
        ab_new = ab_new + hl * (ref_ab - ab_new)
        col_new = oklab_to_linear_gamut_t(torch.cat((lab[..., :1], ab_new), dim=-1))
        col_new = col_new / luminance_t(col_new).clamp_min(1e-6)[..., None] * lum[..., None]
        w = (m * (lum > 1e-6).to(m.dtype))[..., None]
        return shading + w * (col_new - shading)

    def _adjusted_residual(self, level: _Level, product: torch.Tensor, paint: Optional[_Repaint],
                           options: RenderOptions) -> torch.Tensor:
        """Rule 6: the residual that goes with ``product`` (see the module docstring)."""
        res = level.residual
        pos = res.clamp_min(0.0)
        neg = (res - pos) * (product / level.product.clamp_min(1e-6)).clamp(0.0, 1.0)
        if paint is None:
            return pos + neg
        m = paint.coverage[..., None]
        gray = self._edge_limit(level, pos.amin(dim=-1))[..., None]
        excess = pos - gray
        lum_new = luminance_t(product)[..., None]
        follow = (lum_new / luminance_t(level.product)[..., None].clamp_min(1e-6)).clamp(0.0, 1.0)
        spec = self._spec_weight(level)
        # The coloured excess wears the new paint. It is diffuse energy of the old paint, so on
        # the repainted labels a darker target keeps only follow ** EXCESS_FOLLOW_GAMMA of it
        # (kept whole, the yellow BMW's residual lifted a #1b2a57 navy to a medium royal blue,
        # four times the target's lightness); in the outward ramp it keeps its energy.
        keep = 1.0 - paint.hard[..., None] * (1.0 - follow.pow(EXCESS_FOLLOW_GAMMA))
        pos_new = gray * (spec + (1.0 - spec) * follow) + product * (luminance_t(excess)[..., None] * keep
                                                                     / lum_new.clamp_min(1e-6))
        if paint.neutral is not None:
            # A neutral source (a white paint): its neutral floor is the paint's own diffuse energy,
            # not a glint or a veil, and its coloured excess is the light's colour; both are rebuilt in
            # the new paint like the product itself (FLOOR_DIFFUSE_GAMMA, NEUTRAL_EXCESS_GAMMA).
            wn = paint.neutral[..., None]
            keep_n = 1.0 - paint.hard[..., None] * (1.0 - follow.pow(NEUTRAL_EXCESS_GAMMA))
            pos_n = product * ((luminance_t(excess)[..., None] * keep_n + gray * follow.pow(FLOOR_DIFFUSE_GAMMA))
                               / lum_new.clamp_min(1e-6))
            pos_new = torch.where(wn > 0, pos_new + wn * (pos_n - pos_new), pos_new)
        pos = pos + m * (pos_new - pos)
        tint = float(np.clip(options.residual_tint, 0.0, 1.0))
        if tint > 0.0:
            pos_ok = linear_to_oklab_t(pos)
            tinted = oklab_to_linear_gamut_t(torch.cat((pos_ok[..., :1], paint.target[..., 1:]), dim=-1))
            pos = pos + (tint * m) * (tinted - pos)
        # At a clipped highlight the (ratio-scaled) negative residual is the decomposition
        # overshooting the sensor, which does not belong to the new paint.
        hl = paint.highlight if paint.highlight is not None else self._highlight(level)
        return pos + neg - (hl * m) * neg

    def _paint_envelope(self, level: _Level, out: torch.Tensor, paint: _Repaint, params: _GroupParams) -> torch.Tensor:
        """Rule 7: black floor, gloss floor, hue and chroma envelope on the repainted pixels.

        ``out`` is the composed linear image. Only pixels with weight > 0 are converted;
        each moves by its weight toward its enveloped value. The weight is the coverage,
        times rule 8's permission (and the protect mask) outside the mapped labels."""
        hard, m = paint.hard, paint.coverage
        perm = params.allow[level.group_map]
        if level.protect is not None:
            perm = perm * (1.0 - level.protect)
        weight = m * (hard + (1.0 - hard) * perm)
        idx = (weight > 1e-4).nonzero(as_tuple=True)
        if idx[0].numel() == 0:
            return out
        o = out[idx]
        mm = weight[idx][:, None]
        t = paint.target[idx]
        # (a) a target darker than the darkest real paint: that paint's missing reflectance,
        # lit by the photo's own shading shaped toward the shadows.
        floor = (BLACK_FLOOR - t[:, 0].clamp_min(0.0).pow(3.0)).clamp_min(0.0)
        s_px = luminance_t(level.shading[idx])
        s_px = s_px * (s_px / self._shading_median(params.mapped_ids)).clamp_min(0.0).pow(BLACK_FLOOR_GAMMA - 1.0)
        base = o
        if paint.neutral is not None:
            # (a') a neutral source's exposure bound: at a white photo pixel no repaint reflects more
            # than a paint of its albedo under the light a white paint implies there (before the
            # black floor, which a black target's albedo of 0 would otherwise cap away)
            light, near = self._white_light(level)
            w_cap = paint.white[idx] * near[idx]
            y = luminance_t(o).clamp_min(1e-6)
            cap = (light[idx] * luminance_t(paint.albedo[idx].clamp(0.0, 1.0)) / WHITE_REF_Y).clamp_min(1e-6)
            knee = EXPOSURE_KNEE * cap
            yc = torch.where(y > knee, knee + (cap - knee) * torch.tanh((y - knee) / (cap - knee).clamp_min(1e-6)), y)
            base = o * (1.0 + w_cap * (yc / y - 1.0))[:, None]
        new = base + (floor * s_px)[:, None]
        if paint.neutral is not None:
            # (e) a white paint's own glints: the lamp's white light, back on top of the new paint
            new = new + (self._white_glints(level)[idx] * paint.neutral[idx] * hard[idx])[:, None]
        # (b) never below the photo's own white specular (for a neutral source: the neutral-aware
        # estimate, :meth:`_gloss_neutral`).
        gl = self._gloss(level)[idx]
        if paint.neutral is not None:
            wn = paint.neutral[idx]
            gl = torch.where(wn > 0, gl + wn * (self._gloss_neutral(level)[idx] - gl), gl)
        new = torch.maximum(new, gl[:, None])
        # (c) hue held near the target, chroma capped; (d) light targets keep shadow chroma.
        ok = linear_to_oklab_t(new.clamp_min(0.0))
        C = torch.hypot(ok[:, 1], ok[:, 2])
        h = torch.atan2(ok[:, 2], ok[:, 1])
        ct = torch.hypot(t[:, 1], t[:, 2])
        ht = torch.atan2(t[:, 2], t[:, 1])
        conf = _hue_confidence_t(ct)
        dh = torch.remainder(h - ht + math.pi, 2.0 * math.pi) - math.pi
        comp = _soft_compress_t(dh.abs(), math.radians(ENV_HUE_TOL0), math.radians(ENV_HUE_TOL1))
        h_new = ht + (dh + conf * (torch.sign(dh) * comp - dh))
        cap = ENV_CHROMA_K * ct + ENV_CHROMA_C0
        knee = 0.8 * cap
        C_new = torch.where(C <= knee, C, knee + (cap - knee) * torch.tanh((C - knee) / (cap - knee).clamp_min(1e-6)))
        Lt = t[:, 0].clamp_min(1e-3)
        light = _smoothstep_t((Lt - ENV_SHADOW_L0) / ENV_SHADOW_L_RAMP)
        rl = ok[:, 0] / Lt
        c_floor = ENV_SHADOW_CHROMA * ct * rl.clamp(0.0, 1.0) * (rl < 1.0).to(rl.dtype)
        C_new = C_new + conf * light * (torch.maximum(C_new, c_floor) - C_new)
        new = oklab_to_linear_gamut_t(torch.stack((ok[:, 0], C_new * torch.cos(h_new), C_new * torch.sin(h_new)), dim=-1))
        res = out.clone()
        res[idx] = o + mm * (new - o)
        return res

    # ------------------------------------------------------------ rule 8: reflections

    def _reach_falloff(self, level: _Level, rp: _ReflParams, idx: tuple[torch.Tensor, torch.Tensor],
                       px: float) -> torch.Tensor:
        """[n] rule 8's distance weight of the candidates ``idx`` for a small repainted group: full
        within half its reach (``rp.reach``, px at the reference size) of its own pixels, none
        beyond the reach."""
        reach_px = max(1.0, rp.reach * px)
        n = idx[0].numel()
        got = self._group_distance(level, rp.gids, reach_px)
        if got is None:
            return torch.zeros(n, dtype=torch.float32, device=self.device)
        d, (y0, y1, x0, x1) = got
        yy, xx = idx
        inside = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        dist = torch.full((n,), 1e9, dtype=torch.float32, device=self.device)
        dist[inside] = d[yy[inside] - y0, xx[inside] - x0]
        half = max(0.5, 0.5 * reach_px)
        return 1.0 - _smoothstep_t((dist - half) / half)

    def _recolor_reflections(self, level: _Level, params: _GroupParams, m: torch.Tensor,
                             lin: torch.Tensor) -> torch.Tensor:
        """Rule 8: recolour the old paint's reflections outside the repainted parts.

        ``lin`` is the composed linear image, ``m`` the coverage. Pixels with zero gate
        weight are returned bit for bit; only the candidates are converted, so the cost
        scales with the reflections, not the image."""
        if not params.refl:
            return lin
        ok = linear_to_oklab_t(lin.clamp(0.0, 1.0))
        C = torch.hypot(ok[..., 1], ok[..., 2])
        c0 = (REFL_CHROMA_CUT - REFL_CHROMA_RAMP) * OK_PER_LAB_CHROMA
        c1 = (REFL_CHROMA_CUT + REFL_CHROMA_RAMP) * OK_PER_LAB_CHROMA
        w = _smoothstep_t((C - c0) / (c1 - c0)) * (1.0 - m) * params.allow[level.group_map]
        px = self._px(level)
        half = max(0.5, 0.5 * REFL_FALLOFF_PX * px)
        dist = self._mapped_distance(level, params.mapped_ids)
        w = w * (1.0 - _smoothstep_t((dist - half) / half))
        too_bright = _smoothstep_t((ok[..., 0] - self._paint_light_q(params.mapped_ids)) / REFL_LIGHT_RAMP)
        if level.islands is not None:
            too_bright = too_bright * (1.0 - level.islands)
        w = w * (1.0 - too_bright)
        if level.protect is not None:
            w = w * (1.0 - level.protect)
        locked = self._locked_distance(level)
        if locked is not None:
            w = w * _smoothstep_t(locked / max(0.5, LOCK_PROTECT_PX * px))
        idx = (w > 1e-4).nonzero(as_tuple=True)
        if idx[0].numel() == 0:
            return lin
        ok_s = ok[idx]
        lin_s = lin[idx].clamp(0.0, 1.0)
        w_s = w[idx]
        a_s, b_s = ok_s[:, 1], ok_s[:, 2]
        h_s = torch.rad2deg(torch.atan2(b_s, a_s))
        wide = dist[idx] <= EDGE_MIX_PX * px + 1e-6
        isl_s = level.islands[idx] if level.islands is not None else None
        floor_s = lin_s.amin(dim=-1, keepdim=True)
        num = torch.zeros_like(lin_s)
        den = torch.zeros_like(w_s)
        for rp in params.refl:
            dh = (h_s - rp.hue_deg + 180.0) % 360.0 - 180.0
            z = torch.where(dh >= 0.0, torch.full_like(dh, rp.z_pos), torch.full_like(dh, rp.z_neg))
            o = (z - 2.0 * REFL_HUE_RAMP).clamp_min(0.0)
            wr = w_s * (1.0 - _smoothstep_t((dh.abs() - o) / (z - o).clamp_min(1e-3)))
            w_wide = w_s * (1.0 - _smoothstep_t((dh.abs() - EDGE_MIX_HUE0) / (EDGE_MIX_HUE1 - EDGE_MIX_HUE0)))
            wr = torch.where(wide, torch.maximum(wr, w_wide), wr)
            if rp.reach < REFL_FALLOFF_PX - 1e-6:
                wr = wr * self._reach_falloff(level, rp, idx, px)
            if rp.weight < 1.0:
                wr = wr * rp.weight
            ab_new = torch.stack((rp.s * (rp.cos * a_s - rp.sin * b_s), rp.s * (rp.sin * a_s + rp.cos * b_s)), dim=-1)
            new = oklab_to_linear_gamut_t(torch.cat((ok_s[:, :1], ab_new), dim=-1))
            if isl_s is not None:
                # A decal island pixel next to the paint is old paint mixed with the decal's
                # white: rebuild how much old paint it holds (projection of its colour excess
                # on the paint's) in the target colour, on the white floor.
                k = ((lin_s - floor_s) * rp.excess).sum(dim=-1) / rp.excess_sq
                rebuilt = (floor_s + k.clamp_min(0.0)[:, None] * rp.target_offset).clamp(0.0, 1.0)
                new = new + isl_s[:, None] * (rebuilt - new)
            num = num + wr[:, None] * new
            den = den + wr
        wt = den.clamp(max=1.0)[:, None]
        ok_n = linear_to_oklab_t((num / den.clamp_min(1e-6)[:, None]).clamp(0.0, 1.0))
        # Hold the recoloured hue near the target's.
        dirs = [rp.target_dir for rp in params.refl if rp.target_dir is not None]
        cn = torch.hypot(ok_n[:, 1], ok_n[:, 2])
        hn = torch.atan2(ok_n[:, 2], ok_n[:, 1])
        tx, ty = sum(d[0] for d in dirs), sum(d[1] for d in dirs)
        if dirs and math.hypot(tx, ty) > 1e-6:
            ht = math.atan2(ty, tx)
            dh = torch.remainder(hn - ht + math.pi, 2.0 * math.pi) - math.pi
            t1 = math.radians(REFL_TARGET_HUE_TOL)
            hn = ht + torch.sign(dh) * _soft_compress_t(dh.abs(), 0.5 * t1, t1)
        ok_n = torch.stack((ok_n[:, 0], cn * torch.cos(hn), cn * torch.sin(hn)), dim=-1)
        # Blend through neutral: the old chroma fades out before the new one comes in.
        ab = (1.0 - 2.0 * wt).clamp_min(0.0) * ok_s[:, 1:] + (2.0 * wt - 1.0).clamp_min(0.0) * ok_n[:, 1:]
        L = ok_s[:, :1] + wt * (ok_n[:, :1] - ok_s[:, :1])
        out = lin.clone()
        out[idx] = oklab_to_linear_gamut_t(torch.cat((L, ab), dim=-1))
        return out

    # ------------------------------------------------------------ render

    def _render_linear(self, level: _Level, mapping: Mapping, options: RenderOptions) -> torch.Tensor:
        t0 = time.perf_counter()
        params = self._group_params(mapping, options)
        strength = float(options.shading_strength)
        keep_residual = bool(options.keep_residual)
        with torch.no_grad():
            paint = self._repaint(level, params, options)
            shading = level.shading
            if abs(strength - 1.0) >= 1e-6:
                shading = self._pivot * (level.shading / self._pivot).clamp_min(0.0).pow(strength)
            if paint is None:
                if shading is level.shading:
                    out = level.product + level.residual if keep_residual else level.product
                else:
                    product = level.albedo * shading
                    out = product + self._adjusted_residual(level, product, None, options) if keep_residual else product
            else:
                shading = self._retint_shading(level, shading, paint)
                product = paint.albedo * shading
                out = product + self._adjusted_residual(level, product, paint, options) if keep_residual else product
                out = self._paint_envelope(level, out, paint, params)
                out = self._recolor_reflections(level, params, paint.coverage, out)
            out = out.clamp(0.0, 1.0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return out

    @staticmethod
    def _finish(lin: torch.Tensor) -> np.ndarray:
        u8 = (linear_to_srgb_t(lin) * 255.0 + 0.5).to(torch.uint8)
        return u8.cpu().numpy()


def render_once(albedo_lin: np.ndarray, shading_lin: np.ndarray, residual: np.ndarray,
                group_map: np.ndarray, groups: Sequence[ColorGroup], mapping: Mapping,
                options: Optional[RenderOptions] = None, islands: Optional[np.ndarray] = None,
                protect: Optional[np.ndarray] = None, reference_long_side: Optional[int] = None,
                glints: Optional[np.ndarray] = None, neutral: Optional[np.ndarray] = None) -> np.ndarray:
    """One-shot render (uint8 sRGB) that builds a :class:`Renderer` (``glints`` and ``neutral``: a
    working-resolution renderer's :meth:`Renderer.white_glints` and :meth:`Renderer.neutral_weights`, for an
    export that must match the preview), renders at the native resolution and frees the GPU memory again."""
    r = Renderer(albedo_lin, shading_lin, residual, group_map, groups, islands=islands, protect=protect,
                 reference_long_side=reference_long_side, glints=glints, neutral=neutral)
    try:
        return r.render(mapping, options or RenderOptions())
    finally:
        r.free()


def recompose(albedo_lin: np.ndarray, shading_lin: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """uint8 sRGB of ``albedo * shading + residual`` computed in numpy exactly the way
    the renderer's identity path does it (the reference for tests)."""
    lin = np.clip(albedo_lin.astype(np.float32) * shading_lin.astype(np.float32) + residual.astype(np.float32), 0, 1)
    return imageio.to_uint8(imageio.linear_to_srgb(lin))
