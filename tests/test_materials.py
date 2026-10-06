"""Material cues (recolor/segmentation/materials.py): shininess from the layers, the chrome
advisory and the highlight absorb, on synthetic layers. No models."""
from __future__ import annotations

import numpy as np

from recolor import imageio
from recolor.segmentation import grouping, materials
from recolor.types import Region


def _info(n):
    return [{"id": i, "source": "sam", "confidence": 0.9} for i in range(n)]


def _scene():
    """Red paint (region 0), a region of the same paint washed toward white by a broad
    highlight (region 1: light albedo, bright photo, half its pixels blown, the rest red),
    a headlight lamp (region 2: almost entirely clipped), a white decal (region 3, source
    'text') and a light grey backdrop (region 4). A white part elsewhere (region 5) has
    no red under its highlight."""
    h, w = 120, 200
    labels = np.full((h, w), 4, np.int32)
    labels[10:110, 10:90] = 0
    labels[10:110, 90:120] = 1
    labels[20:40, 130:160] = 2
    labels[60:80, 130:160] = 3
    labels[90:110, 130:190] = 5
    lab = np.zeros((h, w, 3), np.float32)
    lab[labels == 4] = (80.0, 0.0, 0.0)
    lab[labels == 0] = (45.0, 60.0, 45.0)
    lab[labels == 1] = (72.0, 22.0, 16.0)                                # washed: chroma 27 < 0.95 x 75, L 72
    lab[labels == 2] = (85.0, 4.0, 4.0)
    lab[labels == 3] = (92.0, 0.0, 0.0)
    lab[labels == 5] = (88.0, 2.0, 1.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    shading = np.full((h, w, 3), 0.8, np.float32)
    residual = np.zeros((h, w, 3), np.float32)
    photo_lin = albedo * shading
    # the highlight region: five pixels in nine blown white, the rest the paint's own red
    red = np.clip(imageio.lab_to_linear(np.array([[[45.0, 60.0, 45.0]]], np.float32)), 0, 1)[0, 0] * 0.8
    m1 = labels == 1
    photo_lin[m1] = red
    yy, xx = np.mgrid[0:h, 0:w]
    blown = (yy * 7 + xx) % 9 < 5
    rows = np.zeros((h, w), bool)
    rows[::2] = True
    photo_lin[m1 & blown] = 1.0
    residual[m1 & blown] = 1.0 - (albedo * shading)[m1 & blown]
    # the lamp: everything clipped
    photo_lin[labels == 2] = 1.0
    residual[labels == 2] = 1.0 - (albedo * shading)[labels == 2]
    # the white part: half blown, the remainder white
    m5 = labels == 5
    photo_lin[m5 & rows] = 1.0
    residual[m5 & rows] = 1.0 - (albedo * shading)[m5 & rows]
    photo = imageio.to_uint8(imageio.linear_to_srgb(np.clip(photo_lin, 0, 1)))
    info = _info(6)
    info[3]["source"] = "text"
    return labels, albedo, photo, residual, info


def test_shine_features_measure_clipping_and_the_remainder():
    labels, albedo, photo, residual, info = _scene()
    f = materials.shine_features(labels, albedo, photo, residual)
    assert f.n == 6 and f.hl[1] > 0.4 and f.clip[2] > 0.95 and f.hl[0] < 0.05
    assert abs(materials._hue(f.rem_lab[1]) - materials._hue((45.0, 60.0, 45.0))) < 10   # the paint under the highlight
    assert f.rem_chroma(1) > 10 and f.rem_chroma(5) < 5
    assert f.touch[1].get(0, 0) > 20
    without = materials.shine_features(labels, albedo, photo, None)
    assert without.spec.max() == 0 and without.hl[2] > 0.95            # clipped pixels still count


def test_absorb_highlights_moves_the_paints_highlight_and_nothing_else():
    labels, albedo, photo, residual, info = _scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    g_of = {r.id: r.group_id for r in regions}
    assert g_of[1] != g_of[0]                                            # clustered as its own light group
    f = materials.shine_features(labels, albedo, photo, residual)
    regions = materials.tag_regions(regions, f)
    r2, g2, gm2, moves = materials.absorb_highlights(regions, groups, labels, f)
    g_of = {r.id: r.group_id for r in r2}
    assert [m["region"] for m in moves] == [1]
    assert g_of[1] == g_of[0]
    assert g_of[2] != g_of[0] and g_of[3] != g_of[0] and g_of[5] != g_of[0]   # the lamp, the decal, the white part
    grouping._check_state(r2, g2)
    paint = g2[g_of[0]]
    assert paint.shiny > 0.1 and paint.finish in ("shiny", "")
    # a highlight in a background group, or of a backdrop region, never moves
    regions_bg = [r if r.id != 1 else Region(**{**r.__dict__, "backdrop": True}) for r in regions]
    _, _, _, moves = materials.absorb_highlights(regions_bg, groups, labels, f)
    assert moves == []


def test_chrome_advisory_tags_a_glinting_neutral_and_never_locks():
    h, w = 80, 120
    labels = np.zeros((h, w), np.int32)
    labels[10:70, 60:110] = 1                                           # a chrome part: mid-grey albedo, glints, every hue
    lab = np.zeros((h, w, 3), np.float32)
    lab[labels == 0] = (45.0, 60.0, 45.0)
    lab[labels == 1] = (55.0, 0.0, 0.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    rng = np.random.default_rng(0)
    shading = np.full((h, w, 3), 0.8, np.float32)
    photo_lin = albedo * shading
    residual = np.zeros((h, w, 3), np.float32)
    m = labels == 1
    n = int(m.sum())
    lum = rng.uniform(0.02, 1.0, n).astype(np.float32)                  # a wide luminance spread
    tint = rng.uniform(0.7, 1.3, (n, 3)).astype(np.float32)             # reflections of every colour
    refl = np.clip(lum[:, None] * tint, 0, 1)
    refl[rng.random(n) < 0.3] = 1.0                                     # 30 % clipped glints
    photo_lin[m] = refl
    residual[m] = refl - (albedo * shading)[m]
    photo = imageio.to_uint8(imageio.linear_to_srgb(np.clip(photo_lin, 0, 1)))
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(2))
    f = materials.shine_features(labels, albedo, photo, residual)
    assert materials.chrome_regions(regions, f) == {1}
    tagged = materials.tag_regions(regions, f)
    assert tagged[1].chrome and not tagged[0].chrome
    r2, g2, gm2 = grouping._finalize(tagged, labels, {r.id: r.group_id for r in tagged})
    chrome = g2[r2[1].group_id]
    assert chrome.finish == "chrome" and not chrome.locked                # a badge, never a lock
    assert g2[r2[0].group_id].finish == ""


def test_the_shiny_badge_follows_the_glint_share_not_the_highlight_share():
    """A strong positive residual is near-universal on glossy paint, so the badge is based
    on the glint share (clipped or specular pixels); the highlight share is kept for the
    tooltip."""
    labels, albedo, photo, residual, info = _scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    feats = materials.shine_features(labels, albedo, photo, residual)
    tagged = {r.id: r for r in materials.tag_regions(regions, feats)}
    assert tagged[2].glint > 0.95 and tagged[1].glint > 0.4 and tagged[0].glint == 0.0
    assert all(0.0 <= r.glint <= 1.0 and r.glint <= max(feats.clip[r.id], feats.spec[r.id]) + 1e-6 for r in tagged.values())
    lamp = grouping._group_from(0, [tagged[2]], labels.size)
    paint = grouping._group_from(1, [tagged[0]], labels.size)
    assert lamp.finish == "shiny" and lamp.glint > 0.95 and paint.finish == "" and paint.glint == 0.0
    # a region with a strong residual everywhere but no glints earns no badge
    from dataclasses import replace
    dull = replace(tagged[0], shiny=0.9, glint=0.05)
    g = grouping._group_from(2, [dull], labels.size)
    assert g.finish == "" and g.shiny == 0.9 and g.glint == 0.05


def test_a_stored_shiny_badge_without_glints_is_dropped_when_the_record_is_read():
    """A job analysed when the badge followed the broader highlight share carries 'shiny'
    on groups whose glint share is below the rule (or missing): the record type applies the
    glint rule again on load, so such a job loses its stale badges without a regroup; a
    badge that meets the rule, and a chrome badge, are kept."""
    from recolor.types import SHINY_GLINT_SHARE, ColorGroup
    base = {"id": 0, "name": "Khaki", "albedo_lab": [60.0, 5.0, 30.0], "albedo_hex": "#a09050", "area": 100,
            "area_frac": 0.1, "region_ids": [0], "hue_family": "yellow"}
    assert ColorGroup.from_dict({**base, "finish": "shiny", "shiny": 0.72}).finish == ""            # no glint key
    assert ColorGroup.from_dict({**base, "finish": "shiny", "shiny": 0.72, "glint": 0.0}).finish == ""
    assert ColorGroup.from_dict({**base, "finish": "shiny", "glint": SHINY_GLINT_SHARE}).finish == "shiny"
    assert ColorGroup.from_dict({**base, "finish": "chrome", "glint": 0.0}).finish == "chrome"
    assert ColorGroup.from_dict({**base}).finish == ""
    assert SHINY_GLINT_SHARE == grouping.SHINY_SHARE
