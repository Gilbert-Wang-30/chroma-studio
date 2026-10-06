"""The regions stage's small and named parts (recolor/segmentation/smallparts.py, wheels.py and
the hierarchy's extras) on synthetic images, with SAM and Florence-2 replaced by stand-ins:
no model, no network."""
from __future__ import annotations

import numpy as np

from recolor import imageio
from recolor.segmentation import hierarchy, smallparts, wheels
from recolor.segmentation.hierarchy import build_regions
from recolor.segmentation.labelops import region_areas


def _lab_image(h, w, base, paint=None):
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = base
    for sl, col in (paint or []):
        lab[sl] = col
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    return albedo, photo


def _mask(seg, iou=0.9):
    ys, xs = np.nonzero(seg)
    return {"segmentation": seg, "area": int(seg.sum()), "bbox": [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)], "predicted_iou": iou, "stability_score": 0.9}


def _check(labels, info):
    assert labels.dtype == np.int32 and labels.min() == 0
    assert int(labels.max()) + 1 == len(info) and [d["id"] for d in info] == list(range(len(info)))
    assert (region_areas(labels, len(info)) > 0).all()


# ------------------------------------------------------------------ distinct small proposals

def test_a_distinct_small_proposal_is_kept_and_a_bland_one_is_not():
    """A proposal below the preset's minimum area is painted when it stands out from its
    own surroundings (an indicator lens on the machinery); one of its host's colour is not."""
    h, w = 120, 160
    whole = np.zeros((h, w), bool)
    whole[10:110, 10:150] = True
    lens = np.zeros((h, w), bool)
    lens[40:50, 60:70] = True                                       # 100 px: below min_px (0.0006 * 19200 = 16 -> use fast: 0.0015 -> 29)
    bland = np.zeros((h, w), bool)
    bland[70:80, 100:110] = True
    albedo, photo = _lab_image(h, w, (50.0, 0.0, 0.0), [(np.s_[10:110, 10:150], (25.0, 0.0, 0.0)),
                                                        (np.s_[40:50, 60:70], (70.0, 40.0, 50.0))])
    masks = [_mask(whole), _mask(lens), _mask(bland)]
    # fast preset: min_px = 0.0015 * 19200 = 29 > 100? no: raise the bar with a bigger image share
    hierarchy_min = hierarchy.REGION_PRESETS["fast"]["min_area_frac"]
    orig = hierarchy.REGION_PRESETS["fast"]["min_area_frac"]
    hierarchy.REGION_PRESETS["fast"]["min_area_frac"] = 0.01           # min_px 192: both small proposals are below it
    try:
        labels, info = build_regions(photo, albedo, masks, detail="fast")
    finally:
        hierarchy.REGION_PRESETS["fast"]["min_area_frac"] = orig
    _check(labels, info)
    small = [d for d in info if d["source"] == "small"]
    assert len(small) == 1 and small[0].get("exempt") is True
    assert len(np.unique(labels[lens])) == 1 and int(labels[45, 65]) == small[0]["id"]
    assert int(labels[75, 105]) == int(labels[15, 15])                  # the bland one merged into its host
    assert hierarchy_min == orig


def test_an_exempt_remnant_below_its_floor_is_merged_away():
    """A word mask painted before its letters leaves slivers between them; a remnant below
    the exemption floor is a speck like any other and joins the paint it is."""
    h, w = 80, 120
    info = [{"id": 0, "source": "sam", "confidence": 0.9}, {"id": 1, "source": "small", "exempt": True, "confidence": 0.9},
            {"id": 2, "source": "small", "exempt": True, "confidence": 0.9}]
    labels = np.zeros((h, w), np.int32)
    labels[20:60, 20:100] = 1                                          # a healthy small region (3200 px)
    labels[30:33, 30:34] = 2                                           # a 12 px remnant of another small region
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (50.0, 0.0, 0.0)
    lab[labels == 1] = (70.0, 40.0, 50.0)
    lab[labels == 2] = (70.0, 40.0, 50.0)
    out, out_info = hierarchy._remove_specks(labels, lab, info, speck_px=200, progress=None)
    _check(out, out_info)
    assert len(out_info) == 2 and sum(d["source"] == "small" for d in out_info) == 1
    assert int(out[31, 31]) == int(out[40, 60])                         # the remnant joined the small region it sat in


# ------------------------------------------------------------------ extras: lettering, wheels, named parts

def _prompter_for(masks_by_job):
    """A SAM box stand-in: for every job returns the candidate masks assigned by index."""
    def prompt(image, jobs):
        out = []
        for i, j in enumerate(jobs):
            x0, y0, x1, y1 = j["crop"]
            cands = []
            for m, score in masks_by_job.get(i, []):
                cands.append({"mask": m[y0:y1, x0:x1], "x0": x0, "y0": y0, "score": score, "clipped": False})
            out.append(cands)
        return out
    return prompt


def test_text_masks_take_sams_mask_or_the_minority_cluster():
    h, w = 90, 140
    albedo, photo = _lab_image(h, w, (45.0, 60.0, 45.0), [(np.s_[30:40, 30:60], (92.0, 0.0, 0.0)),
                                                          (np.s_[50:60, 80:110], (92.0, 0.0, 0.0))])
    lab = imageio.linear_to_lab(albedo)
    quad = lambda x0, y0, x1, y1: [x0, y0, x1, y0, x1, y1, x0, y1]
    ocr = [{"quad": quad(28, 28, 62, 42), "text": "DUCATI"}, {"quad": quad(78, 48, 112, 62), "text": "748"},
           {"quad": quad(10, 70, 40, 80), "text": "--"}]                # no letters or digits: texture
    letters = np.zeros((h, w), bool)
    letters[30:40, 30:60] = True
    prompt = _prompter_for({0: [(letters, 0.95)]})                       # SAM answers the first word only
    out = smallparts.text_masks(photo, lab, ocr, prompt)
    assert [e.source for e in out] == ["text", "text"]
    assert out[0].mask[30:40, 30:60].all() and out[0].mask.sum() == 300  # SAM's mask, kept as is
    assert out[1].mask[50:60, 80:110].mean() > 0.9 and not out[1].mask[:, :70].any()   # the 2-means letters
    assert not smallparts.text_masks(photo, lab, [], prompt)


def test_part_masks_split_a_wheel_and_keep_other_phrases():
    h, w = 160, 160
    yy, xx = np.mgrid[0:h, 0:w]
    outer = ((xx - 80) / 60.0) ** 2 + ((yy - 80) / 60.0) ** 2 <= 1.0
    inner = ((xx - 80) / 42.0) ** 2 + ((yy - 80) / 42.0) ** 2 <= 1.0
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (85.0, 0.0, 0.0)
    lab[outer] = (20.0, 0.0, 0.0)                                       # rubber
    lab[inner] = (60.0, 0.0, 0.0)                                       # a lighter rim
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    lab_a = imageio.linear_to_lab(albedo)
    boxes = [{"box": [22, 22, 138, 138], "label": "tire"}, {"box": [22, 22, 138, 138], "label": "wheel rim"},
             {"box": [0, 0, 30, 20], "label": "headlight"}]
    lamp = np.zeros((h, w), bool)
    lamp[2:16, 2:26] = True
    prompt = _prompter_for({0: [(outer, 0.97)], 1: [(lamp, 0.9)]})
    out = smallparts.part_masks(photo, lab_a, boxes, prompt)
    kinds = sorted(e.source for e in out)
    assert kinds == ["named", "wheel"]
    wheel = next(e for e in out if e.source == "wheel")
    tyre, rim = wheel.parts
    assert rim[inner].mean() > 0.95 and tyre[outer & ~inner].mean() > 0.9 and not (tyre & rim).any()
    named = next(e for e in out if e.source == "named")
    assert named.labels == ["headlight"] and named.mask.sum() == lamp.sum()


def test_build_regions_stamps_extras_and_keeps_the_partition():
    """Lettering overrides the paint; a wheel's tyre and rim take their pixels only from the
    machinery region that reaches outside the wheel; a named part is stamped only where it
    splits a region into two different parts."""
    h, w = 160, 240
    yy, xx = np.mgrid[0:h, 0:w]
    outer = ((xx - 70) / 55.0) ** 2 + ((yy - 80) / 55.0) ** 2 <= 1.0
    inner = ((xx - 70) / 38.0) ** 2 + ((yy - 80) / 38.0) ** 2 <= 1.0
    machine = np.zeros((h, w), bool)
    machine[10:150, 10:140] = True
    paint = np.zeros((h, w), bool)
    paint[10:150, 150:230] = True
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (85.0, 0.0, 0.0)
    lab[machine] = (22.0, 0.0, 0.0)
    lab[outer] = (20.0, 0.0, 0.0)
    lab[inner] = (27.0, 0.0, 0.0)                                       # the rim is nearly the tyre's colour (SAM sees one wheel)
    lab[paint] = (45.0, 60.0, 45.0)
    lab[60:70, 170:210] = (92.0, 0.0, 0.0)                              # white lettering on the paint
    lab[100:140, 160:220] = (45.0, 60.0, 45.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    photo[100:140, 160:220] = (photo[100:140, 160:220] * 0.55).astype(np.uint8)   # a darker panel of the same paint
    masks = [_mask(machine | outer), _mask(paint)]
    letters = np.zeros((h, w), bool)
    letters[60:70, 170:210] = True
    tyre, rim = outer & ~inner, inner
    panel = np.zeros((h, w), bool)
    panel[100:140, 160:220] = True
    extra = [smallparts.Extra(letters, "text", ["DUCATI"]), smallparts.Extra(outer, "wheel", ["tire"], parts=(tyre, rim)),
             smallparts.Extra(panel, "named", ["fork"])]
    labels, info = build_regions(photo, albedo, masks, detail="fast", extra=extra)
    _check(labels, info)
    src = {d["id"]: d["source"] for d in info}
    assert src[int(labels[65, 190])] == "text" and info[int(labels[65, 190])].get("exempt") is True
    assert src[int(labels[80, 70])] == "wheel" and src[int(labels[80, 20])] == "wheel"        # rim and tyre
    assert labels[80, 70] != labels[80, 20]
    assert src[int(labels[120, 190])] == "named" and labels[120, 190] != labels[30, 190]     # the darker panel
    assert labels[30, 20] == labels[145, 130] and src[int(labels[30, 20])] == "sam"          # the machinery stays one region
    # the named part is not stamped where it would not split anything different
    same = build_regions(photo, albedo, masks, detail="fast", extra=[smallparts.Extra(panel, "named", ["fork"])])[1]
    labels2, info2 = build_regions(photo, albedo, masks, detail="fast",
                                   extra=[smallparts.Extra(paint.copy(), "named", ["fork"])])
    assert not any(d["source"] == "named" for d in info2)
    assert len(same) >= 2


def _wheel_scene(disc_l=62.0, rim_l=62.0, notch=False):
    """A wheel: a dark tyre, a rim inside its lip ellipse and optionally a bright disc inside the
    rim (the lip is then the weaker edge on each ray) and a deep notch a swingarm cuts into the
    outline (the mask misses it)."""
    h, w = 220, 220
    yy, xx = np.mgrid[0:h, 0:w]
    outer = ((xx - 110) / 90.0) ** 2 + ((yy - 110) / 84.0) ** 2 <= 1.0
    inner = ((xx - 110) / 70.0) ** 2 + ((yy - 110) / 65.0) ** 2 <= 1.0
    disc = ((xx - 110) / 50.0) ** 2 + ((yy - 110) / 46.0) ** 2 <= 1.0
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (85.0, 0.0, 0.0)
    lab[outer] = (18.0, 0.0, 0.0)
    lab[inner] = (rim_l, 0.0, 0.0)
    lab[disc] = (disc_l, 0.0, 0.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    mask = outer.copy()
    if notch:
        arm = (np.abs(yy - 110) < 26) & (xx < 110)                   # a swingarm from the left to the hub
        mask &= ~arm
        photo[arm & outer] = (60, 60, 60)
    return mask, outer, inner, photo, albedo


def test_split_wheel_passes_a_notched_outline_on_its_angular_support(monkeypatch):
    """The Ducati's rear wheel: the swingarm, the chain guard and the fender cut deep notches into
    the outline and 31 % of the contour lay on the ellipse, yet the ellipse was supported all the
    way round but for the swingarm. Here a deep notch leaves 68 % on it; with the contour share
    asked for above that, the angular support alone lets the wheel pass, and without it the
    wheel is left alone."""
    mask, outer, inner, photo, albedo = _wheel_scene(notch=True)
    monkeypatch.setattr(wheels, "MIN_OUTER_INLIERS", 0.9)
    dbg = {}
    sp = wheels.split_wheel(mask, photo, imageio.linear_to_lab(albedo), debug=dbg)
    assert dbg["fill_iou"] < wheels.MIN_OUTER_INLIERS and dbg["support"] >= wheels.MIN_OUTER_SUPPORT
    assert sp is not None
    tyre, rim = sp
    assert rim[inner & mask].mean() > 0.9 and tyre[outer & ~inner & mask].mean() > 0.9
    monkeypatch.setattr(wheels, "MIN_OUTER_SUPPORT", 1.01)
    assert wheels.split_wheel(mask, photo, imageio.linear_to_lab(albedo)) is None


def test_the_lip_fallback_finds_a_dark_lip_that_a_bright_disc_outshines():
    """A black rim in a black tyre with a silver disc inside it: on most rays the disc's edge is
    the strongest by far and the rim's lip, a weak edge, is below the first fit's 40 %; the
    fallback lets every edge peak of the outer band compete, and the lip is the ellipse the most
    rays support (the disc sits below the band)."""
    import math
    h, w = 220, 220
    yy, xx = np.mgrid[0:h, 0:w]
    outer = ((xx - 110) / 90.0) ** 2 + ((yy - 110) / 84.0) ** 2 <= 1.0
    lip = ((xx - 110) / 72.0) ** 2 + ((yy - 110) / 67.2) ** 2 <= 1.0       # the lip at 0.8 of the radius
    disc = ((xx - 110) / 54.0) ** 2 + ((yy - 110) / 50.4) ** 2 <= 1.0      # the disc at 0.6: below the band
    L = np.full((h, w), 85.0, np.float32)
    L[outer] = 18.0
    L[lip] = 30.0
    L[disc] = 80.0
    wc = outer.copy()
    outer_e = ((110.0, 110.0), (180.0, 168.0), 0.0)
    e = wheels._lip_fallback(outer_e, wc, L, L * 0.0, np.random.default_rng(0))
    assert e is not None
    (cx, cy), (a, b), _ang = e
    assert abs(max(a, b) / 2.0 - 72.0) < 3.0 and abs(min(a, b) / 2.0 - 67.2) < 3.0 and math.hypot(cx - 110, cy - 110) < 3.0
    flat = np.full((h, w), 18.0, np.float32)
    assert wheels._lip_fallback(outer_e, wc, flat, flat * 0.0, np.random.default_rng(0)) is None   # no edge: no lip


def test_split_wheel_finds_the_rim_ellipse():
    h, w = 200, 200
    yy, xx = np.mgrid[0:h, 0:w]
    outer = ((xx - 100) / 80.0) ** 2 + ((yy - 100) / 70.0) ** 2 <= 1.0
    inner = ((xx - 104) / 52.0) ** 2 + ((yy - 100) / 46.0) ** 2 <= 1.0
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (85.0, 0.0, 0.0)
    lab[outer] = (18.0, 0.0, 0.0)
    lab[inner] = (62.0, 0.0, 0.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    sp = wheels.split_wheel(outer, photo, imageio.linear_to_lab(albedo))
    assert sp is not None
    tyre, rim = sp
    assert rim[inner].mean() > 0.95 and (rim & ~inner).sum() < 0.1 * inner.sum()     # the lip sits 1-2 px out
    assert tyre[outer & ~inner].mean() > 0.9 and not (tyre & rim).any()
    assert wheels.split_wheel(np.zeros((h, w), bool), photo, imageio.linear_to_lab(albedo)) is None
    square = np.zeros((h, w), bool)
    square[40:160, 40:160] = True
    assert wheels.split_wheel(square, photo, imageio.linear_to_lab(albedo)) is None  # a square is not a wheel (no rim edge)


# ------------------------------------------------------------------ the matte cut

def test_cut_on_matte_separates_the_object_side_and_keeps_ids_stable():
    h, w = 60, 100
    labels = np.zeros((h, w), np.int32)
    labels[:, 50:] = 1                                                  # region 1 straddles the matte's contour
    labels[10:20, 10:20] = 2
    info = [{"id": 0, "source": "sam", "confidence": 0.9}, {"id": 1, "source": "sam", "confidence": 0.9},
            {"id": 2, "source": "small", "exempt": True, "confidence": 0.9}]
    fg = np.zeros((h, w), np.float32)
    fg[:, 70:] = 1.0                                                    # the object is the right 30 columns
    fg[12:18, 12:18] = 1.0                                              # a small region half in, half out: exempt, left whole
    albedo = np.full((h, w, 3), 0.4, np.float32)
    out, out_info, n_cut = hierarchy.cut_on_matte(labels, info, fg, albedo)
    assert n_cut == 1
    _check(out, out_info)
    assert int(out.max()) == 3 and out_info[3]["source"] == "split" and "exempt" not in out_info[3]
    assert (out[:, 70:] == 3).all() and (out[:, 50:70] == 1).all()     # object side new, backdrop side keeps the id
    assert (out[10:20, 10:20] == 2).all() and out_info[2].get("exempt")
    assert out_info[3]["area"] == 30 * h and len(out_info[3]["albedo_lab"]) == 3
    # nothing to cut: the same partition back
    same, same_info, n = hierarchy.cut_on_matte(labels, info, np.ones((h, w), np.float32), albedo)
    assert n == 0 and np.array_equal(same, labels) and same is not labels


def test_a_two_tone_small_proposal_keeps_only_the_tone_that_is_not_its_surroundings():
    """SAM proposes thin black lettering on the paint with the paint showing between the
    strokes as one small mask: only the strokes are the part; the paint between them stays
    with the paint (painted navy, the "S1000" on the BMW kept yellow letters otherwise)."""
    h, w = 120, 160
    paint = np.zeros((h, w), bool)
    paint[10:110, 10:150] = True
    word = np.zeros((h, w), bool)
    word[50:62, 40:120] = True                                          # a 12 x 80 word box: 960 px
    strokes = np.zeros((h, w), bool)
    for x in range(42, 118, 8):
        strokes[50:62, x:x + 3] = True                                  # thin dark strokes: 10 x 36 = 360 px
    albedo, photo = _lab_image(h, w, (85.0, 0.0, 0.0), [(np.s_[10:110, 10:150], (78.0, 4.0, 78.0))])
    lab = imageio.linear_to_lab(albedo)
    lab[strokes] = (18.0, 0.0, 0.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    orig = hierarchy.REGION_PRESETS["fast"]["min_area_frac"]
    hierarchy.REGION_PRESETS["fast"]["min_area_frac"] = 0.1             # min_px 1920: the word is a small proposal
    try:
        labels, info = build_regions(photo, albedo, [_mask(paint), _mask(word)], detail="fast")
    finally:
        hierarchy.REGION_PRESETS["fast"]["min_area_frac"] = orig
    _check(labels, info)
    small = [d for d in info if d["source"] == "small"]
    assert len(small) == 1
    sid = small[0]["id"]
    assert (labels[strokes] == sid).mean() > 0.95                        # the strokes are the part ...
    assert (labels[word & ~strokes] == sid).mean() < 0.05                # ... the paint between them is not
    assert int(labels[55, 45 + 4]) == int(labels[30, 80])                # it stays with the paint's region


# ------------------------------------------------------------------ bounded work on text-dense photos

def test_merge_quads_is_transitive_deduplicated_capped_and_fast():
    """The full image and the tiles report the same words: overlapping quads become one
    domain (transitively), the texts are kept once, a far word is its own domain, and 300
    quads (a poster wall) merge in milliseconds instead of minutes."""
    import time
    quad = lambda x0, y0, x1, y1: [x0, y0, x1, y0, x1, y1, x0, y1]
    # "DUCATI" from the full image and a tile, "748" over its right half, "SUPERBIKE" over
    # 748's right half (linked to DUCATI only through 748), "SHOWA" far away
    ocr = [{"quad": quad(20, 20, 80, 40), "text": "DUCATI"}, {"quad": quad(22, 21, 82, 41), "text": "DUCATI"},
           {"quad": quad(52, 20, 112, 40), "text": "748"}, {"quad": quad(84, 20, 144, 40), "text": "SUPERBIKE"},
           {"quad": quad(300, 200, 340, 220), "text": "SHOWA"}]
    doms = smallparts._merge_quads(ocr, (300, 400))
    assert len(doms) == 2
    big, far = sorted(doms, key=lambda d: -int(d[0].sum()))
    assert big[1] == ["DUCATI", "748", "SUPERBIKE"] and far[1] == ["SHOWA"]
    assert big[0][20:41, 20:145].all() and big[0].sum() > 124 * 21          # the union, grown by TEXT_EXPAND
    assert not big[0][:, 160:].any() and not far[0][:, :290].any()
    # two words side by side that barely touch stay apart, as the pixel test kept them
    apart = smallparts._merge_quads([{"quad": quad(20, 20, 80, 40), "text": "A"}, {"quad": quad(76, 20, 136, 40), "text": "B"}], (100, 200))
    assert len(apart) == 2
    rng = np.random.default_rng(0)
    many = []
    for _ in range(300):
        x, y = int(rng.integers(0, 1456)), int(rng.integers(0, 994))
        w, h = int(rng.integers(20, 80)), int(rng.integers(8, 30))
        many.append({"quad": quad(x, y, x + w, y + h), "text": "ab12"})
    t0 = time.perf_counter()
    doms = smallparts._merge_quads(many, (1024, 1536))
    assert time.perf_counter() - t0 < 1.0
    assert 0 < len(doms) <= smallparts.TEXT_MAX_DOMAINS
    assert all(m.shape == (1024, 1536) and m.any() for m, _ in doms)
    # a degenerate record is skipped, not fatal: malformed, non-finite, a point or a line
    # (zero area: a point grown by TEXT_EXPAND used to become a 60 px domain), out of the image
    assert smallparts._merge_quads([{"quad": [float("nan")] * 8, "text": "x"}, {"text": "y"}], (50, 50)) == []
    bad = [{"quad": [50, 50, 50, 50, 50, 50, 50, 50], "text": "PT"}, {"quad": quad(10, 30, 40, 30), "text": "LINE"},
           {"quad": quad(300, 300, 320, 310), "text": "OUT"}, {"quad": quad(-30, -30, -10, -20), "text": "NEG"},
           {"quad": [1, 2, 3], "text": "short"}, {"quad": "bad", "text": "str"}]
    assert smallparts._merge_quads(bad, (100, 200)) == []
    doms = smallparts._merge_quads(bad + [{"quad": quad(150, 50, 190, 60), "text": "GH"}], (100, 200))
    assert [w for _, w in doms] == [["GH"]]


def test_find_extras_stops_at_its_budget_and_keeps_what_it_found(caplog):
    """A slow SAM on a photo with many words: the prompts go out in chunks, none after the
    budget, the lettering found so far is returned and the shortfall is logged."""
    import logging
    import time
    h, w = 400, 600
    boxes = [(20 + 70 * (i % 8), 20 + 60 * (i // 8), 60 + 70 * (i % 8), 40 + 60 * (i // 8)) for i in range(40)]
    albedo, photo = _lab_image(h, w, (45.0, 60.0, 45.0),
                               [(np.s_[y0 + 4:y1 - 4, x0 + 4:x1 - 4], (92.0, 0.0, 0.0)) for x0, y0, x1, y1 in boxes])
    quad = lambda x0, y0, x1, y1: [x0, y0, x1, y0, x1, y1, x0, y1]
    ocr = [{"quad": quad(*b), "text": f"W{i:02d}"} for i, b in enumerate(boxes)]
    calls = []

    def slow(image, jobs):
        calls.append(len(jobs))
        time.sleep(0.12)
        out = []
        for j in jobs:
            x0, y0, x1, y1 = j["crop"]
            m = np.zeros((y1 - y0, x1 - x0), bool)
            bx = j["box"]
            m[int(bx[1]) - y0 + 4:int(bx[3]) - y0 - 4, int(bx[0]) - x0 + 4:int(bx[2]) - x0 - 4] = True
            out.append([{"mask": m, "x0": x0, "y0": y0, "score": 0.9, "clipped": False}])
        return out

    analysis = {"ocr": ocr, "grounding": [{"box": [10, 300, 200, 390], "label": "headlight"}]}
    with caplog.at_level(logging.WARNING, logger="recolor.segmentation.smallparts"):
        out = smallparts.find_extras(photo, albedo, analysis, slow, budget_s=0.05)
    assert calls == [smallparts.PROMPT_CHUNK]                                # one chunk, then the deadline
    # the prompted words are examined, the rest fall to the cheap 2-means path: nothing is lost
    assert smallparts.PROMPT_CHUNK <= len(out) <= 40 and all(e.source == "text" for e in out)
    assert any("cut short" in r.message and "16 of 40" in r.message for r in caplog.records)
    # without a budget every domain and box is prompted
    calls.clear()
    out = smallparts.find_extras(photo, albedo, analysis, slow)
    assert sum(calls) == 41 and len(out) >= 40


def test_part_masks_prompt_the_largest_boxes_only(monkeypatch):
    h, w = 200, 200
    albedo, photo = _lab_image(h, w, (60.0, 0.0, 0.0))
    lab = imageio.linear_to_lab(albedo)
    monkeypatch.setattr(smallparts, "PART_MAX_BOXES", 2)
    boxes = [{"box": [0, 0, 20, 20], "label": "logo"}, {"box": [50, 50, 150, 150], "label": "headlight"},
             {"box": [160, 160, 190, 199], "label": "mirror"}]
    seen = []

    def prompt(image, jobs):
        seen.extend(j["box"] for j in jobs)
        return [[] for _ in jobs]

    assert smallparts.part_masks(photo, lab, boxes, prompt) == []
    assert seen == [[50.0, 50.0, 150.0, 150.0], [160.0, 160.0, 190.0, 199.0]]  # the two largest, largest first


# ------------------------------------------------------------------ the matte's compact pieces

def test_cut_on_matte_cuts_a_compact_piece_inside_a_backdrop_region_and_leaves_a_sliver():
    """SAM's backdrop mask swallowed a small object (a reservoir): the matte sees it well
    inside the silhouette, so it becomes a region of its own although it is far below 20 % of
    the backdrop region; a 3 px sliver where the matte and SAM disagree along the object's
    edge stays with the backdrop, and so does a piece too small to count."""
    h, w = 120, 200
    labels = np.zeros((h, w), np.int32)
    labels[20:80, 20:80] = 1                                            # the object SAM found
    info = [{"id": 0, "source": "sam", "confidence": 0.9}, {"id": 1, "source": "sam", "confidence": 0.9}]
    fg = np.zeros((h, w), np.float32)
    fg[20:80, 20:83] = 1.0                                              # the matte: 3 px wider than SAM
    fg[30:50, 120:140] = 1.0                                            # the reservoir, 400 px in the backdrop
    fg[100:110, 150:160] = 1.0                                          # 100 px: below MATTE_CC_MIN_PX
    albedo = np.full((h, w, 3), 0.4, np.float32)
    out, out_info, n_cut = hierarchy.cut_on_matte(labels, info, fg, albedo)
    _check(out, out_info)
    assert n_cut == 1 and int(out.max()) == 2
    assert (out[30:50, 120:140] == 2).all() and out_info[2]["source"] == "split" and out_info[2]["area"] == 400
    assert (out[20:80, 80:83] == 0).all() and (out[100:110, 150:160] == 0).all()
    assert (out[20:80, 20:80] == 1).all()
    # an exempt region (lettering) is never cut, and a matte with no object leaves the map alone
    info[0]["source"], info[0]["exempt"] = "text", True
    same, _, n = hierarchy.cut_on_matte(labels, info, fg, albedo)
    assert n == 0 and np.array_equal(same, labels)
