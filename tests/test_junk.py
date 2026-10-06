"""Junk-group pruning (recolor/segmentation/junk.py) on synthetic scenes built from albedo x
shading: a shadow sliver with a colour shift joins the paint it is a shadow of, a real small
part, lettering, a decal island and a detected part stay, a locked shadow still goes, a
shadow inside a part's mask goes to the part, tiny backdrop crumbs join their backdrop.
No models."""
from __future__ import annotations

import numpy as np
import pytest

from recolor import imageio
from recolor.segmentation import grouping, junk, refine
from recolor.types import ColorGroup

H, W = 200, 200
PAINT = np.s_[25:175, 25:175]                # 22 500 px of red paint on a grey backdrop
SLIVER = np.s_[100:103, 60:80]               # 60 px: below 0.4 % of the object
SPOT = np.s_[120:125, 120:132]               # 60 px of something else


def _paint_lin():
    return imageio.lab_to_linear(np.array([[[45.0, 60.0, 45.0]]], np.float32))[0, 0]


def _scene(sliver_albedo_scale=0.45, sliver_shading=0.3, spot=None, extra_regions=()):
    """labels: 0 backdrop, 1 paint, 2 the sliver (the paint's albedo darkened: a colour-shifted
    shadow the clustering keeps apart), 3 an optional spot of another colour."""
    labels = np.zeros((H, W), np.int32)
    labels[PAINT] = 1
    labels[SLIVER] = 2
    albedo = np.zeros((H, W, 3), np.float32)
    albedo[...] = imageio.lab_to_linear(np.array([[[60.0, 0.0, 0.0]]], np.float32))[0, 0]
    albedo[labels == 1] = _paint_lin()
    albedo[labels == 2] = _paint_lin() * sliver_albedo_scale
    shading = np.full((H, W, 3), 0.9, np.float32)
    shading[labels == 2] = sliver_shading
    if spot is not None:
        labels[SPOT] = 3
        albedo[SPOT] = imageio.lab_to_linear(np.array([[spot]], np.float32))[0, 0]
    for rid, sl in extra_regions:
        labels[sl] = rid
    photo = imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))
    fg = (labels > 0).astype(np.float32)
    return labels, albedo, shading, photo, fg


def _grouped(labels, albedo, info):
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    return regions, groups, gm


def _info(n, **over):
    out = [{"id": i, "source": "sam"} for i in range(n)]
    out[0]["bg"] = 2
    for rid, d in over.items():
        out[int(rid)].update(d)
    return out


def _group_of(regions, groups):
    return {r.id: next(g for g in groups if g.id == r.group_id) for r in regions}


# ------------------------------------------------------------------ the tests on one pair

def _pair(**kw):
    base = dict(nb=1, share=0.9, share_obj=0.9, boundary=20, rho_photo=0.25, rho_shade=0.33,
                photo_g=(20.0, 20.0, 15.0), photo_b=(45.0, 45.0, 34.0), alb_g=(35.0, 45.0, 34.0), alb_b=(45.0, 60.0, 45.0),
                chrom_photo=(0.0, 0.0, 0.0), chrom_shade=(0.0, 0.0, 0.0), e_lum=0.5, e_chr=0.05)
    base.update(kw)
    return junk.PairEvidence(**base)


def test_the_shadow_test_wants_a_darker_photo_the_shading_explains_and_the_hue_kept():
    ok, why = junk.shadow_test(_pair())
    assert ok and why.startswith("shadow")
    assert not junk.shadow_test(_pair(rho_photo=0.95))[0]                            # no step
    assert not junk.shadow_test(_pair(rho_shade=0.98))[0]                            # the shading does not carry it
    assert not junk.shadow_test(_pair(photo_g=(20.0, -30.0, 10.0), chrom_photo=(0.4, -0.1, -0.3)))[0]   # hue moved unlike light
    assert junk.shadow_test(_pair(photo_g=(20.0, -30.0, 10.0)))[0]                   # ... unless the light's own cast explains it
    ok, why = junk.shadow_test(_pair(photo_g=(20.0, 1.0, 1.0)))                      # the hue is lost: undecidable
    assert not ok and "hue lost" in why


def test_the_gradient_and_the_significance_tests():
    flat = _pair(e_lum=0.05, e_chr=0.05, alb_g=(45.0, 58.0, 44.0), photo_g=(45.0, 45.0, 34.0))
    assert junk.gradient_test(flat)[0]
    assert not junk.gradient_test(_pair(e_lum=0.5, alb_g=(45.0, 58.0, 44.0), photo_g=(45.0, 45.0, 34.0)))[0]
    p, why = junk.significance_test(0.001, [flat])
    assert p is flat and "insignificant" in why
    assert junk.significance_test(0.01, [flat])[0] is None                          # too big to be invisible
    assert junk.significance_test(0.001, [_pair(alb_g=(45.0, 10.0, -40.0))])[0] is None   # another colour


# ------------------------------------------------------------------ the pruning

def test_a_colour_shifted_shadow_sliver_joins_its_paint_and_the_labels_stay():
    labels, albedo, shading, photo, fg = _scene()
    regions, groups, gm = _grouped(labels, albedo, _info(3))
    g = _group_of(regions, groups)
    assert g[2].id != g[1].id                                                        # the clustering kept it apart
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg)
    assert [(m["region"], m["rule"]) for m in pr.log] == [(2, "shadow")]
    g = _group_of(pr.regions, pr.groups)
    assert g[2].id == g[1].id and len(pr.groups) == len(groups) - 1
    grouping._check_state(pr.regions, pr.groups)
    lut = np.array([r.group_id for r in pr.regions])
    assert np.array_equal(lut[labels], pr.group_map)                                 # the label map never changes
    assert pr.protect.shape == labels.shape


def test_a_real_small_part_stays():
    labels, albedo, shading, photo, fg = _scene(spot=(55.0, -20.0, -45.0))            # a blue indicator lens
    regions, groups, gm = _grouped(labels, albedo, _info(4))
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, diagnostics=True)
    assert 3 not in [m["region"] for m in pr.log]
    assert any(k["region"] == 3 for k in pr.kept)


@pytest.mark.parametrize("why,over,islands", [
    ("exempt source text", {"2": {"source": "text"}}, False),
    ("decal island", {}, True),
    ("detected part", {"2": {"source": "kind", "part_kind": "badge", "part_label": "Badge", "part_plural": "Badges",
                             "part_instance": 0}}, False),
])
def test_lettering_islands_and_detected_parts_are_never_pruned(why, over, islands):
    labels, albedo, shading, photo, fg = _scene()
    regions, groups, gm = _grouped(labels, albedo, _info(3, **over))
    isl = (labels == 2) if islands else None
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, isl, fg=fg, diagnostics=True)
    assert pr.log == [] and any(k["region"] == 2 and why in k["reason"] for k in pr.kept)
    if why == "detected part":
        assert pr.guarded and pr.guarded[0]["name"] == "Badge"


def test_a_locked_shadow_still_goes_but_a_locked_group_is_never_insignificant():
    labels, albedo, shading, photo, fg = _scene()
    regions, groups, gm = _grouped(labels, albedo, _info(3))
    sliver = _group_of(regions, groups)[2]
    sliver.locked = True                                                             # the material lock took it
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg)
    assert [m["rule"] for m in pr.log] == ["shadow"]
    # a sliver the decomposition split differently (darker albedo, brighter shading, the same photo):
    # only the significance test can move it
    labels, albedo, shading, photo, fg = _scene(sliver_albedo_scale=0.5, sliver_shading=1.8)
    regions, groups, gm = _grouped(labels, albedo, _info(3))
    params = junk.JunkParams(gradient=False)
    free = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, params=params)
    _group_of(regions, groups)[2].locked = True
    locked = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, params=params)
    assert [m["rule"] for m in free.log] == ["significance"] and locked.log == []


def test_a_shadow_inside_a_detected_part_goes_to_the_part():
    seat = np.s_[103:130, 60:80]                                                     # a detected seat below the sliver
    labels, albedo, shading, photo, fg = _scene(extra_regions=[(3, seat)])
    albedo[labels == 3] = imageio.lab_to_linear(np.array([[[20.0, 2.0, 2.0]]], np.float32))[0, 0]
    photo = imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))
    info = _info(4, **{"3": {"source": "kind", "part_kind": "seat", "part_label": "Seat", "part_plural": "Seats",
                             "part_instance": 0}})
    regions, groups, gm = _grouped(labels, albedo, info)
    seat_mask = np.zeros((H, W), bool)
    seat_mask[99:130, 60:80] = True                                                  # SAM's seat mask holds the sliver
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, part_masks=[seat_mask],
                         diagnostics=True)
    g = _group_of(pr.regions, pr.groups)
    assert g[2].id == g[3].id and g[3].part == "seat" and g[3].name == "Seat"
    assert pr.to_part and pr.to_part[0]["region"] == 2
    # without the mask the part group hosts nothing: the shadow joins the paint
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg)
    g = _group_of(pr.regions, pr.groups)
    assert g[2].id == g[1].id and g[3].part == "seat"


def test_backdrop_crumbs_join_their_backdrop_only_when_asked():
    labels, albedo, shading, photo, fg = _scene(extra_regions=[(3, np.s_[5:10, 5:15])])  # 50 px of the backdrop
    albedo[labels == 3] = imageio.lab_to_linear(np.array([[[75.0, 0.0, 0.0]]], np.float32))[0, 0]
    photo = imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))
    info = _info(4, **{"3": {"bg": 2}})
    regions, groups, gm = _grouped(labels, albedo, info)
    g = _group_of(regions, groups)
    assert g[3].is_background and g[3].id != g[0].id
    off = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg)
    assert 3 not in [m["region"] for m in off.log]
    on = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg,
                         params=junk.JunkParams(backdrop_crumbs=True))
    assert any(m["region"] == 3 and m["rule"] == "backdrop_crumbs" for m in on.log)
    g = _group_of(on.regions, on.groups)
    assert g[3].id == g[0].id and g[0].is_background


def test_refine_groups_runs_the_pruning_last_and_reports_it():
    labels, albedo, shading, photo, fg = _scene()
    regions, groups, gm = _grouped(labels, albedo, _info(3))

    def snap(image, labels, group_map, groups, protect=None, progress=None):
        return labels.copy(), "none"

    plain = refine.refine_groups(photo, albedo, labels, regions, groups, gm, snap)
    assert "pruned" not in plain.report
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm, snap, shading=shading, fg=fg,
                               prune=junk.DEFAULT)
    assert [m["rule"] for m in res.report["pruned"]] == ["shadow"] and len(res.groups) == len(plain.groups) - 1
    # a regroup with the analysis's options and its pruning gives the same groups back
    r2, g2, _, _ = refine.regroup_refined(photo, albedo, labels, res.origin, res.labels, res.regions, res.islands,
                                          None, 10.0, bg=np.array([2, 0, 0], np.int8), shading=shading, fg=fg > 0.5,
                                          prune=junk.DEFAULT)
    assert sorted(tuple(g.region_ids) for g in g2) == sorted(tuple(g.region_ids) for g in res.groups)


# ------------------------------------------------------------------ this round: the part rim, locks, users

def _rim_scene(rim_colour=(35.0, 32.0, 14.0), gap=0, off_matte=True):
    """A red detected part (3) on the grey backdrop beside the paint, and a 2 px band (2) just
    outside the part's SAM mask: the part's shadowed rim, darker and duller red, which the matte
    (tighter than SAM) calls backdrop. ``gap`` moves the band away from the part."""
    labels = np.zeros((H, W), np.int32)
    labels[PAINT] = 1
    labels[40:80, 150:190] = 0                                       # the part sits off the paint ...
    labels[45:75, 155:185] = 3                                       # ... a 900 px red foot
    labels[75 + gap:77 + gap, 155:185] = 2                           # the rim under it
    albedo = np.zeros((H, W, 3), np.float32)
    albedo[...] = imageio.lab_to_linear(np.array([[[60.0, 0.0, 0.0]]], np.float32))[0, 0]
    albedo[labels == 1] = _paint_lin()
    albedo[labels == 3] = imageio.lab_to_linear(np.array([[[48.0, 64.0, 41.0]]], np.float32))[0, 0]
    albedo[labels == 2] = imageio.lab_to_linear(np.array([[rim_colour]], np.float32))[0, 0]
    shading = np.full((H, W, 3), 0.9, np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))
    fg = (labels > 0).astype(np.float32)
    if off_matte:
        fg[labels == 2] = 0.0
    info = _info(4, **{"3": {"source": "kind", "part_kind": "foot", "part_label": "Foot", "part_plural": "Feet",
                             "part_instance": 0}})
    mask = labels == 3
    return labels, albedo, shading, photo, fg, info, mask


def test_the_shadowed_rim_of_a_detected_part_joins_the_part():
    labels, albedo, shading, photo, fg, info, mask = _rim_scene()
    regions, groups, gm = _grouped(labels, albedo, info)
    g = _group_of(regions, groups)
    assert g[2].id not in (g[1].id, g[3].id)                         # the clustering kept the rim apart
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, part_masks=[mask])
    assert [(m["region"], m["rule"]) for m in pr.log] == [(2, "part_rim")]
    g = _group_of(pr.regions, pr.groups)
    assert g[2].id == g[3].id and g[3].part == "foot"
    # without the part's mask the rim has no geometric evidence and, off the matte, stays
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg)
    assert pr.log == []


@pytest.mark.parametrize("rim_colour,gap,why", [
    ((35.0, 1.0, 1.0), 0, "lost the part's colour"),                # a neutral band beside a red part: a seal
    ((35.0, -20.0, 30.0), 0, "another hue"),                         # an olive band
    ((70.0, 40.0, 25.0), 0, "lighter than the part"),                # a lighter band
    ((35.0, 32.0, 14.0), 6, "no larger object neighbour"),           # a band 6 px away from the part
])
def test_what_is_no_part_rim(rim_colour, gap, why):
    labels, albedo, shading, photo, fg, info, mask = _rim_scene(rim_colour, gap)
    regions, groups, gm = _grouped(labels, albedo, info)
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, part_masks=[mask],
                         diagnostics=True)
    assert 2 not in [m["region"] for m in pr.log]
    reason = next(k["reason"] for k in pr.kept if k["region"] == 2)
    assert why in reason or "off the object" in reason


def test_a_locked_group_goes_by_the_gradient_test_but_never_by_significance():
    """The lock semantics: tests 0-2 (physical evidence) may move a material-locked group, test 3
    (a colour argument) never does."""
    labels, albedo, shading, photo, fg = _scene(sliver_albedo_scale=0.9, sliver_shading=0.9)      # a seam
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(3), delta_e=2.0)    # kept apart
    params = junk.JunkParams(shadow=False, significance=False)
    _group_of(regions, groups)[2].locked = True
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, params=params)
    assert [m["rule"] for m in pr.log] == ["gradient"]
    only_sig = junk.JunkParams(shadow=False, gradient=False)
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, params=only_sig)
    assert pr.log == []
    _group_of(regions, groups)[2].locked = False
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, params=only_sig)
    assert [m["rule"] for m in pr.log] == ["significance"]


def test_rule_2_never_sends_a_region_into_a_smaller_part():
    """Rule 2 moves a shadow inside a part's mask to the part, but, like every move, never into a
    smaller group: a 60 px sliver under the mask of a 20 px badge joins the paint."""
    badge = np.s_[98:100, 60:70]                                     # 20 px, just above the sliver
    labels, albedo, shading, photo, fg = _scene(extra_regions=[(3, badge)])
    albedo[labels == 3] = imageio.lab_to_linear(np.array([[[60.0, 5.0, 60.0]]], np.float32))[0, 0]
    photo = imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))
    info = _info(4, **{"3": {"source": "kind", "part_kind": "badge", "part_label": "Badge", "part_plural": "Badges",
                             "part_instance": 0}})
    regions, groups, gm = _grouped(labels, albedo, info)
    mask = np.zeros((H, W), bool)
    mask[97:104, 58:82] = True                                       # SAM's badge mask holds the sliver too
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg, part_masks=[mask])
    g = _group_of(pr.regions, pr.groups)
    assert g[2].id == g[1].id and g[3].part == "badge" and set(g[3].region_ids) == {3}


def test_a_region_the_user_flagged_joins_only_a_host_that_keeps_the_choice():
    """The user locked the sliver (its own group): the paint, unlocked, would lose the choice, so the
    sliver stays. Had the user locked the paint and the sliver with it (the lock of a group the
    pruning had put the sliver into), the paint ends locked and the sliver goes back into it."""
    labels, albedo, shading, photo, fg = _scene()
    regions, groups, gm = _grouped(labels, albedo, _info(3))
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg,
                         user_flags={"locked": {"2": True}}, diagnostics=True)
    assert pr.log == [] and any(k["region"] == 2 and "user" in k["reason"] for k in pr.kept)
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg,
                         user_flags={"locked": {"1": True, "2": True}})
    assert [(m["region"], m["rule"]) for m in pr.log] == [(2, "shadow")]
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg,
                         user_flags={"locked": {"2": False}})                  # unlocked by hand: the paint is unlocked too
    assert [m["region"] for m in pr.log] == [2]


def _crumb_scene(source="small"):
    """The paint with a gold group of six 3x3 crumbs (54 px, region 3): what was left of a small
    distinct part (the gold edging of a decal's letters) after the snap."""
    labels, albedo, shading, photo, fg = _scene()
    crumbs = np.zeros((H, W), bool)
    for k in range(6):
        crumbs[60 + 8 * k:63 + 8 * k, 130:133] = True
    labels[crumbs] = 3
    albedo[crumbs] = imageio.lab_to_linear(np.array([[[66.0, 23.0, 43.0]]], np.float32))[0, 0]
    photo = imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))
    info = _info(4, **{"3": {"source": source}})
    return labels, albedo, shading, photo, fg, info, crumbs


def test_a_group_of_crumbs_joins_the_group_owning_its_ring():
    labels, albedo, shading, photo, fg, info, crumbs = _crumb_scene()
    regions, groups, gm = _grouped(labels, albedo, info)
    isl = crumbs.copy()                                               # a small distinct part is an island
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, isl, fg=fg)
    assert [(m["region"], m["rule"]) for m in pr.log if m["region"] == 3] == [(3, "crumbs")]
    g = _group_of(pr.regions, pr.groups)
    assert g[3].id == g[1].id and not pr.islands[crumbs].any()        # into the paint, out of the islands
    # lettering is never crumbs, and a group with a piece above crumb_piece is not
    labels, albedo, shading, photo, fg, info, crumbs = _crumb_scene(source="text")
    regions, groups, gm = _grouped(labels, albedo, info)
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, crumbs.copy(), fg=fg)
    assert 3 not in [m["region"] for m in pr.log]
    labels, albedo, shading, photo, fg, info, crumbs = _crumb_scene()
    regions, groups, gm = _grouped(labels, albedo, info)
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, crumbs.copy(), fg=fg,
                         params=junk.JunkParams(crumb_pieces=7))
    assert 3 not in [m["region"] for m in pr.log]
    assert junk.LEGACY_OFF["crumb_group"] == 0                          # an older seed reruns without the rule


def test_vote_agrees_follows_the_majority_of_the_hosts_votes():
    host = ColorGroup(id=0, name="Red", albedo_lab=(45.0, 60.0, 45.0), albedo_hex="#b3261e", area=1000,
                               area_frac=0.5, region_ids=[1, 3], hue_family="red", locked=True)
    votes = junk.user_votes({"locked": {"2": False, "3": False}, "is_background": {"x": True}})
    assert votes == {"locked": {2: False, 3: False}, "is_background": {}}
    area = {1: 900, 2: 20, 3: 100}
    assert not junk.vote_agrees(votes, 2, [1, 3], area, host)       # 120 of 1020 px voted: the host's lock decides
    assert junk.vote_agrees(votes, 2, [1, 3], {1: 50, 2: 20, 3: 100}, host)   # the votes are the majority: unlocked
    assert junk.vote_agrees(votes, 1, [1, 3], area, host)           # a region without a vote always may


def test_the_shadow_test_has_no_highlight_branch():
    ok, why = junk.shadow_test(_pair(rho_photo=3.0, rho_shade=3.0))  # a much lighter candidate
    assert not ok and "step too small" in why
    assert not hasattr(junk.DEFAULT, "rho_light")


def test_the_diagnostics_are_left_empty_unless_asked_for():
    labels, albedo, shading, photo, fg = _scene(spot=(55.0, -20.0, -45.0))
    regions, groups, gm = _grouped(labels, albedo, _info(4))
    pr = junk.prune_junk(photo, albedo, shading, labels, regions, groups, gm, None, fg=fg)
    assert pr.kept == [] and pr.to_part == [] and pr.guarded == [] and pr.log
